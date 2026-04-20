import h5wasm from 'https://cdn.jsdelivr.net/npm/h5wasm@0.10.1/dist/esm/hdf5_hl.js';

let mod, segF, instF, sensorF, lablF, optF, maxTime, nEvents, nVolumes, nPlanes;
let hasLabl = false;
let hasOptical = false;
let isPixel = false;     // pixel vs wire readout
let numPy = 0, numPz = 0; // pixel grid dimensions (when isPixel)
let velocityMmUs = 1.6; // default, read from config
let volAnodes = [];      // x_anode per volume in mm
let volDriftDirs = [];   // drift direction per volume
const WIRE_LABELS = ['U', 'V', 'Y'];
const PIXEL_LABELS = ['Y-T', 'Z-T', 'Y-Z'];

function readAttr(grp, name) {
  const a = grp.attrs[name];
  if (!a) return undefined;
  const v = a.value;
  return typeof v === 'bigint' ? Number(v) : v;
}

function readF16(ds) {
  const meta = ds.metadata, nEl = meta.total_size, nb = meta.size * nEl;
  const ptr = mod._malloc(nb);
  mod.get_dataset_data(ds.file_id, ds.path, null, null, null, BigInt(ptr));
  const raw = new Uint8Array(mod.HEAPU8.buffer, ptr, nb).slice();
  mod._free(ptr);
  const dv = new DataView(raw.buffer), out = new Float32Array(nEl);
  for (let i = 0; i < nEl; i++) {
    const h = dv.getUint16(i * 2, true), s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (e === 0) out[i] = (s ? -1 : 1) * (m / 1024) * Math.pow(2, -14);
    else if (e === 31) out[i] = m === 0 ? (s ? -Infinity : Infinity) : NaN;
    else out[i] = (s ? -1 : 1) * (1 + m / 1024) * Math.pow(2, e - 15);
  }
  return out;
}

function mountUrl(url, wasmName) {
  const xhr = new XMLHttpRequest();
  xhr.open('HEAD', url, false);
  xhr.send();
  const fileSize = parseInt(xhr.getResponseHeader('Content-Length'));
  if (!fileSize || fileSize <= 0) throw new Error('Cannot get size for ' + url);
  const node = mod.FS.createFile('/', wasmName, {}, true, false);
  node.usedBytes = fileSize;
  Object.defineProperty(node, 'size', { get: () => fileSize });
  let cache = null, cacheStart = 0, cacheEnd = 0;
  const BLOCK = 1024 * 1024;
  node.stream_ops = {
    read(s, buf, off, len, pos) {
      if (cache && pos >= cacheStart && pos + len <= cacheEnd) {
        const co = pos - cacheStart;
        for (let i = 0; i < len; i++) buf[off + i] = cache[co + i];
        return len;
      }
      const fetchEnd = Math.min(pos + Math.max(len, BLOCK), fileSize) - 1;
      const r = new XMLHttpRequest();
      r.open('GET', url, false);
      r.responseType = 'arraybuffer';
      r.setRequestHeader('Range', 'bytes=' + pos + '-' + fetchEnd);
      r.send();
      cache = new Uint8Array(r.response);
      cacheStart = pos;
      cacheEnd = pos + cache.length;
      const n = Math.min(len, cache.length);
      for (let i = 0; i < n; i++) buf[off + i] = cache[i];
      return n;
    },
    llseek(s, off, w) { return w === 1 ? s.position + off : w === 2 ? fileSize + off : off; },
  };
}

function aggregateDisplay(pk, ch, mod2) {
  // mod2 = second dimension size (maxTime for wire/Y-T/Z-T, numPz for Y-Z)
  const n = pk.length, idx = Array.from({length: n}, (_, i) => i);
  idx.sort((a, b) => pk[a] - pk[b]);
  const uPk = [], uCh = [];
  let cur = -1, sum = 0;
  for (const i of idx) {
    if (pk[i] !== cur) { if (cur >= 0) { uPk.push(cur); uCh.push(sum); } cur = pk[i]; sum = 0; }
    sum += ch[i];
  }
  if (cur >= 0) { uPk.push(cur); uCh.push(sum); }
  const nd = uPk.length;
  const w = new Int32Array(nd), t = new Int32Array(nd), c = new Float32Array(nd);
  for (let i = 0; i < nd; i++) { w[i] = Math.floor(uPk[i] / mod2); t[i] = uPk[i] % mod2; c[i] = uCh[i]; }
  return { w, t, c, n: nd };
}

function decodeEvent(idx) {
  const key = 'event_' + String(idx).padStart(3, '0');
  const sEvt = segF.get(key);
  const cEvt = instF.get(key);
  const lEvt = hasLabl ? lablF.get(key) : null;
  const srcIdx = readAttr(sEvt, 'source_event_idx') || idx;
  const nVol = readAttr(sEvt, 'n_volumes') || nVolumes;

  const volumes = [];
  for (let v = 0; v < nVol; v++) {
    const vKey = 'volume_' + v;
    const vg = sEvt.get(vKey);
    if (!vg) { volumes.push({ n: 0, planes: {} }); continue; }

    const n = readAttr(vg, 'n_actual') || 0;
    if (n === 0) { volumes.push({ n: 0, planes: {} }); continue; }

    const posStep = readAttr(vg, 'pos_step_mm');
    const origin = [readAttr(vg, 'pos_origin_x'), readAttr(vg, 'pos_origin_y'), readAttr(vg, 'pos_origin_z')];
    const posRaw = vg.get('positions').value;
    const pos = new Float32Array(n * 3);
    for (let i = 0; i < n * 3; i++) pos[i] = posRaw[i] * posStep + origin[i % 3];

    const de = readF16(vg.get('de'));

    // Per-volume per-instance decomposition (inst owns segment_to_group)
    const volPlanes = {};
    const iVol = cEvt.get(vKey);
    let gids;
    if (iVol && iVol.get('segment_to_group')) {
      gids = new Int32Array(iVol.get('segment_to_group').value);
    } else {
      gids = new Int32Array(n);  // no inst data: zeros (highlighting will no-op)
    }

    // Per-deposit labels via labl (optional). labl carries
    // segment_to_track (N,) FK + per-unique-track dimension table
    // (track_ids, track_pdg, track_interaction, track_ancestor).
    let tids = null, pdg = null, ancTids = null, intIds = null;
    if (lEvt) {
      const lVol = lEvt.get(vKey);
      if (lVol && lVol.get('segment_to_track')) {
        tids = new Int32Array(lVol.get('segment_to_track').value);
        const trackIds = lVol.get('track_ids') ? new Int32Array(lVol.get('track_ids').value) : null;
        if (trackIds && trackIds.length > 0) {
          // Build track_id -> dim-table index map (T is small: ~few thousand)
          const tidToRow = new Map();
          for (let r = 0; r < trackIds.length; r++) tidToRow.set(trackIds[r], r);
          const trkPdg = lVol.get('track_pdg') ? new Int32Array(lVol.get('track_pdg').value) : null;
          const trkInt = lVol.get('track_interaction') ? new Int32Array(lVol.get('track_interaction').value) : null;
          const trkAnc = lVol.get('track_ancestor') ? new Int32Array(lVol.get('track_ancestor').value) : null;
          if (trkPdg) {
            pdg = new Int32Array(n);
            for (let i = 0; i < n; i++) { const r = tidToRow.get(tids[i]); if (r !== undefined) pdg[i] = trkPdg[r]; }
          }
          if (trkAnc) {
            ancTids = new Int32Array(n);
            for (let i = 0; i < n; i++) { const r = tidToRow.get(tids[i]); if (r !== undefined) ancTids[i] = trkAnc[r]; }
          }
          if (trkInt) {
            intIds = new Int32Array(n);
            for (let i = 0; i < n; i++) { const r = tidToRow.get(tids[i]); if (r !== undefined) intIds[i] = trkInt[r]; }
          }
        }
      }
    }

    // t0_us (float16) and drift arrival time
    let t0 = null, arrivalTime = null;
    if (vg.get('t0_us')) {
      t0 = readF16(vg.get('t0_us'));
      arrivalTime = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const driftDist = Math.abs(pos[i * 3] - volAnodes[v]);
        arrivalTime[i] = t0[i] + driftDist / velocityMmUs;
      }
    }
    if (iVol) {
      if (isPixel) {
        // Pixel mode: decode (py, pz, time) voxels → 3 projected planes
        const g = iVol.get('Pixel');
        if (g && g.get('group_ids')) {
          const grpIds = new Int32Array(g.get('group_ids').value);
          const grpSz = new Uint8Array(g.get('group_sizes').value);
          const cpy = new Int16Array(g.get('center_py').value);
          const cpz = new Int16Array(g.get('center_pz').value);
          const ct = new Int16Array(g.get('center_times').value);
          const pc = new Float32Array(g.get('peak_charges').value);
          const dpy = new Int8Array(g.get('delta_py').value);
          const dpz = new Int8Array(g.get('delta_pz').value);
          const dti = new Int8Array(g.get('delta_times').value);
          const cu16 = new Uint16Array(g.get('charges_u16').value);
          const nG = grpIds.length;
          let nEnt = 0; for (let i = 0; i < nG; i++) nEnt += grpSz[i];

          // Decode all voxels
          const allPy = new Int32Array(nEnt), allPz = new Int32Array(nEnt);
          const allT = new Int32Array(nEnt), allGid = new Int32Array(nEnt);
          const allCh = new Float32Array(nEnt);
          let s = 0;
          for (let i = 0; i < nG; i++) {
            const sz = grpSz[i];
            for (let j = 0; j < sz; j++) {
              allPy[s+j] = cpy[i] + dpy[s+j];
              allPz[s+j] = cpz[i] + dpz[s+j];
              allT[s+j] = ct[i] + dti[s+j];
              allGid[s+j] = grpIds[i];
              allCh[s+j] = pc[i] * cu16[s+j] / 65535;
            }
            s += sz;
          }

          // Project into 3 planes
          // Y-T: pk = py * maxTime + time
          const pkYT = new Int32Array(nEnt);
          for (let i = 0; i < nEnt; i++) pkYT[i] = allPy[i] * maxTime + allT[i];
          const dispYT = aggregateDisplay(pkYT, allCh, maxTime);
          volPlanes['Y-T'] = { corrPK: pkYT, corrGID: allGid, corrCH: allCh, nCorr: nEnt,
            dispW: dispYT.w, dispT: dispYT.t, dispCH: dispYT.c, nDisp: dispYT.n };

          // Z-T: pk = pz * maxTime + time
          const pkZT = new Int32Array(nEnt);
          for (let i = 0; i < nEnt; i++) pkZT[i] = allPz[i] * maxTime + allT[i];
          const dispZT = aggregateDisplay(pkZT, allCh, maxTime);
          volPlanes['Z-T'] = { corrPK: pkZT, corrGID: allGid, corrCH: allCh, nCorr: nEnt,
            dispW: dispZT.w, dispT: dispZT.t, dispCH: dispZT.c, nDisp: dispZT.n };

          // Y-Z: pk = py * numPz + pz
          const pkYZ = new Int32Array(nEnt);
          for (let i = 0; i < nEnt; i++) pkYZ[i] = allPy[i] * numPz + allPz[i];
          const dispYZ = aggregateDisplay(pkYZ, allCh, numPz);
          volPlanes['Y-Z'] = { corrPK: pkYZ, corrGID: allGid, corrCH: allCh, nCorr: nEnt,
            dispW: dispYZ.w, dispT: dispYZ.t, dispCH: dispYZ.c, nDisp: dispYZ.n };
        }
      } else {
        // Wire mode: decode per-plane (wire, time) per-instance entries
        for (let p = 0; p < nPlanes; p++) {
          const pl = WIRE_LABELS[p];
          const g = iVol.get(pl);
          if (!g || !g.get('group_ids')) continue;

          const grpIds = new Int32Array(g.get('group_ids').value);
          const grpSz = new Uint8Array(g.get('group_sizes').value);
          const cw = new Int16Array(g.get('center_wires').value);
          const ct = new Int16Array(g.get('center_times').value);
          const pc = new Float32Array(g.get('peak_charges').value);
          const dwi = new Int8Array(g.get('delta_wires').value);
          const dti = new Int8Array(g.get('delta_times').value);
          const cu16 = new Uint16Array(g.get('charges_u16').value);
          const nG = grpIds.length;
          let nEnt = 0; for (let i = 0; i < nG; i++) nEnt += grpSz[i];
          const pk = new Int32Array(nEnt), gid = new Int32Array(nEnt), ch = new Float32Array(nEnt);
          let s = 0;
          for (let i = 0; i < nG; i++) {
            const sz = grpSz[i];
            for (let j = 0; j < sz; j++) {
              pk[s+j] = (cw[i] + dwi[s+j]) * maxTime + (ct[i] + dti[s+j]);
              gid[s+j] = grpIds[i];
              ch[s+j] = pc[i] * cu16[s+j] / 65535;
            }
            s += sz;
          }
          const disp = aggregateDisplay(pk, ch, maxTime);
          volPlanes[pl] = { corrPK: pk, corrGID: gid, corrCH: ch, nCorr: nEnt,
            dispW: disp.w, dispT: disp.t, dispCH: disp.c, nDisp: disp.n };
        }
      }
    }

    volumes.push({ pos, de, tids, gids, pdg, ancTids, intIds, t0, arrivalTime, n, planes: volPlanes });
  }

  return { volumes, config: { event_idx: srcIdx, max_time: maxTime, n_volumes: nVol } };
}

function decodeSensor(idx) {
  const key = 'event_' + String(idx).padStart(3, '0');
  const rEvt = sensorF.get(key);
  const nVol = readAttr(rEvt, 'n_volumes') || nVolumes;
  const labels = isPixel ? PIXEL_LABELS : WIRE_LABELS;

  const sensorVols = [];
  const norms = {};
  for (let v = 0; v < nVol; v++) {
    const vKey = 'volume_' + v;
    const vg = rEvt.get(vKey);
    const volPlanes = {};
    if (vg) {
      if (isPixel) {
        // Pixel mode: read (py, pz, time, value) and project into 3 2D planes
        const g = vg.get('Pixel');
        if (g && g.get('delta_py')) {
          const dpy = new Int16Array(g.get('delta_py').value);
          const dpz = new Int16Array(g.get('delta_pz').value);
          const dt = new Int16Array(g.get('delta_time').value);
          const rawVals = g.get('values').value;
          const nn = rawVals.length;
          let vals;
          if (rawVals instanceof Uint16Array) {
            const ped = readAttr(g, 'pedestal') || 0;
            vals = new Float32Array(nn);
            for (let i = 0; i < nn; i++) vals[i] = rawVals[i] - ped;
          } else {
            vals = new Float32Array(rawVals);
          }
          const allPy = new Int32Array(nn), allPz = new Int32Array(nn), allT = new Int32Array(nn);
          let cpy = readAttr(g, 'py_start') || 0, cpz = readAttr(g, 'pz_start') || 0, ct = readAttr(g, 'time_start') || 0;
          for (let i = 0; i < nn; i++) { cpy += dpy[i]; cpz += dpz[i]; ct += dt[i]; allPy[i] = cpy; allPz[i] = cpz; allT[i] = ct; }

          // Y-T projection: aggregate over pz
          const ytMap = new Map();
          for (let i = 0; i < nn; i++) {
            const k = allPy[i] * maxTime + allT[i];
            ytMap.set(k, (ytMap.get(k) || 0) + vals[i]);
          }
          const ytN = ytMap.size;
          const ytW = new Int32Array(ytN), ytT = new Int32Array(ytN), ytV = new Float32Array(ytN);
          let yi = 0;
          for (const [k, val] of ytMap) { ytW[yi] = Math.floor(k / maxTime); ytT[yi] = k % maxTime; ytV[yi] = val; yi++; }
          volPlanes['Y-T'] = { wires: ytW, times: ytT, values: ytV, n: ytN };

          // Z-T projection: aggregate over py
          const ztMap = new Map();
          for (let i = 0; i < nn; i++) {
            const k = allPz[i] * maxTime + allT[i];
            ztMap.set(k, (ztMap.get(k) || 0) + vals[i]);
          }
          const ztN = ztMap.size;
          const ztW = new Int32Array(ztN), ztT = new Int32Array(ztN), ztV = new Float32Array(ztN);
          let zi = 0;
          for (const [k, val] of ztMap) { ztW[zi] = Math.floor(k / maxTime); ztT[zi] = k % maxTime; ztV[zi] = val; zi++; }
          volPlanes['Z-T'] = { wires: ztW, times: ztT, values: ztV, n: ztN };

          // Y-Z projection: aggregate over time
          const yzMap = new Map();
          for (let i = 0; i < nn; i++) {
            const k = allPy[i] * numPz + allPz[i];
            yzMap.set(k, (yzMap.get(k) || 0) + vals[i]);
          }
          const yzN = yzMap.size;
          const yzW = new Int32Array(yzN), yzT = new Int32Array(yzN), yzV = new Float32Array(yzN);
          let yzi = 0;
          for (const [k, val] of yzMap) { yzW[yzi] = Math.floor(k / numPz); yzT[yzi] = k % numPz; yzV[yzi] = val; yzi++; }
          volPlanes['Y-Z'] = { wires: yzW, times: yzT, values: yzV, n: yzN };
        }
      } else {
        // Wire mode
        for (let p = 0; p < nPlanes; p++) {
          const pl = WIRE_LABELS[p];
          const g = vg.get(pl);
          if (!g || !g.get('delta_wire')) continue;
          const dw = new Int16Array(g.get('delta_wire').value);
          const dt = new Int16Array(g.get('delta_time').value);
          const rawVals = g.get('values').value;
          const nn = rawVals.length;
          let vals;
          if (rawVals instanceof Uint16Array) {
            const ped = readAttr(g, 'pedestal') || 0;
            vals = new Float32Array(nn);
            for (let i = 0; i < nn; i++) vals[i] = rawVals[i] - ped;
          } else {
            vals = new Float32Array(rawVals);
          }
          const wires = new Int32Array(nn), times = new Int32Array(nn);
          let cw2 = readAttr(g, 'wire_start'), ct2 = readAttr(g, 'time_start');
          for (let i = 0; i < nn; i++) { cw2 += dw[i]; ct2 += dt[i]; wires[i] = cw2; times[i] = ct2; }
          volPlanes[pl] = { wires, times, values: vals, n: nn };
        }
      }
    }
    sensorVols.push(volPlanes);
  }

  // Per-plane-type norms across all volumes
  for (let p = 0; p < labels.length; p++) {
    const pl = labels[p];
    let mn = Infinity, mx = -Infinity;
    for (let v = 0; v < nVol; v++) {
      const d = sensorVols[v][pl]; if (!d) continue;
      for (let i = 0; i < d.n; i++) { if (d.values[i] < mn) mn = d.values[i]; if (d.values[i] > mx) mx = d.values[i]; }
    }
    if (mn === Infinity) { mn = -25; mx = 25; }
    // Symmetric norms for Y collection plane (wire mode only)
    if (!isPixel && pl === 'Y') { const m = Math.max(Math.abs(mn), Math.abs(mx)); mn = -m; mx = m; }
    norms[pl] = [mn, mx];
  }

  return { sensorVols, sensorNorms: norms };
}

function decodeLight(idx, activityThresh, gapNs) {
  if (!optF) return { stitched: null };
  const key = 'event_' + String(idx).padStart(3, '0');
  const evt = optF.get(key);
  if (!evt) return { stitched: null };

  const oc = optF.get('config');
  const tickNs = readAttr(oc, 'tick_ns') || 1.0;
  const nCh = readAttr(oc, 'n_channels') || 162;
  const pedestal = readAttr(oc, 'pedestal') || 0;
  const ACTIVITY_THRESH = activityThresh || 100;
  const GAP_NS = gapNs || 5000;
  const PAD_NS = 0;

  // Discover labels
  const labelKeys = evt.keys().filter(k => k.startsWith('label_'));
  const labels = labelKeys.map(k => parseInt(k.split('_')[1])).sort((a, b) => a - b);

  // Load all label chunks
  const allChunks = {}; // labelId -> [{chId, t0, wf}]
  const pePerLabel = {}; // labelId -> {total, perChannel}
  for (const lid of labels) {
    const g = evt.get('label_' + lid);
    if (!g) continue;
    const adc = new Uint16Array(g.get('adc').value);
    const offsets = new BigInt64Array(g.get('offsets').value);
    const t0arr = new Float32Array(g.get('t0_ns').value);
    const pmtIds = new Int32Array(g.get('pmt_id').value);
    const pe = g.get('pe_counts') ? new Int32Array(g.get('pe_counts').value) : new Int32Array(nCh);

    const chunks = [];
    for (let i = 0; i < pmtIds.length; i++) {
      const s = Number(offsets[i]), e = Number(offsets[i + 1]);
      const wf = new Float32Array(e - s);
      for (let j = 0; j < wf.length; j++) wf[j] = adc[s + j] - pedestal;
      chunks.push({ chId: pmtIds[i], t0: t0arr[i], wf });
    }
    allChunks[lid] = chunks;
    pePerLabel[lid] = { total: 0, perChannel: Array.from(pe) };
    for (let i = 0; i < pe.length; i++) pePerLabel[lid].total += pe[i];
  }

  // Find activity regions (skip label -1)
  const activeLabels = labels.filter(l => l !== -1);
  const actTimes = [];
  for (const lid of activeLabels) {
    for (const { t0, wf } of allChunks[lid]) {
      for (let i = 0; i < wf.length; i++) {
        if (Math.abs(wf[i]) > ACTIVITY_THRESH) actTimes.push(t0 + i * tickNs);
      }
    }
  }
  if (actTimes.length === 0) return { stitched: null };

  actTimes.sort((a, b) => a - b);
  // Split into regions by gap
  const regions = [];
  let rStart = actTimes[0];
  let rEnd = actTimes[0];
  for (let i = 1; i < actTimes.length; i++) {
    if (actTimes[i] - rEnd > GAP_NS) {
      regions.push([rStart - PAD_NS, rEnd + PAD_NS]);
      rStart = actTimes[i];
    }
    rEnd = actTimes[i];
  }
  regions.push([rStart - PAD_NS, rEnd + PAD_NS]);

  // Build stitched array + interaction map
  let totalBins = 0;
  const regionInfo = [];
  for (const [rs, re] of regions) {
    const nb = Math.ceil((re - rs) / tickNs);
    regionInfo.push({ tStart: rs, tEnd: re, nBins: nb });
    totalBins += nb;
  }

  const stitched = new Float32Array(nCh * totalBins);
  const intMap = new Int32Array(nCh * totalBins).fill(-1);
  // Track per-interaction absolute contribution per pixel for dominant detection
  // Use compact per-interaction arrays but only write non-zero pixels
  const intAbsMax = new Float32Array(nCh * totalBins); // best abs contribution seen

  let binOff = 0;
  for (let ri = 0; ri < regions.length; ri++) {
    const { tStart, tEnd, nBins } = regionInfo[ri];
    for (const lid of activeLabels) {
      for (const { chId, t0, wf } of allChunks[lid]) {
        const wfEnd = t0 + wf.length * tickNs;
        const ovStart = Math.max(tStart, t0);
        const ovEnd = Math.min(tEnd, wfEnd);
        if (ovStart >= ovEnd) continue;
        const wfI0 = Math.floor((ovStart - t0) / tickNs);
        const arrI0 = Math.floor((ovStart - tStart) / tickNs);
        const n = Math.min(Math.floor((ovEnd - ovStart) / tickNs), nBins - arrI0);
        for (let j = 0; j < n; j++) {
          const idx2 = chId * totalBins + binOff + arrI0 + j;
          stitched[idx2] += wf[wfI0 + j];
          const av = Math.abs(wf[wfI0 + j]);
          if (av > intAbsMax[idx2]) { intAbsMax[idx2] = av; intMap[idx2] = lid; }
        }
      }
    }
    binOff += nBins;
  }

  // For multi-int hover: store per-interaction chunk metadata (compact)
  // so the viewer can compute fractions on demand for a single pixel
  const chunkMeta = {}; // lid -> [{chId, t0, wfLen}]
  for (const lid of activeLabels) {
    chunkMeta[lid] = allChunks[lid].map(c => ({ chId: c.chId, t0: c.t0, wfLen: c.wf.length }));
  }
  // Multi-int breakdown computed on demand at hover time using chunkMeta + regionInfo

  // Region bounds (cumulative bin positions)
  const regionBounds = new Int32Array(regions.length + 1);
  let cum = 0;
  for (let i = 0; i < regionInfo.length; i++) {
    regionBounds[i] = cum;
    cum += regionInfo[i].nBins;
  }
  regionBounds[regions.length] = cum;

  // Which interactions are in each region
  const regionInteractions = regionInfo.map((ri, idx) => {
    const ids = new Set();
    for (const lid of activeLabels) {
      const info = allChunks[lid];
      for (const { t0, wf } of info) {
        const wfEnd = t0 + wf.length * tickNs;
        if (t0 < ri.tEnd && wfEnd > ri.tStart) { ids.add(lid); break; }
      }
    }
    return { tStartUs: ri.tStart / 1000, tEndUs: ri.tEnd / 1000, ids: [...ids] };
  });

  return {
    stitched, intMap, regionBounds,
    chunkMeta, regionInfo: regionInfo.map(r => ({ tStart: r.tStart, tEnd: r.tEnd, nBins: r.nBins })),
    nChannels: nCh, totalBins,
    regionInteractions,
    pePerLabel,
    activeLabels,
  };
}

function collectTransfers(obj) {
  const t = [];
  (function walk(o) {
    if (o instanceof ArrayBuffer) t.push(o);
    else if (ArrayBuffer.isView(o) && !t.includes(o.buffer)) t.push(o.buffer);
    else if (typeof o === 'object' && o !== null) for (const v of Object.values(o)) walk(v);
  })(obj);
  return t;
}

self.onmessage = async function(e) {
  const { action } = e.data;
  if (action === 'init') {
    mod = await h5wasm.ready;
    const base = e.data.base;
    const manifest = e.data.manifest;
    mountUrl(base + '/' + manifest.seg, 'seg.h5');
    mountUrl(base + '/' + manifest.inst, 'inst.h5');
    mountUrl(base + '/' + manifest.sensor, 'sensor.h5');
    segF = new h5wasm.File('/seg.h5', 'r');
    instF = new h5wasm.File('/inst.h5', 'r');
    sensorF = new h5wasm.File('/sensor.h5', 'r');

    // Optional labl file (per-track label dimension table)
    if (manifest.labl) {
      mountUrl(base + '/' + manifest.labl, 'labl.h5');
      lablF = new h5wasm.File('/labl.h5', 'r');
      hasLabl = true;
    }

    // Verify run_id consistency across files
    const segRid = readAttr(segF.get('config'), 'run_id');
    const instRid = readAttr(instF.get('config'), 'run_id');
    const sensorRid = readAttr(sensorF.get('config'), 'run_id');
    if (segRid != null && instRid != null && sensorRid != null) {
      if (segRid !== instRid || segRid !== sensorRid) {
        throw new Error(`run_id mismatch: seg=${segRid}, inst=${instRid}, sensor=${sensorRid}`);
      }
    }
    if (hasLabl) {
      const lablRid = readAttr(lablF.get('config'), 'run_id');
      if (segRid != null && lablRid != null && segRid !== lablRid) {
        throw new Error(`run_id mismatch: seg=${segRid}, labl=${lablRid}`);
      }
    }

    maxTime = readAttr(instF.get('config'), 'num_time_steps');
    nEvents = readAttr(segF.get('config'), 'n_events');
    nVolumes = readAttr(segF.get('config'), 'n_volumes') || 2;
    const nwRaw = instF.get('config/num_wires').value;
    nPlanes = nwRaw.length / nVolumes;
    isPixel = (nPlanes === 0);

    const numWires = [];
    if (isPixel) {
      // Infer pixel grid from first event's sensor data
      nPlanes = 3; // 3 projections: Y-T, Z-T, Y-Z
      const rEvt = sensorF.get('event_000');
      let maxPyVal = 100, maxPzVal = 100;
      for (let v = 0; v < nVolumes; v++) {
        const pg = rEvt.get('volume_' + v + '/Pixel');
        if (!pg) continue;
        const dpy = pg.get('delta_py').value;
        const dpz = pg.get('delta_pz').value;
        let pyS = readAttr(pg, 'py_start') || 0, pzS = readAttr(pg, 'pz_start') || 0;
        let pyMax = pyS, pzMax = pzS;
        for (let i = 0; i < dpy.length; i++) { pyS += dpy[i]; pzS += dpz[i]; if (pyS > pyMax) pyMax = pyS; if (pzS > pzMax) pzMax = pzS; }
        if (pyMax + 1 > maxPyVal) maxPyVal = pyMax + 1;
        if (pzMax + 1 > maxPzVal) maxPzVal = pzMax + 1;
      }
      // Round up to nearest 100 (pixel grids are typically round numbers)
      numPy = Math.ceil(maxPyVal / 100) * 100;
      numPz = Math.ceil(maxPzVal / 100) * 100;
      for (let v = 0; v < nVolumes; v++) numWires.push([numPy, numPz, numPy]);
    } else {
      for (let v = 0; v < nVolumes; v++) {
        const row = [];
        for (let p = 0; p < nPlanes; p++) row.push(nwRaw[v * nPlanes + p]);
        numWires.push(row);
      }
    }
    // Read volume ranges (n_volumes, 3, 2) in mm
    let volRanges = null;
    const vrDs = segF.get('config/volume_ranges');
    if (vrDs) {
      const vr = new Float32Array(vrDs.value);
      volRanges = [];
      for (let v = 0; v < nVolumes; v++) {
        const r = [];
        for (let ax = 0; ax < 3; ax++) r.push([vr[(v*3+ax)*2], vr[(v*3+ax)*2+1]]);
        volRanges.push(r);
      }
    }
    // Read drift velocity from sensor config
    const sensorCfg = sensorF.get('config');
    const vCmUs = readAttr(sensorCfg, 'velocity_cm_us') || 0.16;
    velocityMmUs = vCmUs * 10;
    const timeStepUs = readAttr(sensorCfg, 'time_step_us') || 0.5;

    // Compute per-volume anode positions and drift directions from ranges
    // Vol with x_max <= 0: anode at x_min (drift toward -x), drift_dir = -1
    // Vol with x_min >= 0: anode at x_max (drift toward +x), drift_dir = +1
    volAnodes = []; volDriftDirs = [];
    if (volRanges) {
      for (let v = 0; v < nVolumes; v++) {
        const xMin = volRanges[v][0][0], xMax = volRanges[v][0][1];
        if (xMax <= 0) { volAnodes.push(xMin); volDriftDirs.push(-1); }
        else { volAnodes.push(xMax); volDriftDirs.push(1); }
      }
    }

    // Mount optical file if present
    if (manifest.optical) {
      mountUrl(base + '/' + manifest.optical, 'opt.h5');
      optF = new h5wasm.File('/opt.h5', 'r');
      hasOptical = true;
    }

    let optConfig = null;
    if (hasOptical) {
      const oc = optF.get('config');
      const lkAttr = oc.attrs['label_key'];
      optConfig = {
        tickNs: readAttr(oc, 'tick_ns') || 1.0,
        nChannels: readAttr(oc, 'n_channels') || 162,
        pedestal: readAttr(oc, 'pedestal') || 0,
        nBits: readAttr(oc, 'n_bits') || 15,
        labelKey: lkAttr ? (typeof lkAttr === 'object' ? lkAttr.value : lkAttr) : 'interaction',
      };
    }

    const planeLabels = isPixel ? PIXEL_LABELS : WIRE_LABELS.slice(0, nPlanes);
    const readoutWindowUs = maxTime * timeStepUs;
    self.postMessage({ action: 'ready', nEvents, maxTime, nVolumes, numWires, planeLabels, volRanges,
      velocityMmUs, volAnodes, volDriftDirs, timeStepUs, readoutWindowUs,
      hasLabl, hasOptical, optConfig, isPixel, numPy, numPz });
  } else if (action === 'loadEvent') {
    const d = decodeEvent(e.data.idx);
    self.postMessage({ action: 'eventLoaded', ...d }, collectTransfers(d));
  } else if (action === 'loadSensor') {
    const d = decodeSensor(e.data.idx);
    self.postMessage({ action: 'sensorLoaded', ...d }, collectTransfers(d));
  } else if (action === 'loadLight') {
    const d = decodeLight(e.data.idx, e.data.activityThresh, e.data.gapNs);
    self.postMessage({ action: 'lightLoaded', ...d }, collectTransfers(d));
  }
};
