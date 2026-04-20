import { VS, FS } from './shaders.js';
import {
  WARM_STOPS, OBSIDIAN_STOPS, SEISMIC_STOPS, INFERNO_R_STOPS,
  warmRGB, obsidianRGB, seismicRGB, infernoRGB,
  hsl2rgb, parseHex, dbNorm, dbInverse
} from './colormaps.js';

// ============================================================
// GLOBALS
// ============================================================
let lightMode = false;
const BG_DARK = 0x080808, BG_LIGHT = 0xf0f0f0;
let BG = BG_DARK;
let D = null;
let isPixelMode = false;
let numPy = 0, numPz = 0;
let worker = null;
let nEvents = 0, nVolumes = 0, planeLabels = [];
let curEvent = 0;
let curVol = -1; // -1 = All, 0..N-1 = specific volume
let curColorMode = 'de'; // 'de' or 'label'
let curLabel = 'track'; // 'track','pdg','ancestor','interaction'
let curColor = 'de'; // effective: 'de' or curLabel value
let corrMode = true;
let autoRotate = true;
let selectedTrack = null;
let initCamPos = null, initCamTarget = null;
let topTracks = [];
let vizReady = false;

let curViewMode = 'hits';
let sensorVols = []; // [ {U:{wires,times,values,n},...}, ... ] per volume
let sensorLoadedFor = -1;
let sensorGamma = 0.2;
let sensorDeadband = 2.0;
let deEmphPow = 5.0;
let deEmphAmt = 0.75;

// Labels (per-track metadata: track/pdg/ancestor/interaction)
// Sourced from the optional labl/ file. When absent, label color modes
// and the LABEL UI are hidden; correspondence highlighting still works
// because gids come from inst/.
let hasLabl = false;

// Optical
let hasOptical = false;
let optConfig = null; // {tickNs, nChannels, pedestal, nBits}
let lightData = null; // {stitched, intMap, regionBounds, nChannels, totalBins, regionInteractions, pePerLabel, activeLabels}
let lightLoadedFor = -1;
let nPmtsPerSide = 81;
let optLabelKey = 'interaction'; // from optical config: 'interaction' or 'ancestor'
function getOptIds(vol){
  if(optLabelKey==='ancestor') return vol.ancTids;
  return vol.intIds;
}
function getOptId(vol,si){
  const ids=getOptIds(vol);
  return ids?ids[si]:-1;
}
let optActivityThresh = 50;  // ADC (~19 sigma above noise)
let optGapUs = 5.0;          // µs (min inter-interaction gap is ~9 µs)

// Drift animation
let driftMode = false;
let driftPlaying = false;
let simTime = 0;
let simTimeMin = 0, simTimeMax = 0;
let driftSpeed = 200;
let fadeDuration = 250;
let loopPauseMs = 1500;
let loopPaused = false, loopPauseStart = 0;
let lastFrameTime = 0;
let velocityMmUs = 1.6;
let volAnodes = [], volDriftDirs = [];
let timeStepUs = 0.5;
// Stored 3D arrays for color mode switching
let _3dPdg=null,_3dAnc=null,_3dInt=null,_3dTids=null;
let _3dTrkWt=null,_3dAncWt=null,_3dIntWt=null,_3dPdgWt=null;
let sensorNorms = {};

let volumes = []; // [ {pos,de,gids,n,planes:{U:{corrPK,...},...}, tids?,pdg?,ancTids?,intIds?}, ... ]
                  // gids comes from inst/segment_to_group; tids/pdg/ancTids/intIds
                  // are populated only when labl/ is present.
let volRanges = null; // [ [[xmin,xmax],[ymin,ymax],[zmin,zmax]], ... ] in mm

let renderer, scene, camera, controls;
let ptGeo, ptMat, ptMesh, boxMesh;
let anodeMesh = null, cathodeMesh = null;
let showVolBounds = true;
let showAnodeCathode = false;
let pickTarget, pickScene, pickMat, pickMesh;
let hlBuf;

// PDG code lookup
const PDG_NAMES=new Map([
  [11,'e\u207B'],[-11,'e\u207A'],[13,'\u03BC\u207B'],[-13,'\u03BC\u207A'],
  [211,'\u03C0\u207A'],[-211,'\u03C0\u207B'],[111,'\u03C0\u2070'],
  [2212,'p'],[2112,'n'],[22,'\u03B3'],
  [321,'K\u207A'],[-321,'K\u207B'],[310,'K\u2070_S'],[130,'K\u2070_L'],
  [3122,'\u039B'],[3222,'\u03A3\u207A'],[3112,'\u03A3\u207B'],
  [12,'\u03BD_e'],[-12,'\u03BD\u0305_e'],[14,'\u03BD_\u03BC'],[-14,'\u03BD\u0305_\u03BC'],
]);
function pdgName(c){
  if(PDG_NAMES.has(c)) return PDG_NAMES.get(c);
  if(Math.abs(c)>1e9) return 'nucleus';
  if(c===2147483647) return 'unknown';
  return 'PDG:'+c;
}
let c2d, ctx, offC, offCtx;
let margin = {t:22, r:42, b:46, l:48};

// Multi-panel system
let panels = []; // [{vol,plane,view,viewMax,dispColors,chargeMin,chargeMax,grpToPx,pkToDispIdx,pixelDomTrack,rect}]
let panelGrid = {rows:0, cols:0};
let expandedPanel = -1; // -1 = grid view, >=0 = index of expanded panel
let grpToSegPerVol = []; // per-volume: Map<groupId, [segIdx,...]>

let hovGrp = -1;
let pendingPick = null;
let isPan = false, panLast = {x:0,y:0}, panPanel = -1;
let is2dHover = false;

// Camera animation
let camAnim = null; // {startPos,startTarget,endPos,endTarget,startTime,duration}

// ============================================================
// COLORMAPS (stops + functions imported from colormaps.js)
// ============================================================
// Theme colors
function bgCol(){return lightMode?'#fafafa':'#080808';}
function axCol(){return lightMode?'#cccccc':'#2a2a2a';}
function txCol(){return lightMode?'#777777':'#555555';}
function sweepCol(){return lightMode?'rgba(250,250,250,0.88)':'rgba(8,8,8,0.85)';}

// Mode-aware colormap selectors
function sensorCmapRGB(t){return lightMode?seismicRGB(t):obsidianRGB(t);}
function hitsCmapRGB(t){return lightMode?infernoRGB(t):warmRGB(t);}

function getDispForPanel(p){
  if(!p) return null;
  if(curViewMode==='sensor'&&sensorVols[p.vol]&&sensorVols[p.vol][p.plane]){
    const rp=sensorVols[p.vol][p.plane];
    return{w:rp.wires,t:rp.times,v:rp.values,n:rp.n};
  }
  const pd=volumes[p.vol]&&volumes[p.vol].planes[p.plane];
  if(!pd) return null;
  return{w:pd.dispW,t:pd.dispT,v:pd.dispCH,n:pd.nDisp};
}
function curVolData(){ return curVol>=0&&volumes[curVol]?volumes[curVol]:null; }

function makeColormapTex(stops){
  const cv=document.createElement('canvas');cv.width=256;cv.height=1;
  const cx=cv.getContext('2d'),gr=cx.createLinearGradient(0,0,256,0);
  for(const[t,c] of stops) gr.addColorStop(t,c);
  cx.fillStyle=gr;cx.fillRect(0,0,256,1);
  const tx=new THREE.CanvasTexture(cv);
  tx.minFilter=THREE.LinearFilter;tx.magFilter=THREE.LinearFilter;
  return tx;
}

function makeTrackTex(){
  const cv=document.createElement('canvas');cv.width=256;cv.height=1;
  const cx=cv.getContext('2d'),id=cx.createImageData(256,1);
  for(let i=0;i<256;i++){
    const[r,g,b]=hsl2rgb(i/256,.78,.55);
    id.data[i*4]=r;id.data[i*4+1]=g;id.data[i*4+2]=b;id.data[i*4+3]=255;
  }
  cx.putImageData(id,0,0);
  const tx=new THREE.CanvasTexture(cv);
  tx.minFilter=THREE.NearestFilter;tx.magFilter=THREE.NearestFilter;
  return tx;
}

let warmTex, infernoTex, trackTex;
function deTex(){return lightMode?infernoTex:warmTex;}

// ============================================================
// DATA LOADING (via Worker)
// ============================================================
function updateDriftUI(){
  const btn=document.getElementById('driftPlayPause');
  btn.innerHTML=driftPlaying?'&#x23F8;':'&#x25B6;';
  btn.classList.toggle('active',driftPlaying);
  document.getElementById('driftScrubber').value=simTime;
  const tUs=simTime.toFixed(0);
  const tBin=Math.max(0,simTime/timeStepUs).toFixed(0);
  document.getElementById('driftTimeLabel').textContent=tUs+' \u03BCs (bin '+tBin+')';
}

function showOverlay(msg){
  document.getElementById('overlayMsg').textContent=msg||'Loading...';
  document.getElementById('overlay').classList.add('visible');
}
function hideOverlay(){document.getElementById('overlay').classList.remove('visible');}

function createWorker(){
  worker=new Worker('h5_worker.js',{type:'module'});
}

function workerCall(action, data){
  return new Promise((resolve)=>{
    const handler=(e)=>{
      worker.removeEventListener('message',handler);
      resolve(e.data);
    };
    worker.addEventListener('message',handler);
    worker.postMessage({action,...data});
  });
}

async function loadEvent(idx){
  showOverlay('Loading event '+idx+'...');
  const d=await workerCall('loadEvent',{idx});
  D.config.event_idx=d.config.event_idx;
  D.config.max_time=d.config.max_time;
  volumes=d.volumes;
  curEvent=idx;
  sensorLoadedFor=-1;
  document.getElementById('evInput').value=idx;
  selectedTrack=null;
  document.getElementById('catFilterSelect').value='all';
  document.getElementById('catFilterInput').value='';
  populateVolSelect();
  buildPanels();
  rebuildAllLookups();
  computeTopTracks();
  populateCatFilter();
  computePanelDomIds();
  buildPoints();
  if(driftMode){
    simTime=0;
    const scrub=document.getElementById('driftScrubber');
    scrub.min=0; scrub.max=simTimeMax; scrub.step=simTimeMax/1000;
    scrub.value=0;
    if(ptMat) ptMat.uniforms.simTime.value=0;
    loopPaused=false;
    updateDriftUI();
  }
  await loadSensor(idx);
  lightLoadedFor=-1; // invalidate cached light, load on demand
  if(curViewMode==='optical'){
    await loadLight(idx);
    if(lightData){ renderOptical(); renderOpticalFrame(null); }
  } else {
    precomputeAllPanelColors();
    resetPanelViews();
    render2DBase();
    render2DFrame(null);
  }
  hideOverlay();
  setStatus('Event '+idx+' (source '+d.config.event_idx+')');
}

async function loadSensor(idx){
  if(sensorLoadedFor===idx) return;
  const d=await workerCall('loadSensor',{idx});
  sensorVols=d.sensorVols;
  sensorNorms=d.sensorNorms;
  sensorLoadedFor=idx;
}

async function loadLight(idx,forceReload){
  if(!hasOptical) return;
  if(!forceReload&&lightLoadedFor===idx) return;
  const d=await workerCall('loadLight',{idx,activityThresh:optActivityThresh,gapNs:optGapUs*1000});
  lightData=d.stitched?d:null;
  lightLoadedFor=idx;
}

// ============================================================
// LOOKUP BUILDERS
// ============================================================
function buildGroupToSeg(sd){
  const m=new Map();
  for(let i=0;i<sd.n;i++){
    const g=sd.gids[i]; if(g===0) continue;
    if(!m.has(g)) m.set(g,[]);
    m.get(g).push(i);
  }
  return m;
}

function buildPlaneLookups(pd, planeLabel){
  const p2g=new Map(), g2p=new Map();
  for(let i=0;i<pd.nCorr;i++){
    const pk=pd.corrPK[i], gid=pd.corrGID[i], ch=pd.corrCH[i];
    if(!p2g.has(pk)) p2g.set(pk,[]);
    p2g.get(pk).push({gid,ch});
    if(!g2p.has(gid)) g2p.set(gid,[]);
    g2p.get(gid).push({pk,ch});
  }
  const pk2di=new Map();
  const mod2=(isPixelMode&&planeLabel==='Y-Z')?numPz:D.config.max_time;
  for(let i=0;i<pd.nDisp;i++) pk2di.set(pd.dispW[i]*mod2+pd.dispT[i], i);
  return {p2g, g2p, pk2di};
}

function rebuildAllLookups(){
  // Build per-volume grpToSeg
  grpToSegPerVol=[];
  for(let v=0;v<volumes.length;v++){
    grpToSegPerVol.push(volumes[v].n>0?buildGroupToSeg(volumes[v]):new Map());
  }
  // Build per-panel lookups
  for(const p of panels){
    const pd=volumes[p.vol]&&volumes[p.vol].planes[p.plane];
    if(pd){
      const lk=buildPlaneLookups(pd, p.plane);
      p.grpToPx=lk.g2p; p.pkToDispIdx=lk.pk2di; p.pxToGrps=lk.p2g;
    } else {
      p.grpToPx=new Map(); p.pkToDispIdx=new Map(); p.pxToGrps=new Map();
    }
  }
}

function getIdsForLabel(vol,label){
  // All label arrays are sourced from labl/. Returns null when the
  // labl file is absent; callers must null-check.
  if(label==='track') return vol.tids||null;
  if(label==='pdg') return vol.pdg||null;
  if(label==='ancestor') return vol.ancTids||null;
  if(label==='interaction') return vol.intIds||null;
  return vol.tids||null;
}

function computeTopTracks(){
  const counts=new Map();
  const label=curColorMode==='label'?curLabel:'track';
  if(curVol<0){
    for(const vol of volumes){if(!vol||vol.n===0) continue;
      const ids=getIdsForLabel(vol,label);
      if(ids) for(let i=0;i<vol.n;i++) counts.set(ids[i],(counts.get(ids[i])||0)+1);
    }
  } else {
    const vd=curVolData();
    if(vd&&vd.n>0){const ids=getIdsForLabel(vd,label);
      if(ids) for(let i=0;i<vd.n;i++) counts.set(ids[i],(counts.get(ids[i])||0)+1);
    }
  }
  topTracks=[...counts.entries()].sort((a,b)=>b[1]-a[1]).slice(0,25).map(([tid,cnt])=>({tid,cnt}));
}

function labelForId(label,id){
  if(label==='pdg') return pdgName(id)+' ('+id+')';
  if(label==='track') return 'Track '+id;
  if(label==='ancestor') return 'Anc '+id;
  if(label==='interaction') return 'Int '+id;
  return ''+id;
}

function populateCatFilter(){
  computeTopTracks();
  const sel=document.getElementById('catFilterSelect');
  const label=curLabel;
  sel.innerHTML='<option value="all">All</option>';
  for(const{tid,cnt} of topTracks){
    const o=document.createElement('option');
    o.value=tid; o.textContent=labelForId(label,tid)+' ('+cnt.toLocaleString()+')';
    sel.appendChild(o);
  }
  sel.value='all';
  document.getElementById('catFilterInput').value='';
}

function panelMod2(p){
  return (isPixelMode&&p.plane==='Y-Z')?numPz:D.config.max_time;
}
function computePanelDomIds(){
  for(const p of panels){
    const vd=volumes[p.vol], pd=vd&&vd.planes[p.plane];
    if(!pd||pd.nDisp===0){p.pixelDomTrack=null;p.pixelDomPdg=null;p.pixelDomAnc=null;p.pixelDomInt=null;continue;}
    const g2s=grpToSegPerVol[p.vol];
    const n=pd.nDisp;
    const mod2=panelMod2(p);
    p.pixelDomTrack=new Int32Array(n);
    p.pixelDomPdg=new Int32Array(n);
    p.pixelDomAnc=new Int32Array(n);
    p.pixelDomInt=new Int32Array(n);
    for(let i=0;i<n;i++){
      const pk=pd.dispW[i]*mod2+pd.dispT[i];
      const groups=p.pxToGrps.get(pk);
      if(!groups||groups.length===0) continue;
      let maxCh=-1, maxGid=0;
      for(const{gid,ch} of groups){if(ch>maxCh){maxCh=ch;maxGid=gid;}}
      const segs=g2s.get(maxGid);
      if(segs&&segs.length>0){
        const si=segs[0];
        p.pixelDomTrack[i]=vd.tids?vd.tids[si]:0;
        p.pixelDomPdg[i]=vd.pdg?vd.pdg[si]:0;
        p.pixelDomAnc[i]=vd.ancTids?vd.ancTids[si]:0;
        p.pixelDomInt[i]=vd.intIds?vd.intIds[si]:0;
      }
    }
  }
}

function applyTrackFilter(){
  if(!ptMat||!hlBuf) return;
  if(selectedTrack===null){
    ptMat.uniforms.noHover.value=1.0;
    hlBuf.fill(0);
    setStatus('');
  } else {
    ptMat.uniforms.noHover.value=0.0;
    const label=curColorMode==='label'?curLabel:'track';
    let nHit=0;
    if(curVol<0){
      let off=0;
      for(const vol of volumes){
        if(!vol||vol.n===0) continue;
        const ids=getIdsForLabel(vol,label);
        for(let i=0;i<vol.n;i++){hlBuf[off+i]=(ids&&ids[i]===selectedTrack)?1.0:0.0;if(hlBuf[off+i])nHit++;}
        off+=vol.n;
      }
    } else {
      const vd=curVolData();
      if(vd){const ids=getIdsForLabel(vd,label);
        for(let i=0;i<vd.n;i++){hlBuf[i]=(ids&&ids[i]===selectedTrack)?1.0:0.0;if(hlBuf[i])nHit++;}
      }
    }
    const lbl=labelForId(label,selectedTrack);
    if(nHit===0) setStatus(`${lbl}: no deposits`);
    else setStatus(`${lbl}: ${nHit.toFixed(0)} deposits`);
  }
  ptGeo.attributes.hl.needsUpdate=true;
  if(curViewMode==='optical'&&lightData){
    renderOpticalFrame(null);
  } else {
    render2DBase();
    render2DFrame(null);
  }
}

// ============================================================
// THREE.JS SETUP (shaders imported from shaders.js)
// ============================================================
function initThree(){
  const el=document.getElementById('panel3d');
  renderer=new THREE.WebGLRenderer({antialias:true,alpha:false,preserveDrawingBuffer:true});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
  renderer.setSize(el.clientWidth,el.clientHeight);
  renderer.setClearColor(BG);
  el.appendChild(renderer.domElement);

  scene=new THREE.Scene();
  camera=new THREE.PerspectiveCamera(42,el.clientWidth/el.clientHeight,10,80000);
  camera.position.set(4000,2500,6000);

  controls=new THREE.OrbitControls(camera,renderer.domElement);
  controls.enableDamping=true;
  controls.dampingFactor=0.12;
  controls.autoRotate=true;
  controls.autoRotateSpeed=0.4;
  controls.target.set(0,0,0);

  warmTex=makeColormapTex(WARM_STOPS);
  infernoTex=makeColormapTex(INFERNO_R_STOPS);
  trackTex=makeTrackTex();

  // pick target — must match renderer's actual pixel size
  const dpr=renderer.getPixelRatio();
  pickTarget=new THREE.WebGLRenderTarget(el.clientWidth*dpr,el.clientHeight*dpr);
  pickScene=new THREE.Scene();

  renderer.domElement.addEventListener('mousemove',on3dMove);
  renderer.domElement.addEventListener('mouseleave',()=>{pendingPick=null;clearHL();});
}

function boxColors(){
  return lightMode
    ?[0x7799bb,0xbb8888,0x88aa88,0xbbaa77,0x9988bb,0x88aabb]
    :[0x334466,0x663333,0x336633,0x665533,0x443366,0x336666];
}

function buildVolBoxes(){
  if(boxMesh){scene.remove(boxMesh);boxMesh=null;}
  if(!showVolBounds) return;
  const colors=boxColors();
  const boxGroup=new THREE.Group();
  if(volRanges){
    const volIndices=curVol<0?Array.from({length:nVolumes},(_,i)=>i):[curVol];
    for(const vi of volIndices){
      const r=volRanges[vi]; if(!r) continue;
      const xn=r[0][0],xx=r[0][1],yn=r[1][0],yx=r[1][1],zn=r[2][0],zx=r[2][1];
      const c=[[xn,yn,zn],[xx,yn,zn],[xx,yx,zn],[xn,yx,zn],[xn,yn,zx],[xx,yn,zx],[xx,yx,zx],[xn,yx,zx]];
      const edges=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
      const bv=[];for(const[a,b] of edges) bv.push(...c[a],...c[b]);
      const bg=new THREE.BufferGeometry();
      bg.setAttribute('position',new THREE.Float32BufferAttribute(bv,3));
      boxGroup.add(new THREE.LineSegments(bg,new THREE.LineBasicMaterial({color:colors[vi%colors.length]})));
    }
  }
  boxMesh=boxGroup;
  scene.add(boxMesh);
}

function buildAnodeCathode(){
  if(anodeMesh){scene.remove(anodeMesh);anodeMesh=null;}
  if(cathodeMesh){scene.remove(cathodeMesh);cathodeMesh=null;}
  if(!volRanges||!showAnodeCathode) return;
  const anodeGroup=new THREE.Group(), cathodeGroup=new THREE.Group();
  const volIndices=curVol<0?Array.from({length:nVolumes},(_,i)=>i):[curVol];
  const anodeColor=lightMode?0x1a73e8:0x4488cc;
  const cathodeColor=lightMode?0xcc3333:0xcc6644;
  for(const vi of volIndices){
    const r=volRanges[vi]; if(!r) continue;
    const yn=r[1][0],yx=r[1][1],zn=r[2][0],zx=r[2][1];
    const dd=volDriftDirs[vi]||1;
    const anodeX=dd<0?r[0][0]:r[0][1];
    const cathodeX=dd<0?r[0][1]:r[0][0];
    // Anode plane
    const ag=new THREE.BufferGeometry();
    ag.setAttribute('position',new THREE.Float32BufferAttribute([anodeX,yn,zn,anodeX,yx,zn,anodeX,yx,zx,anodeX,yn,zn,anodeX,yx,zx,anodeX,yn,zx],3));
    anodeGroup.add(new THREE.Mesh(ag,new THREE.MeshBasicMaterial({color:anodeColor,transparent:true,opacity:0.08,side:THREE.DoubleSide,depthWrite:false})));
    // Cathode plane
    const cg=new THREE.BufferGeometry();
    cg.setAttribute('position',new THREE.Float32BufferAttribute([cathodeX,yn,zn,cathodeX,yx,zn,cathodeX,yx,zx,cathodeX,yn,zn,cathodeX,yx,zx,cathodeX,yn,zx],3));
    cathodeGroup.add(new THREE.Mesh(cg,new THREE.MeshBasicMaterial({color:cathodeColor,transparent:true,opacity:0.08,side:THREE.DoubleSide,depthWrite:false})));
  }
  anodeMesh=anodeGroup; cathodeMesh=cathodeGroup;
  scene.add(anodeMesh); scene.add(cathodeMesh);
}

// ============================================================
// BUILD 3D POINTS
// ============================================================
function buildPoints(){
  // dispose old
  if(ptMesh){scene.remove(ptMesh);ptGeo.dispose();ptMesh=null;}
  if(pickMesh){pickScene.remove(pickMesh);pickMesh=null;}

  // Collect segments: All mode = concatenate all volumes, single = one volume
  const volsToShow=curVol<0?volumes.filter(v=>v.n>0):(curVolData()&&curVolData().n>0?[curVolData()]:[]);
  let totalN=0; for(const v of volsToShow) totalN+=v.n;
  if(totalN===0){
    document.getElementById('label3dText').textContent='3D SEGMENTS \u2014 0 deposits';
    return;
  }

  const pos=new Float32Array(totalN*3), de=new Float32Array(totalN);
  const tids=new Int32Array(totalN), gids=new Int32Array(totalN);
  const allPdg=new Int32Array(totalN), allAnc=new Int32Array(totalN), allInt=new Int32Array(totalN);
  let off=0;
  for(const v of volsToShow){
    pos.set(v.pos,off*3); de.set(v.de,off);
    if(v.tids) tids.set(v.tids,off);
    if(v.gids) gids.set(v.gids,off);
    if(v.pdg) allPdg.set(v.pdg,off);
    if(v.ancTids) allAnc.set(v.ancTids,off);
    if(v.intIds) allInt.set(v.intIds,off);
    off+=v.n;
  }

  const deVal=new Float32Array(totalN), trackVal=new Float32Array(totalN);
  const trackWt=new Float32Array(totalN);
  hlBuf=new Float32Array(totalN);

  let deMin=Infinity, deMax=-Infinity;
  for(let i=0;i<totalN;i++){if(de[i]<deMin)deMin=de[i];if(de[i]>deMax)deMax=de[i];}
  if(deMax<=deMin) deMax=deMin+1;
  const deFloor=Math.max(deMin,1e-4);
  const logDeMin=Math.log10(deFloor), logDeRange=Math.log10(deMax)-logDeMin||1;
  // Weight per category: log(count) normalized
  function catWeight(ids){
    const c=new Map();
    for(let i=0;i<totalN;i++) c.set(ids[i],(c.get(ids[i])||0)+1);
    let mx=1; for(const v of c.values()) if(v>mx) mx=v; const lm=Math.log10(mx+1);
    const w=new Float32Array(totalN);
    for(let i=0;i<totalN;i++) w[i]=Math.log10((c.get(ids[i])||1)+1)/lm;
    return w;
  }
  const trkWtArr=catWeight(tids);
  const ancWtArr=catWeight(allAnc);
  const intWtArr=catWeight(allInt);
  const pdgWtArr=catWeight(allPdg);
  // Store arrays for color mode switching (used by switchColorMode)
  _3dPdg=allPdg; _3dAnc=allAnc; _3dInt=allInt;
  _3dTids=tids; _3dTrkWt=trkWtArr; _3dAncWt=ancWtArr; _3dIntWt=intWtArr; _3dPdgWt=pdgWtArr;
  // Set trackVal/trackWt based on current color mode
  let curIds=tids, curWts=trkWtArr;
  if(curColor==='pdg'){curIds=allPdg;curWts=pdgWtArr;}
  else if(curColor==='ancestor'){curIds=allAnc;curWts=ancWtArr;}
  else if(curColor==='interaction'){curIds=allInt;curWts=intWtArr;}
  for(let i=0;i<totalN;i++){
    deVal[i]=Math.max(0,Math.min(1,(Math.log10(Math.max(de[i],deFloor))-logDeMin)/logDeRange));
    trackVal[i]=(Math.abs(curIds[i])*0.618033988749895)%1.0;
    trackWt[i]=curWts[i];
  }

  ptGeo=new THREE.BufferGeometry();
  ptGeo.setAttribute('position',new THREE.BufferAttribute(pos,3));
  ptGeo.setAttribute('deVal',new THREE.BufferAttribute(deVal,1));
  ptGeo.setAttribute('trackVal',new THREE.BufferAttribute(trackVal,1));
  ptGeo.setAttribute('trackWt',new THREE.BufferAttribute(trackWt,1));
  const hlAttr=new THREE.BufferAttribute(hlBuf,1);
  hlAttr.setUsage(THREE.DynamicDrawUsage);
  ptGeo.setAttribute('hl',hlAttr);

  // Drift animation attributes
  const segT0Arr=new Float32Array(totalN);
  const segArrArr=new Float32Array(totalN);
  const segXAnodeArr=new Float32Array(totalN);
  let dOff=0;
  // Compute drift attributes and animation time range
  const rwUs=D.config.max_time*timeStepUs; // readout window in us
  simTimeMin=0; simTimeMax=rwUs;
  for(const v of volsToShow){
    const vi=volumes.indexOf(v);
    const xa=volAnodes[vi]||0;
    for(let i=0;i<v.n;i++){
      segT0Arr[dOff+i]=v.t0?v.t0[i]:0;
      segArrArr[dOff+i]=v.arrivalTime?v.arrivalTime[i]:(segT0Arr[dOff+i]+Math.abs(v.pos[i*3]-xa)/velocityMmUs);
      segXAnodeArr[dOff+i]=xa;
    }
    dOff+=v.n;
  }
  ptGeo.setAttribute('segT0',new THREE.BufferAttribute(segT0Arr,1));
  ptGeo.setAttribute('segArrival',new THREE.BufferAttribute(segArrArr,1));
  ptGeo.setAttribute('segXAnode',new THREE.BufferAttribute(segXAnodeArr,1));

  ptMat=new THREE.ShaderMaterial({
    vertexShader:VS, fragmentShader:FS,
    uniforms:{
      cmap:{value:curColor==='de'?deTex():trackTex},
      noHover:{value:1.0},
      useTrack:{value:curColor!=='de'?1.0:0.0},
      emphPow:{value:deEmphPow},
      emphAmt:{value:deEmphAmt},
      simTime:{value:-99999},
      driftOn:{value:0.0},
      fadeDur:{value:fadeDuration},
      softHL:{value:0.0},
    },
    transparent:true, depthWrite:false,
  });
  ptMesh=new THREE.Points(ptGeo,ptMat);
  scene.add(ptMesh);

  // Bounding boxes from config geometry ranges (not data extents)
  buildVolBoxes();
  buildAnodeCathode();

  // Camera
  let xMin=Infinity,xMax=-Infinity,yMin=Infinity,yMax=-Infinity,zMin=Infinity,zMax=-Infinity;
  for(let i=0;i<totalN;i++){
    const x=pos[i*3],y=pos[i*3+1],z=pos[i*3+2];
    if(x<xMin)xMin=x;if(x>xMax)xMax=x;if(y<yMin)yMin=y;if(y>yMax)yMax=y;if(z<zMin)zMin=z;if(z>zMax)zMax=z;
  }
  const cx=(xMin+xMax)/2,cy=(yMin+yMax)/2,cz=(zMin+zMax)/2;
  const diag=Math.sqrt((xMax-xMin)**2+(yMax-yMin)**2+(zMax-zMin)**2)||1;
  const newPos=new THREE.Vector3(cx+diag*.55,cy+diag*.35,cz+diag*.65);
  const newTarget=new THREE.Vector3(cx,cy,cz);
  camera.near=diag*.001;camera.far=diag*5;
  camera.updateProjectionMatrix();
  if(vizReady){
    animateCamera(newPos,newTarget,500);
  } else {
    camera.position.copy(newPos);
    controls.target.copy(newTarget);
  }
  initCamPos=newPos.clone();
  initCamTarget=newTarget.clone();

  // Pick geometry: encode concatenated segment index (+1) as color
  // From the index we can determine volume + local segment + groupId
  {
    const pickColors=new Float32Array(totalN*3);
    for(let i=0;i<totalN;i++){
      const id=i+1; // 0 = background
      pickColors[i*3]=((id)&0xFF)/255;
      pickColors[i*3+1]=((id>>8)&0xFF)/255;
      pickColors[i*3+2]=((id>>16)&0xFF)/255;
    }
    const pGeo=new THREE.BufferGeometry();
    pGeo.setAttribute('position',new THREE.BufferAttribute(pos,3));
    pGeo.setAttribute('pickColor',new THREE.BufferAttribute(pickColors,3));
    if(!pickMat){
      pickMat=new THREE.ShaderMaterial({
        vertexShader:`attribute vec3 pickColor;varying vec3 vPC;
          void main(){vPC=pickColor;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);gl_PointSize=8.0;}`,
        fragmentShader:`varying vec3 vPC;void main(){
          vec2 d=gl_PointCoord-vec2(0.5);if(length(d)>0.5)discard;gl_FragColor=vec4(vPC,1.0);}`,
      });
    }
    pickMesh=new THREE.Points(pGeo,pickMat);
    pickScene.add(pickMesh);
  }

  const label=curVol<0?'ALL VOLUMES':'VOL '+curVol;
  document.getElementById('label3dText').textContent=
    `3D SEGMENTS (${label}) \u2014 ${totalN.toLocaleString()} deposits`;
}

// ============================================================
// 2D CANVAS — MULTI-PANEL
// ============================================================
function init2D(){
  c2d=document.getElementById('canvas2d');
  ctx=c2d.getContext('2d');
  offC=document.createElement('canvas');
  offCtx=offC.getContext('2d');
  resize2D();
  c2d.addEventListener('mousemove',on2dMove);
  c2d.addEventListener('mouseleave',()=>{is2dHover=false;clearHL();});
  c2d.addEventListener('wheel',on2dWheel,{passive:false});
  c2d.addEventListener('mousedown',on2dDown);
  window.addEventListener('mouseup',()=>{isPan=false;panPanel=-1;c2d.style.cursor='crosshair';});
  c2d.addEventListener('dblclick',(e)=>{
    if(curViewMode==='optical') return;
    if(expandedPanel>=0){
      const p=panels[expandedPanel];
      p.view={...p.viewMax};
      render2DBase();render2DFrame(null);
    } else {
      const pi=panelAtMouse(e.offsetX,e.offsetY);
      if(pi>=0) expandPanel(pi);
    }
  });
  document.getElementById('panelCloseBtn').addEventListener('click',()=>collapsePanel());
}

function resize2D(){
  const el=document.getElementById('panel2d');
  const dpr=Math.min(window.devicePixelRatio||1,2);
  const w=el.clientWidth, h=el.clientHeight;
  c2d.width=w*dpr; c2d.height=h*dpr;
  c2d.style.width=w+'px'; c2d.style.height=h+'px';
  offC.width=c2d.width; offC.height=c2d.height;
  ctx.setTransform(dpr,0,0,dpr,0,0);
  offCtx.setTransform(dpr,0,0,dpr,0,0);
  updatePanelRects();
}

function buildPanels(){
  panels=[];
  expandedPanel=-1;
  document.getElementById('panelCloseBtn')?.classList.remove('visible');
  if(volumes.length===0) return;
  const nP=planeLabels.length;
  if(curVol<0){
    const MAX_VIS=3;
    let visVols;
    if(nVolumes<=MAX_VIS){
      visVols=Array.from({length:nVolumes},(_,i)=>i);
    } else {
      // Show top MAX_VIS volumes by 2D display pixel count
      visVols=volumes.map((v,i)=>{
        let nPx=0;
        for(const pl of planeLabels) if(v.planes[pl]) nPx+=v.planes[pl].nDisp||0;
        return {i, nPx};
      }).filter(x=>x.nPx>0)
        .sort((a,b)=>b.nPx-a.nPx).slice(0,MAX_VIS)
        .map(x=>x.i).sort((a,b)=>a-b);
    }
    panelGrid={rows:visVols.length, cols:nP};
    for(const vi of visVols) for(let p=0;p<nP;p++) panels.push({vol:vi, plane:planeLabels[p]});
  } else {
    panelGrid={rows:1, cols:nP};
    for(let p=0;p<nP;p++) panels.push({vol:curVol, plane:planeLabels[p]});
  }
  for(const p of panels){
    p.view={x0:0,x1:100,y0:0,y1:100}; p.viewMax={...p.view};
    p.dispColors=null; p.chargeMin=1; p.chargeMax=1;
    p.grpToPx=new Map(); p.pkToDispIdx=new Map(); p.pxToGrps=new Map();
    p.pixelDomTrack=null; p.rect={x:0,y:0,w:0,h:0,pw:0,ph:0};
  }
  updatePanelRects();
}

function updatePanelRects(){
  if(panels.length===0||panelGrid.rows===0) return;
  const el=document.getElementById('panel2d');
  const cw=el.clientWidth, ch=el.clientHeight;
  const isExp=expandedPanel>=0&&expandedPanel<panels.length;
  const headerH=isExp?50:40;
  const usableH=ch-headerH;

  if(isExp){
    // Expanded: one panel fills the area
    const plotSize=Math.min(cw-margin.l-margin.r, usableH-margin.t-margin.b);
    const padX=(cw-plotSize-margin.l-margin.r)/2;
    const padY=(usableH-plotSize-margin.t-margin.b)/2;
    for(let i=0;i<panels.length;i++){
      if(i===expandedPanel){
        panels[i].rect={x:padX, y:headerH+padY, w:plotSize+margin.l+margin.r, h:plotSize+margin.t+margin.b, pw:plotSize, ph:plotSize};
      } else {
        panels[i].rect={x:-9999, y:-9999, w:0, h:0, pw:0, ph:0}; // offscreen
      }
    }
  } else {
    // Grid: all panels visible
    const cellW=cw/panelGrid.cols, cellH=usableH/panelGrid.rows;
    const plotAvailW=cellW-margin.l-margin.r;
    const plotAvailH=cellH-margin.t-margin.b;
    const plotSize=Math.min(plotAvailW,plotAvailH);
    const padX=(cellW-plotSize-margin.l-margin.r)/2;
    const padY=(cellH-plotSize-margin.t-margin.b)/2;
    for(let i=0;i<panels.length;i++){
      const row=Math.floor(i/panelGrid.cols), col=i%panelGrid.cols;
      panels[i].rect={x:col*cellW+padX, y:headerH+row*cellH+padY,
        w:plotSize+margin.l+margin.r, h:plotSize+margin.t+margin.b,
        pw:plotSize, ph:plotSize};
    }
  }
}

function panelAtMouse(mx,my){
  if(expandedPanel>=0) return expandedPanel; // only one visible
  for(let i=0;i<panels.length;i++){
    const r=panels[i].rect;
    if(mx>=r.x&&mx<r.x+r.w&&my>=r.y&&my<r.y+r.h) return i;
  }
  return -1;
}

function expandPanel(pi){
  expandedPanel=pi;
  updatePanelRects();
  document.getElementById('panelCloseBtn').classList.add('visible');
  const p=panels[pi];
  render2DBase();render2DFrame(null);
}

function collapsePanel(){
  expandedPanel=-1;
  updatePanelRects();
  document.getElementById('panelCloseBtn').classList.remove('visible');
  render2DBase();render2DFrame(null);
}

function panelDims(p){
  // Returns [xMax, yMax] for a panel based on plane type
  const pi=planeLabels.indexOf(p.plane);
  if(isPixelMode){
    if(p.plane==='Y-T') return [numPy, D.config.max_time];
    if(p.plane==='Z-T') return [numPz, D.config.max_time];
    if(p.plane==='Y-Z') return [numPy, numPz];
  }
  const nw=D.config.num_wires_actual[p.vol]?D.config.num_wires_actual[p.vol][pi]:100;
  return [nw, D.config.max_time];
}
function resetPanelViews(){
  for(const p of panels){
    const [xMax, yMax]=panelDims(p);
    p.view={x0:0, x1:xMax, y0:0, y1:yMax};
    p.viewMax={...p.view};
  }
}

function precomputeAllPanelColors(){
  for(const p of panels){
    const dd=getDispForPanel(p);
    if(!dd||dd.n===0){p.dispColors=null;continue;}
    const n=dd.n;
    p.dispColors=new Uint8Array(n*3);
    p.dispAlpha=null; // per-pixel alpha (only for track mode)
    if(curViewMode==='sensor'){
      const norms=sensorNorms[p.plane]||[-25,25];
      for(let i=0;i<n;i++){
        const t=dbNorm(dd.v[i],norms[0],norms[1],sensorDeadband,sensorGamma);
        const[r,g,b]=sensorCmapRGB(t);
        p.dispColors[i*3]=r;p.dispColors[i*3+1]=g;p.dispColors[i*3+2]=b;
      }
    } else if(curColor!=='de'){
      // Categorical mode: track, pdg, ancestor, interaction
      let domIds;
      if(curColor==='track') domIds=p.pixelDomTrack;
      else if(curColor==='pdg') domIds=p.pixelDomPdg;
      else if(curColor==='ancestor') domIds=p.pixelDomAnc;
      else domIds=p.pixelDomInt;
      if(!domIds){/* fallback to charge */ domIds=null;}
      if(domIds){
        // Emphasis based on charge (2D equivalent of dE), not category count
        const ch=dd.v;
        let chMin=Infinity,chMax=-Infinity;
        for(let i=0;i<n;i++){if(ch[i]<chMin)chMin=ch[i];if(ch[i]>chMax)chMax=ch[i];}
        if(chMin<=0)chMin=1;if(chMax<=chMin)chMax=chMin*10;
        const logMin=Math.log10(chMin),logRange=Math.log10(chMax)-logMin||1;
        p.dispAlpha=new Float32Array(n);
        for(let i=0;i<n;i++){
          const id=domIds[i];
          const hue=(Math.abs(id)*0.618033988749895)%1.0;
          const[r,g,b]=hsl2rgb(hue,.78,.55);
          p.dispColors[i*3]=r;p.dispColors[i*3+1]=g;p.dispColors[i*3+2]=b;
          const t=Math.max(0,Math.min(1,(Math.log10(Math.max(ch[i],chMin))-logMin)/logRange));
          p.dispAlpha[i]=0.15+0.85*Math.pow(t,deEmphPow);
        }
      }
    } else {
      const ch=dd.v;
      p.chargeMin=Infinity;p.chargeMax=-Infinity;
      for(let i=0;i<n;i++){if(ch[i]<p.chargeMin)p.chargeMin=ch[i];if(ch[i]>p.chargeMax)p.chargeMax=ch[i];}
      if(p.chargeMin<=0)p.chargeMin=1;if(p.chargeMax<=p.chargeMin)p.chargeMax=p.chargeMin*10;
      const logMin=Math.log10(p.chargeMin),logRange=Math.log10(p.chargeMax)-logMin||1;
      for(let i=0;i<n;i++){
        const t=Math.max(0,Math.min(1,(Math.log10(Math.max(ch[i],p.chargeMin))-logMin)/logRange));
        const[r,g,b]=hitsCmapRGB(t);
        p.dispColors[i*3]=r;p.dispColors[i*3+1]=g;p.dispColors[i*3+2]=b;
      }
    }
  }
}

function renderPanelBase(cx,p){
  const dd=getDispForPanel(p);
  const r=p.rect;
  cx.fillStyle=bgCol();cx.fillRect(r.x,r.y,r.w,r.h);
  if(!dd||dd.n===0||!p.dispColors) return;
  const vw=p.view.x1-p.view.x0, vh=p.view.y1-p.view.y0;
  const dotW=Math.max(1,Math.ceil(r.pw/vw*.92)), dotH=Math.max(1,Math.ceil(r.ph/vh*.92));
  const ox=r.x+margin.l, oy=r.y+margin.t;
  let filterIds=null;
  if(curViewMode==='hits'&&selectedTrack!==null){
    if(curColor==='track') filterIds=p.pixelDomTrack;
    else if(curColor==='pdg') filterIds=p.pixelDomPdg;
    else if(curColor==='ancestor') filterIds=p.pixelDomAnc;
    else if(curColor==='interaction') filterIds=p.pixelDomInt;
    else filterIds=p.pixelDomTrack;
  }
  const hasAlpha=p.dispAlpha&&deEmphAmt>0;
  for(let i=0;i<dd.n;i++){
    const wi=dd.w[i],ti=dd.t[i];
    if(wi<p.view.x0||wi>p.view.x1||ti<p.view.y0||ti>p.view.y1) continue;
    if(filterIds&&filterIds[i]!==selectedTrack) continue;
    if(hasAlpha) cx.globalAlpha=0.15+(p.dispAlpha[i]-0.15)*deEmphAmt;
    const px=ox+((wi-p.view.x0)/vw)*r.pw;
    const py=oy+((p.view.y1-ti)/vh)*r.ph;
    cx.fillStyle=`rgb(${p.dispColors[i*3]},${p.dispColors[i*3+1]},${p.dispColors[i*3+2]})`;
    cx.fillRect(px,py,dotW,dotH);
  }
  if(hasAlpha) cx.globalAlpha=1.0;
}

function render2DBase(){
  const el=document.getElementById('panel2d');
  const w=el.clientWidth, h=el.clientHeight;
  offCtx.fillStyle=bgCol();offCtx.fillRect(0,0,w,h);
  for(const p of panels) renderPanelBase(offCtx,p);
}

function render2DFrame(hlMap){
  // hlMap: null or Map<panelIdx, [{idx,intensity}]>
  const el=document.getElementById('panel2d');
  const w=el.clientWidth, h=el.clientHeight;
  ctx.clearRect(0,0,w,h);

  if(hlMap&&hlMap.size>0){
    ctx.globalAlpha=lightMode?0.25:0.20;
    ctx.drawImage(offC,0,0,w,h);
    ctx.globalAlpha=1.0;
    for(const[pi,hlPx] of hlMap){
      const p=panels[pi]; if(!p) continue;
      const dd=getDispForPanel(p); if(!dd) continue;
      const r=p.rect, vw=p.view.x1-p.view.x0, vh=p.view.y1-p.view.y0;
      const dotW=Math.max(2,Math.ceil(r.pw/vw*.92)+1), dotH=Math.max(2,Math.ceil(r.ph/vh*.92)+1);
      const ox=r.x+margin.l, oy=r.y+margin.t;
      for(const{idx,intensity} of hlPx){
        const wi=dd.w[idx],ti=dd.t[idx];
        if(wi<p.view.x0||wi>p.view.x1||ti<p.view.y0||ti>p.view.y1) continue;
        const px=ox+((wi-p.view.x0)/vw)*r.pw;
        const py=oy+((p.view.y1-ti)/vh)*r.ph;
        ctx.globalAlpha=0.4+0.6*intensity;
        ctx.fillStyle=`rgb(${p.dispColors[idx*3]},${p.dispColors[idx*3+1]},${p.dispColors[idx*3+2]})`;
        ctx.fillRect(px-1,py-1,dotW+2,dotH+2);
      }
    }
    ctx.globalAlpha=1.0;
  } else {
    ctx.drawImage(offC,0,0,w,h);
  }
  // Drift sweep: mask future time with dark overlay
  // simTime is in absolute us; readout time_bin = max(0, simTime) / timeStepUs
  if(driftMode){
    const sweepTimeBin=Math.max(0,simTime)/timeStepUs;
    ctx.fillStyle=lightMode?'rgba(240,240,240,0.85)':'rgba(8,8,8,0.85)';
    for(const p of panels){
      if(isPixelMode&&p.plane==='Y-Z') continue; // no time axis
      const r=p.rect; if(r.pw<=0) continue;
      const vh=p.view.y1-p.view.y0;
      if(sweepTimeBin>=p.view.y1) continue; // all visible
      if(sweepTimeBin<=p.view.y0){
        // All masked (readout hasn't reached this view range yet)
        ctx.fillRect(r.x+margin.l,r.y+margin.t,r.pw,r.ph);
        continue;
      }
      const ox=r.x+margin.l, oy=r.y+margin.t;
      // Higher time = toward top of canvas. Mask everything above the sweep line.
      // sweepTimeBin maps to y position: oy + (p.view.y1 - sweepTimeBin)/vh * r.ph
      const sweepY=oy+((p.view.y1-sweepTimeBin)/vh)*r.ph;
      if(sweepY>oy) ctx.fillRect(ox,oy,r.pw,sweepY-oy);
    }
  }
  drawAllPanelAxes();
}

function drawPanelAxes(p,pi){
  const r=p.rect, ox=r.x, oy=r.y;
  const pW=r.pw, pH=r.ph;
  const isExpanded=expandedPanel>=0;
  const fsPx=Math.max(9,Math.min(15,Math.round(pH/16)));
  const fs=fsPx+'px';
  ctx.strokeStyle=axCol();ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(ox+margin.l,oy+margin.t);
  ctx.lineTo(ox+margin.l,oy+r.h-margin.b);ctx.lineTo(ox+r.w-margin.r,oy+r.h-margin.b);ctx.stroke();

  ctx.fillStyle=txCol();ctx.font=fs+' monospace';ctx.textAlign='center';
  const nT=pW<150?3:5;
  for(let i=0;i<=nT;i++){
    const f=i/nT, v=Math.round(p.view.x0+f*(p.view.x1-p.view.x0));
    ctx.fillText(v,ox+margin.l+f*pW,oy+r.h-margin.b+fsPx+2);
  }
  ctx.textAlign='right';
  const nY=pH<150?2:3;
  for(let i=0;i<=nY;i++){
    const f=i/nY, v=Math.round(p.view.y1-f*(p.view.y1-p.view.y0));
    ctx.fillText(v,ox+margin.l-4,oy+margin.t+f*pH+fsPx/3);
  }
  // Axis labels — show when panel is large enough
  if(pH>=100){
    const labelFs=Math.max(9,fsPx)+'px';
    ctx.font=labelFs+' sans-serif';ctx.fillStyle=txCol();
    // Determine axis labels based on plane type
    let xLabel='Wire', yLabel='Time';
    if(isPixelMode){
      if(p.plane==='Y-T'){xLabel='Pixel Y';yLabel='Time';}
      else if(p.plane==='Z-T'){xLabel='Pixel Z';yLabel='Time';}
      else if(p.plane==='Y-Z'){xLabel='Pixel Y';yLabel='Pixel Z';}
    }
    ctx.textAlign='center';
    ctx.fillText(xLabel,ox+margin.l+pW/2,oy+r.h-4);
    ctx.save();
    ctx.translate(ox+14,oy+margin.t+pH/2);
    ctx.rotate(-Math.PI/2);
    ctx.textAlign='center';
    ctx.fillText(yLabel,0,0);
    ctx.restore();
  }
  // Colorbar (skip in categorical label modes — colorbar is meaningless for hashed hues)
  if(curColor==='de'||curViewMode==='sensor'){
    const bx=ox+r.w-margin.r+4, bw=8, by=oy+margin.t, bh=pH;
    if(curViewMode==='sensor'){
      const norms=sensorNorms[p.plane]||[-25,25];
      for(let y=0;y<bh;y++){const t=1-y/bh;const[cr,cg,cb]=sensorCmapRGB(t);ctx.fillStyle=`rgb(${cr},${cg},${cb})`;ctx.fillRect(bx,by+y,bw,1);}
      ctx.strokeStyle=axCol();ctx.strokeRect(bx,by,bw,bh);
      ctx.textAlign='left';ctx.font=fs+' monospace';ctx.fillStyle=txCol();
      for(let i=0;i<=4;i++){const tn=i/4;const v=dbInverse(tn,norms[0],norms[1],sensorDeadband,sensorGamma);ctx.fillText(v.toFixed(0),bx+bw+2,by+bh*(1-tn)+fsPx/3);}
    } else {
      for(let y=0;y<bh;y++){const t=1-y/bh;const[cr,cg,cb]=hitsCmapRGB(t);ctx.fillStyle=`rgb(${cr},${cg},${cb})`;ctx.fillRect(bx,by+y,bw,1);}
      ctx.strokeStyle=axCol();ctx.strokeRect(bx,by,bw,bh);
      if(p.chargeMin<p.chargeMax){
        ctx.textAlign='left';ctx.font=fs+' monospace';ctx.fillStyle=txCol();
        const lMin=Math.log10(Math.max(p.chargeMin,1)),lMax=Math.log10(Math.max(p.chargeMax,1));
        for(let i=0;i<=3;i++){const f=i/3,lv=lMin+f*(lMax-lMin),v=Math.pow(10,lv);
          const label=v>=1e4?(v/1e3).toFixed(0)+'k':v>=100?v.toFixed(0):v.toFixed(1);
          ctx.fillText(label,bx+bw+2,by+bh*(1-f)+fsPx/3);}
      }
    }
  }
  // Title
  const pn={U:'U',V:'V',Y:'Y'}[p.plane]||p.plane;
  const title=panelGrid.rows>1?`V${p.vol} ${pn}`:pn;
  const tfsPx=Math.max(11,Math.min(17,Math.round(pH/12)));
  ctx.textAlign='center';ctx.fillStyle=lightMode?'#444':'#888';ctx.font='bold '+tfsPx+'px monospace';
  ctx.fillText(title,ox+margin.l+pW/2,oy+margin.t-6);
}

function drawAllPanelAxes(){
  for(let i=0;i<panels.length;i++) drawPanelAxes(panels[i],i);
  const modeLabel=curViewMode==='sensor'?'Sensor':'Hits';
  let volLabel=curVol<0?'All Volumes':'Vol '+curVol;
  if(curVol<0&&nVolumes>3){
    const nShown=new Set(panels.map(p=>p.vol)).size;
    volLabel=`Top ${nShown} of ${nVolumes} Volumes`;
  }
  document.getElementById('label2dText').textContent=volLabel+' \u2014 '+modeLabel;
}

// ============================================================
// OPTICAL RENDERING
// ============================================================
function symLogNorm(v,vmax,linthresh){
  if(Math.abs(v)<linthresh) return 0.5+0.5*(v/linthresh)*0.1;
  const sign=v>0?1:-1;
  const t=Math.log10(Math.abs(v)/linthresh+1)/Math.log10(vmax/linthresh+1);
  return 0.5+sign*0.5*Math.min(t,1);
}

function optCmapRGB(t){
  // Use sensor colormap (obsidian/seismic) for optical too
  return sensorCmapRGB(t);
}

function renderOptical(){
  if(!lightData||!lightData.stitched) return;
  const el=document.getElementById('panel2d');
  const cw=el.clientWidth, ch=el.clientHeight;
  const m={t:30,b:35,l:50,r:50};
  const pw=cw-m.l-m.r, ph=ch-m.t-m.b;
  const nCh=lightData.nChannels, tBins=lightData.totalBins;

  let chStart=0, chEnd=nCh;
  if(curVol===0) chEnd=nPmtsPerSide;
  else if(curVol===1) chStart=nPmtsPerSide;
  const chRange=chEnd-chStart;

  let vmax=1;
  for(let c=chStart;c<chEnd;c++)
    for(let t=0;t<tBins;t++){const v=Math.abs(lightData.stitched[c*tBins+t]);if(v>vmax)vmax=v;}
  const linthresh=Math.max(vmax*0.005,1);

  offCtx.fillStyle=bgCol();offCtx.fillRect(0,0,cw,ch);

  // Render heatmap to ImageData then scale to plot area
  const imgW=Math.min(pw,tBins), imgH=Math.min(ph,chRange);
  const imgData=offCtx.createImageData(imgW,imgH);
  const xScale=tBins/imgW;
  for(let iy=0;iy<imgH;iy++){
    const c=chStart+Math.floor((chRange-1)*(1-iy/imgH));
    for(let ix=0;ix<imgW;ix++){
      const t=Math.floor(ix*xScale);
      const v=lightData.stitched[c*tBins+t];
      const norm=symLogNorm(v,vmax,linthresh);
      const[r,g,b]=optCmapRGB(norm);
      const off=(iy*imgW+ix)*4;
      imgData.data[off]=r;imgData.data[off+1]=g;imgData.data[off+2]=b;imgData.data[off+3]=255;
    }
  }
  // Draw scaled
  const tmp=document.createElement('canvas');tmp.width=imgW;tmp.height=imgH;
  tmp.getContext('2d').putImageData(imgData,0,0);
  offCtx.imageSmoothingEnabled=false;
  offCtx.drawImage(tmp,0,0,imgW,imgH,m.l,m.t,pw,ph);

  // Region separators
  const rb=lightData.regionBounds;
  offCtx.strokeStyle=lightMode?'rgba(0,0,0,0.3)':'rgba(255,255,255,0.4)';
  offCtx.lineWidth=1;offCtx.setLineDash([4,3]);
  for(let i=1;i<rb.length-1;i++){
    const x=m.l+(rb[i]/tBins)*pw;
    offCtx.beginPath();offCtx.moveTo(x,m.t);offCtx.lineTo(x,m.t+ph);offCtx.stroke();
  }
  offCtx.setLineDash([]);

  if(curVol<0){
    const divY=m.t+ph*(1-(nPmtsPerSide-chStart)/chRange);
    offCtx.strokeStyle='rgba(0,200,0,0.5)';offCtx.lineWidth=2;
    offCtx.beginPath();offCtx.moveTo(m.l,divY);offCtx.lineTo(m.l+pw,divY);offCtx.stroke();
    offCtx.lineWidth=1;
  }

  // Precompute per-interaction highlight images for fast correspondence
  lightData._layout={m,pw,ph,chStart,chEnd,chRange,vmax,linthresh,imgW,imgH,xScale};
  _precomputeOptHighlights();
}

function _precomputeOptHighlights(){
  // Build a small ImageData per interaction for fast overlay
  if(!lightData||!lightData._layout) return;
  const L=lightData._layout;
  const tBins=lightData.totalBins;
  lightData._hlImages={};
  for(const lid of lightData.activeLabels){
    const img=new ImageData(L.imgW,L.imgH);
    let hasPixels=false;
    for(let iy=0;iy<L.imgH;iy++){
      const c=L.chStart+Math.floor((L.chRange-1)*(1-iy/L.imgH));
      for(let ix=0;ix<L.imgW;ix++){
        const t=Math.floor(ix*L.xScale);
        if(lightData.intMap[c*tBins+t]===lid){
          const v=lightData.stitched[c*tBins+t];
          const norm=symLogNorm(v,L.vmax,L.linthresh);
          const[r,g,b]=optCmapRGB(norm);
          const off=(iy*L.imgW+ix)*4;
          img.data[off]=r;img.data[off+1]=g;img.data[off+2]=b;img.data[off+3]=255;
          hasPixels=true;
        }
      }
    }
    if(hasPixels) lightData._hlImages[lid]=img;
  }
}

function renderOpticalFrame(hlIntId){
  const el=document.getElementById('panel2d');
  const cw=el.clientWidth, ch=el.clientHeight;
  ctx.clearRect(0,0,cw,ch);

  if(hlIntId!==null&&hlIntId!==undefined&&lightData&&lightData._hlImages){
    ctx.globalAlpha=lightMode?0.2:0.15;
    ctx.drawImage(offC,0,0,cw,ch);
    ctx.globalAlpha=1.0;
    const L=lightData._layout;
    const hlImg=lightData._hlImages[hlIntId];
    if(hlImg&&L){
      const tmp2=document.createElement('canvas');tmp2.width=L.imgW;tmp2.height=L.imgH;
      tmp2.getContext('2d').putImageData(hlImg,0,0);
      ctx.imageSmoothingEnabled=false;
      ctx.drawImage(tmp2,0,0,L.imgW,L.imgH,L.m.l,L.m.t,L.pw,L.ph);
    }
  } else {
    ctx.drawImage(offC,0,0,cw,ch);
  }
  drawOpticalAxesAndColorbar();
}

// Optical → 3D correspondence: mouse on optical panel
function opticalAtMouse(mx,my){
  if(!lightData||!lightData._layout) return null;
  const L=lightData._layout;
  const lx=mx-L.m.l, ly=my-L.m.t;
  if(lx<0||lx>=L.pw||ly<0||ly>=L.ph) return null;
  const tBin=Math.floor((lx/L.pw)*lightData.totalBins);
  const ch=L.chStart+Math.floor((L.chRange-1)*(1-ly/L.ph));
  const px=ch*lightData.totalBins+tBin;
  const intId=lightData.intMap[px];
  if(intId<0) return null;
  return intId;
}

function highlightFromOptical(intId){
  if(!ptMat||!hlBuf) return;
  ptMat.uniforms.noHover.value=0.0;
  ptMat.uniforms.softHL.value=1.0;
  hlBuf.fill(0);
  let nHit=0, off=0;
  const volsToShow=curVol<0?volumes.filter(v=>v.n>0):(curVolData()&&curVolData().n>0?[curVolData()]:[]);
  for(const v of volsToShow){
    const ids=getOptIds(v);
    if(ids) for(let i=0;i<v.n;i++){
      if(ids[i]==intId){hlBuf[off+i]=1.0;nHit++;}
    }
    off+=v.n;
  }
  ptGeo.attributes.hl.needsUpdate=true;
  renderOpticalFrame(intId);
  const lbl=optLabelKey==='ancestor'?'Ancestor':'Interaction';
  const pe=lightData.pePerLabel?lightData.pePerLabel[intId]:null;
  const tail=hasLabl?`${nHit} segments`:'no labl: 3D link unavailable';
  setStatus(`${lbl} ${intId} \u00B7 ${tail} \u00B7 ${pe?pe.total.toLocaleString():0} PE`);
}

function renderOpticalDrift(){
  // During drift: show base optical, mask regions not yet reached by simTime
  if(!lightData||!lightData._layout) return;
  const el=document.getElementById('panel2d');
  const cw=el.clientWidth, ch=el.clientHeight;
  ctx.clearRect(0,0,cw,ch);
  ctx.drawImage(offC,0,0,cw,ch);

  // Mask unreached regions with dark overlay (left to right reveal)
  const L=lightData._layout;
  const rb=lightData.regionBounds;
  const ri=lightData.regionInteractions;
  ctx.fillStyle=sweepCol();
  for(let i=0;i<ri.length;i++){
    // Region turns on when simTime >= region's activity start time (in µs)
    const regionStartUs=ri[i].tStartUs;
    if(simTime<regionStartUs){
      // This region hasn't been reached yet — mask it
      const x0=L.m.l+(rb[i]/lightData.totalBins)*L.pw;
      const x1=L.m.l+(rb[i+1]/lightData.totalBins)*L.pw;
      ctx.fillRect(x0,L.m.t,x1-x0,L.ph);
    }
  }

  // Redraw axes and colorbar on top
  drawOpticalAxesAndColorbar();
}

function drawOpticalAxesAndColorbar(){
  if(!lightData||!lightData._layout) return;
  const L=lightData._layout;
  ctx.strokeStyle=axCol();ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(L.m.l,L.m.t);ctx.lineTo(L.m.l,L.m.t+L.ph);
  ctx.lineTo(L.m.l+L.pw,L.m.t+L.ph);ctx.stroke();
  ctx.fillStyle=txCol();ctx.font='13px monospace';ctx.textAlign='center';
  const rb=lightData.regionBounds;
  for(let i=0;i<lightData.regionInteractions.length;i++){
    const ri=lightData.regionInteractions[i];
    const cx2=L.m.l+((rb[i]+rb[i+1])/2/lightData.totalBins)*L.pw;
    ctx.fillText('I:'+ri.ids.join(','),cx2,L.m.t+L.ph+14);
    ctx.font='11px monospace';
    ctx.fillText(ri.tStartUs.toFixed(0)+'\u00B5s',cx2,L.m.t+L.ph+26);
    ctx.font='13px monospace';
  }
  ctx.textAlign='right';
  ctx.fillText(L.chStart,L.m.l-6,L.m.t+L.ph+4);
  ctx.fillText(L.chEnd-1,L.m.l-6,L.m.t+10);
  ctx.save();ctx.translate(12,L.m.t+L.ph/2);ctx.rotate(-Math.PI/2);
  ctx.textAlign='center';ctx.fillText('PMT Channel',0,0);ctx.restore();
  // Colorbar
  const bx=L.m.l+L.pw+6, bw=10, by=L.m.t, bh=L.ph;
  for(let y=0;y<bh;y++){const t=1-y/bh;const[r,g,b]=optCmapRGB(t);ctx.fillStyle=`rgb(${r},${g},${b})`;ctx.fillRect(bx,by+y,bw,1);}
  ctx.strokeStyle=axCol();ctx.strokeRect(bx,by,bw,bh);
  ctx.textAlign='left';ctx.font='12px monospace';ctx.fillStyle=txCol();
  const vm=L.vmax, lt=L.linthresh;
  for(let i=0;i<=4;i++){
    const tn=i/4;let val;
    if(tn===0.5) val=0;
    else if(tn>0.5){const f2=(tn-0.5)/0.5;val=lt*(Math.pow(vm/lt+1,f2)-1);}
    else{const f2=(0.5-tn)/0.5;val=-lt*(Math.pow(vm/lt+1,f2)-1);}
    const py=by+bh*(1-tn);
    const label=Math.abs(val)>=1000?(val/1000).toFixed(0)+'k':val.toFixed(0);
    ctx.fillText(label,bx+bw+3,py+3);
  }
  document.getElementById('label2dText').textContent=(curVol<0?'All':'Vol '+curVol)+' \u2014 Optical';
}

// ============================================================
// HIGHLIGHT / CORRESPONDENCE
// ============================================================
function highlightFromGroup(gid, volIdx){
  if(gid===hovGrp) return;
  hovGrp=gid;
  const vi=volIdx!=null?volIdx:(curVol>=0?curVol:0);
  const g2s=grpToSegPerVol[vi];

  ptMat.uniforms.noHover.value=0.0;
  hlBuf.fill(0);
  const segs=g2s?g2s.get(gid):null;
  // Find offset of this volume's segments in the concatenated buffer
  if(segs){
    let off=0;
    if(curVol<0){ for(let v=0;v<vi;v++) off+=volumes[v].n; }
    for(const si of segs) hlBuf[off+si]=1.0;
  }
  ptGeo.attributes.hl.needsUpdate=true;

  const vd=volumes[vi];
  const si0=segs?segs[0]:0;

  if(curViewMode==='optical'&&lightData){
    // Optical correspondence: find label id (interaction or ancestor) of this segment.
    // Requires labl/ for the per-deposit→track→interaction lookup.
    const intId=vd?getOptId(vd,si0):-1;
    renderOpticalFrame(intId>=0?intId:null);
    const trackId=(segs&&vd&&vd.tids)?vd.tids[si0]:'?';
    const pdgCode=(segs&&vd&&vd.pdg)?vd.pdg[si0]:0;
    const intStr=hasLabl?`Int ${intId}`:'(no labl: optical link unavailable)';
    setStatus(`Grp ${gid} \u00B7 Trk ${trackId} \u00B7 ${pdgName(pdgCode)} \u00B7 ${intStr} \u00B7 ${segs?segs.length:0} seg`);
  } else {
    // Wire plane correspondence
    const hlMap=new Map();
    for(let pi=0;pi<panels.length;pi++){
      if(panels[pi].vol!==vi) continue;
      const gpx=panels[pi].grpToPx.get(gid);
      if(!gpx||gpx.length===0) continue;
      let maxCh=0; for(const{ch} of gpx) if(ch>maxCh) maxCh=ch;
      const arr=[];
      for(const{pk,ch} of gpx){
        const di=panels[pi].pkToDispIdx.get(pk);
        if(di!==undefined) arr.push({idx:di, intensity:ch/maxCh});
      }
      if(arr.length>0) hlMap.set(pi,arr);
    }
    render2DFrame(hlMap);
    const nSeg=segs?segs.length:0;
    let avgDE=0;
    if(segs&&vd){for(const si2 of segs) avgDE+=vd.de[si2]; avgDE/=nSeg;}
    const trackId=(segs&&vd&&vd.tids)?vd.tids[si0]:'?';
    const pdgCode=(segs&&vd&&vd.pdg)?vd.pdg[si0]:0;
    let totalHL=0; for(const a of hlMap.values()) totalHL+=a.length;
    setStatus(`Grp ${gid} \u00B7 Trk ${trackId} \u00B7 ${pdgName(pdgCode)} (${pdgCode}) \u00B7 ${nSeg} seg \u00B7 dE ${avgDE.toFixed(3)} \u00B7 ${totalHL} px`);
  }
}

function highlightFromPixel(panelIdx,wire,time){
  const p=panels[panelIdx]; if(!p) return;
  const mod2=panelMod2(p);
  const pk=wire*mod2+time;
  const groups=p.pxToGrps.get(pk);
  if(!groups||groups.length===0){clearHL();return;}

  const vi=p.vol;
  const g2s=grpToSegPerVol[vi];

  ptMat.uniforms.noHover.value=0.0;
  hlBuf.fill(0);
  let off=0;
  if(curVol<0){ for(let v=0;v<vi;v++) off+=volumes[v].n; }
  let maxGCh=0;
  for(const{ch} of groups) if(ch>maxGCh) maxGCh=ch;
  for(const{gid,ch} of groups){
    const intensity=ch/maxGCh;
    const segs=g2s?g2s.get(gid):null;
    if(segs) for(const si of segs) hlBuf[off+si]=Math.max(hlBuf[off+si],intensity);
  }
  ptGeo.attributes.hl.needsUpdate=true;

  const hlMap=new Map();
  const di=p.pkToDispIdx.get(pk);
  if(di!==undefined) hlMap.set(panelIdx,[{idx:di,intensity:1.0}]);
  render2DFrame(hlMap);

  const pd=volumes[vi]&&volumes[vi].planes[p.plane];
  const totalCh=pd?pd.dispCH[di!==undefined?di:0]:0;
  const topG=groups.slice().sort((a,b)=>b.ch-a.ch).slice(0,5);
  const parts=topG.map(g=>`g${g.gid}:${(g.ch/totalCh*100).toFixed(0)}%`).join(' ');
  setStatus(`V${vi} ${p.plane} Wire ${wire} \u00B7 Time ${time} \u00B7 Charge ${totalCh.toFixed(0)} \u00B7 ${groups.length} groups [${parts}]`);
}

function clearHL(){
  hovGrp=-1;
  if(ptMat){ptMat.uniforms.softHL.value=0.0;}
  if(ptMat&&hlBuf){
    if(selectedTrack!==null){
      ptMat.uniforms.noHover.value=0.0;
      applyTrackFilter();
      return;
    } else {
      ptMat.uniforms.noHover.value=1.0;
      hlBuf.fill(0);
    }
    ptGeo.attributes.hl.needsUpdate=true;
  }
  if(curViewMode==='optical') renderOpticalFrame(null);
  else render2DFrame(null);
  setStatus('');
}

function setStatus(s){document.getElementById('status').textContent=s||'Hover to explore correspondence';}

async function toggleTheme(){
  lightMode=!lightMode;
  showOverlay('Switching to '+(lightMode?'light':'dark')+' mode...');
  await new Promise(r=>setTimeout(r,50));
  BG=lightMode?BG_LIGHT:BG_DARK;
  document.body.classList.toggle('light-mode',lightMode);
  document.querySelector('.ts-dark').classList.toggle('active',!lightMode);
  document.querySelector('.ts-light').classList.toggle('active',lightMode);
  if(renderer) renderer.setClearColor(BG);
  // Rebuild 3D overlays and colormap with theme-appropriate colors
  buildVolBoxes();
  buildAnodeCathode();
  if(ptMat&&curColor==='de') ptMat.uniforms.cmap.value=deTex();
  // Recompute 2D colors with new colormaps
  if(curViewMode==='optical'&&lightData){
    renderOptical();
    renderOpticalFrame(null);
  } else {
    precomputeAllPanelColors();
    render2DBase();
    render2DFrame(null);
  }
  hideOverlay();
}

// ============================================================
// SAVE / COPY
// ============================================================
function saveFilename(panel){
  const evt='evt'+String(curEvent).padStart(3,'0');
  const mode=curViewMode==='sensor'?'sensor':curViewMode==='optical'?'optical':'hits';
  if(panel==='3d') return `${evt}_3d_${curColor==='de'?'de':curColor}.png`;
  if(expandedPanel>=0){
    const p=panels[expandedPanel];
    return `${evt}_V${p.vol}_${p.plane}_${mode}.png`;
  }
  const volSet=[...new Set(panels.map(p=>p.vol))].map(v=>'V'+v).join('-');
  return `${evt}_${volSet}_${mode}.png`;
}

function canvasToBlob(canvas){
  return new Promise(r=>canvas.toBlob(r,'image/png'));
}

function downloadBlob(blob,filename){
  const url=URL.createObjectURL(blob);
  const a=document.createElement('a');
  a.href=url; a.download=filename; a.click();
  URL.revokeObjectURL(url);
}

function showToast(msg){
  let t=document.getElementById('toast');
  if(!t){t=document.createElement('div');t.id='toast';document.body.appendChild(t);}
  t.textContent=msg;t.classList.add('visible');
  clearTimeout(t._tid);
  t._tid=setTimeout(()=>t.classList.remove('visible'),1500);
}

async function save3D(){
  const blob=await canvasToBlob(renderer.domElement);
  downloadBlob(blob,saveFilename('3d'));
}

async function copy3D(){
  const blob=await canvasToBlob(renderer.domElement);
  await navigator.clipboard.write([new ClipboardItem({'image/png':blob})]);
  showToast('3D copied to clipboard');
}

async function save2D(){
  const blob=await canvasToBlob(c2d);
  downloadBlob(blob,saveFilename('2d'));
}

async function copy2D(){
  const blob=await canvasToBlob(c2d);
  await navigator.clipboard.write([new ClipboardItem({'image/png':blob})]);
  showToast('2D copied to clipboard');
}

// ============================================================
// 3D INTERACTION (GPU PICKING)
// ============================================================
function on3dMove(e){
  if(!corrMode||is2dHover||curViewMode==='sensor') return;
  pendingPick={x:e.offsetX, y:e.offsetY};
}

function performPick(x,y){
  const dpr=renderer.getPixelRatio();
  const px=Math.round(x*dpr), py=Math.round(pickTarget.height-y*dpr);

  renderer.setRenderTarget(pickTarget);
  renderer.setClearColor(0x000000,0);
  renderer.clear();
  renderer.render(pickScene,camera);
  renderer.setRenderTarget(null);
  renderer.setClearColor(BG);

  // Read a neighborhood and find the nearest hit
  const R=8;
  const sz=R*2+1;
  const x0=Math.max(0,px-R), y0=Math.max(0,py-R);
  const w=Math.min(sz,pickTarget.width-x0), h=Math.min(sz,pickTarget.height-y0);
  if(w<=0||h<=0){clearHL();return;}
  const buf=new Uint8Array(w*h*4);
  renderer.readRenderTargetPixels(pickTarget,x0,y0,w,h,buf);

  let bestDist=Infinity, bestId=0;
  const cx=px-x0, cy=py-y0;
  for(let iy=0;iy<h;iy++){
    for(let ix=0;ix<w;ix++){
      const off=(iy*w+ix)*4;
      const id=buf[off]|(buf[off+1]<<8)|(buf[off+2]<<16);
      if(id===0) continue;
      const dx=ix-cx, dy=iy-cy, dist=dx*dx+dy*dy;
      if(dist<bestDist){bestDist=dist;bestId=id;}
    }
  }

  if(bestId===0){clearHL();return;}
  is2dHover=false;
  const segIdx=bestId-1; // concatenated segment index
  // Determine volume and local segment index
  let volIdx=-1, localIdx=segIdx;
  const volsUsed=curVol<0?volumes:[(curVol>=0&&volumes[curVol]?volumes[curVol]:{n:0})];
  const volIdxStart=curVol<0?0:curVol;
  let off=0;
  for(let vi=0;vi<volsUsed.length;vi++){
    const v=volsUsed[vi]; if(!v||v.n===0) continue;
    if(segIdx<off+v.n){volIdx=volIdxStart+vi;localIdx=segIdx-off;break;}
    off+=v.n;
  }
  if(volIdx<0){clearHL();return;}
  const gid=volumes[volIdx].gids[localIdx];
  highlightFromGroup(gid,volIdx);
}

// ============================================================
// 2D INTERACTION (panel-aware)
// ============================================================
function mouseToPanel(mx,my,pi){
  const p=panels[pi], r=p.rect;
  const lx=mx-r.x-margin.l, ly=my-r.y-margin.t;
  if(lx<0||lx>r.pw||ly<0||ly>r.ph) return null;
  const fx=lx/r.pw, fy=ly/r.ph;
  return{fx,fy,wire:Math.round(p.view.x0+fx*(p.view.x1-p.view.x0)),
         time:Math.round(p.view.y1-fy*(p.view.y1-p.view.y0))};
}

function on2dMove(e){
  if(isPan&&panPanel>=0&&curViewMode!=='optical'){
    const p=panels[panPanel], r=p.rect, v=p.view, vm=p.viewMax;
    const dx=e.offsetX-panLast.x, dy=e.offsetY-panLast.y;
    panLast={x:e.offsetX,y:e.offsetY};
    const dw=-dx/r.pw*(v.x1-v.x0), dt=dy/r.ph*(v.y1-v.y0);
    v.x0+=dw;v.x1+=dw;v.y0+=dt;v.y1+=dt;
    if(v.x0<vm.x0){v.x1+=vm.x0-v.x0;v.x0=vm.x0;}
    if(v.x1>vm.x1){v.x0-=v.x1-vm.x1;v.x1=vm.x1;}
    if(v.y0<vm.y0){v.y1+=vm.y0-v.y0;v.y0=vm.y0;}
    if(v.y1>vm.y1){v.y0-=v.y1-vm.y1;v.y1=vm.y1;}
    render2DBase();render2DFrame(null);
    return;
  }
  if(!corrMode||curViewMode==='sensor') return;
  // Optical → 3D correspondence
  if(curViewMode==='optical'){
    const optId=opticalAtMouse(e.offsetX,e.offsetY);
    if(optId!==null){is2dHover=true;highlightFromOptical(optId);}
    else if(is2dHover){is2dHover=false;clearHL();}
    return;
  }
  const pi=panelAtMouse(e.offsetX,e.offsetY);
  if(pi<0){if(is2dHover){is2dHover=false;clearHL();}return;}
  const m=mouseToPanel(e.offsetX,e.offsetY,pi);
  if(!m){if(is2dHover){is2dHover=false;clearHL();}return;}
  const p=panels[pi], mod2=panelMod2(p);
  const pk=m.wire*mod2+m.time;
  if(p.pkToDispIdx.has(pk)){
    is2dHover=true;
    highlightFromPixel(pi,m.wire,m.time);
  } else {
    let found=false;
    const vw=p.view.x1-p.view.x0;
    const r2=Math.max(1,Math.ceil(vw/p.rect.pw));
    for(let dw=-r2;dw<=r2&&!found;dw++){
      for(let dt=-r2;dt<=r2&&!found;dt++){
        const npk=(m.wire+dw)*mod2+(m.time+dt);
        if(p.pkToDispIdx.has(npk)){
          is2dHover=true;
          highlightFromPixel(pi,m.wire+dw,m.time+dt);
          found=true;
        }
      }
    }
    if(!found&&is2dHover){is2dHover=false;clearHL();}
  }
}

function on2dWheel(e){
  e.preventDefault();
  if(curViewMode==='optical') return;
  const pi=panelAtMouse(e.offsetX,e.offsetY);
  if(pi<0) return;
  const p=panels[pi], r=p.rect, v=p.view, vm=p.viewMax;
  const m=mouseToPanel(e.offsetX,e.offsetY,pi);
  if(!m) return;
  const factor=e.deltaY>0?1.15:1/1.15;
  let nw=(v.x1-v.x0)*factor, nh=(v.y1-v.y0)*factor;
  const maxW=vm.x1-vm.x0, maxH=vm.y1-vm.y0;
  if(nw>maxW) nw=maxW; if(nh>maxH) nh=maxH;
  v.x0=m.wire-m.fx*nw; v.x1=m.wire+(1-m.fx)*nw;
  v.y0=m.time-(1-m.fy)*nh; v.y1=m.time+m.fy*nh;
  if(v.x0<vm.x0){v.x1+=vm.x0-v.x0;v.x0=vm.x0;}
  if(v.x1>vm.x1){v.x0-=v.x1-vm.x1;v.x1=vm.x1;}
  if(v.y0<vm.y0){v.y1+=vm.y0-v.y0;v.y0=vm.y0;}
  if(v.y1>vm.y1){v.y0-=v.y1-vm.y1;v.y1=vm.y1;}
  render2DBase();render2DFrame(null);
}

function on2dDown(e){
  if(curViewMode==='optical') return;
  if(e.button===0){
    panPanel=panelAtMouse(e.offsetX,e.offsetY);
    isPan=panPanel>=0;
    panLast={x:e.offsetX,y:e.offsetY};
    if(isPan) c2d.style.cursor='grabbing';
  }
}

// ============================================================
// UI CONTROLS
// ============================================================
function setupUI(){
  // Save / Copy buttons
  document.getElementById('save3d').addEventListener('click',save3D);
  document.getElementById('copy3d').addEventListener('click',copy3D);
  document.getElementById('save2d').addEventListener('click',save2D);
  document.getElementById('copy2d').addEventListener('click',copy2D);

  document.getElementById('volSelect').addEventListener('change',e=>{
    const v=e.target.value;
    curVol=v==='all'?-1:parseInt(v);
    switchVolume();
  });
  // Color mode: dE vs LABEL toggle
  document.getElementById('colorModeGrp').addEventListener('click',e=>{
    const b=e.target.closest('button');if(!b) return;
    if(b.dataset.v==='label'&&!hasLabl) return;  // no per-track labels available
    curColorMode=b.dataset.v;
    document.querySelectorAll('#colorModeGrp button').forEach(x=>x.classList.toggle('active',x===b));
    const isLabel=curColorMode==='label';
    document.getElementById('labelSelect').style.display=isLabel?'':'none';
    document.getElementById('catFilterSelect').style.display=isLabel?'':'none';
    document.getElementById('catFilterInput').style.display=isLabel?'':'none';
    curColor=isLabel?curLabel:'de';
    if(isLabel) populateCatFilter();
    selectedTrack=null;
    switchColorMode();
  });
  document.getElementById('labelSelect').addEventListener('change',e=>{
    curLabel=e.target.value;
    curColor=curLabel;
    populateCatFilter();
    selectedTrack=null;
    switchColorMode();
  });
  document.getElementById('catFilterSelect').addEventListener('change',e=>{
    const v=e.target.value;
    selectedTrack=v==='all'?null:parseInt(v);
    applyTrackFilter();
  });
  document.getElementById('catFilterInput').addEventListener('keydown',e=>{
    if(e.key==='Enter'){
      const v=e.target.value.trim();
      selectedTrack=v===''?null:parseInt(v);
      document.getElementById('catFilterSelect').value='all';
      applyTrackFilter();
    }
  });
  // Event navigation
  const evInput=document.getElementById('evInput');
  document.getElementById('evPrev').addEventListener('click',()=>{
    const n=Math.max(0,curEvent-1);
    if(n!==curEvent) loadEvent(n);
  });
  document.getElementById('evNext').addEventListener('click',()=>{
    const n=Math.min(nEvents-1,curEvent+1);
    if(n!==curEvent) loadEvent(n);
  });
  evInput.addEventListener('keydown',e=>{
    if(e.key==='Enter'){
      const n=Math.max(0,Math.min(nEvents-1,parseInt(evInput.value)||0));
      if(n!==curEvent) loadEvent(n);
      else evInput.value=curEvent;
    }
  });

  document.getElementById('driftBtn').addEventListener('click',()=>{
    driftMode=!driftMode;
    document.getElementById('driftBtn').classList.toggle('active',driftMode);
    document.getElementById('driftBar').classList.toggle('visible',driftMode);
    if(driftMode){
      simTime=0; driftPlaying=true; loopPaused=false; lastFrameTime=0;
      updateDriftUI();
      if(ptMat){ptMat.uniforms.driftOn.value=1.0;ptMat.uniforms.simTime.value=simTime;}
      // Set scrubber range
      const scrub=document.getElementById('driftScrubber');
      scrub.min=0; scrub.max=simTimeMax; scrub.step=simTimeMax/1000;
      scrub.value=0;
    } else {
      driftPlaying=false;
      if(ptMat){ptMat.uniforms.driftOn.value=0.0;}
      if(curViewMode==='optical') renderOpticalFrame(null);
      else render2DFrame(null);
    }
  });
  document.getElementById('driftPlayPause').addEventListener('click',()=>{
    driftPlaying=!driftPlaying;
    loopPaused=false;
    lastFrameTime=0;
    updateDriftUI();
  });
  document.getElementById('driftScrubber').addEventListener('input',(e)=>{
    simTime=parseFloat(e.target.value);
    driftPlaying=false;
    loopPaused=false;
    updateDriftUI();
    if(ptMat) ptMat.uniforms.simTime.value=simTime;
    if(curViewMode==='optical') renderOpticalDrift();
    else render2DFrame(null);
  });
  document.getElementById('corrBtn').addEventListener('click',()=>{
    corrMode=!corrMode;
    document.getElementById('corrBtn').classList.toggle('active',corrMode);
    if(!corrMode) clearHL();
  });
  document.getElementById('rotBtn').addEventListener('click',()=>{
    autoRotate=!autoRotate;
    controls.autoRotate=autoRotate;
    document.getElementById('rotBtn').classList.toggle('active',autoRotate);
  });
  document.getElementById('fsBtn').addEventListener('click',()=>{
    if(!document.fullscreenElement) document.documentElement.requestFullscreen();
    else document.exitFullscreen();
  });
  document.addEventListener('fullscreenchange',()=>{
    document.getElementById('fsBtn').classList.toggle('active',!!document.fullscreenElement);
  });
  // View mode (HITS / SENSOR)
  document.getElementById('viewGrp').addEventListener('click',e=>{
    const b=e.target.closest('button');if(!b) return;
    const newMode=b.dataset.v;
    if(newMode===curViewMode) return;
    curViewMode=newMode;
    document.querySelectorAll('#viewGrp button').forEach(x=>x.classList.toggle('active',x===b));
    switchViewMode();
  });

  // Settings panel
  document.getElementById('themeSwitch').addEventListener('click',toggleTheme);
  document.getElementById('settingsBtn').addEventListener('click',()=>{
    document.getElementById('settingsPanel').classList.toggle('visible');
  });
  document.getElementById('settingsClose').addEventListener('click',()=>{
    document.getElementById('settingsPanel').classList.remove('visible');
  });
  document.getElementById('gammaSlider').addEventListener('input',e=>{
    sensorGamma=parseFloat(e.target.value);
    document.getElementById('gammaVal').textContent=sensorGamma.toFixed(2);
    if(curViewMode==='sensor'){
      showOverlay('Recomputing...');
      setTimeout(()=>{precomputeAllPanelColors();render2DBase();render2DFrame(null);hideOverlay();},50);
    }
  });
  document.getElementById('deadbandInput').addEventListener('change',e=>{
    sensorDeadband=parseFloat(e.target.value)||0;
    if(curViewMode==='sensor'){
      showOverlay('Recomputing...');
      setTimeout(()=>{precomputeAllPanelColors();render2DBase();render2DFrame(null);hideOverlay();},50);
    }
  });

  // Optical settings
  document.getElementById('optThreshInput').addEventListener('change',async e=>{
    optActivityThresh=parseFloat(e.target.value)||100;
    if(curViewMode==='optical'){
      showOverlay('Recomputing optical...');
      lightLoadedFor=-1;
      await loadLight(curEvent,true);
      if(lightData){renderOptical();renderOpticalFrame(null);}
      hideOverlay();
    }
  });
  document.getElementById('optGapInput').addEventListener('change',async e=>{
    optGapUs=parseFloat(e.target.value)||5.0;
    if(curViewMode==='optical'){
      showOverlay('Recomputing optical...');
      lightLoadedFor=-1;
      await loadLight(curEvent,true);
      if(lightData){renderOptical();renderOpticalFrame(null);}
      hideOverlay();
    }
  });

  document.getElementById('emphSlider').addEventListener('input',e=>{
    deEmphAmt=parseFloat(e.target.value);
    document.getElementById('emphVal').textContent=Math.round(deEmphAmt*100)+'%';
    if(ptMat) ptMat.uniforms.emphAmt.value=deEmphAmt;
  });
  document.getElementById('emphPowSlider').addEventListener('input',e=>{
    deEmphPow=parseFloat(e.target.value);
    document.getElementById('emphPowVal').textContent=deEmphPow.toFixed(1);
    if(ptMat) ptMat.uniforms.emphPow.value=deEmphPow;
  });

  document.getElementById('volBoundsChk').addEventListener('change',e=>{
    showVolBounds=e.target.checked;
    buildVolBoxes();
  });
  document.getElementById('anodeCathodeChk').addEventListener('change',e=>{
    showAnodeCathode=e.target.checked;
    buildAnodeCathode();
  });

  // Drift settings
  document.getElementById('driftSpeedSlider').addEventListener('input',e=>{
    driftSpeed=parseFloat(e.target.value);
    document.getElementById('driftSpeedVal').textContent=driftSpeed.toFixed(0);
  });
  document.getElementById('fadeDurSlider').addEventListener('input',e=>{
    fadeDuration=parseFloat(e.target.value);
    document.getElementById('fadeDurVal').textContent=fadeDuration.toFixed(0);
    if(ptMat) ptMat.uniforms.fadeDur.value=fadeDuration;
  });
  document.getElementById('loopPauseSlider').addEventListener('input',e=>{
    loopPauseMs=parseFloat(e.target.value)*1000;
    document.getElementById('loopPauseVal').textContent=parseFloat(e.target.value).toFixed(1);
  });

  document.getElementById('resetBtn').addEventListener('click',()=>{
    selectedTrack=null;
    curColorMode='de'; curColor='de';
    document.querySelectorAll('#colorModeGrp button').forEach(b=>b.classList.toggle('active',b.dataset.v==='de'));
    document.getElementById('labelSelect').style.display='none';
    document.getElementById('catFilterSelect').style.display='none';
    document.getElementById('catFilterInput').style.display='none';
    curViewMode='hits';
    document.querySelectorAll('#viewGrp button').forEach(b=>b.classList.toggle('active',b.dataset.v==='hits'));
    curVol=-1;
    document.getElementById('volSelect').value='all';
    driftMode=false;
    document.getElementById('driftBtn').classList.remove('active');
    document.getElementById('driftBar').classList.remove('visible');
    if(ptMat) ptMat.uniforms.driftOn.value=0.0;
    updateCorrBtnState();
    switchVolume();
  });
  window.addEventListener('resize',onResize);
}

async function switchVolume(){
  showOverlay('Loading volume...');
  await new Promise(r=>setTimeout(r,50));
  clearHL();
  expandedPanel=-1;
  document.getElementById('panelCloseBtn').classList.remove('visible');
  selectedTrack=null;
  document.getElementById('catFilterSelect').value='all';
  document.getElementById('catFilterInput').value='';
  updateCorrBtnState();
  buildPanels();
  rebuildAllLookups();
  computeTopTracks();
  populateCatFilter();
  computePanelDomIds();
  buildPoints();
  // Update drift scrubber range and reset time
  if(driftMode){
    simTime=0;
    const scrub=document.getElementById('driftScrubber');
    scrub.min=0; scrub.max=simTimeMax; scrub.step=simTimeMax/1000;
    scrub.value=0;
    if(ptMat) ptMat.uniforms.simTime.value=0;
    loopPaused=false;
    updateDriftUI();
  }
  precomputeAllPanelColors();
  resetPanelViews();
  if(curViewMode==='optical'&&lightData){
    renderOptical();
    renderOpticalFrame(null);
  } else {
    render2DBase();
    render2DFrame(null);
  }
  hideOverlay();
}


function switchColorMode(){
  const isCat=curColor!=='de'; // categorical mode
  if(ptMat&&ptGeo&&_3dTids){
    ptMat.uniforms.useTrack.value=isCat?1.0:0.0;
    ptMat.uniforms.cmap.value=isCat?trackTex:deTex();
    // For categorical modes: rewrite trackVal (hue) and trackWt (emphasis) buffers
    if(isCat){
      const n=_3dTids.length;
      const tvAttr=ptGeo.attributes.trackVal;
      const twAttr=ptGeo.attributes.trackWt;
      const tv=tvAttr.array, tw=twAttr.array;
      let ids,wt;
      if(curColor==='track'){ids=_3dTids;wt=_3dTrkWt;}
      else if(curColor==='pdg'){ids=_3dPdg;wt=_3dPdgWt;}
      else if(curColor==='ancestor'){ids=_3dAnc;wt=_3dAncWt;}
      else{ids=_3dInt;wt=_3dIntWt;}
      for(let i=0;i<n;i++){
        tv[i]=(Math.abs(ids[i])*0.618033988749895)%1.0;
        tw[i]=wt[i];
      }
      tvAttr.needsUpdate=true;
      twAttr.needsUpdate=true;
    }
  }
  if(curViewMode==='optical'&&lightData){
    renderOptical();
    renderOpticalFrame(null);
  } else {
    precomputeAllPanelColors();
    render2DBase();
    render2DFrame(null);
  }
}

function updateCorrBtnState(){
  const btn=document.getElementById('corrBtn');
  if(curViewMode==='sensor'){
    btn.classList.remove('active');
    btn.classList.add('disabled');
    btn.textContent='\u2715 CORR';
  } else {
    btn.classList.remove('disabled');
    btn.classList.toggle('active',corrMode);
    btn.textContent='CORR';
  }
}

async function switchViewMode(){
  const label=curViewMode==='sensor'?'sensor':curViewMode==='optical'?'optical':'hits';
  showOverlay('Switching to '+label+'...');
  await new Promise(r=>setTimeout(r,50));
  // Collapse expanded panel when switching modes
  if(expandedPanel>=0){expandedPanel=-1;document.getElementById('panelCloseBtn').classList.remove('visible');updatePanelRects();}
  clearHL();
  updateCorrBtnState();
  if(curViewMode==='sensor') await loadSensor(curEvent);
  if(curViewMode==='optical') await loadLight(curEvent);
  if(curViewMode==='optical'&&lightData){
    renderOptical();
    renderOpticalFrame(null);
  } else {
    precomputeAllPanelColors();
    resetPanelViews();
    render2DBase();
    render2DFrame(null);
  }
  hideOverlay();
}

function onResize(){
  const el=document.getElementById('panel3d');
  renderer.setSize(el.clientWidth,el.clientHeight);
  camera.aspect=el.clientWidth/el.clientHeight;
  camera.updateProjectionMatrix();
  pickTarget.setSize(el.clientWidth*renderer.getPixelRatio(), el.clientHeight*renderer.getPixelRatio());
  resize2D();
  render2DBase();
  render2DFrame(null);
}

// ============================================================
// ANIMATION
// ============================================================
// CAMERA ANIMATION
// ============================================================
function animateCamera(targetPos,targetLookAt,dur){
  camAnim={startPos:camera.position.clone(),startTarget:controls.target.clone(),
    endPos:targetPos.clone(),endTarget:targetLookAt.clone(),
    startTime:performance.now(),duration:dur||500};
}

// ============================================================
// ANIMATION
// ============================================================
function animate(){
  requestAnimationFrame(animate);
  const now=performance.now();
  const dt=lastFrameTime?now-lastFrameTime:16;
  lastFrameTime=now;

  // Camera animation
  if(camAnim){
    const t=Math.min((now-camAnim.startTime)/camAnim.duration,1);
    const e=t*(2-t);
    camera.position.lerpVectors(camAnim.startPos,camAnim.endPos,e);
    controls.target.lerpVectors(camAnim.startTarget,camAnim.endTarget,e);
    if(t>=1) camAnim=null;
  }

  // Drift animation
  if(driftMode&&ptMat){
    ptMat.uniforms.driftOn.value=1.0;
    ptMat.uniforms.fadeDur.value=fadeDuration;
    if(driftPlaying){
      if(!loopPaused){
        simTime+=driftSpeed*(dt/1000);
        if(simTime>=simTimeMax){
          simTime=simTimeMax;
          loopPaused=true; loopPauseStart=now;
        }
      } else {
        if(now-loopPauseStart>loopPauseMs){
          simTime=0;
          loopPaused=false;
        }
      }
    }
    ptMat.uniforms.simTime.value=simTime;
    updateDriftUI();
    if(curViewMode==='optical') renderOpticalDrift();
    else render2DFrame(null);
  }

  controls.update();
  if(pendingPick){performPick(pendingPick.x,pendingPick.y);pendingPick=null;}
  renderer.render(scene,camera);
}

function populateVolSelect(){
  const sel=document.getElementById('volSelect');
  const prev=sel.value;
  sel.innerHTML='<option value="all">All Volumes</option>';
  for(let v=0;v<nVolumes;v++){
    const o=document.createElement('option');
    o.value=v;
    const vol=volumes[v];
    const n=vol?vol.n:0;
    let nPx=0;
    if(vol) for(const pl of planeLabels) if(vol.planes[pl]) nPx+=vol.planes[pl].nDisp||0;
    if(n>0){
      const label=n>=1000?(n/1000).toFixed(1)+'k':String(n);
      o.textContent='Vol '+v+' ('+label+(nPx===0?' \u2013 no signal':'')+ ')';
      if(nPx===0){ o.disabled=true; o.style.color='#888'; }
    } else {
      o.textContent='Vol '+v+' (empty)';
      o.disabled=true;
      o.style.color='#888';
    }
    sel.appendChild(o);
  }
  // Restore previous selection if still valid
  if(prev&&prev!=='all'&&sel.querySelector(`option[value="${prev}"]`)&&!sel.querySelector(`option[value="${prev}"][disabled]`)) sel.value=prev;
  else sel.value='all';
}

// ============================================================
// INIT
// ============================================================
(async function(){
  const ls=document.getElementById('loadStatus');
  try{
    ls.textContent='Initializing h5wasm...';
    createWorker();

    const params=new URLSearchParams(window.location.search);
    const base=params.get('base')||'';

    ls.textContent='Fetching manifest...';
    const mr=await fetch(base+'/manifest.json');
    if(!mr.ok) throw new Error('Cannot fetch manifest.json');
    const manifest=await mr.json();

    ls.textContent='Mounting HDF5 files...';
    const cfg=await workerCall('init',{base,manifest});
    nEvents=cfg.nEvents;
    nVolumes=cfg.nVolumes;
    planeLabels=cfg.planeLabels;
    volRanges=cfg.volRanges;
    velocityMmUs=cfg.velocityMmUs||1.6;
    volAnodes=cfg.volAnodes||[];
    volDriftDirs=cfg.volDriftDirs||[];
    timeStepUs=cfg.timeStepUs||0.5;
    isPixelMode=!!cfg.isPixel;
    numPy=cfg.numPy||0;
    numPz=cfg.numPz||0;
    const readoutWindowUs=cfg.readoutWindowUs||1350;
    hasLabl=!!cfg.hasLabl;
    hasOptical=!!cfg.hasOptical;
    optConfig=cfg.optConfig||null;
    if(optConfig){nPmtsPerSide=Math.floor(optConfig.nChannels/2);optLabelKey=optConfig.labelKey||'interaction';}
    // Show/hide optical button
    const ob=document.getElementById('opticalBtn');
    if(ob) ob.style.display=hasOptical?'':'none';
    // Hide LABEL color mode + selects when no labl file is present.
    // Per-track metadata (track/pdg/ancestor/interaction) is unavailable.
    if(!hasLabl){
      const lblBtn=document.querySelector('#colorModeGrp button[data-v="label"]');
      if(lblBtn) lblBtn.style.display='none';
      const ls2=document.getElementById('labelSelect');
      if(ls2) ls2.style.display='none';
      const cfs=document.getElementById('catFilterSelect');
      if(cfs) cfs.style.display='none';
      const cfi=document.getElementById('catFilterInput');
      if(cfi) cfi.style.display='none';
      curColorMode='de'; curColor='de';
    }

    document.getElementById('loading').style.display='none';
    document.getElementById('app').style.display='flex';
    document.getElementById('evInput').max=nEvents-1;
    document.getElementById('evTotal').textContent='/ '+(nEvents-1);

    // Build min_wire_indices (assume 0 for U/V, 1 for Y — standard)
    const mwi=[];
    for(let v=0;v<nVolumes;v++){const row=[];for(let p=0;p<planeLabels.length;p++) row.push(0);mwi.push(row);}
    D={config:{max_time:cfg.maxTime,num_wires_actual:cfg.numWires,min_wire_indices:mwi}};

    populateVolSelect();

    initThree();
    init2D();
    setupUI();

    showOverlay('Loading event 0...');
    const d=await workerCall('loadEvent',{idx:0});
    D.config.event_idx=d.config.event_idx;
    volumes=d.volumes;
    populateVolSelect();

    // Start in All mode
    curVol=-1;
    updateCorrBtnState();
    buildPanels();
    rebuildAllLookups();
    computeTopTracks();
    populateCatFilter();
    computePanelDomIds();
    buildPoints();
    precomputeAllPanelColors();
    resetPanelViews();
    render2DBase();
    render2DFrame(null);
    vizReady=true;
    animate();
    hideOverlay();
    setStatus('Event 0 (source '+d.config.event_idx+') \u2014 All volumes');
  }catch(e){
    ls.textContent='Error: '+e.message+'. Run: ./view.sh production_run/';
    console.error(e);
  }
})();
