// Colormap definitions and interpolation for JAXTPC viewer.

// --- Stop definitions ---
// Each stop is [t, color] where color is either a hex string or [r,g,b] array.

// Dark mode
export const WARM_STOPS = [[0,'#080808'],[.12,'#2a0a02'],[.3,'#7a2200'],[.5,'#b85800'],[.7,'#ee8800'],[.88,'#ffcc55'],[1,'#fffde0']];
export const OBSIDIAN_STOPS = [[0,[224,255,255]],[.2,[0,229,255]],[.35,[0,136,170]],[.5,[10,10,10]],[.65,[170,85,0]],[.8,[255,136,0]],[1,[255,238,204]]];
// Light mode
export const SEISMIC_STOPS = [[0,[0,0,128]],[.25,[0,0,255]],[.5,[255,255,255]],[.75,[255,0,0]],[1,[128,0,0]]];
export const INFERNO_R_STOPS = [[0,'#fcffa4'],[.2,'#fca50a'],[.4,'#dd513a'],[.6,'#932667'],[.8,'#420a68'],[1,'#0d0829']];

// --- Utilities ---

export function parseHex(h) {
  return [parseInt(h.slice(1,3),16), parseInt(h.slice(3,5),16), parseInt(h.slice(5,7),16)];
}

function toRGB(c) { return typeof c === 'string' ? parseHex(c) : c; }

/** Generic piecewise-linear colormap interpolation. */
export function interpolate(stops, t) {
  t = Math.max(0, Math.min(1, t));
  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, c0] = stops[i], [t1, c1] = stops[i+1];
    if (t <= t1) {
      const f = (t - t0) / (t1 - t0);
      const a = toRGB(c0), b = toRGB(c1);
      return [Math.round(a[0]+f*(b[0]-a[0])), Math.round(a[1]+f*(b[1]-a[1])), Math.round(a[2]+f*(b[2]-a[2]))];
    }
  }
  return toRGB(stops[stops.length - 1][1]);
}

// --- Named colormap functions ---

export function warmRGB(t)     { return interpolate(WARM_STOPS, t); }
export function obsidianRGB(t) { return interpolate(OBSIDIAN_STOPS, t); }
export function seismicRGB(t)  { return interpolate(SEISMIC_STOPS, t); }
export function infernoRGB(t)  { return interpolate(INFERNO_R_STOPS, t); }

// --- HSL → RGB ---

export function hsl2rgb(h, s, l) {
  const c = (1 - Math.abs(2*l - 1)) * s, x = c * (1 - Math.abs((h*6) % 2 - 1)), m = l - c/2;
  let r = 0, g = 0, b = 0;
  const i = Math.floor(h * 6) % 6;
  if (i === 0) { r = c; g = x; } else if (i === 1) { r = x; g = c; } else if (i === 2) { g = c; b = x; }
  else if (i === 3) { g = x; b = c; } else if (i === 4) { r = x; b = c; } else { r = c; b = x; }
  return [Math.round((r+m)*255), Math.round((g+m)*255), Math.round((b+m)*255)];
}

// --- Deadband normalization ---

export function dbNorm(v, vmin, vmax, db, gamma) {
  const df = db > 0 ? .08 : 0, half = df/2, sr = .5 - half;
  if (db > 0) {
    if (v < -db) { const d = -db - vmin; if (Math.abs(d) < 1e-30) return .5; return (.5 - half) - sr * Math.pow(Math.min(Math.max((-db - v) / d, 0), 1), gamma); }
    if (v > db)  { const d = vmax - db;   if (Math.abs(d) < 1e-30) return .5; return (.5 + half) + sr * Math.pow(Math.min(Math.max((v - db) / d, 0), 1), gamma); }
    return .5;
  }
  if (v < 0) { if (Math.abs(vmin) < 1e-30) return .5; return .5 - .5 * Math.pow(Math.min(Math.max(-v / (-vmin), 0), 1), gamma); }
  if (v > 0) { if (Math.abs(vmax) < 1e-30) return .5; return .5 + .5 * Math.pow(Math.min(Math.max(v / vmax, 0), 1), gamma); }
  return .5;
}

export function dbInverse(y, vmin, vmax, db, gamma) {
  const df = db > 0 ? .08 : 0, half = df/2, sr = .5 - half;
  if (db > 0) {
    if (y < .5 - half) { const tg = Math.min(Math.max(((.5 - half) - y) / sr, 0), 1); return -db - Math.pow(tg, 1/gamma) * (-db - vmin); }
    if (y > .5 + half) { const tg = Math.min(Math.max((y - (.5 + half)) / sr, 0), 1); return db + Math.pow(tg, 1/gamma) * (vmax - db); }
    return ((y - (.5 - half)) / df) * 2 * db - db;
  }
  if (y < .5) { const tg = Math.min(Math.max((.5 - y) / .5, 0), 1); return -Math.pow(tg, 1/gamma) * (-vmin); }
  if (y > .5) { const tg = Math.min(Math.max((y - .5) / .5, 0), 1); return Math.pow(tg, 1/gamma) * vmax; }
  return 0;
}
