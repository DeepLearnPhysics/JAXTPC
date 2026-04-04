// GLSL shaders for the JAXTPC 3D point cloud renderer.

export const VS = `
attribute float deVal;
attribute float trackVal;
attribute float trackWt;
attribute float hl;
attribute float segT0;
attribute float segArrival;
attribute float segXAnode;
uniform float noHover;
uniform float useTrack;
uniform float emphPow;
uniform float emphAmt;
uniform float simTime;
uniform float driftOn;
uniform float fadeDur;
uniform float softHL;
varying float vT;
varying float vA;
varying float vHL;
void main(){
  vT = mix(deVal, trackVal, useTrack);
  vHL = hl;
  float h = max(hl, noHover);
  float emph = pow(clamp(deVal, 0.001, 1.0), emphPow);
  float eActive = emphAmt * noHover;
  float eFactor = mix(1.0, emph, eActive);
  vA = (mix(0.02, 0.85, h) + hl * 0.15) * max(eFactor, 0.03);
  float sz = (mix(1.5, 4.0, noHover) + hl * 6.0) * mix(1.0, max(eFactor, 0.2), eActive);
  // softHL=1: optical mode — identical to idle state (noHover=1), color-only correspondence
  if(softHL > 0.5) {
    float idleEmph = pow(clamp(deVal, 0.001, 1.0), emphPow);
    float idleFactor = mix(1.0, idleEmph, emphAmt);
    vA = 0.85 * max(idleFactor, 0.03);
    sz = 4.0 * mix(1.0, max(idleFactor, 0.2), emphAmt);
  }

  vec3 pos = position;
  if(driftOn > 0.5) {
    float created = step(segT0, simTime);
    float driftDur = max(segArrival - segT0, 0.001);
    float driftFrac = clamp((simTime - segT0) / driftDur, 0.0, 1.0);
    float arrived = step(segArrival, simTime);
    float fadeOut = 1.0 - clamp((simTime - segArrival) / max(fadeDur, 1.0), 0.0, 1.0);
    float animAlpha = created * mix(1.0, fadeOut, arrived);
    pos.x = mix(position.x, segXAnode, driftFrac * created);
    vA *= animAlpha;
    sz *= max(animAlpha, 0.0);
  }

  vec4 mv = modelViewMatrix * vec4(pos, 1.0);
  gl_Position = projectionMatrix * mv;
  gl_PointSize = sz;
}`;

export const FS = `
uniform sampler2D cmap;
uniform float softHL;
varying float vT;
varying float vA;
varying float vHL;
void main(){
  vec2 d = gl_PointCoord - vec2(0.5);
  float r = length(d);
  if(r > 0.5) discard;
  float edge = 1.0 - smoothstep(0.38, 0.5, r);
  vec3 c = texture2D(cmap, vec2(clamp(vT,0.005,0.995), 0.5)).rgb;
  if(softHL > 0.5) {
    // Optical: relative dimming only — matching stays as-is, non-matching fades
    c *= mix(0.25, 1.0, vHL);
  } else {
    c = mix(c, vec3(1.0), vHL * 0.3);
  }
  gl_FragColor = vec4(c, vA * edge);
}`;
