const PLOT_MAX_FT = 500;   // equidistant outer plot boundary
const PPF         = 0.78;  // pixels per foot
const TOP_PAD      = 22;   // room above the arc's apex
const BOTTOM_PAD   = 30;   // room below home plate for the plate icon
const SIDE_PAD     = 18;   // room beside the arc's widest points

const PLOT_R = PLOT_MAX_FT * PPF;
const HY     = TOP_PAD + PLOT_R;
const SVG_H  = HY + BOTTOM_PAD;
const HX     = PLOT_R * Math.sin(Math.PI / 4) + SIDE_PAD;
const SVG_W  = HX * 2;

const SC_HOME_X = 126.3436, SC_HOME_Y = 209.8488;
const SC_TO_FT_X = 2.301098;
const SC_TO_FT_Y = 2.299718;

const BASE_FT  = 90;
const CF_FT    = 400;   // actual CF fence, used for rendering/labels only
const CORN_FT  = 330;   // actual corner fence, used for rendering/labels only
const TRACK_FT = 15;

const FOUL_ANG = Math.PI / 4;

function ftToSvg(ftUp, ftRight) {
  return { x: HX + ftRight * PPF, y: HY - ftUp * PPF };
}

function scToSvg(sc_x, sc_y) {
  return ftToSvg((SC_HOME_Y - sc_y) * SC_TO_FT_Y, (sc_x - SC_HOME_X) * SC_TO_FT_X);
}

function svgToSc(svgX, svgY) {
  const ftRight = (svgX - HX) / PPF;
  const ftUp    = (HY - svgY) / PPF;
  return {
    x: Math.round((SC_HOME_X + ftRight / SC_TO_FT_X) * 10) / 10,
    y: Math.round((SC_HOME_Y - ftUp    / SC_TO_FT_Y) * 10) / 10,
  };
}

function scDist(sc_x, sc_y) {
  const dx = (sc_x - SC_HOME_X) * SC_TO_FT_X;
  const dy = (SC_HOME_Y - sc_y) * SC_TO_FT_Y;
  return Math.round(Math.sqrt(dx * dx + dy * dy));
}

function polar(r, angFromVert) {
  return { x: HX + r * Math.sin(angFromVert), y: HY - r * Math.cos(angFromVert) };
}

// Move a point radially toward home plate by shrinkFt feet
function shrink(pt, shrinkFt) {
  const dx = pt.x - HX, dy = pt.y - HY;
  const d  = Math.sqrt(dx * dx + dy * dy);
  const s  = Math.max(0, d - shrinkFt * PPF) / d;
  return { x: HX + dx * s, y: HY + dy * s };
}

// Returns true if the SVG point is in fair territory: between the foul
// lines, in front of home, and inside the equidistant plot boundary.
function isInFairTerritory(svgX, svgY) {
  if (svgY >= HY) return false;
  const dx = svgX - HX;
  const dy = HY - svgY; // positive = upward
  if (dy <= 0) return false;
  const angle = Math.abs(Math.atan2(dx, dy));
  if (angle > FOUL_ANG) return false;
  const r = Math.sqrt(dx * dx + dy * dy);
  return r <= PLOT_R + 0.5; // small epsilon for rounding
}

function buildField() {
  const svg = document.getElementById('field-svg');

  // Own the aspect ratio directly rather than relying on markup elsewhere
  // matching these constants exactly.
  svg.setAttribute('viewBox', `0 0 ${SVG_W} ${SVG_H}`);
  svg.style.aspectRatio = `${SVG_W} / ${SVG_H}`;

  const DIAG   = BASE_FT / Math.SQRT2;
  const home   = { x: HX, y: HY };
  const first  = ftToSvg(DIAG,  DIAG);
  const second = ftToSvg(BASE_FT * Math.SQRT2, 0);
  const third  = ftToSvg(DIAG, -DIAG);
  const mound  = ftToSvg(60.5, 0);

  // Outfield wall: foul poles at 330ft on the 45° foul lines
  const lfPole = polar(CORN_FT * PPF, -FOUL_ANG);
  const rfPole = polar(CORN_FT * PPF,  FOUL_ANG);
  // CF wall at 400ft straight up
  const cf     = polar(CF_FT * PPF, 0);

  // Single quadratic bezier per side: control point is on the wall arc
  // midpoint between pole and cf at roughly the right distance.
  const MID_ANG = Math.PI / 8;   // 22.5° from vertical
  const MID_R   = 355 * PPF;
  const lfMid   = polar(MID_R, -MID_ANG);
  const rfMid   = polar(MID_R,  MID_ANG);

  const lfCtrl = {
    x: 2 * lfMid.x - 0.5 * (lfPole.x + cf.x),
    y: 2 * lfMid.y - 0.5 * (lfPole.y + cf.y),
  };
  const rfCtrl = {
    x: 2 * rfMid.x - 0.5 * (rfPole.x + cf.x),
    y: 2 * rfMid.y - 0.5 * (rfPole.y + cf.y),
  };

  // Warning track: same bezier shapes, every point shrunk 15ft toward home
  const lfPoleT = shrink(lfPole, TRACK_FT);
  const rfPoleT = shrink(rfPole, TRACK_FT);
  const cfT     = shrink(cf,     TRACK_FT);
  const lfCtrlT = shrink(lfCtrl, TRACK_FT);
  const rfCtrlT = shrink(rfCtrl, TRACK_FT);

  // Infield dirt
  const dirtCX = (home.x + second.x) / 2;
  const dirtCY = (home.y + second.y) / 2;
  const dirtR  = 95 * PPF;

  // Outfield grass
  const grassPath = [
    `M ${lfPole.x},${lfPole.y}`,
    `Q ${lfCtrl.x},${lfCtrl.y} ${cf.x},${cf.y}`,
    `Q ${rfCtrl.x},${rfCtrl.y} ${rfPole.x},${rfPole.y}`,
    `L ${home.x},${home.y} Z`
  ].join(' ');

  // Warning track band (outer bezier → straight sides → inner bezier back)
  const trackPath = [
    `M ${lfPole.x},${lfPole.y}`,
    `Q ${lfCtrl.x},${lfCtrl.y} ${cf.x},${cf.y}`,
    `Q ${rfCtrl.x},${rfCtrl.y} ${rfPole.x},${rfPole.y}`,
    `L ${rfPoleT.x},${rfPoleT.y}`,
    `Q ${rfCtrlT.x},${rfCtrlT.y} ${cfT.x},${cfT.y}`,
    `Q ${lfCtrlT.x},${lfCtrlT.y} ${lfPoleT.x},${lfPoleT.y}`,
    `Z`
  ].join(' ');

  // Equidistant plot boundary: the fan/wedge shape the whole panel is
  // clipped and drawn to. Same pole → mid → apex bezier technique as the
  // fence above, just at PLOT_R instead of the fence distances, and
  // closed back to home so it can double as the panel's own background.
  const plotLeft  = polar(PLOT_R, -FOUL_ANG);
  const plotRight = polar(PLOT_R,  FOUL_ANG);
  const plotTop   = polar(PLOT_R,  0);
  const PLOT_MID_ANG = Math.PI / 8;
  const plotLeftMid  = polar(PLOT_R, -PLOT_MID_ANG);
  const plotRightMid = polar(PLOT_R,  PLOT_MID_ANG);
  const plotLeftCtrl = {
    x: 2 * plotLeftMid.x - 0.5 * (plotLeft.x + plotTop.x),
    y: 2 * plotLeftMid.y - 0.5 * (plotLeft.y + plotTop.y),
  };
  const plotRightCtrl = {
    x: 2 * plotRightMid.x - 0.5 * (plotTop.x + plotRight.x),
    y: 2 * plotRightMid.y - 0.5 * (plotTop.y + plotRight.y),
  };
  const fanPath = [
    `M ${home.x},${home.y}`,
    `L ${plotLeft.x},${plotLeft.y}`,
    `Q ${plotLeftCtrl.x},${plotLeftCtrl.y} ${plotTop.x},${plotTop.y}`,
    `Q ${plotRightCtrl.x},${plotRightCtrl.y} ${plotRight.x},${plotRight.y}`,
    `L ${home.x},${home.y}`,
    `Z`
  ].join(' ');

  svg.innerHTML = `
    <defs>
      <clipPath id="fieldClip"><path d="${fanPath}"/></clipPath>
      <radialGradient id="grassGrad" cx="50%" cy="100%" r="85%">
        <stop offset="0%" stop-color="#1e4a24"/>
        <stop offset="100%" stop-color="#102214"/>
      </radialGradient>
      <radialGradient id="dirtGrad" cx="50%" cy="40%" r="60%">
        <stop offset="0%" stop-color="#9a7050"/>
        <stop offset="100%" stop-color="#6b4a28"/>
      </radialGradient>
    </defs>

    <path d="${fanPath}" fill="#0d1a0f" stroke="var(--border)" stroke-width="1.5" stroke-linejoin="round"/>

    <path d="${grassPath}" fill="url(#grassGrad)" clip-path="url(#fieldClip)"/>

    <path d="${trackPath}" fill="#5c4020" opacity="0.85"/>

    <circle cx="${dirtCX}" cy="${dirtCY}" r="${dirtR}" fill="url(#dirtGrad)" opacity="0.9"/>

    <polygon points="
      ${home.x},${home.y}
      ${first.x},${first.y}
      ${second.x},${second.y}
      ${third.x},${third.y}
    " fill="#1a3e20"/>

    <ellipse cx="${mound.x}" cy="${mound.y}" rx="${10 * PPF}" ry="${7 * PPF}"
      fill="#8b6340" opacity="0.95"/>

    <line x1="${home.x}" y1="${home.y}" x2="${lfPole.x}" y2="${lfPole.y}"
      stroke="rgba(255,255,255,0.55)" stroke-width="1.5"/>
    <line x1="${home.x}" y1="${home.y}" x2="${rfPole.x}" y2="${rfPole.y}"
      stroke="rgba(255,255,255,0.55)" stroke-width="1.5"/>

    <line x1="${home.x}"   y1="${home.y}"   x2="${first.x}"  y2="${first.y}"  stroke="rgba(255,255,255,0.3)" stroke-width="1"/>
    <line x1="${first.x}"  y1="${first.y}"  x2="${second.x}" y2="${second.y}" stroke="rgba(255,255,255,0.3)" stroke-width="1"/>
    <line x1="${second.x}" y1="${second.y}" x2="${third.x}"  y2="${third.y}"  stroke="rgba(255,255,255,0.3)" stroke-width="1"/>
    <line x1="${third.x}"  y1="${third.y}"  x2="${home.x}"   y2="${home.y}"   stroke="rgba(255,255,255,0.3)" stroke-width="1"/>

    <rect x="${first.x - 5}"  y="${first.y - 5}"  width="10" height="10" rx="1.5" fill="white" opacity="0.9"/>
    <rect x="${second.x - 5}" y="${second.y - 5}" width="10" height="10" rx="1.5" fill="white" opacity="0.9"/>
    <rect x="${third.x - 5}"  y="${third.y - 5}"  width="10" height="10" rx="1.5" fill="white" opacity="0.9"/>

    <polygon points="
      ${home.x},${home.y - 7}
      ${home.x + 6},${home.y - 2}
      ${home.x + 6},${home.y + 4}
      ${home.x - 6},${home.y + 4}
      ${home.x - 6},${home.y - 2}
    " fill="white" opacity="0.9"/>

    <text x="${cf.x}"          y="${cf.y + 16}"     text-anchor="middle" font-family="DM Mono,monospace" font-size="12" fill="rgba(255,255,255,0.7)" letter-spacing="1">CF</text>
    <text x="${lfPole.x + 38}" y="${lfPole.y - 8}"  text-anchor="middle" font-family="DM Mono,monospace" font-size="12" fill="rgba(255,255,255,0.7)" letter-spacing="1">LF</text>
    <text x="${rfPole.x - 38}" y="${rfPole.y - 8}"  text-anchor="middle" font-family="DM Mono,monospace" font-size="12" fill="rgba(255,255,255,0.7)" letter-spacing="1">RF</text>
    <text x="${cf.x}"          y="${cf.y + 28}"     text-anchor="middle" font-family="DM Mono,monospace" font-size="11" fill="rgba(255,255,255,0.6)">400 ft</text>
    <text x="${lfPole.x + 38}" y="${lfPole.y + 6}"  text-anchor="middle" font-family="DM Mono,monospace" font-size="11" fill="rgba(255,255,255,0.6)">330 ft</text>
    <text x="${rfPole.x - 38}" y="${rfPole.y + 6}"  text-anchor="middle" font-family="DM Mono,monospace" font-size="11" fill="rgba(255,255,255,0.6)">330 ft</text>

    <g id="hit-marker" opacity="0">
      <circle id="hit-ring" cx="0" cy="0" r="11" fill="none" stroke="#00c46a" stroke-width="1.5" opacity="0.5"/>
      <circle id="hit-dot"  cx="0" cy="0" r="4.5" fill="#00c46a"/>
    </g>
  `;

  svg.addEventListener('click', function(e) {
    const rect = svg.getBoundingClientRect();
    const svgX = (e.clientX - rect.left) * (SVG_W / rect.width);
    const svgY = (e.clientY - rect.top)  * (SVG_H / rect.height);

    if (!isInFairTerritory(svgX, svgY)) {
      showFoulError();
      return;
    }
    hideFoulError();

    const sc   = svgToSc(svgX, svgY);
    const dist = scDist(sc.x, sc.y);

    document.getElementById('hc_x').value = sc.x;
    document.getElementById('hc_y').value = sc.y;

    placeMarker(svgX, svgY);
    refreshPills(sc.x, sc.y, dist);
  });
}

function placeMarker(svgX, svgY) {
  const marker = document.getElementById('hit-marker');
  const dot    = document.getElementById('hit-dot');
  const ring   = document.getElementById('hit-ring');
  marker.setAttribute('opacity', '1');
  dot.setAttribute('cx', svgX);  dot.setAttribute('cy', svgY);
  ring.setAttribute('cx', svgX); ring.setAttribute('cy', svgY);
}

function refreshPills(x, y, dist) {
  document.getElementById('display-x').textContent    = x;
  document.getElementById('display-y').textContent    = y;
  document.getElementById('display-dist').textContent = (dist === '—') ? dist : `~${dist}`;
}

function showFoulError() {
  const el = document.getElementById('foul-msg');
  if (el) { el.classList.add('visible'); setTimeout(() => el.classList.remove('visible'), 2000); }
}

function hideFoulError() {
  const el = document.getElementById('foul-msg');
  if (el) el.classList.remove('visible');
}

function onCoordInput(bypassFoulCheck = false) {
  const xVal = parseFloat(document.getElementById('hc_x').value);
  const yVal = parseFloat(document.getElementById('hc_y').value);

  if (isNaN(xVal) || isNaN(yVal)) {
    document.getElementById('hit-marker').setAttribute('opacity', '0');
    refreshPills('—', '—', '—');
    hideFoulError();
    return;
  }

  const svgPos = scToSvg(xVal, yVal);
  const inFair = isInFairTerritory(svgPos.x, svgPos.y);

  if (!inFair && !bypassFoulCheck) {
    showFoulError();
    document.getElementById('hit-marker').setAttribute('opacity', '0');
    refreshPills('—', '—', '—');
    return;
  }
  hideFoulError();

  const dist = scDist(xVal, yVal);
  placeMarker(svgPos.x, svgPos.y);
  refreshPills(xVal, yVal, dist);
}