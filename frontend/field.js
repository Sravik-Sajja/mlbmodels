/* field.js — Baseball field SVG rendering & coordinate mapping */

const SVG_W = 400, SVG_H = 420;

const SC_HOME_X = 125, SC_HOME_Y = 205;

const HX = SVG_W / 2;
const HY = SVG_H - 28;

const BASE_FT  = 90;
const CF_FT    = 400;
const CORN_FT  = 330;
const TRACK_FT = 15;

const PPF      = (HY - 18) / CF_FT;
const FOUL_ANG = Math.PI / 4;
const SC_TO_FT = 2.0;

function ftToSvg(ftUp, ftRight) {
  return { x: HX + ftRight * PPF, y: HY - ftUp * PPF };
}

function scToSvg(sc_x, sc_y) {
  return ftToSvg((SC_HOME_Y - sc_y) * SC_TO_FT, (sc_x - SC_HOME_X) * SC_TO_FT);
}

function svgToSc(svgX, svgY) {
  const ftRight = (svgX - HX) / PPF;
  const ftUp    = (HY - svgY) / PPF;
  return {
    x: Math.round((SC_HOME_X + ftRight / SC_TO_FT) * 10) / 10,
    y: Math.round((SC_HOME_Y - ftUp    / SC_TO_FT) * 10) / 10,
  };
}

function scDist(sc_x, sc_y) {
  const dx = (sc_x - SC_HOME_X) * SC_TO_FT;
  const dy = (SC_HOME_Y - sc_y) * SC_TO_FT;
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

// Returns true if the SVG point is in fair territory (between foul lines, in front of home)
function isInFairTerritory(svgX, svgY) {
  // Must be above home plate (smaller y = higher on screen)
  if (svgY >= HY) return false;
  // Vector from home plate
  const dx = svgX - HX;
  const dy = HY - svgY; // positive = upward
  if (dy <= 0) return false;
  // Angle from vertical: foul lines are at ±45°
  const angle = Math.abs(Math.atan2(dx, dy));
  return angle <= FOUL_ANG;
}

function buildField() {
  const svg = document.getElementById('field-svg');

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
  // A point at 45° halfway (22.5°) from vertical at ~355ft looks natural.
  const MID_ANG = Math.PI / 8;   // 22.5° from vertical
  const MID_R   = 355 * PPF;
  const lfMid   = polar(MID_R, -MID_ANG);
  const rfMid   = polar(MID_R,  MID_ANG);

  // For a smooth arc through lfPole, lfMid, cf we need the bezier control point.
  // Quadratic bezier: B(t) = (1-t)²P0 + 2t(1-t)C + t²P1
  // At t=0.5: midpoint = 0.25*P0 + 0.5*C + 0.25*P1  →  C = 2*mid - 0.5*(P0+P1)
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

  svg.innerHTML = `
    <defs>
      <clipPath id="fieldClip"><rect width="${SVG_W}" height="${SVG_H}"/></clipPath>
      <radialGradient id="grassGrad" cx="50%" cy="100%" r="85%">
        <stop offset="0%" stop-color="#1e4a24"/>
        <stop offset="100%" stop-color="#102214"/>
      </radialGradient>
      <radialGradient id="dirtGrad" cx="50%" cy="40%" r="60%">
        <stop offset="0%" stop-color="#9a7050"/>
        <stop offset="100%" stop-color="#6b4a28"/>
      </radialGradient>
    </defs>

    <rect width="${SVG_W}" height="${SVG_H}" fill="#0d1a0f"/>

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

    <text x="${cf.x}"          y="${cf.y + 16}"     text-anchor="middle" font-family="DM Mono,monospace" font-size="8" fill="rgba(255,255,255,0.3)" letter-spacing="1">CF</text>
    <text x="${lfPole.x + 38}" y="${lfPole.y - 8}"  text-anchor="middle" font-family="DM Mono,monospace" font-size="8" fill="rgba(255,255,255,0.3)" letter-spacing="1">LF</text>
    <text x="${rfPole.x - 38}" y="${rfPole.y - 8}"  text-anchor="middle" font-family="DM Mono,monospace" font-size="8" fill="rgba(255,255,255,0.3)" letter-spacing="1">RF</text>
    <text x="${cf.x}"          y="${cf.y + 28}"     text-anchor="middle" font-family="DM Mono,monospace" font-size="7" fill="rgba(255,255,255,0.2)">400 ft</text>
    <text x="${lfPole.x + 38}" y="${lfPole.y + 6}"  text-anchor="middle" font-family="DM Mono,monospace" font-size="7" fill="rgba(255,255,255,0.2)">330 ft</text>
    <text x="${rfPole.x - 38}" y="${rfPole.y + 6}"  text-anchor="middle" font-family="DM Mono,monospace" font-size="7" fill="rgba(255,255,255,0.2)">330 ft</text>

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

    document.getElementById('hc_x').value            = sc.x;
    document.getElementById('hc_y').value            = sc.y;
    document.getElementById('hit_distance_sc').value = dist;

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
  document.getElementById('display-dist').textContent = dist;
}

function showFoulError() {
  const el = document.getElementById('foul-msg');
  if (el) { el.classList.add('visible'); setTimeout(() => el.classList.remove('visible'), 2000); }
}

function hideFoulError() {
  const el = document.getElementById('foul-msg');
  if (el) el.classList.remove('visible');
}

function onCoordInput() {
  const xVal = parseFloat(document.getElementById('hc_x').value);
  const yVal = parseFloat(document.getElementById('hc_y').value);

  // Either field empty/invalid — clear derived outputs and wait
  if (isNaN(xVal) || isNaN(yVal)) {
    document.getElementById('hit_distance_sc').value = '';
    document.getElementById('hit-marker').setAttribute('opacity', '0');
    refreshPills('—', '—', '—');
    hideFoulError();
    return;
  }

  const svgPos = scToSvg(xVal, yVal);

  if (!isInFairTerritory(svgPos.x, svgPos.y)) {
    showFoulError();
    document.getElementById('hit-marker').setAttribute('opacity', '0');
    document.getElementById('hit_distance_sc').value = '';
    refreshPills('—', '—', '—');
    return;
  }
  hideFoulError();

  const dist = scDist(xVal, yVal);
  document.getElementById('hit_distance_sc').value = dist;
  placeMarker(svgPos.x, svgPos.y);
  refreshPills(xVal, yVal, dist);
}