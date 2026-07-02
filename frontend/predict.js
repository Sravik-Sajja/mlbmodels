//const API_BASE = 'http://localhost:5000';
const API_BASE = 'https://mlbmodels-production.up.railway.app'
const COLOR_MAP = {
    Single: 'single-fill',
    Double: 'double-fill',
    Triple: 'triple-fill',
    HR:     'hr-fill',
  };

  async function predict() {
    const fieldIds = ['hc_x', 'hc_y', 'launch_speed', 'launch_angle'];
    const raw = {};

    for (const id of fieldIds) {
      const val = document.getElementById(id).value;
      if (val === '') { showError('Please fill in all fields.'); return; }
      raw[id] = parseFloat(val);
    }

    if (raw.launch_angle > 90 || raw.launch_angle < -90) {
      showError('Please enter valid launch angle'); return;
    }
    if (raw.launch_speed < 1 || raw.launch_speed > 130) {
      showError('Please enter valid exit velocity'); return;
    }

    const svgPos = scToSvg(raw.hc_x, raw.hc_y);
    if (!isInFairTerritory(svgPos.x, svgPos.y)) {
      showError('Field position must be in play.');
      return;
    }

    const data = raw;

    const btn = document.getElementById('predict-btn');
    btn.disabled    = true;
    btn.textContent = 'Predicting...';
    hideError();

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(data),
      });

      if (!res.ok) {
        const err = await res.json();
        showError(err.error || 'Server error.');
        return;
      }

      renderResults(await res.json());
    } catch {
      showError('Could not reach server.');
    } finally {
      btn.disabled    = false;
      btn.textContent = 'Predict outcome';
    }
  }

  function renderResults(result) {
    document.getElementById('hit-pct').textContent = result.hit_probability + '%';
    document.getElementById('out-pct').textContent = result.out_probability + '%';

    setTimeout(() => {
      document.getElementById('hit-bar').style.width = result.hit_probability + '%';
    }, 50);

    const container = document.getElementById('breakdown-rows');
    container.innerHTML = '';

    const breakdown = result.bases_breakdown;
    const total = Object.values(breakdown).reduce((a, b) => a + b, 0);
    const max   = Math.max(...Object.values(breakdown));

    for (const label of ['Single', 'Double', 'Triple', 'HR']) {
      if (!(label in breakdown)) continue;
      const pct       = Math.round((breakdown[label] / total) * 1000) / 10;
      const colorClass = COLOR_MAP[label] || 'single-fill';
      const row = document.createElement('div');
      row.className = 'b-row';
      row.innerHTML = `
        <span class="b-label">${label}</span>
        <div class="b-bar">
          <div class="b-fill ${colorClass}" data-pct="${pct}" data-max="${max}"></div>
        </div>
        <span class="b-pct">${pct}%</span>
      `;
      container.appendChild(row);
    }

    setTimeout(() => {
      document.querySelectorAll('.b-fill').forEach(el => {
        el.style.width = (parseFloat(el.dataset.pct) / parseFloat(el.dataset.max) * 100) + '%';
      });
    }, 50);

    document.getElementById('results').classList.add('visible');
  }

  function showError(msg) {
    const el = document.getElementById('error-msg');
    el.textContent = msg;
    el.classList.add('visible');
  }

  function hideError() {
    document.getElementById('error-msg').classList.remove('visible');
  }