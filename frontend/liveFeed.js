const API_BASE = 'http://localhost:5000';
//const API_BASE = 'https://mlbmodels-production.up.railway.app'

let selectedDate = null;
let selectedGamePk = null;

function pad(n) { return n.toString().padStart(2, '0'); }

function toISODate(d) {
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

function dayLabel(d) {
  return d.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
}

function initLiveFeed() {
  const scroller = document.getElementById('day-scroller');
  if (!scroller) return;
  scroller.innerHTML = '';

  const today = new Date();
  for (let i = 0; i < 30; i++) {
    const d = new Date(today);
    d.setDate(today.getDate() - i);
    const iso = toISODate(d);

    const btn = document.createElement('button');
    btn.className = 'day-pill';
    btn.textContent = i === 0 ? 'Today' : dayLabel(d);
    btn.dataset.date = iso;
    btn.addEventListener('click', () => selectDay(iso, btn));
    scroller.appendChild(btn);
  }

  const firstBtn = scroller.querySelector('.day-pill');
  if (firstBtn) selectDay(firstBtn.dataset.date, firstBtn);
}

async function selectDay(iso, btnEl) {
  selectedDate = iso;
  selectedGamePk = null;

  document.querySelectorAll('.day-pill').forEach(b => b.classList.remove('active'));
  if (btnEl) btnEl.classList.add('active');

  const gamesList = document.getElementById('games-list');
  const playsList = document.getElementById('plays-list');
  gamesList.innerHTML = '<p class="live-hint">Loading games...</p>';
  playsList.innerHTML = '<p class="live-hint">Select a game to see batted balls</p>';

  try {
    const res = await fetch(`${API_BASE}/games?date=${iso}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Failed to load games');

    if (!data.games.length) {
      gamesList.innerHTML = '<p class="live-hint">No games on this date</p>';
      return;
    }

    gamesList.innerHTML = '';
    data.games.forEach(g => {
      const row = document.createElement('button');
      row.className = 'game-row';
      row.innerHTML = `
        <span class="game-teams">${g.away} @ ${g.home}</span>
        <span class="game-status">${g.status}</span>
      `;
      row.addEventListener('click', () => selectGame(g.gamePk, row));
      gamesList.appendChild(row);
    });
  } catch (err) {
    gamesList.innerHTML = '<p class="live-hint error">Could not load games</p>';
  }
}

async function selectGame(gamePk, rowEl) {
  selectedGamePk = gamePk;

  document.querySelectorAll('.game-row').forEach(r => r.classList.remove('active'));
  if (rowEl) rowEl.classList.add('active');

  const playsList = document.getElementById('plays-list');
  playsList.innerHTML = '<p class="live-hint">Loading batted balls...</p>';

  try {
    const res = await fetch(`${API_BASE}/plays?gamePk=${gamePk}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Failed to load plays');

    if (!data.plays.length) {
      playsList.innerHTML = '<p class="live-hint">No batted balls recorded yet for this game</p>';
      return;
    }

    playsList.innerHTML = '';
    data.plays.forEach(p => {
      const row = document.createElement('button');
      row.className = 'play-row';
      row.innerHTML = `
        <span class="play-batter">${p.batter}</span>
        <span class="play-event">${p.event || p.description}</span>
        <span class="play-inning">${p.half === 'top' ? '\u25B2' : '\u25BC'}${p.inning}</span>
      `;
      row.addEventListener('click', () => applyPlay(p, row));
      playsList.appendChild(row);
    });
  } catch (err) {
    playsList.innerHTML = '<p class="live-hint error">Could not load plays</p>';
  }
}

function applyPlay(p, rowEl) {
  document.getElementById('hc_x').value = p.hc_x;
  document.getElementById('hc_y').value = p.hc_y;
  document.getElementById('launch_speed').value = p.launch_speed;
  document.getElementById('launch_angle').value = p.launch_angle;

  // Reuses field.js logic to place the marker + update the coord pills
  onCoordInput();

  document.querySelectorAll('.play-row').forEach(r => r.classList.remove('active'));
  if (rowEl) rowEl.classList.add('active');
}