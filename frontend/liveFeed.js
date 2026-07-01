let selectedDate = null;
let selectedGamePk = null;

function pad(n) { return n.toString().padStart(2, '0'); }

function toISODate(d) {
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

function dayLabel(d) {
  return d.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
}

function showGamesView() {
  document.getElementById('day-scroller').classList.remove('hidden');
  document.getElementById('games-list').classList.remove('hidden');
  document.getElementById('plays-list').classList.add('hidden');
  document.getElementById('back-btn').classList.remove('visible');
  document.getElementById('live-panel-title').textContent = 'Pull a real play';
  selectedGamePk = null;
  document.querySelectorAll('.game-row').forEach(r => r.classList.remove('active'));
}

function showPlaysView(label) {
  document.getElementById('day-scroller').classList.add('hidden');
  document.getElementById('games-list').classList.add('hidden');
  document.getElementById('plays-list').classList.remove('hidden');
  document.getElementById('back-btn').classList.add('visible');
  document.getElementById('live-panel-title').textContent = label || 'Batted balls';
}

function resetResults() {
  const results = document.getElementById('results');
  const errorMsg = document.getElementById('error-msg');
  const hitBar = document.getElementById('hit-bar');
  const hitPct = document.getElementById('hit-pct');
  const outPct = document.getElementById('out-pct');
  const breakdown = document.getElementById('breakdown-rows');

  if (results) results.classList.remove('visible');
  if (errorMsg) errorMsg.classList.remove('visible');
  if (hitBar) hitBar.style.width = '0%';
  if (hitPct) hitPct.textContent = '—';
  if (outPct) outPct.textContent = '—';
  if (breakdown) breakdown.innerHTML = '';
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

  const backBtn = document.getElementById('back-btn');
  if (backBtn) backBtn.addEventListener('click', showGamesView);

  const firstBtn = scroller.querySelector('.day-pill');
  if (firstBtn) selectDay(firstBtn.dataset.date, firstBtn);
}

async function selectDay(iso, btnEl) {
  selectedDate = iso;
  selectedGamePk = null;
  showGamesView();

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

        const awayScore = g.awayScore ?? '–';
        const homeScore = g.homeScore ?? '–';
        const awayLogo  = `https://www.mlbstatic.com/team-logos/${g.awayId}.svg`;
        const homeLogo  = `https://www.mlbstatic.com/team-logos/${g.homeId}.svg`;

        let statusText = g.status;
        if (g.abstractState === 'Live' && g.inning) {
            const arrow = g.inningHalf === 'Bottom' ? '\u25BC' : '\u25B2';
            statusText = `${arrow}${g.inning}`;
        }

        row.innerHTML = `
            <span class="game-teams-compact">
            <img class="team-logo" src="${awayLogo}" alt="${g.away}" title="${g.away}" onerror="this.style.display='none'">
            <span class="at-sign">@</span>
            <img class="team-logo" src="${homeLogo}" alt="${g.home}" title="${g.home}" onerror="this.style.display='none'">
            </span>
            <span class="game-score-compact">${awayScore}–${homeScore}</span>
            <span class="game-status">${statusText}</span>
        `;
        row.addEventListener('click', () => selectGame(g, row));
        gamesList.appendChild(row);
        });
  } catch (err) {
    gamesList.innerHTML = '<p class="live-hint error">Could not load games</p>';
  }
}

async function selectGame(g, rowEl) {
  selectedGamePk = g.gamePk;

  document.querySelectorAll('.game-row').forEach(r => r.classList.remove('active'));
  if (rowEl) rowEl.classList.add('active');

  showPlaysView(`${g.away} @ ${g.home}`);

  const playsList = document.getElementById('plays-list');
  playsList.innerHTML = '<p class="live-hint">Loading batted balls...</p>';

  try {
    const res = await fetch(`${API_BASE}/plays?gamePk=${g.gamePk}`);
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
  resetResults();

  document.getElementById('hc_x').value = p.hc_x;
  document.getElementById('hc_y').value = p.hc_y;
  document.getElementById('launch_speed').value = p.launch_speed;
  document.getElementById('launch_angle').value = p.launch_angle;

  onCoordInput();

  document.querySelectorAll('.play-row').forEach(r => r.classList.remove('active'));
  if (rowEl) rowEl.classList.add('active');
}