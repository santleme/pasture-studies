(() => {
  'use strict';

  const ACTIONS = ['graze', 'drink', 'walk', 'rest', 'ruminate', 'herd_interact'];
  const ACTION_LABELS = { graze: 'Grazing', drink: 'Drinking', walk: 'Walking', rest: 'Resting', ruminate: 'Ruminating', herd_interact: 'Herd interaction' };
  const FEATURES = ['hunger', 'thirst', 'fatigue', 'social_need', 'forage_visible', 'water_visible', 'forage_distance_norm', 'water_distance_norm', 'forage_available', 'water_available', 'local_blocked_fraction', 'herd_count_norm', 'mem_water_success', 'mem_forage_success', 'mem_water_blocked', 'mem_forage_depleted', 'mem_event_recency', 'tick_phase_sin', 'tick_phase_cos', 'self_in_herd'];
  const DRIVE_LABELS = { hunger: 'Hunger', thirst: 'Thirst', fatigue: 'Fatigue', social_need: 'Social need' };
  const COLORS = ['#526950', '#8a6854', '#737c91', '#b58945', '#9b735e'];
  const $ = (id) => document.getElementById(id);
  const app = { state: null, selected: 'cow-1', cursor: -1, events: [], eventGapNotice: '', pollTimer: 0, polling: false, canvasObserver: null, seedInputDirty: false };

  const canvas = $('pasture-map');
  const ctx = canvas.getContext('2d');

  function esc(value) {
    return String(value ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }
  function request(path, options = {}) {
    return fetch(path, { headers: { 'Content-Type': 'application/json' }, ...options }).then(async (response) => {
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
      return data;
    });
  }
  function animalNumber(id) { return Math.max(1, Number(String(id).split('-').pop()) || 1); }
  function selectedAnimal() { return app.state?.animals?.find((a) => a.id === app.selected) || app.state?.animals?.[0] || null; }
  function setConnection(online, message) {
    $('connection-dot').className = `live-dot ${online ? 'online' : 'offline'}`;
    $('connection-label').textContent = message;
  }
  function updateClock() {
    const state = app.state;
    if (!state) return;
    const tick = Number(state.clock?.tick ?? state.tick ?? 0);
    const day = Number(state.clock?.synthetic_day_index ?? state.synthetic_day_index ?? Math.floor(tick / 24));
    const withinDay = Number(state.clock?.tick_within_day ?? state.tick_within_day ?? (tick % 24));
    const ticksPerDay = Number(state.clock?.ticks_per_virtual_day ?? state.ticks_per_virtual_day ?? 24);
    const paused = Boolean(state.paused ?? state.clock?.paused);
    $('clock-time').textContent = `DAY ${day} · TICK ${String(withinDay).padStart(3, '0')}/${String(ticksPerDay).padStart(3, '0')}`;
    $('clock-time').title = `Absolute synthetic tick ${tick}; ${ticksPerDay} abstract ticks per virtual day.`;
    $('clock-state').textContent = paused ? 'PAUSED' : 'RUNNING';
    $('clock-state').classList.toggle('running', !paused);
    const button = $('toggle-run');
    button.querySelector('.button-icon').textContent = paused ? '▶' : 'Ⅱ';
    button.querySelector('span:last-child').textContent = paused ? 'Resume' : 'Pause';
    $('seed-value').textContent = state.seed ?? '—';
    if (!app.seedInputDirty && state.seed !== undefined && state.seed !== null) $('seed-input').value = String(state.seed);
    $('map-meta').textContent = `${state.width ?? 14} × ${state.height ?? 10} GRID`;
    const policy = state.policy_id || '—';
    $('policy-short').textContent = policy.length > 22 ? `${policy.slice(0, 19)}…` : policy;
    $('policy-short').title = policy;
    if (state.speed !== undefined) $('speed-select').value = String(state.speed);
  }
  function renderSelector() {
    const animals = app.state?.animals || [];
    if (!animals.some((a) => a.id === app.selected) && animals[0]) app.selected = animals[0].id;
    $('animal-selector').innerHTML = animals.map((animal) => {
      const n = animalNumber(animal.id);
      return `<button type="button" class="animal-tab" data-animal="${esc(animal.id)}" aria-pressed="${animal.id === app.selected}" aria-label="Inspect cow ${n}, currently ${esc(animal.action || 'rest')}"><span class="mini-cow">${String(n).padStart(2, '0')}</span><span><strong>Cow ${String(n).padStart(2, '0')}</strong><small>${esc(ACTION_LABELS[animal.action] || animal.action || '—')}</small></span></button>`;
    }).join('');
    $('record-index').textContent = `${String(animalNumber(app.selected)).padStart(2, '0')} / ${String(animals.length).padStart(2, '0')}`;
  }
  function renderInspector(animal) {
    if (!animal) return;
    const n = animalNumber(animal.id);
    $('animal-avatar').textContent = String(n).padStart(2, '0');
    $('animal-name').textContent = `Cow ${String(n).padStart(2, '0')}`;
    $('animal-action').textContent = ACTION_LABELS[animal.action] || animal.action || '—';
    $('animal-reward').innerHTML = `REWARD <b>${Number(animal.reward || 0).toFixed(2)}</b>`;
    $('policy-id').textContent = animal.policy_id || app.state?.policy_id || '—';

    const physiology = animal.physiology || {};
    $('drive-list').innerHTML = Object.entries(DRIVE_LABELS).map(([key, label]) => {
      const value = Number(physiology[key] ?? 0);
      const width = Math.min(100, Math.max(0, value * 100));
      return `<div class="drive-row"><span>${label}</span><div class="meter"><i class="${key}" style="width:${width}%"></i></div><b>${value.toFixed(2)}</b></div>`;
    }).join('');

    const hasDecision = animal.decision_tick !== null && animal.decision_tick !== undefined && Array.isArray(animal.decision_probabilities);
    const probs = hasDecision ? animal.decision_probabilities : (animal.probabilities || []);
    $('decision-context').textContent = hasDecision
      ? `Decision at tick ${animal.decision_tick} · current observation tick ${animal.observation?.tick ?? app.state?.tick ?? 0}. Distribution uses the recorded decision features below.`
      : `Current observation at tick ${animal.observation?.tick ?? app.state?.tick ?? 0}; preview only, no decision yet.`;
    $('probability-list').innerHTML = ACTIONS.map((action, index) => {
      const value = Number(probs[index] || 0);
      return `<div class="prob-row ${hasDecision && action === animal.action ? 'chosen' : ''}"><span>${esc(action.replace('_', ' '))}</span><div class="meter"><i style="width:${Math.max(0, Math.min(100, value * 100))}%"></i></div><b>${(value * 100).toFixed(1)}%</b></div>`;
    }).join('');

    const currentFeatures = animal.current_features || animal.features || [];
    const features = hasDecision ? (animal.decision_features || []) : currentFeatures;
    const feature = (key) => Number(currentFeatures[FEATURES.indexOf(key)] || 0);
    $('feature-vector-label').firstChild.textContent = hasDecision
      ? `Decision features · tick ${animal.decision_tick} `
      : 'Current preview features · no decision yet ';
    const summary = [
      ['Forage visible', animal.observation?.visible_forage?.length || 0], ['Water visible', animal.observation?.visible_water?.length || 0],
      ['Nearby herd', animal.observation?.local_herd_count || 0],
      ['Water memory', feature('mem_water_success')], ['Forage memory', feature('mem_forage_success')],
      ['Blocked water', feature('mem_water_blocked')], ['Depleted forage', feature('mem_forage_depleted')],
      ['Herd nearby', feature('self_in_herd')],
    ];
    $('feature-summary').innerHTML = summary.map(([label, value]) => `<span class="feature-pill">${label}<b>${Number(value).toFixed(typeof value === 'number' && !Number.isInteger(value) ? 2 : 0)}</b></span>`).join('');
    $('feature-list').innerHTML = FEATURES.map((name, i) => `<li>${esc(name)} <b>${Number(features[i] ?? 0).toFixed(3)}</b></li>`).join('');
    const memoryEvents = animal.memory_features?.retrieved_events || [];
    $('memory-events').innerHTML = memoryEvents.length ? memoryEvents.map((event) => `<div class="memory-item"><b class="${event.success ? '' : 'fail'}">${event.success ? 'SUCCESS' : 'NO ACCESS'}</b><span>${esc(event.kind.replaceAll('_', ' '))} · tick ${Number(event.tick)}</span><span>(${event.x}, ${event.y})</span></div>`).join('') : '<span class="empty-memory">No retrieved events for this observation.</span>';
  }
  function roundRect(context, x, y, width, height, radius) {
    context.beginPath(); context.roundRect(x, y, width, height, radius);
  }
  function drawMap() {
    if (!app.state) return;
    const { width = 14, height = 10, animals = [], forage = [], water = [], obstacles = [] } = app.state;
    const rect = canvas.getBoundingClientRect();
    if (!rect.width) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.width * (height / width) * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const cw = rect.width / width;
    const ch = rect.width * (height / width) / height;
    ctx.clearRect(0, 0, rect.width, rect.width * (height / width));

    // Muted, hand-drawn field blocks provide spatial texture without external imagery.
    ctx.fillStyle = '#cbd4b9'; ctx.fillRect(0, 0, rect.width, ch * height);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        ctx.fillStyle = ((x + y) % 2 === 0) ? 'rgba(238,235,211,.18)' : 'rgba(104,130,87,.045)';
        ctx.fillRect(x * cw, y * ch, cw, ch);
        ctx.strokeStyle = 'rgba(76,94,67,.075)'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x * cw, (y + 1) * ch); ctx.lineTo((x + 1) * cw, (y + 1) * ch); ctx.stroke();
      }
    }
    // Outer ranch rail and internal fence line are visually distinct from blocked terrain.
    ctx.strokeStyle = '#796f56'; ctx.lineWidth = 3;
    ctx.strokeRect(2, 2, rect.width - 4, ch * height - 4);
    for (const p of obstacles) {
      const x = (p.x + .5) * cw, y = (p.y + .5) * ch;
      ctx.fillStyle = '#887b60'; ctx.fillRect(p.x * cw + 2, p.y * ch + 2, cw - 4, ch - 4);
      ctx.strokeStyle = '#b5a17b'; ctx.lineWidth = 2;
      for (let k = -ch; k < cw; k += 8) { ctx.beginPath(); ctx.moveTo(p.x * cw + k, p.y * ch); ctx.lineTo(p.x * cw + k + ch, p.y * ch + ch); ctx.stroke(); }
      ctx.fillStyle = '#5e5746'; ctx.beginPath(); ctx.arc(x, y, 1.5, 0, Math.PI * 2); ctx.fill();
    }
    for (const patch of forage) {
      const x = (patch.x + .5) * cw, y = (patch.y + .5) * ch;
      ctx.fillStyle = patch.available ? '#d4a850' : '#928d70'; ctx.beginPath(); ctx.arc(x, y, Math.min(cw, ch) * .25, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = '#f1e6c4'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x, y, Math.min(cw, ch) * .36, 0, Math.PI * 2); ctx.stroke();
      if (!patch.available) { ctx.strokeStyle = '#655f50'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x - 4, y + 4); ctx.lineTo(x + 4, y - 4); ctx.stroke(); }
    }
    for (const patch of water) {
      const x = (patch.x + .5) * cw, y = (patch.y + .5) * ch;
      roundRect(ctx, x - cw * .28, y - ch * .2, cw * .56, ch * .4, 4);
      ctx.fillStyle = patch.available ? '#719fa2' : '#939e95'; ctx.fill(); ctx.strokeStyle = '#e1e9dc'; ctx.lineWidth = 2; ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x - cw * .16, y); ctx.lineTo(x + cw * .16, y); ctx.stroke();
    }
    animals.forEach((animal, index) => {
      const x = (animal.x + .5) * cw, y = (animal.y + .5) * ch;
      const active = animal.id === app.selected;
      const r = Math.min(cw, ch) * (active ? .34 : .29);
      if (active) { ctx.beginPath(); ctx.arc(x, y, r + 6, 0, Math.PI * 2); ctx.fillStyle = 'rgba(48,68,52,.14)'; ctx.fill(); ctx.strokeStyle = '#34483d'; ctx.lineWidth = 1.5; ctx.stroke(); }
      ctx.beginPath(); ctx.ellipse(x, y, r * 1.2, r * .86, -.12, 0, Math.PI * 2); ctx.fillStyle = '#f4efdc'; ctx.fill(); ctx.strokeStyle = COLORS[index % COLORS.length]; ctx.lineWidth = 2; ctx.stroke();
      // Small head/ear silhouette keeps animal markers readable at small canvas sizes.
      ctx.fillStyle = '#eadfc3'; ctx.beginPath(); ctx.arc(x + r * .82, y - r * .23, r * .36, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      ctx.fillStyle = COLORS[index % COLORS.length]; ctx.beginPath(); ctx.arc(x + r * .9, y - r * .27, 1.4, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#344339'; ctx.font = `700 ${Math.max(8, Math.min(10, r * .85))}px ui-monospace,monospace`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(String(animalNumber(animal.id)), x - r * .1, y + .5);
    });
  }
  function renderEvents() {
    const list = $('event-list');
    const items = app.events.slice(-80).reverse();
    $('event-count').textContent = `${app.events.length} ${app.events.length === 1 ? 'ENTRY' : 'ENTRIES'}`;
    $('empty-events').hidden = items.length > 0;
    const gapNotice = $('event-gap-notice');
    gapNotice.hidden = !app.eventGapNotice;
    gapNotice.textContent = app.eventGapNotice;
    list.innerHTML = items.map((event) => {
      const n = animalNumber(event.animal_id);
      const type = event.kind === 'decision' ? (event.detail.split(' via ')[0] || 'decision') : event.kind;
      const detail = event.kind === 'decision' ? `Selected <strong>${esc(ACTION_LABELS[type] || type)}</strong> using the local policy` : esc(event.detail || event.kind.replaceAll('_', ' '));
      return `<li class="event-row"><time>T${String(event.tick).padStart(3, '0')}</time><span class="event-animal">Cow ${String(n).padStart(2, '0')}</span><span><span class="event-kind">${esc(event.kind.replaceAll('_', ' '))}</span>${detail}</span><span class="event-reward">${Number(event.reward || 0) ? Number(event.reward).toFixed(2) : '—'}</span></li>`;
    }).join('');
  }
  function render() {
    if (!app.state) return;
    $('map-loading').classList.add('hidden');
    updateClock(); renderSelector(); drawMap(); renderInspector(selectedAnimal()); renderEvents();
  }
  async function refresh() {
    if (app.polling) return;
    app.polling = true;
    try {
      const state = await request('/api/state');
      app.state = state;
      if (Number(state.tick || 0) === 0 && Number(state.seed) !== Number($('seed-value').textContent)) app.cursor = -1;
      const result = await request(`/api/events?after=${encodeURIComponent(app.cursor)}`);
      const incoming = result.events || [];
      if (result.retention?.has_gap_after_cursor && !app.eventGapNotice) {
        const oldestTick = incoming[0]?.tick;
        const tickLabel = oldestTick === undefined ? '' : ` The retained log begins at tick ${oldestTick}.`;
        app.eventGapNotice = `Some earlier event entries are no longer retained.${tickLabel}`;
      }
      if (incoming.length) {
        app.events.push(...incoming); app.cursor = Math.max(app.cursor, ...incoming.map((e) => Number(e.id)));
        if (app.events.length > 250) app.events.splice(0, app.events.length - 250);
      }
      setConnection(true, 'Local simulation connected');
      render();
    } catch (error) {
      setConnection(false, 'Local simulation unavailable');
      $('map-loading').textContent = error.message || 'Unable to connect';
      $('map-loading').classList.remove('hidden');
    } finally { app.polling = false; }
  }
  async function post(path, body = {}) {
    try {
      await request(path, { method: 'POST', body: JSON.stringify(body) });
      if (path === '/api/reset') { app.cursor = -1; app.events = []; app.eventGapNotice = ''; app.seedInputDirty = false; }
      await refresh();
    } catch (error) { setConnection(false, error.message || 'Request failed'); }
  }
  function schedule() {
    clearTimeout(app.pollTimer);
    const delay = app.state && !app.state.paused ? Math.max(350, 1000 / (Number(app.state.speed) || 1)) : 1400;
    app.pollTimer = window.setTimeout(async () => { await refresh(); schedule(); }, delay);
  }

  $('animal-selector').addEventListener('click', (event) => {
    const button = event.target.closest('[data-animal]');
    if (!button) return;
    app.selected = button.dataset.animal; render();
  });
  $('toggle-run').addEventListener('click', () => post('/api/control', { action: app.state?.paused ? 'resume' : 'pause' }));
  $('step-once').addEventListener('click', () => post('/api/step'));
  $('speed-select').addEventListener('change', (event) => post('/api/control', { action: 'speed', speed: Number(event.target.value) }));
  $('seed-input').addEventListener('input', () => { app.seedInputDirty = true; });
  $('reset-world').addEventListener('click', () => {
    const value = Number($('seed-input').value);
    if (Number.isInteger(value)) post('/api/reset', { seed: value });
    else setConnection(false, 'Seed must be a whole number');
  });
  if ('ResizeObserver' in window) {
    app.canvasObserver = new ResizeObserver(drawMap); app.canvasObserver.observe(canvas);
  } else window.addEventListener('resize', drawMap);

  refresh().then(schedule);
})();
