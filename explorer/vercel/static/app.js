'use strict';

// ── Runtime mode (FastAPI backend vs. Vercel static bundle vs. ?api=<url>) ──
const _urlParams = new URLSearchParams(window.location.search);
const _paramApi = (_urlParams.get('api') || '').trim().replace(/\/+$/, '');
const _cfgApi = (window.EXPLORER_API_BASE || '').trim().replace(/\/+$/, '');
const API_BASE = _paramApi || _cfgApi;
const IS_STATIC_MODE = Boolean(window.EXPLORER_STATIC_MODE) && !API_BASE;

const _staticState = {
  modelsCatalog: null,
  udaCatalog: null,
  activeBundleId: null,
  bundleData: null,
  cellCache: new Map(),
  lastUdaDemo: null,
};

// ── Constants ───────────────────────────────────────────────────────────────
const LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'];
// 16 distinct editorial colours for PTB-XL diagnostic subclasses (light theme)
const CLASS_COLORS = [
  '#b83b2a', '#2b6cb0', '#2f855a', '#7b4b94',
  '#b7791f', '#0d7a82', '#c05621', '#9b2c5e',
  '#3b82c4', '#4d8b55', '#d06a38', '#8c5ba8',
  '#2d8a8a', '#c49022', '#4a6fa5', '#a6546a',
];

const state = {
  loaded: false,
  somDim: null,   // [H, W]
  classNames: null,
  cells: null,   // flat array [{row,col,count,majority,majority_label,class_counts}]
  colorMode: 'majority',
  selectedLeads: [0],    // default: Lead I only
  currentCellData: null,   // last /api/cell response
  predictedCells: null,   // Set of "r,c" keys highlighted from /api/predict_*
  dominantPred: null,   // "r,c" of dominant predicted SOM cell
  modelsCatalog: null,  // cached /api/models response
};

function switchTab(tabName) {
  document.querySelectorAll('.nav-tab').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === tabName);
  });
  document.querySelectorAll('.tab-view').forEach(view => {
    view.classList.toggle('active', view.id === `view-${tabName}`);
  });
  hideTip();
  if (tabName === 'compare' && typeof window.initCellCompare === 'function') {
    window.initCellCompare();
  } else if (tabName === 'uda' && typeof window.initUdaExplorer === 'function') {
    window.initUdaExplorer();
  } else if (tabName === 'overview' && typeof window.initSystemOverview === 'function') {
    window.initSystemOverview();
  } else if (tabName === 'som' && state.loaded) {
    drawHeatmap();
  }
}

function _syncPredictPanelMode() {
  const panel = document.getElementById('predict-panel');
  const outBox = document.getElementById('predict-result');
  panel.style.display = 'block';
  if (IS_STATIC_MODE) {
    document.getElementById('new-ecg-path').disabled = true;
    document.getElementById('new-ecg-file').disabled = true;
    document.getElementById('new-ecg-fs').disabled = true;
    document.getElementById('predict-btn').disabled = true;
    outBox.style.display = 'block';
    outBox.innerHTML = 'Live ECG file inference requires the local FastAPI/Docker backend (or append <code>?api=https://…</code> to connect to one).';
  } else {
    outBox.style.display = 'none';
  }
}

function applyAdaptedToSom(data) {
  if (!data || data.error) return;
  state.loaded = true;
  state.somDim = data.som_dim;
  state.classNames = data.class_names;
  state.cells = data.cells;
  state.predictedCells = null;
  state.dominantPred = null;
  setStatus(data.info);
  buildClassLegend(data.class_names);
  _syncPredictPanelMode();
  document.getElementById('cell-detail').style.display = 'none';
  window.dispatchEvent(new CustomEvent('som-loaded'));
  switchTab('som');
}

// ── Bootstrap ────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  const metaEl = document.querySelector('.topbar-meta');
  if (metaEl) {
    if (IS_STATIC_MODE) metaEl.textContent += ' · Static Preview';
    else if (API_BASE) metaEl.textContent += ' · Remote API';
  }
  document.querySelectorAll('.nav-tab').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });
  await fetchModels();
  document.getElementById('load-btn').addEventListener('click', loadData);
  document.getElementById('predict-btn').addEventListener('click', predictNewECG);
  document.getElementById('add-to-compare-btn').addEventListener('click', () => {
    if (state.currentCellData && typeof window.addCellToCompare === 'function') {
      window.addCellToCompare(state.currentCellData.row, state.currentCellData.col);
    }
  });
  document.getElementById('color-mode').addEventListener('change', e => {
    state.colorMode = e.target.value;
    if (state.loaded) drawHeatmap();
  });
  // Auto-load the best (top-ranked) checkpoint on startup
  const sel = document.getElementById('ckpt-select');
  const dp = document.getElementById('data-path');
  if (sel.options.length > 0 && dp.value) {
    loadData();
  } else if (IS_STATIC_MODE) {
    setStatus('Lightweight static deploy (no precomputed SOM bundles). Explore UDA Explorer & System Overview above.');
    const hm = document.getElementById('heatmap-container');
    if (hm) {
      hm.innerHTML =
        '<div style="padding:28px;color:var(--muted);line-height:1.6;">' +
        '<strong style="color:var(--text);">Static Preview Notice</strong><br>' +
        'No precomputed SOM bundles were included in this lightweight build. ' +
        'Switch to <strong>UDA Explorer</strong> or <strong>System Overview</strong> above, or run ' +
        '<code>python explorer/build_static_site.py &amp;&amp; vercel --prod</code> locally to include full interactive SOM &amp; Cell Compare matrices, ' +
        'or pass <code>?api=https://your-backend</code>.' +
        '</div>';
    }
  }
});

// ── Model list ───────────────────────────────────────────────────────────────
async function fetchModels() {
  try {
    const data = await apiFetch('/api/models');
    state.modelsCatalog = data;
    const sel = document.getElementById('ckpt-select');
    data.models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.value; opt.textContent = m.label;
      sel.appendChild(opt);
    });
    const dsSel = document.getElementById('data-path');
    dsSel.innerHTML = '';
    (data.datasets || []).forEach(d => {
      const opt = document.createElement('option');
      opt.value = d.value;
      opt.textContent = d.label;
      dsSel.appendChild(opt);
    });
    if (data.default_data) {
      dsSel.value = data.default_data;
    }
  } catch (e) {
    setStatus('Could not fetch model list: ' + e.message);
  }
}

// ── Load & inference ─────────────────────────────────────────────────────────
async function loadData() {
  const ckpt = document.getElementById('ckpt-select').value;
  const dataPath = document.getElementById('data-path').value;
  const split = document.querySelector('input[name="split"]:checked').value;
  const forceRecompute = document.getElementById('force-recompute').checked;
  if (!ckpt || !dataPath) { setStatus('Select a checkpoint and data path.'); return; }

  const btn = document.getElementById('load-btn');
  btn.disabled = true;
  setStatus('<span class="spinner"></span>' + (forceRecompute ? 'Running inference…' : 'Loading…'), true);

  try {
    const url = '/api/load' + (forceRecompute ? '?force_recompute=true' : '');
    const data = await apiFetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ckpt, data_path: dataPath, split }),
    });
    if (data.error) { setStatus('Error: ' + data.error); return; }

    state.loaded = true;
    state.somDim = data.som_dim;
    state.classNames = data.class_names;
    state.cells = data.cells;
    state.predictedCells = null;
    state.dominantPred = null;
    setStatus(data.info);
    buildClassLegend(data.class_names);
    _syncPredictPanelMode();

    drawHeatmap();
    document.getElementById('cell-detail').style.display = 'none';
    window.dispatchEvent(new CustomEvent('som-loaded'));
  } catch (e) {
    setStatus('Error: ' + e.message);
  } finally {
    btn.disabled = false;
  }
}

// ── Predict on new ECG ───────────────────────────────────────────────────────
async function predictNewECG() {
  const pathVal = document.getElementById('new-ecg-path').value.trim();
  const fileEl = document.getElementById('new-ecg-file');
  const fsVal = parseFloat(document.getElementById('new-ecg-fs').value) || 100;
  const outBox = document.getElementById('predict-result');
  const btn = document.getElementById('predict-btn');

  if (!pathVal && (!fileEl.files || fileEl.files.length === 0)) {
    outBox.style.display = 'block';
    outBox.textContent = 'Enter a file path or choose a .npy / .npz / .csv file.';
    return;
  }

  btn.disabled = true;
  outBox.style.display = 'block';
  outBox.innerHTML = '<span class="spinner"></span>Predicting…';

  try {
    let data;
    if (fileEl.files && fileEl.files.length > 0 && !pathVal) {
      const f = fileEl.files[0];
      const buf = new Uint8Array(await f.arrayBuffer());
      let bin = '';
      for (let i = 0; i < buf.length; i += 0x8000) {
        bin += String.fromCharCode(...buf.subarray(i, i + 0x8000));
      }
      data = await apiFetch('/api/predict_upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: f.name, content_b64: btoa(bin), fs: fsVal }),
      });
    } else {
      data = await apiFetch('/api/predict_path', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: pathVal, fs: fsVal }),
      });
    }

    if (data.error) {
      outBox.textContent = 'Error: ' + data.error;
      return;
    }

    const dom = data.dominant_cell;
    state.dominantPred = `${dom.row},${dom.col}`;
    state.predictedCells = new Set(data.beat_cells.map(b => `${b.row},${b.col}`));

    const sexPct = data.predicted_sex === 'Female'
      ? (data.predicted_sex_prob_female * 100).toFixed(1)
      : ((1 - data.predicted_sex_prob_female) * 100).toFixed(1);

    outBox.innerHTML =
      `<strong>${data.record_id}</strong> (${data.num_beats} beats)<br>` +
      `Age: <strong>${data.predicted_age.toFixed(1)} yrs</strong> · ` +
      `Sex: <strong>${data.predicted_sex}</strong> (${sexPct}%)<br>` +
      `SOM cell: <strong>(${dom.row}, ${dom.col})</strong>` +
      (data.dominant_label ? ` · <strong>${data.dominant_label}</strong>` : '');

    drawHeatmap();
    showCellDetail(dom.row, dom.col);
  } catch (e) {
    outBox.textContent = 'Error: ' + e.message;
  } finally {
    btn.disabled = false;
  }
}

// ── Heatmap ──────────────────────────────────────────────────────────────────
function buildClassLegend(classNames) {
  const el = document.getElementById('class-legend');
  el.innerHTML = '';
  classNames.forEach((name, i) => {
    const row = document.createElement('div');
    row.className = 'class-legend-row';
    const swatch = document.createElement('span');
    swatch.className = 'class-legend-swatch';
    swatch.style.background = CLASS_COLORS[i % CLASS_COLORS.length];
    const label = document.createElement('span');
    label.textContent = name;
    row.appendChild(swatch);
    row.appendChild(label);
    el.appendChild(row);
  });
  el.style.display = 'flex';
}

// ResizeObserver to redraw heatmap when container width changes
let _heatmapRO = null;

function drawHeatmap() {
  const [H, W] = state.somDim;
  const cells = state.cells;
  const mode = state.colorMode;
  const classes = state.classNames;

  const container = document.getElementById('heatmap-container');
  container.innerHTML = '';

  // Attach resize observer once
  if (!_heatmapRO) {
    _heatmapRO = new ResizeObserver(() => { if (state.loaded) drawHeatmap(); });
    _heatmapRO.observe(container);
  }

  const M = { t: 28, r: 16, b: 36, l: 38 };
  const availW = (container.clientWidth || container.getBoundingClientRect().width || 600) - M.l - M.r;
  const CELL = Math.max(48, Math.floor(availW / W));
  const svgW = W * CELL + M.l + M.r;
  const svgH = H * CELL + M.t + M.b;

  const svg = d3.select(container).append('svg')
    .attr('width', svgW).attr('height', svgH);
  const g = svg.append('g').attr('transform', `translate(${M.l},${M.t})`);

  // Color helpers (editorial light theme)
  const maxCount = d3.max(cells, d => d.count) || 1;
  const countScale = d3.scaleSequential(d3.interpolateOrRd).domain([0, maxCount]);
  // Soft editorial tint so charcoal text + ECG sparkline remain ultra-legible
  const softTint = hex => {
    const c = d3.hsl(hex);
    c.s = Math.min(c.s, 0.45);
    c.l = 0.92;
    return c.formatHex();
  };
  const cellColor = d => {
    if (d.count === 0) return '#fafafa';
    if (mode === 'count') return countScale(d.count);
    const cls = mode === 'predicted' ? d.predicted : d.majority;
    return cls < 0 ? '#fafafa' : softTint(CLASS_COLORS[cls % CLASS_COLORS.length]);
  };
  const defaultStroke = d => {
    const key = `${d.row},${d.col}`;
    if (state.dominantPred === key) return '#b83b2a';
    if (state.predictedCells && state.predictedCells.has(key)) return '#222222';
    return '#e2e2e2';
  };
  const defaultStrokeW = d => {
    const key = `${d.row},${d.col}`;
    if (state.dominantPred === key) return 2.6;
    if (state.predictedCells && state.predictedCells.has(key)) return 1.8;
    return 1;
  };

  // Cell groups
  const cellG = g.selectAll('.som-cell').data(cells).enter()
    .append('g').attr('class', 'som-cell')
    .attr('transform', d => `translate(${d.col * CELL},${d.row * CELL})`)
    .style('cursor', 'pointer')
    .on('click', (_, d) => showCellDetail(d.row, d.col))
    .on('mouseover', (ev, d) => {
      d3.select(ev.currentTarget).select('.crect')
        .attr('stroke', '#b83b2a').attr('stroke-width', 2);
      showTip(ev, d);
    })
    .on('mousemove', ev => moveTip(ev))
    .on('mouseout', (ev, d) => {
      d3.select(ev.currentTarget).select('.crect')
        .attr('stroke', defaultStroke(d)).attr('stroke-width', defaultStrokeW(d));
      hideTip();
    });

  cellG.append('rect').attr('class', 'crect')
    .attr('width', CELL - 2).attr('height', CELL - 2).attr('rx', 3)
    .attr('fill', cellColor)
    .attr('stroke', defaultStroke)
    .attr('stroke-width', defaultStrokeW);

  // Primary label — majority class name (top of cell)
  const fs = Math.min(11, CELL / 5.5);
  const textY = (CELL - 2) * 0.20;
  cellG.append('text')
    .attr('x', (CELL - 2) / 2)
    .attr('y', textY)
    .attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
    .style('font-size', fs + 'px')
    .style('fill', d => {
      if (d.count === 0) return '#d0d0d0';
      if (mode === 'count' && d.count > maxCount * 0.55) return '#ffffff';
      return '#222222';
    })
    .style('pointer-events', 'none').style('font-weight', '600')
    .text(d => {
      if (mode === 'count') return d.count || '';
      return (mode === 'predicted' ? d.predicted_label : d.majority_label) || '';
    });

  // Count label (below name, majority/predicted modes only)
  if (mode === 'majority' || mode === 'predicted') {
    cellG.append('text')
      .attr('x', (CELL - 2) / 2).attr('y', textY + Math.min(11, CELL / 5.5) + 1)
      .attr('text-anchor', 'middle').attr('dominant-baseline', 'hanging')
      .style('font-size', Math.min(8, CELL / 7.5) + 'px')
      .style('font-family', 'var(--font-mono)')
      .style('fill', '#767676')
      .style('pointer-events', 'none')
      .text(d => d.count > 0 ? d.count : '');
  }

  // SOM centroid sparklines
  const leadIdx = state.selectedLeads[0];
  const sparkPad = 3;
  const sparkTop = (CELL - 2) * 0.40;
  const sparkBot = (CELL - 2) * 0.78;
  const sparkH = sparkBot - sparkTop;
  const sparkW = (CELL - 2) - sparkPad * 2;
  const lineGen = d3.line();
  cellG.each(function (d) {
    if (!d.proto_leads || d.count === 0) return;
    const sig = d.proto_leads[leadIdx];
    const xSc = d3.scaleLinear().domain([0, sig.length - 1]).range([sparkPad, sparkPad + sparkW]);
    const ext = d3.extent(sig);
    const span = (ext[1] - ext[0]) || 1;
    const ySc = d3.scaleLinear()
      .domain([ext[0] - span * 0.08, ext[1] + span * 0.08])
      .range([sparkTop + sparkH, sparkTop]);
    const pts = sig.map((v, i) => [xSc(i), ySc(v)]);
    const sparkStroke = (mode === 'count' && d.count > maxCount * 0.55)
      ? 'rgba(255,255,255,0.88)'
      : 'rgba(34,34,34,0.82)';
    d3.select(this).append('path')
      .attr('d', lineGen(pts))
      .attr('fill', 'none')
      .attr('stroke', sparkStroke)
      .attr('stroke-width', 1.15)
      .style('pointer-events', 'none');
  });

  // Class-distribution stacked bar
  const barTop = (CELL - 2) * 0.83;
  const barHeight = Math.max(4, (CELL - 2) * 0.10);
  const barWidth = (CELL - 2) - sparkPad * 2;
  cellG.each(function (d) {
    if (d.count === 0 || !d.class_counts) return;
    const nC = state.classNames ? state.classNames.length : d.class_counts.length;
    let xOff = sparkPad;
    for (let ci = 0; ci < nC; ci++) {
      const n = d.class_counts[ci] || 0;
      if (n === 0) continue;
      const w = (n / d.count) * barWidth;
      d3.select(this).append('rect')
        .attr('x', xOff).attr('y', barTop)
        .attr('width', w).attr('height', barHeight)
        .attr('fill', CLASS_COLORS[ci % CLASS_COLORS.length])
        .attr('opacity', 0.9)
        .style('pointer-events', 'none');
      xOff += w;
    }
  });

  // Axes
  const xAxis = d3.axisBottom(
    d3.scaleLinear().domain([-0.5, W - 0.5]).range([0, W * CELL])
  ).ticks(W).tickFormat(d3.format('d')).tickSize(3);
  const yAxis = d3.axisLeft(
    d3.scaleLinear().domain([-0.5, H - 0.5]).range([0, H * CELL])
  ).ticks(H).tickFormat(d3.format('d')).tickSize(3);

  g.append('g').attr('transform', `translate(0,${H * CELL + 4})`).call(xAxis)
    .selectAll('text').style('font-size', '9px').style('fill', '#767676');
  g.append('g').attr('transform', 'translate(-4,0)').call(yAxis)
    .selectAll('text').style('font-size', '9px').style('fill', '#767676');

  // Legend
  if (mode === 'count') {
    drawColorbar(svg, M.l + W * CELL + 16, M.t, H * CELL, 0, maxCount,
      d3.interpolateOrRd, '# samples');
  }
}

// ── Tooltip ──────────────────────────────────────────────────────────────────
function showTip(ev, d) {
  let html = `<b>Cell (${d.row}, ${d.col})</b><br>Samples: <b>${d.count}</b>`;
  if (d.majority >= 0 && state.classNames) {
    const cn = state.classNames;
    const isPred = state.colorMode === 'predicted';
    const cls = isPred ? d.predicted : d.majority;
    const n = d.class_counts[cls] || 0;
    html += `<br>${isPred ? 'Predicted' : 'Majority'}: ${cn[cls]} (${n}/${d.count})`;
    const parts = cn.map((c, i) => d.class_counts[i] > 0 ? `${c}:${d.class_counts[i]}` : null)
      .filter(Boolean);
    if (parts.length) html += '<br><span style="color:#767676;font-family:var(--font-mono);font-size:11px">' + parts.join('  ') + '</span>';
  }
  const tip = document.getElementById('tooltip');
  tip.innerHTML = html;
  tip.style.display = 'block';
  moveTip(ev);
}
function moveTip(ev) {
  const tip = document.getElementById('tooltip');
  tip.style.left = (ev.clientX + 14) + 'px';
  tip.style.top = (ev.clientY - 10) + 'px';
}
function hideTip() { document.getElementById('tooltip').style.display = 'none'; }

// ── Colorbar ─────────────────────────────────────────────────────────────────
function drawColorbar(svg, x, y, h, lo, hi, interp, title) {
  const BW = 14, steps = 24;
  const id = 'cbg' + Math.random().toString(36).slice(2);
  const defs = svg.select('defs').empty() ? svg.append('defs') : svg.select('defs');
  const grad = defs.append('linearGradient').attr('id', id)
    .attr('x1', '0%').attr('y1', '100%').attr('x2', '0%').attr('y2', '0%');
  d3.range(steps + 1).forEach(i =>
    grad.append('stop').attr('offset', `${i * 100 / steps}%`)
      .attr('stop-color', interp(i / steps))
  );
  const g = svg.append('g').attr('transform', `translate(${x},${y})`);
  g.append('rect').attr('width', BW).attr('height', h).attr('fill', `url(#${id})`);
  const sc = d3.scaleLinear().domain([lo, hi]).range([h, 0]);
  g.append('g').attr('transform', `translate(${BW},0)`)
    .call(d3.axisRight(sc).ticks(5).tickSize(3))
    .selectAll('text').style('font-size', '9px').style('fill', '#767676');
  g.append('text')
    .attr('transform', `translate(${BW + 48},${h / 2}) rotate(-90)`)
    .attr('text-anchor', 'middle').style('font-size', '9px').style('fill', '#767676')
    .text(title);
}

function drawClassLegend(svg, x, y, classNames) {
  const g = svg.append('g').attr('transform', `translate(${x},${y})`);
  classNames.forEach((cn, i) => {
    g.append('rect').attr('x', 0).attr('y', i * 22).attr('width', 12).attr('height', 12)
      .attr('fill', CLASS_COLORS[i % CLASS_COLORS.length]).attr('rx', 2);
    g.append('text').attr('x', 18).attr('y', i * 22 + 10)
      .style('font-size', '11px').style('fill', '#222222').text(cn);
  });
}

// ── Lead toggle bar ──────────────────────────────────────────────────────────
function buildLeadToggleBar(nLeads) {
  const bar = document.getElementById('lead-toggle-bar');
  bar.innerHTML = '';

  const label = document.createElement('span');
  label.textContent = 'Leads:';
  label.style.cssText = 'font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin-right:6px;white-space:nowrap';
  bar.appendChild(label);

  for (let li = 0; li < nLeads; li++) {
    const btn = document.createElement('button');
    btn.textContent = LEAD_NAMES[li] ?? `L${li}`;
    btn.dataset.lead = li;
    btn.className = 'lead-btn' + (state.selectedLeads.includes(li) ? ' active' : '');
    btn.addEventListener('click', () => {
      if (state.selectedLeads.includes(li)) {
        // keep at least one lead selected
        if (state.selectedLeads.length === 1) return;
        state.selectedLeads = state.selectedLeads.filter(x => x !== li);
      } else {
        state.selectedLeads = [...state.selectedLeads, li].sort((a, b) => a - b);
      }
      document.querySelectorAll('.lead-btn').forEach(b => {
        b.classList.toggle('active', state.selectedLeads.includes(+b.dataset.lead));
      });
      renderCellCharts();
      drawHeatmap();
    });
    bar.appendChild(btn);
  }

  // "All" and "Reset" shortcuts
  const all = document.createElement('button');
  all.textContent = 'All'; all.className = 'lead-btn lead-btn-action';
  all.addEventListener('click', () => {
    state.selectedLeads = d3.range(nLeads);
    document.querySelectorAll('.lead-btn[data-lead]').forEach(b => b.classList.add('active'));
    renderCellCharts();
    drawHeatmap();
  });
  bar.appendChild(all);

  const reset = document.createElement('button');
  reset.textContent = 'I only'; reset.className = 'lead-btn lead-btn-action';
  reset.addEventListener('click', () => {
    state.selectedLeads = [0];
    document.querySelectorAll('.lead-btn[data-lead]').forEach(b =>
      b.classList.toggle('active', b.dataset.lead === '0'));
    renderCellCharts();
    drawHeatmap();
  });
  bar.appendChild(reset);
}

// ── Render cell charts with current selectedLeads ─────────────────────────────
function renderCellCharts() {
  const data = state.currentCellData;
  if (!data) return;
  const { row, col } = data;
  const sel = state.selectedLeads;
  const names = sel.map(i => LEAD_NAMES[i] ?? `L${i}`);

  // Prototype
  drawSignalGrid('proto-chart',
    sel.map(i => data.proto_leads[i]),
    data.t_axis,
    { title: `Prototype — Cell (${row},${col})`, lineColor: '#b83b2a', leadNames: names }
  );

  // Percentile bands
  if (data.percentiles) {
    const pctSel = {};
    for (const k of ['p5', 'p25', 'p50', 'p75', 'p95'])
      pctSel[k] = sel.map(i => data.percentiles[k][i]);
    drawPercentileGrid('pct-chart', pctSel, data.t_axis,
      { title: `Percentile bands  N=${data.n_total}`, leadNames: names }
    );
  } else {
    document.getElementById('pct-chart').innerHTML = dim('No samples in this cell.');
  }

  // Sample list
  const listEl = document.getElementById('sample-list');
  listEl.innerHTML = '';
  data.samples.forEach((s, i) => {
    const card = document.createElement('div');
    card.className = 'sample-card';
    const meta = document.createElement('div');
    meta.className = 'sample-meta';
    meta.textContent = `Rec ${s.record_id}  ·  Beat ${s.beat_idx}  ·  ${s.label}  ·  ${s.sex}, ${Math.round(s.age)} y`;
    card.appendChild(meta);
    const wrap = document.createElement('div');
    wrap.id = `sc${i}`;
    card.appendChild(wrap);
    listEl.appendChild(card);
    drawSignalGrid(`sc${i}`, sel.map(j => s.leads[j]), data.t_axis, {
      lineColor: '#222222',
      cellH: 88,
      margin: { t: 16, r: 4, b: 14, l: 30 },
      leadNames: names,
    });
  });
}

// ── Cell detail ───────────────────────────────────────────────────────────────
async function showCellDetail(row, col) {
  const detail = document.getElementById('cell-detail');
  detail.style.display = 'block';
  detail.scrollIntoView({ behavior: 'smooth', block: 'start' });

  try {
    const data = await apiFetch(`/api/cell/${row}/${col}`);
    state.currentCellData = data;
    document.getElementById('cell-title').textContent =
      `Cell (${row}, ${col})  —  ${data.n_total} samples`;
    document.getElementById('samples-title').textContent =
      `Sample Beats  (showing ${data.n_shown} of ${data.n_total})`;

    const nLeads = data.proto_leads.length;
    buildLeadToggleBar(nLeads);
    renderCellCharts();

  } catch (e) {
    document.getElementById('cell-title').textContent =
      `Cell (${row}, ${col}) — Error: ${e.message}`;
  }
}

// ── Signal grid ───────────────────────────────────────────────────────────────
function drawSignalGrid(containerId, leadsData, tAxis, cfg = {}) {
  const {
    title = '',
    lineColor = '#89b4fa',
    cellH = 108,
    margin: m = { t: 20, r: 6, b: 16, l: 34 },
    leadNames = leadsData.map((_, i) => LEAD_NAMES[i] ?? `L${i}`),
  } = cfg;

  const container = document.getElementById(containerId);
  if (!container) return;
  container.innerHTML = '';

  const nLeads = leadsData.length;
  const cols = 4;
  const rows = Math.ceil(nLeads / cols);
  const titleH = title ? 18 : 0;
  const svgW = Math.max(container.clientWidth || 400, 360);
  const cellW = svgW / cols;
  const totalH = rows * cellH + titleH + 6;

  const svg = d3.select(container).append('svg')
    .attr('width', svgW).attr('height', totalH).style('max-width', '100%');

  if (title) {
    svg.append('text').attr('x', svgW / 2).attr('y', 13)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px').style('fill', '#222222').style('font-weight', '600')
      .text(title);
  }

  for (let li = 0; li < nLeads; li++) {
    const row = Math.floor(li / cols);
    const col = li % cols;
    const gx = col * cellW + m.l;
    const gy = row * cellH + m.t + titleH;
    const w = cellW - m.l - m.r;
    const h = cellH - m.t - m.b;
    const g = svg.append('g').attr('transform', `translate(${gx},${gy})`);

    g.append('rect').attr('width', w).attr('height', h)
      .attr('fill', '#ffffff').attr('stroke', '#e2e2e2').attr('stroke-width', 0.8);

    const xSc = d3.scaleLinear().domain([tAxis[0], tAxis[tAxis.length - 1]]).range([0, w]);
    const ext = d3.extent(leadsData[li]);
    const yPad = ((ext[1] - ext[0]) || 0.2) * 0.12;
    const ySc = d3.scaleLinear().domain([ext[0] - yPad, ext[1] + yPad]).range([h, 0]);

    // Zero line
    if (ext[0] < 0 && ext[1] > 0) {
      g.append('line').attr('x1', 0).attr('x2', w)
        .attr('y1', ySc(0)).attr('y2', ySc(0))
        .attr('stroke', '#d0d0d0').attr('stroke-width', 0.6).attr('stroke-dasharray', '3,2');
    }

    // ECG trace
    const lineFn = d3.line().x((d, i) => xSc(tAxis[i])).y(d => ySc(d));
    g.append('path').datum(leadsData[li]).attr('d', lineFn)
      .attr('fill', 'none').attr('stroke', lineColor).attr('stroke-width', 1.4);

    // Lead label
    g.append('text').attr('x', w / 2).attr('y', -5)
      .attr('text-anchor', 'middle')
      .style('font-size', '9px').style('fill', '#444444').style('font-weight', '600')
      .text(leadNames[li] ?? `L${li}`);

    // Y-axis (leftmost column)
    if (col === 0) {
      g.append('g').call(
        d3.axisLeft(ySc).ticks(2).tickSize(2).tickFormat(v => v.toFixed(1))
      ).style('color', '#767676').selectAll('text').style('font-size', '6.5px');
    }
    // X-axis (bottom row)
    if (row === rows - 1) {
      g.append('g').attr('transform', `translate(0,${h})`).call(
        d3.axisBottom(xSc).ticks(3).tickSize(2).tickFormat(v => v.toFixed(1) + 's')
      ).style('color', '#767676').selectAll('text').style('font-size', '6.5px');
    }
  }
}

// ── Percentile band grid ──────────────────────────────────────────────────────
function drawPercentileGrid(containerId, pcts, tAxis, cfg = {}) {
  const {
    title = '',
    cellH = 108,
    margin: m = { t: 20, r: 6, b: 16, l: 34 },
    leadNames = pcts.p50.map((_, i) => LEAD_NAMES[i] ?? `L${i}`),
  } = cfg;

  const container = document.getElementById(containerId);
  if (!container) return;
  container.innerHTML = '';

  const nLeads = pcts.p50.length;
  const cols = 4;
  const rows = Math.ceil(nLeads / cols);
  const N = tAxis.length;
  const idx = d3.range(N);
  const titleH = title ? 18 : 0;
  const svgW = Math.max(container.clientWidth || 400, 360);
  const cellW = svgW / cols;
  const totalH = rows * cellH + titleH + 6;

  const svg = d3.select(container).append('svg')
    .attr('width', svgW).attr('height', totalH).style('max-width', '100%');

  if (title) {
    svg.append('text').attr('x', svgW / 2).attr('y', 13)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px').style('fill', '#222222').style('font-weight', '600')
      .text(title);
  }

  // Legend
  const legY = totalH - 6;
  [
    { col: 'rgba(184,59,42,0.12)', label: '5–95 pct' },
    { col: 'rgba(184,59,42,0.26)', label: 'IQR' },
    { col: '#b83b2a', label: 'Median' },
  ].forEach((e, i) => {
    svg.append('rect').attr('x', 10 + i * 80).attr('y', legY - 7)
      .attr('width', 12).attr('height', 6).attr('rx', 1).attr('fill', e.col);
    svg.append('text').attr('x', 26 + i * 80).attr('y', legY)
      .style('font-size', '8px').style('fill', '#767676').text(e.label);
  });

  for (let li = 0; li < nLeads; li++) {
    const row = Math.floor(li / cols);
    const col = li % cols;
    const gx = col * cellW + m.l;
    const gy = row * cellH + m.t + titleH;
    const w = cellW - m.l - m.r;
    const h = cellH - m.t - m.b;
    const g = svg.append('g').attr('transform', `translate(${gx},${gy})`);

    g.append('rect').attr('width', w).attr('height', h)
      .attr('fill', '#ffffff').attr('stroke', '#e2e2e2').attr('stroke-width', 0.8);

    const xSc = d3.scaleLinear().domain([tAxis[0], tAxis[tAxis.length - 1]]).range([0, w]);
    const allV = [...pcts.p5[li], ...pcts.p95[li]];
    const ext = d3.extent(allV);
    const yPad = ((ext[1] - ext[0]) || 0.2) * 0.12;
    const ySc = d3.scaleLinear().domain([ext[0] - yPad, ext[1] + yPad]).range([h, 0]);

    const areaFn = (lo, hi) =>
      d3.area().x(i => xSc(tAxis[i])).y0(i => ySc(lo[i])).y1(i => ySc(hi[i]))(idx);
    const lineFn = arr =>
      d3.line().x(i => xSc(tAxis[i])).y(i => ySc(arr[i]))(idx);

    // 5–95 band
    g.append('path').attr('d', areaFn(pcts.p5[li], pcts.p95[li]))
      .attr('fill', 'rgba(184,59,42,0.10)').attr('stroke', 'none');
    // IQR band
    g.append('path').attr('d', areaFn(pcts.p25[li], pcts.p75[li]))
      .attr('fill', 'rgba(184,59,42,0.22)').attr('stroke', 'none');
    // Median
    g.append('path').attr('d', lineFn(pcts.p50[li]))
      .attr('fill', 'none').attr('stroke', '#b83b2a').attr('stroke-width', 1.7);

    // Lead label
    g.append('text').attr('x', w / 2).attr('y', -5)
      .attr('text-anchor', 'middle')
      .style('font-size', '9px').style('fill', '#444444').style('font-weight', '600')
      .text(leadNames[li] ?? `L${li}`);

    if (col === 0) {
      g.append('g').call(
        d3.axisLeft(ySc).ticks(2).tickSize(2).tickFormat(v => v.toFixed(1))
      ).style('color', '#767676').selectAll('text').style('font-size', '6.5px');
    }
    if (row === rows - 1) {
      g.append('g').attr('transform', `translate(0,${h})`).call(
        d3.axisBottom(xSc).ticks(3).tickSize(2).tickFormat(v => v.toFixed(1) + 's')
      ).style('color', '#767676').selectAll('text').style('font-size', '6.5px');
    }
  }
}

// ── Static-Mode Client Adapter (for Vercel / static CDN hosting) ─────────────
async function _fetchJsonDirect(url) {
  const resp = await fetch(url);
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`${resp.status}: ${text}`);
  }
  return resp.json();
}

function _jsDivergence(countsP, countsQ) {
  const sp = countsP.reduce((s, v) => s + v, 0);
  const sq = countsQ.reduce((s, v) => s + v, 0);
  if (sp <= 0 || sq <= 0) return 0;
  let sumP = 0;
  let sumQ = 0;
  for (let k = 0; k < countsP.length; k++) {
    const p = countsP[k] / sp;
    const q = countsQ[k] / sq;
    const m = 0.5 * (p + q);
    if (p > 0 && m > 0) sumP += p * Math.log2(p / m);
    if (q > 0 && m > 0) sumQ += q * Math.log2(q / m);
  }
  return Math.max(0, Math.min(1, 0.5 * (sumP + sumQ)));
}

function _toroidalDist(r1, c1, r2, c2, H, W) {
  const dr = Math.min(Math.abs(r1 - r2), H - Math.abs(r1 - r2));
  const dc = Math.min(Math.abs(c1 - c2), W - Math.abs(c1 - c2));
  return [dr + dc, Math.hypot(dr, dc)];
}

function _l2Norm(vec) {
  let acc = 0;
  for (let i = 0; i < vec.length; i++) acc += vec[i] * vec[i];
  return Math.sqrt(acc);
}

function _l2Dist(a, b) {
  let acc = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    acc += d * d;
  }
  return Math.sqrt(acc);
}

function _cosSim(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot / Math.max(_l2Norm(a) * _l2Norm(b), 1e-8);
}

function _waveformRmse(p1, p2) {
  let acc = 0;
  let cnt = 0;
  for (let li = 0; li < p1.length; li++) {
    const a = p1[li];
    const b = p2[li];
    for (let ti = 0; ti < a.length; ti++) {
      const d = a[ti] - b[ti];
      acc += d * d;
      cnt += 1;
    }
  }
  return cnt > 0 ? Math.sqrt(acc / cnt) : 0;
}

async function _loadStaticSomBundle(bundleId, infoSuffix = '') {
  const data = await _fetchJsonDirect(`./data/som/${bundleId}.json`);
  _staticState.activeBundleId = bundleId;
  _staticState.bundleData = data;
  _staticState.cellCache.clear();
  if (infoSuffix && data.info && !data.info.includes(infoSuffix.trim())) {
    data.info += infoSuffix;
  }
  return data;
}

async function _staticCompareCells(body) {
  const bundle = _staticState.bundleData;
  if (!bundle) throw new Error('400: No SOM bundle loaded.');
  const reqCells = body?.cells || [];
  if (!reqCells.length || reqCells.length > 6) {
    throw new Error('400: Select between 1 and 6 SOM cells to compare.');
  }
  const [H, W] = bundle.som_dim;
  const coords = [];
  const seen = new Set();
  for (const pair of reqCells) {
    if (!Array.isArray(pair) || pair.length !== 2) throw new Error('400: Invalid cell coordinate.');
    const r = Number(pair[0]);
    const c = Number(pair[1]);
    if (r < 0 || r >= H || c < 0 || c >= W) throw new Error(`404: Cell (${r}, ${c}) out of range.`);
    const key = `${r},${c}`;
    if (!seen.has(key)) {
      seen.add(key);
      coords.push([r, c]);
    }
  }

  const details = await Promise.all(coords.map(([r, c]) => staticApiFetch(`/api/cell/${r}/${c}`)));
  const defaultAgeLabels = ['<30', '30s', '40s', '50s', '60s', '70s', '80+'];
  const cellPayloads = coords.map(([r, c], idx) => {
    const flat = r * W + c;
    const summary = bundle.cells[flat] || {};
    const stats = (bundle.cell_stats && bundle.cell_stats[flat]) || {};
    const detail = details[idx] || {};
    return {
      row:            r,
      col:            c,
      count:          detail.n_total ?? summary.count ?? 0,
      majority_label: summary.majority_label || '',
      purity:         stats.purity ?? 0,
      class_counts:   summary.class_counts || [],
      top_classes:    stats.top_classes || [],
      age_mean:       stats.age_mean ?? null,
      age_std:        stats.age_std ?? null,
      age_median:     stats.age_median ?? null,
      age_bins:       stats.age_bins || [0, 0, 0, 0, 0, 0, 0],
      age_bin_labels: stats.age_bin_labels || defaultAgeLabels,
      female_pct:     stats.female_pct ?? null,
      proto_leads:    detail.proto_leads || summary.proto_leads,
      percentiles:    detail.percentiles || null,
    };
  });

  const emb = bundle.embeddings || null;
  const pairwise = coords.map(([r1, c1], i) => coords.map(([r2, c2], j) => {
    const flat1 = r1 * W + c1;
    const flat2 = r2 * W + c2;
    const [tl1, tl2] = _toroidalDist(r1, c1, r2, c2, H, W);
    const e1 = emb && emb[r1]?.[c1];
    const e2 = emb && emb[r2]?.[c2];
    const p1 = bundle.cells[flat1]?.proto_leads || [];
    const p2 = bundle.cells[flat2]?.proto_leads || [];
    return {
      toroidal_l1:   tl1,
      toroidal_l2:   tl2,
      latent_l2:     (e1 && e2) ? _l2Dist(e1, e2) : 0,
      latent_cos:    (e1 && e2) ? _cosSim(e1, e2) : 1,
      waveform_rmse: (p1.length && p2.length) ? _waveformRmse(p1, p2) : 0,
      js_divergence: _jsDivergence(cellPayloads[i].class_counts, cellPayloads[j].class_counts),
    };
  }));

  return {
    som_dim:     bundle.som_dim,
    class_names: bundle.class_names,
    t_axis:      bundle.t_axis || details[0]?.t_axis || [],
    cells:       cellPayloads,
    pairwise,
  };
}

async function staticApiFetch(url, opts = {}) {
  const path = url.split('?')[0];
  const body = opts.body ? JSON.parse(opts.body) : {};

  if (path === '/api/models') {
    if (!_staticState.modelsCatalog) {
      _staticState.modelsCatalog = await _fetchJsonDirect('./data/models.json');
    }
    return _staticState.modelsCatalog;
  }

  if (path === '/api/load') {
    const cat = _staticState.modelsCatalog || await staticApiFetch('/api/models');
    const ckpt = body.ckpt || '';
    const dataPath = body.data_path || '';
    const split = body.split || 'test';
    const bundles = cat.bundles || {};
    let bundleId = bundles[`${ckpt}|${dataPath}|${split}`];
    let suffix = '';
    if (!bundleId && split !== 'test') {
      bundleId = bundles[`${ckpt}|${dataPath}|test`];
      if (bundleId) suffix = '  |  (static preview: test split)';
    }
    if (!bundleId) {
      const meta = cat.bundle_meta || {};
      const match = Object.entries(meta).find(([, m]) => m.ckpt === ckpt)
        || Object.entries(meta).find(([, m]) => m.data_path === dataPath)
        || Object.entries(meta)[0];
      if (!match) {
        throw new Error('No precomputed SOM bundle exists for this lightweight static build.');
      }
      bundleId = match[0];
      suffix = `  |  (static preview bundle: ${match[1].data_path})`;
    }
    return _loadStaticSomBundle(bundleId, suffix);
  }

  const cellMatch = path.match(/^\/api\/cell\/(\d+)\/(\d+)$/);
  if (cellMatch) {
    if (!_staticState.activeBundleId) {
      throw new Error('400: No SOM bundle loaded yet.');
    }
    const r = Number(cellMatch[1]);
    const c = Number(cellMatch[2]);
    const key = `${_staticState.activeBundleId}:${r},${c}`;
    if (!_staticState.cellCache.has(key)) {
      const detail = await _fetchJsonDirect(`./data/cells/${_staticState.activeBundleId}/${r}_${c}.json`);
      _staticState.cellCache.set(key, detail);
    }
    return _staticState.cellCache.get(key);
  }

  if (path === '/api/cells/compare') {
    return _staticCompareCells(body);
  }

  if (path === '/api/overview') {
    return _fetchJsonDirect('./data/overview.json');
  }

  if (path === '/api/uda/results') {
    if (!_staticState.udaCatalog) {
      _staticState.udaCatalog = await _fetchJsonDirect('./data/uda_results.json');
    }
    return _staticState.udaCatalog;
  }

  if (path === '/api/uda/run') {
    const udaCat = _staticState.udaCatalog || await staticApiFetch('/api/uda/results');
    const demos = udaCat.static_demos || [];
    if (!demos.length) {
      throw new Error('Live UDA Playground requires a FastAPI backend or precomputed UDA demo snapshots.');
    }
    const method = body.method || 'uda_bw_ot';
    const ckpt = body.ckpt || '';
    const tgt = body.tgt_data || '';
    const match = demos.find(d => d.ckpt === ckpt && d.tgt_data === tgt && d.method === method)
      || demos.find(d => d.tgt_data === tgt && d.method === method)
      || demos.find(d => d.method === method)
      || demos[0];
    const demoData = await _fetchJsonDirect(match.url);
    _staticState.lastUdaDemo = match;
    const isExact = match.tgt_data === tgt && match.method === method;
    demoData.note = isExact
      ? 'Precomputed Vercel demo snapshot (default params). Run locally or pass ?api= for custom sweeps.'
      : `Matched closest static demo (${match.tgt_data}, ${match.method}). Pass ?api= for custom runs.`;
    return demoData;
  }

  if (path === '/api/uda/apply') {
    const demo = _staticState.lastUdaDemo;
    if (!demo || !demo.adapted_bundle_id) {
      throw new Error('400: Run a UDA preview first before applying its adapted SOM.');
    }
    return _loadStaticSomBundle(demo.adapted_bundle_id);
  }

  if (path === '/api/predict_path' || path === '/api/predict_upload') {
    throw new Error('Live ECG inference requires local FastAPI/Docker or passing ?api=https://your-backend.');
  }

  throw new Error(`Unsupported static endpoint: ${path}`);
}

// ── Utilities ─────────────────────────────────────────────────────────────────
async function apiFetch(url, opts) {
  if (IS_STATIC_MODE) {
    return staticApiFetch(url, opts);
  }
  const target = API_BASE ? `${API_BASE}${url}` : url;
  const resp = await fetch(target, opts);
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`${resp.status}: ${text}`);
  }
  return resp.json();
}

function setStatus(msg, isHtml = false) {
  const el = document.getElementById('status');
  if (isHtml) el.innerHTML = msg; else el.textContent = msg;
}

function showTooltipHtml(ev, html) {
  const tip = document.getElementById('tooltip');
  tip.innerHTML = html;
  tip.style.display = 'block';
  moveTip(ev);
}

function dim(msg) {
  return `<p style="color:var(--muted);padding:10px 0">${msg}</p>`;
}

window.explorerShared = {
  apiFetch,
  getApiBase: () => API_BASE,
  isStaticMode: () => IS_STATIC_MODE,
  CLASS_COLORS,
  LEAD_NAMES,
  showTooltipHtml,
  moveTip,
  hideTip,
  switchTab,
  applyAdaptedToSom,
  getModelsCatalog: () => state.modelsCatalog,
  getSomState: () => ({
    loaded: state.loaded,
    somDim: state.somDim,
    classNames: state.classNames,
    cells: state.cells,
  }),
};
