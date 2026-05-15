'use strict';

// ── Constants ───────────────────────────────────────────────────────────────
const LEAD_NAMES  = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6'];
// Catppuccin Mocha-inspired palette — distinct on dark cell backgrounds
const CLASS_COLORS = ['#e06c75','#61afef','#98c379','#c678dd','#e5c07b'];

const state = {
  loaded:          false,
  somDim:          null,   // [H, W]
  classNames:      null,
  cells:           null,   // flat array [{row,col,count,majority,majority_label,class_counts}]
  colorMode:       'majority',
  selectedLeads:   [0],    // default: Lead I only
  currentCellData: null,   // last /api/cell response
};

// ── Bootstrap ────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  await fetchModels();
  document.getElementById('load-btn').addEventListener('click', loadData);
  document.getElementById('color-mode').addEventListener('change', e => {
    state.colorMode = e.target.value;
    if (state.loaded) drawHeatmap();
  });
});

// ── Model list ───────────────────────────────────────────────────────────────
async function fetchModels() {
  try {
    const data = await apiFetch('/api/models');
    const sel  = document.getElementById('ckpt-select');
    data.models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.value; opt.textContent = m.label;
      sel.appendChild(opt);
    });
    if (data.default_data)
      document.getElementById('data-path').value = data.default_data;
  } catch (e) {
    setStatus('Could not fetch model list: ' + e.message);
  }
}

// ── Load & inference ─────────────────────────────────────────────────────────
async function loadData() {
  const ckpt          = document.getElementById('ckpt-select').value;
  const dataPath      = document.getElementById('data-path').value;
  const split         = document.querySelector('input[name="split"]:checked').value;
  const forceRecompute = document.getElementById('force-recompute').checked;
  if (!ckpt || !dataPath) { setStatus('Select a checkpoint and data path.'); return; }

  const btn = document.getElementById('load-btn');
  btn.disabled = true;
  setStatus('<span class="spinner"></span>' + (forceRecompute ? 'Running inference…' : 'Loading…'), true);

  try {
    const url = '/api/load' + (forceRecompute ? '?force_recompute=true' : '');
    const data = await apiFetch(url, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ckpt, data_path: dataPath, split}),
    });
    if (data.error) { setStatus('Error: ' + data.error); return; }

    state.loaded     = true;
    state.somDim     = data.som_dim;
    state.classNames = data.class_names;
    state.cells      = data.cells;
    setStatus(data.info);
    buildClassLegend(data.class_names);

    drawHeatmap();
    document.getElementById('cell-detail').style.display = 'none';
  } catch (e) {
    setStatus('Error: ' + e.message);
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
  const [H, W]   = state.somDim;
  const cells     = state.cells;
  const mode      = state.colorMode;
  const classes   = state.classNames;

  const container = document.getElementById('heatmap-container');
  container.innerHTML = '';

  // Attach resize observer once
  if (!_heatmapRO) {
    _heatmapRO = new ResizeObserver(() => { if (state.loaded) drawHeatmap(); });
    _heatmapRO.observe(container);
  }

  const M         = {t: 28, r: 16, b: 36, l: 38};
  const availW    = (container.clientWidth || container.getBoundingClientRect().width || 600) - M.l - M.r;
  const CELL      = Math.max(48, Math.floor(availW / W));
  const svgW      = W * CELL + M.l + M.r;
  const svgH      = H * CELL + M.t + M.b;

  const svg = d3.select(container).append('svg')
    .attr('width', svgW).attr('height', svgH);
  const g = svg.append('g').attr('transform', `translate(${M.l},${M.t})`);

  // Color helpers
  const maxCount   = d3.max(cells, d => d.count) || 1;
  const countScale = d3.scaleSequential(d3.interpolateYlGnBu).domain([0, maxCount]);
  // Darken majority-class color slightly so white text + sparkline stay readable
  const darken = hex => {
    const c = d3.hsl(hex);
    c.l = Math.max(0.18, c.l - 0.18);
    return c.formatHex();
  };
  const cellColor  = d => {
    if (d.count === 0) return '#1e1e2e';
    return mode === 'count'
      ? countScale(d.count)
      : (d.majority < 0 ? '#1e1e2e' : darken(CLASS_COLORS[d.majority % CLASS_COLORS.length]));
  };

  // Cell groups
  const cellG = g.selectAll('.som-cell').data(cells).enter()
    .append('g').attr('class', 'som-cell')
    .attr('transform', d => `translate(${d.col * CELL},${d.row * CELL})`)
    .style('cursor', 'pointer')
    .on('click',     (_, d) => showCellDetail(d.row, d.col))
    .on('mouseover', (ev, d) => {
      d3.select(ev.currentTarget).select('.crect')
        .attr('stroke', '#cdd6f4').attr('stroke-width', 2);
      showTip(ev, d);
    })
    .on('mousemove', ev => moveTip(ev))
    .on('mouseout',  ev => {
      d3.select(ev.currentTarget).select('.crect')
        .attr('stroke', '#45475a').attr('stroke-width', 1);
      hideTip();
    });

  cellG.append('rect').attr('class', 'crect')
    .attr('width', CELL - 2).attr('height', CELL - 2).attr('rx', 3)
    .attr('fill', cellColor)
    .attr('stroke', '#45475a').attr('stroke-width', 1);

  // Primary label — majority class name (top of cell)
  const fs = Math.min(11, CELL / 5.5);
  const textY = (CELL - 2) * 0.20;
  cellG.append('text')
    .attr('x', (CELL - 2) / 2)
    .attr('y', textY)
    .attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
    .style('font-size', fs + 'px').style('fill', d => d.count === 0 ? '#45475a' : '#eee')
    .style('pointer-events', 'none').style('font-weight', '600')
    .text(d => mode === 'count' ? (d.count || '') : (d.majority_label || ''));

  // Count label (below name, majority mode only)
  if (mode === 'majority') {
    cellG.append('text')
      .attr('x', (CELL - 2) / 2).attr('y', textY + Math.min(11, CELL / 5.5) + 1)
      .attr('text-anchor', 'middle').attr('dominant-baseline', 'hanging')
      .style('font-size', Math.min(8, CELL / 7.5) + 'px').style('fill', 'rgba(255,255,255,0.65)')
      .style('pointer-events', 'none')
      .text(d => d.count > 0 ? d.count : '');
  }

  // SOM centroid sparklines
  const leadIdx  = state.selectedLeads[0];
  const sparkPad = 3;
  const sparkTop = (CELL - 2) * 0.40;
  const sparkBot = (CELL - 2) * 0.78;
  const sparkH   = sparkBot - sparkTop;
  const sparkW   = (CELL - 2) - sparkPad * 2;
  const lineGen  = d3.line();
  cellG.each(function(d) {
    if (!d.proto_leads || d.count === 0) return;
    const sig  = d.proto_leads[leadIdx];
    const xSc  = d3.scaleLinear().domain([0, sig.length - 1]).range([sparkPad, sparkPad + sparkW]);
    const ext  = d3.extent(sig);
    const span = (ext[1] - ext[0]) || 1;
    const ySc  = d3.scaleLinear()
      .domain([ext[0] - span * 0.08, ext[1] + span * 0.08])
      .range([sparkTop + sparkH, sparkTop]);
    const pts  = sig.map((v, i) => [xSc(i), ySc(v)]);
    d3.select(this).append('path')
      .attr('d', lineGen(pts))
      .attr('fill', 'none')
      .attr('stroke', 'rgba(255,255,255,0.85)')
      .attr('stroke-width', 1.2)
      .style('pointer-events', 'none');
  });

  // Class-distribution stacked bar
  const barTop    = (CELL - 2) * 0.83;
  const barHeight = Math.max(4, (CELL - 2) * 0.10);
  const barWidth  = (CELL - 2) - sparkPad * 2;
  cellG.each(function(d) {
    if (d.count === 0 || !d.class_counts) return;
    const nC    = state.classNames ? state.classNames.length : d.class_counts.length;
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
    .selectAll('text').style('font-size', '9px').style('fill', '#a6adc8');
  g.append('g').attr('transform', 'translate(-4,0)').call(yAxis)
    .selectAll('text').style('font-size', '9px').style('fill', '#a6adc8');

  // Legend
  if (mode === 'count') {
    drawColorbar(svg, M.l + W * CELL + 16, M.t, H * CELL, 0, maxCount,
                 d3.interpolateYlGnBu, '# samples');
  }
}

// ── Tooltip ──────────────────────────────────────────────────────────────────
function showTip(ev, d) {
  let html = `<b>Cell (${d.row}, ${d.col})</b><br>Samples: <b>${d.count}</b>`;
  if (d.majority >= 0 && state.classNames) {
    const cn = state.classNames;
    const n  = d.class_counts[d.majority] || 0;
    html    += `<br>Majority: ${cn[d.majority]} (${n}/${d.count})`;
    const parts = cn.map((c, i) => d.class_counts[i] > 0 ? `${c}:${d.class_counts[i]}` : null)
                    .filter(Boolean);
    if (parts.length) html += '<br><span style="color:#6c7086">' + parts.join('  ') + '</span>';
  }
  const tip = document.getElementById('tooltip');
  tip.innerHTML = html;
  tip.style.display = 'block';
  moveTip(ev);
}
function moveTip(ev) {
  const tip = document.getElementById('tooltip');
  tip.style.left = (ev.clientX + 14) + 'px';
  tip.style.top  = (ev.clientY - 10) + 'px';
}
function hideTip() { document.getElementById('tooltip').style.display = 'none'; }

// ── Colorbar ─────────────────────────────────────────────────────────────────
function drawColorbar(svg, x, y, h, lo, hi, interp, title) {
  const BW = 14, steps = 24;
  const id  = 'cbg' + Math.random().toString(36).slice(2);
  const defs = svg.select('defs').empty() ? svg.append('defs') : svg.select('defs');
  const grad = defs.append('linearGradient').attr('id', id)
    .attr('x1','0%').attr('y1','100%').attr('x2','0%').attr('y2','0%');
  d3.range(steps + 1).forEach(i =>
    grad.append('stop').attr('offset', `${i * 100 / steps}%`)
        .attr('stop-color', interp(i / steps))
  );
  const g  = svg.append('g').attr('transform', `translate(${x},${y})`);
  g.append('rect').attr('width', BW).attr('height', h).attr('fill', `url(#${id})`);
  const sc = d3.scaleLinear().domain([lo, hi]).range([h, 0]);
  g.append('g').attr('transform', `translate(${BW},0)`)
   .call(d3.axisRight(sc).ticks(5).tickSize(3))
   .selectAll('text').style('font-size','9px').style('fill','#a6adc8');
  g.append('text')
   .attr('transform', `translate(${BW + 48},${h / 2}) rotate(-90)`)
   .attr('text-anchor','middle').style('font-size','9px').style('fill','#a6adc8')
   .text(title);
}

function drawClassLegend(svg, x, y, classNames) {
  const g = svg.append('g').attr('transform', `translate(${x},${y})`);
  classNames.forEach((cn, i) => {
    g.append('rect').attr('x',0).attr('y',i*22).attr('width',12).attr('height',12)
     .attr('fill', CLASS_COLORS[i % CLASS_COLORS.length]).attr('rx', 2);
    g.append('text').attr('x',18).attr('y',i*22+10)
     .style('font-size','11px').style('fill','#cdd6f4').text(cn);
  });
}

// ── Lead toggle bar ──────────────────────────────────────────────────────────
function buildLeadToggleBar(nLeads) {
  const bar = document.getElementById('lead-toggle-bar');
  bar.innerHTML = '';

  const label = document.createElement('span');
  label.textContent = 'Leads:';
  label.style.cssText = 'font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--subtext);margin-right:6px;white-space:nowrap';
  bar.appendChild(label);

  for (let li = 0; li < nLeads; li++) {
    const btn = document.createElement('button');
    btn.textContent  = LEAD_NAMES[li] ?? `L${li}`;
    btn.dataset.lead = li;
    btn.className    = 'lead-btn' + (state.selectedLeads.includes(li) ? ' active' : '');
    btn.addEventListener('click', () => {
      if (state.selectedLeads.includes(li)) {
        // keep at least one lead selected
        if (state.selectedLeads.length === 1) return;
        state.selectedLeads = state.selectedLeads.filter(x => x !== li);
      } else {
        state.selectedLeads = [...state.selectedLeads, li].sort((a,b) => a-b);
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
  const sel   = state.selectedLeads;
  const names = sel.map(i => LEAD_NAMES[i] ?? `L${i}`);

  // Prototype
  drawSignalGrid('proto-chart',
    sel.map(i => data.proto_leads[i]),
    data.t_axis,
    { title: `Prototype — Cell (${row},${col})`, lineColor: '#89b4fa', leadNames: names }
  );

  // Percentile bands
  if (data.percentiles) {
    const pctSel = {};
    for (const k of ['p5','p25','p50','p75','p95'])
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
      lineColor: '#a6e3a1',
      cellH: 88,
      margin: {t:16, r:4, b:14, l:30},
      leadNames: names,
    });
  });
}

// ── Cell detail ───────────────────────────────────────────────────────────────
async function showCellDetail(row, col) {
  const detail = document.getElementById('cell-detail');
  detail.style.display = 'block';
  detail.scrollIntoView({behavior:'smooth', block:'start'});

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
    title     = '',
    lineColor = '#89b4fa',
    cellH     = 108,
    margin: m = {t:20, r:6, b:16, l:34},
    leadNames = leadsData.map((_, i) => LEAD_NAMES[i] ?? `L${i}`),
  } = cfg;

  const container = document.getElementById(containerId);
  if (!container) return;
  container.innerHTML = '';

  const nLeads = leadsData.length;
  const cols   = 4;
  const rows   = Math.ceil(nLeads / cols);
  const titleH = title ? 18 : 0;
  const svgW   = Math.max(container.clientWidth || 400, 360);
  const cellW  = svgW / cols;
  const totalH = rows * cellH + titleH + 6;

  const svg = d3.select(container).append('svg')
    .attr('width', svgW).attr('height', totalH).style('max-width','100%');

  if (title) {
    svg.append('text').attr('x', svgW/2).attr('y', 13)
      .attr('text-anchor','middle')
      .style('font-size','11px').style('fill','#cdd6f4').style('font-weight','700')
      .text(title);
  }

  for (let li = 0; li < nLeads; li++) {
    const row  = Math.floor(li / cols);
    const col  = li % cols;
    const gx   = col * cellW + m.l;
    const gy   = row * cellH + m.t + titleH;
    const w    = cellW - m.l - m.r;
    const h    = cellH - m.t - m.b;
    const g    = svg.append('g').attr('transform', `translate(${gx},${gy})`);

    g.append('rect').attr('width', w).attr('height', h)
      .attr('fill','#181825').attr('stroke','#313244').attr('stroke-width',0.5);

    const xSc = d3.scaleLinear().domain([tAxis[0], tAxis[tAxis.length - 1]]).range([0, w]);
    const ext  = d3.extent(leadsData[li]);
    const yPad = ((ext[1] - ext[0]) || 0.2) * 0.12;
    const ySc  = d3.scaleLinear().domain([ext[0] - yPad, ext[1] + yPad]).range([h, 0]);

    // Zero line
    if (ext[0] < 0 && ext[1] > 0) {
      g.append('line').attr('x1',0).attr('x2',w)
       .attr('y1',ySc(0)).attr('y2',ySc(0))
       .attr('stroke','#45475a').attr('stroke-width',0.4).attr('stroke-dasharray','3,2');
    }

    // ECG trace
    const lineFn = d3.line().x((d,i) => xSc(tAxis[i])).y(d => ySc(d));
    g.append('path').datum(leadsData[li]).attr('d', lineFn)
      .attr('fill','none').attr('stroke', lineColor).attr('stroke-width', 1.4);

    // Lead label
    g.append('text').attr('x', w / 2).attr('y', -5)
      .attr('text-anchor','middle')
      .style('font-size','8.5px').style('fill','#a6adc8')
      .text(leadNames[li] ?? `L${li}`);

    // Y-axis (leftmost column)
    if (col === 0) {
      g.append('g').call(
        d3.axisLeft(ySc).ticks(2).tickSize(2).tickFormat(v => v.toFixed(1))
      ).style('color','#6c7086').selectAll('text').style('font-size','6px');
    }
    // X-axis (bottom row)
    if (row === rows - 1) {
      g.append('g').attr('transform', `translate(0,${h})`).call(
        d3.axisBottom(xSc).ticks(3).tickSize(2).tickFormat(v => v.toFixed(1) + 's')
      ).style('color','#6c7086').selectAll('text').style('font-size','6px');
    }
  }
}

// ── Percentile band grid ──────────────────────────────────────────────────────
function drawPercentileGrid(containerId, pcts, tAxis, cfg = {}) {
  const {
    title  = '',
    cellH  = 108,
    margin: m = {t:20, r:6, b:16, l:34},
    leadNames = pcts.p50.map((_, i) => LEAD_NAMES[i] ?? `L${i}`),
  } = cfg;

  const container = document.getElementById(containerId);
  if (!container) return;
  container.innerHTML = '';

  const nLeads = pcts.p50.length;
  const cols   = 4;
  const rows   = Math.ceil(nLeads / cols);
  const N      = tAxis.length;
  const idx    = d3.range(N);
  const titleH = title ? 18 : 0;
  const svgW   = Math.max(container.clientWidth || 400, 360);
  const cellW  = svgW / cols;
  const totalH = rows * cellH + titleH + 6;

  const svg = d3.select(container).append('svg')
    .attr('width', svgW).attr('height', totalH).style('max-width','100%');

  if (title) {
    svg.append('text').attr('x', svgW/2).attr('y', 13)
      .attr('text-anchor','middle')
      .style('font-size','11px').style('fill','#cdd6f4').style('font-weight','700')
      .text(title);
  }

  // Legend (first paint only)
  const legY = totalH - 6;
  [
    {col:'rgba(137,180,250,0.18)', label:'5–95 pct'},
    {col:'rgba(137,180,250,0.35)', label:'IQR'},
    {col:'#cba6f7',                label:'Median'},
  ].forEach((e, i) => {
    svg.append('rect').attr('x', 10 + i * 80).attr('y', legY - 7)
      .attr('width', 12).attr('height', 6).attr('rx', 1).attr('fill', e.col);
    svg.append('text').attr('x', 26 + i * 80).attr('y', legY)
      .style('font-size','8px').style('fill','#6c7086').text(e.label);
  });

  for (let li = 0; li < nLeads; li++) {
    const row  = Math.floor(li / cols);
    const col  = li % cols;
    const gx   = col * cellW + m.l;
    const gy   = row * cellH + m.t + titleH;
    const w    = cellW - m.l - m.r;
    const h    = cellH - m.t - m.b;
    const g    = svg.append('g').attr('transform', `translate(${gx},${gy})`);

    g.append('rect').attr('width', w).attr('height', h)
      .attr('fill','#181825').attr('stroke','#313244').attr('stroke-width',0.5);

    const xSc  = d3.scaleLinear().domain([tAxis[0], tAxis[tAxis.length-1]]).range([0, w]);
    const allV = [...pcts.p5[li], ...pcts.p95[li]];
    const ext  = d3.extent(allV);
    const yPad = ((ext[1] - ext[0]) || 0.2) * 0.12;
    const ySc  = d3.scaleLinear().domain([ext[0] - yPad, ext[1] + yPad]).range([h, 0]);

    const areaFn = (lo, hi) =>
      d3.area().x(i => xSc(tAxis[i])).y0(i => ySc(lo[i])).y1(i => ySc(hi[i]))(idx);
    const lineFn = arr =>
      d3.line().x(i => xSc(tAxis[i])).y(i => ySc(arr[i]))(idx);

    // 5–95 band
    g.append('path').attr('d', areaFn(pcts.p5[li], pcts.p95[li]))
      .attr('fill','rgba(137,180,250,0.13)').attr('stroke','none');
    // IQR band
    g.append('path').attr('d', areaFn(pcts.p25[li], pcts.p75[li]))
      .attr('fill','rgba(137,180,250,0.30)').attr('stroke','none');
    // Median
    g.append('path').attr('d', lineFn(pcts.p50[li]))
      .attr('fill','none').attr('stroke','#cba6f7').attr('stroke-width', 1.8);

    // Lead label
    g.append('text').attr('x', w/2).attr('y', -5)
      .attr('text-anchor','middle')
      .style('font-size','8.5px').style('fill','#a6adc8')
      .text(leadNames[li] ?? `L${li}`);

    if (col === 0) {
      g.append('g').call(
        d3.axisLeft(ySc).ticks(2).tickSize(2).tickFormat(v => v.toFixed(1))
      ).style('color','#6c7086').selectAll('text').style('font-size','6px');
    }
    if (row === rows - 1) {
      g.append('g').attr('transform', `translate(0,${h})`).call(
        d3.axisBottom(xSc).ticks(3).tickSize(2).tickFormat(v => v.toFixed(1) + 's')
      ).style('color','#6c7086').selectAll('text').style('font-size','6px');
    }
  }
}

// ── Utilities ─────────────────────────────────────────────────────────────────
async function apiFetch(url, opts) {
  const resp = await fetch(url, opts);
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

function dim(msg) {
  return `<p style="color:var(--muted);padding:10px 0">${msg}</p>`;
}
