'use strict';

(function () {
  const COMPARE_COLORS = [
    '#b83b2a', // terracotta
    '#2b6cb0', // steel blue
    '#2f855a', // forest green
    '#7b4b94', // plum
    '#b7791f', // ochre
    '#0d7a82', // teal
  ];
  const MAX_COMPARE = 6;

  const cmpState = {
    initialized: false,
    selected: [],           // [[row, col], ...]
    selectedLeads: [0, 1, 6, 10], // I, II, V1, V5
    waveMode: 'proto',      // 'proto' | 'median' | 'diff'
    distMetric: 'latent_l2',
    payload: null,          // last /api/cells/compare response
    requestSeq: 0,
  };

  function shared() {
    return window.explorerShared;
  }

  function softTint(hex) {
    const c = d3.hsl(hex);
    c.s = Math.min(c.s, 0.45);
    c.l = 0.91;
    return c.formatHex();
  }

  function sameCoord(a, b) {
    return a[0] === b[0] && a[1] === b[1];
  }

  function findSlot(r, c) {
    return cmpState.selected.findIndex(p => p[0] === r && p[1] === c);
  }

  function updateTabBadge() {
    const badge = document.getElementById('compare-tab-badge');
    if (!badge) return;
    const n = cmpState.selected.length;
    badge.textContent = String(n);
    badge.style.display = n > 0 ? 'inline-block' : 'none';
  }

  function bindOnce() {
    if (cmpState.initialized) return;
    cmpState.initialized = true;

    document.getElementById('cmp-go-som-btn').addEventListener('click', () => {
      shared().switchTab('som');
    });
    document.getElementById('cmp-preset-distinct').addEventListener('click', () => applyPreset('distinct'));
    document.getElementById('cmp-preset-top').addEventListener('click', () => applyPreset('top'));
    document.getElementById('cmp-preset-nbrs').addEventListener('click', () => applyPreset('nbrs'));
    document.getElementById('cmp-clear-btn').addEventListener('click', () => {
      cmpState.selected = [];
      cmpState.payload = null;
      refreshSelectionViews();
    });

    document.getElementById('cmp-dist-metric').addEventListener('change', e => {
      cmpState.distMetric = e.target.value;
      renderPairwiseMatrix();
    });
    document.getElementById('cmp-wave-mode').addEventListener('change', e => {
      cmpState.waveMode = e.target.value;
      renderWaveComparison();
    });

    window.addEventListener('som-loaded', () => {
      revalidateOrSeedSelection();
      if (document.getElementById('view-compare').classList.contains('active')) {
        initCellCompare();
      }
    });
  }

  function pickPresetCoords(mode) {
    const som = shared().getSomState();
    if (!som.loaded || !som.cells) return [];
    const [H, W] = som.somDim;
    const populated = som.cells.filter(c => c.count > 0).sort((a, b) => b.count - a.count);
    if (!populated.length) return [];

    if (mode === 'distinct') {
      const seenLabels = new Set();
      const chosen = [];
      for (const c of populated) {
        if (c.majority_label && !seenLabels.has(c.majority_label)) {
          seenLabels.add(c.majority_label);
          chosen.push([c.row, c.col]);
          if (chosen.length >= 4) break;
        }
      }
      for (const c of populated) {
        if (chosen.length >= Math.min(4, populated.length)) break;
        if (!chosen.some(p => sameCoord(p, [c.row, c.col]))) {
          chosen.push([c.row, c.col]);
        }
      }
      return chosen;
    }

    if (mode === 'top') {
      return populated.slice(0, 4).map(c => [c.row, c.col]);
    }

    // Toroidal neighbors around anchor (current first selection or top populated cell)
    const anchor = cmpState.selected[0] || [populated[0].row, populated[0].col];
    const [ar, ac] = anchor;
    const offsets = [[0, 0], [0, 1], [1, 0], [0, -1], [-1, 0]];
    const chosen = [];
    for (const [dr, dc] of offsets) {
      const rr = (ar + dr + H) % H;
      const cc = (ac + dc + W) % W;
      const cell = som.cells[rr * W + cc];
      if (cell && cell.count > 0 && !chosen.some(p => sameCoord(p, [rr, cc]))) {
        chosen.push([rr, cc]);
      }
      if (chosen.length >= 4) break;
    }
    return chosen;
  }

  function applyPreset(mode) {
    cmpState.selected = pickPresetCoords(mode);
    refreshSelectionViews();
  }

  function revalidateOrSeedSelection() {
    const som = shared().getSomState();
    if (!som.loaded || !som.somDim) {
      cmpState.selected = [];
      cmpState.payload = null;
      updateTabBadge();
      return;
    }
    const [H, W] = som.somDim;
    cmpState.selected = cmpState.selected.filter(([r, c]) => r >= 0 && r < H && c >= 0 && c < W);
    if (cmpState.selected.length === 0) {
      cmpState.selected = pickPresetCoords('distinct');
    }
    cmpState.payload = null;
    updateTabBadge();
  }

  function addCellToCompare(row, col) {
    bindOnce();
    const som = shared().getSomState();
    if (!som.loaded || !som.somDim) return;
    const existing = findSlot(row, col);
    if (existing >= 0) {
      shared().switchTab('compare');
      return;
    }
    if (cmpState.selected.length >= MAX_COMPARE) {
      cmpState.selected.shift();
    }
    cmpState.selected.push([row, col]);
    updateTabBadge();

    const btn = document.getElementById('add-to-compare-btn');
    if (btn) {
      btn.textContent = `Added (${cmpState.selected.length}/${MAX_COMPARE})`;
      setTimeout(() => {
        if (document.getElementById('add-to-compare-btn')) {
          document.getElementById('add-to-compare-btn').textContent = '+ Add to Compare';
        }
      }, 1200);
    }

    if (document.getElementById('view-compare').classList.contains('active')) {
      refreshSelectionViews();
    }
  }

  function toggleCell(row, col) {
    const idx = findSlot(row, col);
    if (idx >= 0) {
      cmpState.selected.splice(idx, 1);
    } else {
      if (cmpState.selected.length >= MAX_COMPARE) {
        cmpState.selected.shift();
      }
      cmpState.selected.push([row, col]);
    }
    refreshSelectionViews();
  }

  async function initCellCompare() {
    bindOnce();
    const som = shared().getSomState();
    const emptyEl = document.getElementById('cmp-empty-state');
    const workEl = document.getElementById('cmp-workspace');

    if (!som.loaded || !som.cells) {
      emptyEl.style.display = 'flex';
      workEl.style.display = 'none';
      return;
    }

    emptyEl.style.display = 'none';
    workEl.style.display = 'block';
    revalidateOrSeedSelection();
    buildLeadBar(12);
    await refreshSelectionViews();
  }

  async function refreshSelectionViews() {
    updateTabBadge();
    renderChips();
    renderPickerGrid();

    if (cmpState.selected.length === 0) {
      cmpState.payload = null;
      renderEmptyComparison('Click cells in the SOM Matrix Picker or choose a preset above.');
      return;
    }

    const seq = ++cmpState.requestSeq;
    try {
      const data = await shared().apiFetch('/api/cells/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cells: cmpState.selected }),
      });
      if (seq !== cmpState.requestSeq) return;
      if (data.error) {
        renderEmptyComparison('Error: ' + data.error);
        return;
      }
      cmpState.payload = data;
      renderSummaryTable();
      renderPairwiseMatrix();
      renderWaveComparison();
      renderClassComposition();
      renderDemographics();
    } catch (err) {
      if (seq !== cmpState.requestSeq) return;
      renderEmptyComparison('Error: ' + err.message);
    }
  }

  function renderEmptyComparison(msg) {
    const html = `<p style="color:var(--muted);padding:10px 0">${msg}</p>`;
    ['cmp-summary-table', 'cmp-pairwise-wrap', 'cmp-wave-legend', 'cmp-wave-chart', 'cmp-class-chart', 'cmp-demo-chart']
      .forEach(id => {
        const el = document.getElementById(id);
        if (el) el.innerHTML = (id === 'cmp-wave-legend' ? '' : html);
      });
  }

  function renderChips() {
    const wrap = document.getElementById('cmp-selected-chips');
    const som = shared().getSomState();
    const [H, W] = som.somDim || [8, 8];
    if (!cmpState.selected.length) {
      wrap.innerHTML = '<span style="font-size:12px;color:var(--muted)">No cells selected (max 6)</span>';
      return;
    }
    wrap.innerHTML = '';
    cmpState.selected.forEach(([r, c], idx) => {
      const cell = som.cells ? som.cells[r * W + c] : null;
      const lbl = cell && cell.majority_label ? ` · ${cell.majority_label}` : '';
      const chip = document.createElement('span');
      chip.className = 'cmp-chip';
      chip.innerHTML =
        `<span class="cmp-chip-dot" style="background:${COMPARE_COLORS[idx % COMPARE_COLORS.length]}"></span>` +
        `<span>#${idx + 1} (${r},${c})${lbl}</span>`;
      const rm = document.createElement('button');
      rm.type = 'button';
      rm.className = 'cmp-chip-rm';
      rm.title = 'Remove cell';
      rm.textContent = '×';
      rm.addEventListener('click', () => toggleCell(r, c));
      chip.appendChild(rm);
      wrap.appendChild(chip);
    });
  }

  function renderPickerGrid() {
    const wrap = document.getElementById('cmp-grid-wrap');
    wrap.innerHTML = '';
    const som = shared().getSomState();
    if (!som.loaded || !som.cells) return;
    const [H, W] = som.somDim;
    const colors = shared().CLASS_COLORS;

    const avail = Math.max((wrap.clientWidth || 440) - 20, 260);
    const cell = Math.max(32, Math.min(54, Math.floor(avail / W)));
    const svgW = W * cell + 20;
    const svgH = H * cell + 20;

    const svg = d3.select(wrap).append('svg')
      .attr('viewBox', `0 0 ${svgW} ${svgH}`)
      .style('width', '100%').style('max-width', `${svgW}px`).style('height', 'auto');
    const g = svg.append('g').attr('transform', 'translate(10,10)');

    som.cells.forEach(d => {
      const slot = findSlot(d.row, d.col);
      const isSel = slot >= 0;
      const selCol = isSel ? COMPARE_COLORS[slot % COMPARE_COLORS.length] : '#dcdcdc';
      const fill = (d.count === 0 || d.majority < 0)
        ? '#fafafa'
        : softTint(colors[d.majority % colors.length]);

      const cg = g.append('g')
        .attr('transform', `translate(${d.col * cell},${d.row * cell})`)
        .style('cursor', 'pointer')
        .on('click', () => toggleCell(d.row, d.col))
        .on('mouseover', ev => {
          shared().showTooltipHtml(ev,
            `<b>Cell (${d.row}, ${d.col})</b> · n=${d.count}<br>` +
            `${d.majority_label ? `Majority: <b>${d.majority_label}</b>` : 'Empty cell'}<br>` +
            `<span style="color:#767676;font-size:11px">${isSel ? `Selected as #${slot + 1} (click to remove)` : 'Click to add'}</span>`
          );
        })
        .on('mousemove', shared().moveTip)
        .on('mouseout', shared().hideTip);

      cg.append('rect')
        .attr('x', 1).attr('y', 1)
        .attr('width', cell - 2).attr('height', cell - 2).attr('rx', 3)
        .attr('fill', fill)
        .attr('stroke', selCol)
        .attr('stroke-width', isSel ? 2.5 : 0.9);

      if (d.count > 0) {
        cg.append('text')
          .attr('x', (cell - 2) / 2).attr('y', (cell - 2) * 0.48)
          .attr('text-anchor', 'middle')
          .style('font-size', Math.min(9.5, cell / 5) + 'px')
          .style('font-weight', '600').style('fill', '#222222')
          .text((d.majority_label || '').slice(0, 6));
        cg.append('text')
          .attr('x', (cell - 2) / 2).attr('y', (cell - 2) * 0.80)
          .attr('text-anchor', 'middle')
          .style('font-family', 'var(--font-mono)')
          .style('font-size', Math.min(8, cell / 6) + 'px').style('fill', '#767676')
          .text(d.count);
      }

      if (isSel) {
        cg.append('circle')
          .attr('cx', 9).attr('cy', 9).attr('r', 7)
          .attr('fill', selCol);
        cg.append('text')
          .attr('x', 9).attr('y', 12)
          .attr('text-anchor', 'middle')
          .style('font-family', 'var(--font-mono)').style('font-size', '9px')
          .style('font-weight', '700').style('fill', '#ffffff')
          .text(String(slot + 1));
      }
    });
  }

  function renderSummaryTable() {
    const wrap = document.getElementById('cmp-summary-table');
    const cells = cmpState.payload?.cells || [];
    if (!cells.length) return;

    const rowsHtml = cells.map((c, idx) => {
      const col = COMPARE_COLORS[idx % COMPARE_COLORS.length];
      const ageStr = c.age_mean == null ? '—' : `${c.age_mean.toFixed(1)} ± ${c.age_std.toFixed(1)}`;
      const femStr = c.female_pct == null ? '—' : `${c.female_pct.toFixed(0)}%`;
      const purStr = c.count > 0 ? `${(c.purity * 100).toFixed(0)}%` : '—';
      return (
        `<tr>
          <td><span class="cmp-chip-dot" style="background:${col};display:inline-block;margin-right:6px"></span>#${idx + 1} (${c.row},${c.col})</td>
          <td class="txt">${c.majority_label || 'empty'}</td>
          <td>${c.count}</td>
          <td>${purStr}</td>
          <td>${ageStr}</td>
          <td>${femStr}</td>
        </tr>`
      );
    }).join('');

    wrap.innerHTML =
      `<table class="cmp-table">
        <thead>
          <tr><th>Cell</th><th>Majority</th><th>Beats</th><th>Purity</th><th>Age (yr)</th><th>Female</th></tr>
        </thead>
        <tbody>${rowsHtml}</tbody>
      </table>`;
  }

  function renderPairwiseMatrix() {
    const wrap = document.getElementById('cmp-pairwise-wrap');
    wrap.innerHTML = '';
    const data = cmpState.payload;
    if (!data || !data.cells || data.cells.length < 2) {
      wrap.innerHTML = '<p style="color:var(--muted);font-size:12px">Select 2+ cells to view the pairwise distance matrix.</p>';
      return;
    }

    const cells = data.cells;
    const mat = data.pairwise;
    const metric = cmpState.distMetric;
    const labels = cells.map((c, i) => `#${i + 1} (${c.row},${c.col})`);

    const offDiag = [];
    for (let i = 0; i < cells.length; i++) {
      for (let j = 0; j < cells.length; j++) {
        if (i !== j) offDiag.push(mat[i][j][metric]);
      }
    }
    const ext = d3.extent(offDiag);
    const scale = metric === 'latent_cos'
      ? d3.scaleSequential(d3.interpolateYlGnBu).domain([ext[0] ?? 0, ext[1] ?? 1])
      : d3.scaleSequential(d3.interpolateYlOrRd).domain([0, ext[1] || 1]);

    const M = { t: 28, r: 12, b: 10, l: 74 };
    const avail = Math.max((wrap.clientWidth || 440) - M.l - M.r, 200);
    const colW = Math.min(82, Math.floor(avail / cells.length));
    const rowH = 38;
    const svgW = M.l + cells.length * colW + M.r;
    const svgH = M.t + cells.length * rowH + M.b;

    const svg = d3.select(wrap).append('svg')
      .attr('viewBox', `0 0 ${svgW} ${svgH}`)
      .style('width', '100%').style('max-width', `${svgW}px`).style('height', 'auto');
    const g = svg.append('g').attr('transform', `translate(${M.l},${M.t})`);

    labels.forEach((lbl, idx) => {
      g.append('text')
        .attr('x', idx * colW + colW / 2).attr('y', -10)
        .attr('text-anchor', 'middle')
        .style('font-family', 'var(--font-mono)').style('font-size', '10.5px')
        .style('font-weight', '700').style('fill', COMPARE_COLORS[idx % COMPARE_COLORS.length])
        .text(lbl);
      g.append('text')
        .attr('x', -8).attr('y', idx * rowH + rowH / 2 + 4)
        .attr('text-anchor', 'end')
        .style('font-family', 'var(--font-mono)').style('font-size', '10.5px')
        .style('font-weight', '700').style('fill', COMPARE_COLORS[idx % COMPARE_COLORS.length])
        .text(lbl);
    });

    for (let i = 0; i < cells.length; i++) {
      for (let j = 0; j < cells.length; j++) {
        const pair = mat[i][j];
        const v = pair[metric];
        const fill = i === j ? '#fafafa' : scale(v);
        const cObj = d3.color(fill);
        const lum = cObj ? (0.299 * cObj.rgb().r + 0.587 * cObj.rgb().g + 0.114 * cObj.rgb().b) : 255;
        const txtCol = i === j ? '#999999' : (lum < 145 ? '#ffffff' : '#222222');
        const fmt = metric === 'toroidal_l1' ? String(v) : v.toFixed(2);

        const cg = g.append('g')
          .attr('transform', `translate(${j * colW},${i * rowH})`)
          .on('mouseover', ev => {
            shared().showTooltipHtml(ev,
              `<b>${labels[i]} ↔ ${labels[j]}</b><br>` +
              `Latent L2: <b>${pair.latent_l2.toFixed(3)}</b> · Cos: <b>${pair.latent_cos.toFixed(3)}</b><br>` +
              `Wave RMSE: <b>${pair.waveform_rmse.toFixed(3)}</b> · JS Div: <b>${pair.js_divergence.toFixed(3)}</b><br>` +
              `Toroidal: <b>${pair.toroidal_l1} hops</b> (L2 ${pair.toroidal_l2.toFixed(2)})`
            );
          })
          .on('mousemove', shared().moveTip)
          .on('mouseout', shared().hideTip);

        cg.append('rect')
          .attr('x', 1).attr('y', 1)
          .attr('width', colW - 2).attr('height', rowH - 2).attr('rx', 2.5)
          .attr('fill', fill).attr('stroke', '#e2e2e2').attr('stroke-width', 0.8);

        cg.append('text')
          .attr('x', colW / 2).attr('y', rowH / 2 + 4)
          .attr('text-anchor', 'middle')
          .style('font-family', 'var(--font-mono)').style('font-size', '11px')
          .style('fill', txtCol)
          .text(fmt);
      }
    }
  }

  function buildLeadBar(nLeads) {
    const bar = document.getElementById('cmp-lead-bar');
    bar.innerHTML = '';
    const leadNames = shared().LEAD_NAMES;

    const lbl = document.createElement('span');
    lbl.textContent = 'Leads:';
    lbl.style.cssText = 'font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin-right:6px';
    bar.appendChild(lbl);

    for (let li = 0; li < nLeads; li++) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.textContent = leadNames[li] || `L${li}`;
      btn.dataset.lead = li;
      btn.className = 'lead-btn' + (cmpState.selectedLeads.includes(li) ? ' active' : '');
      btn.addEventListener('click', () => {
        if (cmpState.selectedLeads.includes(li)) {
          if (cmpState.selectedLeads.length === 1) return;
          cmpState.selectedLeads = cmpState.selectedLeads.filter(x => x !== li);
        } else {
          cmpState.selectedLeads = [...cmpState.selectedLeads, li].sort((a, b) => a - b);
        }
        syncLeadButtons();
        renderWaveComparison();
      });
      bar.appendChild(btn);
    }

    const presets = [
      ['4-Lead (I,II,V1,V5)', [0, 1, 6, 10]],
      ['Limb', [0, 1, 2, 3, 4, 5]],
      ['Precordial', [6, 7, 8, 9, 10, 11]],
      ['All 12', d3.range(nLeads)],
    ];
    presets.forEach(([text, arr]) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.textContent = text;
      b.className = 'lead-btn lead-btn-action';
      b.addEventListener('click', () => {
        cmpState.selectedLeads = arr.slice();
        syncLeadButtons();
        renderWaveComparison();
      });
      bar.appendChild(b);
    });
  }

  function syncLeadButtons() {
    document.querySelectorAll('#cmp-lead-bar .lead-btn[data-lead]').forEach(b => {
      b.classList.toggle('active', cmpState.selectedLeads.includes(Number(b.dataset.lead)));
    });
  }

  function renderWaveComparison() {
    const legWrap = document.getElementById('cmp-wave-legend');
    const chartWrap = document.getElementById('cmp-wave-chart');
    legWrap.innerHTML = '';
    chartWrap.innerHTML = '';

    const data = cmpState.payload;
    if (!data || !data.cells || !data.cells.length) return;
    const cells = data.cells;
    const tAxis = data.t_axis || [];
    const sel = cmpState.selectedLeads.filter(li => li < cells[0].proto_leads.length);
    const leadNames = shared().LEAD_NAMES;
    const mode = cmpState.waveMode;

    if (mode === 'diff' && cells.length < 2) {
      chartWrap.innerHTML = '<p style="color:var(--muted);padding:10px 0">Select at least 2 cells to plot the difference waveform (Cell #1 − Cell #2).</p>';
      return;
    }

    if (mode === 'diff') {
      const c1 = cells[0], c2 = cells[1];
      document.getElementById('cmp-wave-sub').textContent =
        `Difference waveform Δ = Cell #1 (${c1.row},${c1.col}) − Cell #2 (${c2.row},${c2.col}) with shaded polarity.`;
      legWrap.innerHTML =
        `<span class="cmp-leg-item"><span class="cmp-leg-line" style="background:#b83b2a"></span>Δ (#1 − #2)</span>` +
        `<span class="cmp-leg-item"><span class="cmp-leg-line" style="background:${COMPARE_COLORS[0]};opacity:0.45"></span>#1 (${c1.row},${c1.col}) ${c1.majority_label}</span>` +
        `<span class="cmp-leg-item"><span class="cmp-leg-line" style="background:${COMPARE_COLORS[1]};opacity:0.45"></span>#2 (${c2.row},${c2.col}) ${c2.majority_label}</span>`;
    } else {
      document.getElementById('cmp-wave-sub').textContent =
        mode === 'median'
          ? 'Empirical median waveforms (solid) with 25–75th percentile IQR bands on a shared per-lead voltage axis.'
          : 'Decoded SOM prototype waveforms across selected leads on a shared per-lead voltage axis.';
      legWrap.innerHTML = cells.map((c, idx) => (
        `<span class="cmp-leg-item">` +
        `<span class="cmp-leg-line" style="background:${COMPARE_COLORS[idx % COMPARE_COLORS.length]}"></span>` +
        `#${idx + 1} (${c.row},${c.col}) ${c.majority_label || 'empty'} (n=${c.count})` +
        `</span>`
      )).join('');
    }

    const cols = sel.length <= 2 ? sel.length : (sel.length === 4 ? 2 : 4);
    const rows = Math.ceil(sel.length / cols);
    const svgW = Math.max(chartWrap.clientWidth || 920, 480);
    const cellW = svgW / cols;
    const cellH = 152;
    const m = { t: 22, r: 10, b: 22, l: 38 };
    const totalH = rows * cellH + 8;

    const svg = d3.select(chartWrap).append('svg')
      .attr('viewBox', `0 0 ${svgW} ${totalH}`)
      .style('width', '100%').style('height', 'auto');

    sel.forEach((li, plotIdx) => {
      const r = Math.floor(plotIdx / cols);
      const c = plotIdx % cols;
      const gx = c * cellW + m.l;
      const gy = r * cellH + m.t;
      const w = cellW - m.l - m.r;
      const h = cellH - m.t - m.b;
      const g = svg.append('g').attr('transform', `translate(${gx},${gy})`);

      g.append('rect').attr('width', w).attr('height', h)
        .attr('fill', '#ffffff').attr('stroke', '#e2e2e2').attr('stroke-width', 0.9);

      const xSc = d3.scaleLinear().domain([tAxis[0], tAxis[tAxis.length - 1]]).range([0, w]);

      const allVals = [];
      if (mode === 'diff') {
        const s1 = cells[0].proto_leads[li];
        const s2 = cells[1].proto_leads[li];
        const diff = s1.map((v, k) => v - s2[k]);
        allVals.push(...s1, ...s2, ...diff);
      } else {
        cells.forEach(cellObj => {
          if (mode === 'median' && cellObj.percentiles) {
            allVals.push(...cellObj.percentiles.p25[li], ...cellObj.percentiles.p75[li], ...cellObj.percentiles.p50[li]);
          } else {
            allVals.push(...cellObj.proto_leads[li]);
          }
        });
      }

      const ext = d3.extent(allVals);
      const yPad = ((ext[1] - ext[0]) || 0.2) * 0.12;
      const ySc = d3.scaleLinear().domain([ext[0] - yPad, ext[1] + yPad]).range([h, 0]);

      if (ext[0] < 0 && ext[1] > 0) {
        g.append('line').attr('x1', 0).attr('x2', w)
          .attr('y1', ySc(0)).attr('y2', ySc(0))
          .attr('stroke', '#d4d4d4').attr('stroke-width', 0.7).attr('stroke-dasharray', '3,2');
      }

      const lineFn = d3.line().x((d, k) => xSc(tAxis[k])).y(d => ySc(d));
      if (mode === 'diff') {
        const s1 = cells[0].proto_leads[li];
        const s2 = cells[1].proto_leads[li];
        const diff = s1.map((v, k) => v - s2[k]);
        const areaFn = d3.area().x((_, k) => xSc(tAxis[k])).y0(ySc(0)).y1((_, k) => ySc(diff[k]));
        g.append('path').datum(diff).attr('d', areaFn).attr('fill', 'rgba(184,59,42,0.16)');
        g.append('path').datum(s1).attr('d', lineFn)
          .attr('fill', 'none').attr('stroke', COMPARE_COLORS[0]).attr('stroke-width', 1.1).attr('stroke-dasharray', '4,3').attr('opacity', 0.55);
        g.append('path').datum(s2).attr('d', lineFn)
          .attr('fill', 'none').attr('stroke', COMPARE_COLORS[1]).attr('stroke-width', 1.1).attr('stroke-dasharray', '4,3').attr('opacity', 0.55);
        g.append('path').datum(diff).attr('d', lineFn)
          .attr('fill', 'none').attr('stroke', '#b83b2a').attr('stroke-width', 1.9);
      } else {
        const idxRange = d3.range(tAxis.length);
        cells.forEach((cellObj, idx) => {
          const stroke = COMPARE_COLORS[idx % COMPARE_COLORS.length];
          if (mode === 'median' && cellObj.percentiles) {
            const lo = cellObj.percentiles.p25[li];
            const hi = cellObj.percentiles.p75[li];
            const areaFn = d3.area().x(k => xSc(tAxis[k])).y0(k => ySc(lo[k])).y1(k => ySc(hi[k]));
            g.append('path').datum(idxRange).attr('d', areaFn)
              .attr('fill', stroke).attr('opacity', 0.14);
            g.append('path').datum(cellObj.percentiles.p50[li]).attr('d', lineFn)
              .attr('fill', 'none').attr('stroke', stroke).attr('stroke-width', 1.65);
          } else {
            g.append('path').datum(cellObj.proto_leads[li]).attr('d', lineFn)
              .attr('fill', 'none').attr('stroke', stroke).attr('stroke-width', 1.65).attr('opacity', 0.92);
          }
        });
      }

      g.append('text').attr('x', w / 2).attr('y', -6)
        .attr('text-anchor', 'middle')
        .style('font-size', '10px').style('font-weight', '700').style('fill', '#222222')
        .text(leadNames[li] || `L${li}`);

      g.append('g').call(d3.axisLeft(ySc).ticks(3).tickSize(2).tickFormat(v => v.toFixed(1)))
        .style('color', '#767676').selectAll('text').style('font-size', '7.5px');
      if (r === rows - 1) {
        g.append('g').attr('transform', `translate(0,${h})`)
          .call(d3.axisBottom(xSc).ticks(4).tickSize(2).tickFormat(v => v.toFixed(1) + 's'))
          .style('color', '#767676').selectAll('text').style('font-size', '7.5px');
      }
    });
  }

  function renderClassComposition() {
    const wrap = document.getElementById('cmp-class-chart');
    wrap.innerHTML = '';
    const data = cmpState.payload;
    if (!data || !data.cells || !data.cells.length) return;
    const cells = data.cells;
    const classNames = data.class_names || [];
    const colors = shared().CLASS_COLORS;

    const M = { t: 16, r: 72, b: 28, l: 108 };
    const svgW = Math.max(wrap.clientWidth || 500, 400);
    const rowH = 34;
    const h = cells.length * rowH;
    const svgH = h + M.t + M.b;
    const w = svgW - M.l - M.r;

    const svg = d3.select(wrap).append('svg')
      .attr('viewBox', `0 0 ${svgW} ${svgH}`)
      .style('width', '100%').style('height', 'auto');
    const g = svg.append('g').attr('transform', `translate(${M.l},${M.t})`);

    const x = d3.scaleLinear().domain([0, 100]).range([0, w]);
    g.append('g').attr('transform', `translate(0,${h})`)
      .call(d3.axisBottom(x).ticks(5).tickFormat(d => d + '%'))
      .call(ax => ax.select('.domain').attr('stroke', '#d0d0d0'));

    cells.forEach((c, idx) => {
      const y = idx * rowH;
      const total = d3.sum(c.class_counts) || 0;
      g.append('text')
        .attr('x', -10).attr('y', y + rowH / 2 + 4)
        .attr('text-anchor', 'end')
        .style('font-family', 'var(--font-mono)').style('font-size', '11px')
        .style('font-weight', '700').style('fill', COMPARE_COLORS[idx % COMPARE_COLORS.length])
        .text(`#${idx + 1} (${c.row},${c.col})`);

      let xOff = 0;
      if (total > 0) {
        c.class_counts.forEach((cnt, ci) => {
          if (!cnt) return;
          const pct = (cnt / total) * 100;
          g.append('rect')
            .attr('x', x(xOff))
            .attr('y', y + 5)
            .attr('width', Math.max(1, x(xOff + pct) - x(xOff)))
            .attr('height', rowH - 10)
            .attr('fill', colors[ci % colors.length])
            .attr('opacity', 0.90)
            .on('mouseover', ev => {
              const topStr = (c.top_classes || []).map(t => `${t.label}: ${t.pct}%`).join(' · ');
              shared().showTooltipHtml(ev,
                `<b>Cell (${c.row}, ${c.col}) — ${classNames[ci]}</b><br>` +
                `Share: <b>${pct.toFixed(1)}%</b> (${cnt}/${total})<br>` +
                `<span style="color:#767676;font-size:11px">Top: ${topStr}</span>`
              );
            })
            .on('mousemove', shared().moveTip)
            .on('mouseout', shared().hideTip);
          xOff += pct;
        });
      } else {
        g.append('rect').attr('x', 0).attr('y', y + 5).attr('width', w).attr('height', rowH - 10)
          .attr('fill', '#fafafa').attr('stroke', '#e2e2e2');
      }

      g.append('text')
        .attr('x', w + 8).attr('y', y + rowH / 2 + 4)
        .style('font-family', 'var(--font-mono)').style('font-size', '11px').style('fill', '#444444')
        .text(total > 0 ? `${(c.purity * 100).toFixed(0)}% ${c.majority_label}` : 'empty');
    });
  }

  function renderDemographics() {
    const wrap = document.getElementById('cmp-demo-chart');
    wrap.innerHTML = '';
    const data = cmpState.payload;
    if (!data || !data.cells || !data.cells.length) return;
    const cells = data.cells;
    const binLabels = cells[0].age_bin_labels || ['<30', '30s', '40s', '50s', '60s', '70s', '80+'];

    const series = cells.map((c, idx) => {
      const tot = d3.sum(c.age_bins) || 1;
      return {
        idx,
        key: `#${idx + 1} (${c.row},${c.col})`,
        color: COMPARE_COLORS[idx % COMPARE_COLORS.length],
        pcts: c.age_bins.map(n => (n / tot) * 100),
        ageMean: c.age_mean,
        femalePct: c.female_pct,
      };
    });

    const M = { t: 24, r: 16, b: 30, l: 44 };
    const svgW = Math.max(wrap.clientWidth || 500, 400);
    const svgH = 240;
    const w = svgW - M.l - M.r;
    const h = svgH - M.t - M.b;

    const maxPct = Math.max(25, d3.max(series.flatMap(s => s.pcts)) || 40);
    const x0 = d3.scaleBand().domain(binLabels).range([0, w]).padding(0.22);
    const x1 = d3.scaleBand().domain(series.map(s => s.key)).range([0, x0.bandwidth()]).padding(0.08);
    const y = d3.scaleLinear().domain([0, maxPct * 1.12]).range([h, 0]);

    const svg = d3.select(wrap).append('svg')
      .attr('viewBox', `0 0 ${svgW} ${svgH}`)
      .style('width', '100%').style('height', 'auto');
    const g = svg.append('g').attr('transform', `translate(${M.l},${M.t})`);

    g.append('g').call(d3.axisLeft(y).ticks(4).tickSize(-w).tickFormat(d => d + '%'))
      .call(ax => ax.selectAll('.tick line').attr('stroke', '#eeeeee'))
      .call(ax => ax.select('.domain').attr('stroke', '#d0d0d0'));
    g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x0));

    binLabels.forEach((bl, bi) => {
      const bg = g.append('g').attr('transform', `translate(${x0(bl)},0)`);
      series.forEach(s => {
        const val = s.pcts[bi];
        bg.append('rect')
          .attr('x', x1(s.key))
          .attr('y', y(val))
          .attr('width', x1.bandwidth())
          .attr('height', Math.max(0.5, h - y(val)))
          .attr('fill', s.color)
          .attr('opacity', 0.86)
          .on('mouseover', ev => {
            shared().showTooltipHtml(ev, `<b>${s.key} · Age ${bl}</b><br>Share: <b>${val.toFixed(1)}%</b>`);
          })
          .on('mousemove', shared().moveTip)
          .on('mouseout', shared().hideTip);
      });
    });

    // Demographic summary chips below chart
    const chipsDiv = document.createElement('div');
    chipsDiv.className = 'pair-demo-strip';
    chipsDiv.innerHTML = series.map(s => (
      `<span class="demo-chip">` +
      `<span class="cmp-chip-dot" style="background:${s.color};display:inline-block;margin-right:5px"></span>` +
      `${s.key}: Age ${s.ageMean == null ? '—' : s.ageMean.toFixed(1) + 'y'} · ${s.femalePct == null ? '—' : s.femalePct.toFixed(0) + '% F'}` +
      `</span>`
    )).join('');
    wrap.appendChild(chipsDiv);
  }

  window.initCellCompare = initCellCompare;
  window.addCellToCompare = addCellToCompare;
})();
