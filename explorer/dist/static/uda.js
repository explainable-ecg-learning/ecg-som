'use strict';

(function () {
  const UDA_PAIRS = [
    ['ptbxl_full', 'georgia', 'PTB-XL Full → Georgia'],
    ['ptbxl_full', 'chapman', 'PTB-XL Full → Chapman'],
    ['ptbxl_2k',   'georgia', 'PTB-XL 2k → Georgia'],
    ['ptbxl_2k',   'chapman', 'PTB-XL 2k → Chapman'],
    ['georgia',    'ptbxl',   'Georgia → PTB-XL'],
    ['chapman',    'ptbxl',   'Chapman → PTB-XL'],
    ['georgia',    'chapman', 'Georgia → Chapman'],
    ['chapman',    'georgia', 'Chapman → Georgia'],
  ];

  const REGIME_ORDER = [
    'zero_shot',
    'uda_vanilla',
    'uda_bw_ot',
    'uda_sf_strong',
    'uda_ot_strong',
    'uda_replay_strong',
    'fewshot_100_head',
  ];

  const XDS_SOURCES = [
    ['ptbxl_full', 'PTB-XL Full'],
    ['ptbxl_2k',   'PTB-XL 2k'],
    ['georgia',    'Georgia'],
    ['chapman',    'Chapman'],
  ];
  const XDS_TARGETS = [
    ['ptbxl',   'PTB-XL'],
    ['georgia', 'Georgia'],
    ['chapman', 'Chapman'],
  ];

  const LOWER_BETTER = new Set(['MAE_Age', 'TE', 'QE']);

  const uState = {
    initialized: false,
    loading: false,
    payload: null,
    selectedPair: ['ptbxl_full', 'georgia'],
    metric: 'delta_Rec_AMI',
    xdsMetric: 'Rec_AMI',
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

  async function initUdaExplorer() {
    if (uState.initialized || uState.loading) return;
    uState.loading = true;
    try {
      const { apiFetch, getModelsCatalog } = shared();
      const [udaRes, catalog] = await Promise.all([
        apiFetch('/api/uda/results'),
        getModelsCatalog() ? Promise.resolve(getModelsCatalog()) : apiFetch('/api/models'),
      ]);
      uState.payload = udaRes;
      uState.initialized = true;

      renderUdaKpis(udaRes.uda_ablation.overall_means || []);
      renderBenchmarkHeatmap();
      renderPairDeepDive();
      renderCrossDatasetMatrix();
      setupPlayground(catalog, udaRes.pkl_datasets || [], Boolean(udaRes.static_mode));

      document.getElementById('uda-metric-select').addEventListener('change', e => {
        uState.metric = e.target.value;
        renderBenchmarkHeatmap();
        renderPairDeepDive();
      });
      document.getElementById('xds-metric-select').addEventListener('change', e => {
        uState.xdsMetric = e.target.value;
        renderCrossDatasetMatrix();
      });
    } catch (err) {
      const grid = document.getElementById('uda-kpi-grid');
      if (grid) grid.innerHTML = `<div class="kpi-card"><div class="kpi-label">Error</div><div class="kpi-foot">${err.message}</div></div>`;
    } finally {
      uState.loading = false;
    }
  }

  function renderUdaKpis(means) {
    const byReg = Object.fromEntries(means.map(m => [m.regime, m]));
    const zs = byReg.zero_shot || {};
    const van = byReg.uda_vanilla || {};
    const bw = byReg.uda_bw_ot || {};
    const sf = byReg.uda_sf_strong || {};
    const ot = byReg.uda_ot_strong || {};
    const fmt = (v, d = 4) => (v == null ? '—' : Number(v).toFixed(d));

    const cards = [
      {
        label: 'Zero-Shot Baseline',
        val: fmt(zs.Rec_AMI, 4),
        accent: false,
        foot: `8-pair Rec AMI · Sex AUC ${fmt(zs.AUC_Sex, 4)} · TE ${fmt(zs.TE, 3)}`,
      },
      {
        label: 'Vanilla UDA (8 Pairs)',
        val: fmt(van.Rec_AMI, 4),
        accent: false,
        foot: `Beat NMI ${fmt(van.NMI, 4)} · TE ${fmt(van.TE, 3)} · 0 labels`,
      },
      {
        label: 'BW-OT Sex AUROC (0.1 s)',
        val: fmt(bw.AUC_Sex, 4),
        accent: true,
        foot: `+${fmt((bw.AUC_Sex || 0) - (zs.AUC_Sex || 0), 4)} across 8/8 pairs · 0 backprop`,
      },
      {
        label: 'SF Hybrid (PTB-XL Full → *)',
        val: fmt(sf.strong_Rec_AMI, 4),
        accent: true,
        foot: `vs Zero-Shot ${fmt(zs.strong_Rec_AMI, 4)} · QE ${fmt(sf.QE, 2)}`,
      },
      {
        label: 'OT + Hybrid DEC+SWA',
        val: `TE ${fmt(ot.TE, 3)}`,
        accent: false,
        foot: `Sex AUC ${fmt(ot.AUC_Sex, 4)} · PTB-XL→Geo ${fmt(lookupSummary('ptbxl_full', 'georgia', 'uda_ot_strong')?.Rec_AMI_mean, 4)}`,
      },
    ];

    document.getElementById('uda-kpi-grid').innerHTML = cards.map(c => (
      `<div class="kpi-card">
        <div class="kpi-label">${c.label}</div>
        <div class="kpi-val ${c.accent ? 'accent' : ''}">${c.val}</div>
        <div class="kpi-foot">${c.foot}</div>
      </div>`
    )).join('');
  }

  function lookupSummary(src, tgt, reg) {
    const sum = uState.payload?.uda_ablation?.summary || [];
    return sum.find(r => r.source === src && r.target === tgt && r.regime === reg) || null;
  }

  function cellMetricValue(src, tgt, reg, metric) {
    const row = lookupSummary(src, tgt, reg);
    if (!row) return { val: null, std: 0 };
    if (metric === 'delta_Rec_AMI') {
      const zs = lookupSummary(src, tgt, 'zero_shot');
      if (!zs) return { val: null, std: 0 };
      return {
        val: row.Rec_AMI_mean - zs.Rec_AMI_mean,
        std: row.Rec_AMI_std || 0,
      };
    }
    return {
      val: row[`${metric}_mean`] ?? null,
      std: row[`${metric}_std`] ?? 0,
    };
  }

  function formatMetric(metric, val) {
    if (val == null || !Number.isFinite(val)) return '—';
    if (metric === 'delta_Rec_AMI') return (val >= 0 ? '+' : '') + val.toFixed(3);
    if (metric === 'MAE_Age' || metric === 'QE') return val.toFixed(2);
    if (metric === 'Active_Nodes') return val.toFixed(1);
    if (metric === 'TE') return val.toFixed(3);
    return val.toFixed(4);
  }

  function renderBenchmarkHeatmap() {
    const wrap = document.getElementById('uda-heatmap-wrap');
    wrap.innerHTML = '';
    const metric = uState.metric;
    const regMeta = uState.payload?.uda_ablation?.regime_meta || {};

    const vals = [];
    UDA_PAIRS.forEach(([s, t]) => {
      REGIME_ORDER.forEach(reg => {
        const { val } = cellMetricValue(s, t, reg, metric);
        if (val != null) vals.push(val);
      });
    });

    const M = { t: 46, r: 22, b: 14, l: 170 };
    const availW = Math.max((wrap.clientWidth || 920) - M.l - M.r, 560);
    const colW = Math.floor(availW / REGIME_ORDER.length);
    const rowH = 42;
    const svgW = M.l + REGIME_ORDER.length * colW + M.r;
    const svgH = M.t + UDA_PAIRS.length * rowH + M.b;

    let colorScale;
    if (metric === 'delta_Rec_AMI') {
      const lim = Math.max(0.015, d3.max(vals, v => Math.abs(v)) || 0.015);
      colorScale = d3.scaleDiverging(d3.interpolateRdBu).domain([-lim, 0, lim]);
    } else {
      const ext = d3.extent(vals);
      const lo = ext[0] ?? 0;
      const hi = ext[1] ?? 1;
      colorScale = LOWER_BETTER.has(metric)
        ? d3.scaleSequential(d3.interpolateYlOrRd).domain([lo, hi])
        : d3.scaleSequential(d3.interpolateYlGnBu).domain([lo, hi]);
    }

    const svg = d3.select(wrap).append('svg')
      .attr('viewBox', `0 0 ${svgW} ${svgH}`)
      .style('width', '100%').style('height', 'auto');
    const g = svg.append('g').attr('transform', `translate(${M.l},${M.t})`);

    // Column headers
    REGIME_ORDER.forEach((reg, j) => {
      const label = regMeta[reg]?.label || reg;
      g.append('text')
        .attr('x', j * colW + colW / 2)
        .attr('y', -14)
        .attr('text-anchor', 'middle')
        .style('font-size', '11.5px')
        .style('font-weight', '700')
        .style('fill', '#222222')
        .text(label);
      const tLbl = regMeta[reg]?.target_labels ? `${regMeta[reg].target_labels} labels` : '0 labels';
      g.append('text')
        .attr('x', j * colW + colW / 2)
        .attr('y', -2)
        .attr('text-anchor', 'middle')
        .style('font-size', '9.5px')
        .style('font-family', 'var(--font-mono)')
        .style('fill', '#767676')
        .text(tLbl);
    });

    const [selS, selT] = uState.selectedPair;
    UDA_PAIRS.forEach(([s, t, lbl], i) => {
      const isSelected = (s === selS && t === selT);
      const rowG = g.append('g')
        .attr('transform', `translate(0,${i * rowH})`)
        .style('cursor', 'pointer')
        .on('click', () => {
          uState.selectedPair = [s, t];
          renderBenchmarkHeatmap();
          renderPairDeepDive();
        });

      rowG.append('text')
        .attr('x', -10)
        .attr('y', rowH / 2 + 4)
        .attr('text-anchor', 'end')
        .style('font-size', '12px')
        .style('font-weight', isSelected ? '700' : '500')
        .style('fill', isSelected ? '#b83b2a' : '#222222')
        .text(lbl);

      // Best cell index in row
      let bestVal = null;
      REGIME_ORDER.forEach(reg => {
        if (metric === 'delta_Rec_AMI' && reg === 'zero_shot') return;
        const { val } = cellMetricValue(s, t, reg, metric);
        if (val == null) return;
        if (bestVal == null || (LOWER_BETTER.has(metric) ? val < bestVal : val > bestVal)) {
          bestVal = val;
        }
      });

      REGIME_ORDER.forEach((reg, j) => {
        const { val, std } = cellMetricValue(s, t, reg, metric);
        const fill = val == null ? '#fafafa' : colorScale(val);
        const cObj = d3.color(fill);
        const lum = cObj ? (0.299 * cObj.rgb().r + 0.587 * cObj.rgb().g + 0.114 * cObj.rgb().b) : 255;
        const txtCol = lum < 145 ? '#ffffff' : '#222222';
        const isBest = val != null && bestVal != null && Math.abs(val - bestVal) < 1e-7;

        const cellG = rowG.append('g')
          .attr('transform', `translate(${j * colW},0)`)
          .on('mouseover', ev => {
            const fullRow = lookupSummary(s, t, reg);
            const html =
              `<b>${lbl} · ${regMeta[reg]?.label || reg}</b><br>` +
              `${metric}: <b>${formatMetric(metric, val)}</b>` +
              (std ? ` ± ${std.toFixed(3)}` : '') +
              (fullRow ? `<br><span style="color:#767676;font-family:var(--font-mono);font-size:11px">RecAMI ${fullRow.Rec_AMI_mean.toFixed(4)} · SexAUC ${fullRow.AUC_Sex_mean.toFixed(4)} · TE ${fullRow.TE_mean.toFixed(3)}</span>` : '');
            shared().showTooltipHtml(ev, html);
          })
          .on('mousemove', shared().moveTip)
          .on('mouseout', shared().hideTip);

        cellG.append('rect')
          .attr('x', 1.5).attr('y', 2)
          .attr('width', colW - 3).attr('height', rowH - 4).attr('rx', 3)
          .attr('fill', fill)
          .attr('stroke', isSelected ? '#b83b2a' : '#e2e2e2')
          .attr('stroke-width', isSelected ? 1.8 : 0.8);

        cellG.append('text')
          .attr('x', colW / 2)
          .attr('y', rowH / 2 + 4)
          .attr('text-anchor', 'middle')
          .style('font-family', 'var(--font-mono)')
          .style('font-size', '11.5px')
          .style('font-weight', isBest ? '700' : '400')
          .style('fill', txtCol)
          .text(formatMetric(metric, val));
      });
    });
  }

  function renderPairDeepDive() {
    const [src, tgt] = uState.selectedPair;
    const pairMeta = UDA_PAIRS.find(([s, t]) => s === src && t === tgt);
    const pairLabel = pairMeta ? pairMeta[2] : `${src} → ${tgt}`;
    const metric = uState.metric === 'delta_Rec_AMI' ? 'Rec_AMI' : uState.metric;

    document.getElementById('uda-pair-title').textContent = `Pair Deep-Dive: ${pairLabel}`;
    const chance = uState.payload?.uda_ablation?.chance?.[tgt];
    const chanceStr = chance ? ` (random Rec AMI chance = ${chance.random_record_AMI.toFixed(4)})` : '';
    document.getElementById('uda-pair-sub').textContent =
      `Showing ${metric} across all 7 regimes (bars = 3-seed mean ± std, dots = seeds)${chanceStr}`;

    const wrap = document.getElementById('uda-pair-bar-wrap');
    wrap.innerHTML = '';

    const runs = (uState.payload?.uda_ablation?.runs || []).filter(r => r.source === src && r.target === tgt);
    const regMeta = uState.payload?.uda_ablation?.regime_meta || {};
    const stats = REGIME_ORDER.map(reg => {
      const srow = lookupSummary(src, tgt, reg);
      const mean = srow ? (srow[`${metric}_mean`] ?? 0) : 0;
      const std = srow ? (srow[`${metric}_std`] ?? 0) : 0;
      const pts = runs.filter(r => r.regime === reg).map(r => r[metric]).filter(Number.isFinite);
      return { reg, label: regMeta[reg]?.label || reg, mean, std, pts, row: srow };
    });

    const M = { t: 20, r: 16, b: 56, l: 52 };
    const svgW = Math.max(wrap.clientWidth || 540, 460);
    const svgH = 265;
    const w = svgW - M.l - M.r;
    const h = svgH - M.t - M.b;

    const allV = stats.flatMap(s => [s.mean + s.std, s.mean - s.std, ...s.pts]);
    let lo = d3.min(allV) ?? 0;
    let hi = d3.max(allV) ?? 1;
    const pad = (hi - lo || 0.05) * 0.16;
    lo = Math.max(0, lo - pad);
    hi = hi + pad;

    const x = d3.scaleBand().domain(stats.map(s => s.label)).range([0, w]).padding(0.28);
    const y = d3.scaleLinear().domain([lo, hi]).range([h, 0]);

    const svg = d3.select(wrap).append('svg')
      .attr('viewBox', `0 0 ${svgW} ${svgH}`)
      .style('width', '100%').style('height', 'auto');
    const g = svg.append('g').attr('transform', `translate(${M.l},${M.t})`);

    g.append('g').call(d3.axisLeft(y).ticks(5).tickSize(-w))
      .call(ax => ax.selectAll('.tick line').attr('stroke', '#eeeeee'))
      .call(ax => ax.select('.domain').attr('stroke', '#d0d0d0'));

    g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x))
      .selectAll('text')
      .attr('transform', 'rotate(-16)')
      .style('text-anchor', 'end')
      .style('font-size', '10px');

    stats.forEach(s => {
      const cx = x(s.label) + x.bandwidth() / 2;
      const fillCol = s.reg === 'zero_shot' ? '#767676'
        : s.reg === 'uda_bw_ot' || s.reg === 'uda_ot_strong' ? '#b83b2a'
        : s.reg === 'fewshot_100_head' ? '#0d7a82' : '#2b6cb0';

      g.append('rect')
        .attr('x', x(s.label))
        .attr('y', y(s.mean))
        .attr('width', x.bandwidth())
        .attr('height', Math.max(1, h - y(s.mean)))
        .attr('rx', 2)
        .attr('fill', fillCol)
        .attr('opacity', 0.82);

      if (s.std > 0) {
        g.append('line')
          .attr('x1', cx).attr('x2', cx)
          .attr('y1', y(Math.min(hi, s.mean + s.std)))
          .attr('y2', y(Math.max(lo, s.mean - s.std)))
          .attr('stroke', '#222222').attr('stroke-width', 1.3);
      }

      s.pts.forEach((pv, idx) => {
        const jx = (idx - (s.pts.length - 1) / 2) * 5;
        g.append('circle')
          .attr('cx', cx + jx)
          .attr('cy', y(pv))
          .attr('r', 3.0)
          .attr('fill', '#ffffff')
          .attr('stroke', '#222222')
          .attr('stroke-width', 1.2);
      });
    });

    const zsRow = lookupSummary(src, tgt, 'zero_shot');
    const otRow = lookupSummary(src, tgt, 'uda_ot_strong');
    const bwRow = lookupSummary(src, tgt, 'uda_bw_ot');
    const fsRow = lookupSummary(src, tgt, 'fewshot_100_head');
    const chips = [
      zsRow ? `Zero-Shot: RecAMI ${zsRow.Rec_AMI_mean.toFixed(4)} · AgeMAE ${zsRow.MAE_Age_mean.toFixed(2)} · SexAUC ${zsRow.AUC_Sex_mean.toFixed(4)}` : '',
      bwRow ? `BW-OT (0.1s): SexAUC ${bwRow.AUC_Sex_mean.toFixed(4)} · AgeMAE ${bwRow.MAE_Age_mean.toFixed(2)}` : '',
      otRow ? `OT+Hybrid: RecAMI ${otRow.Rec_AMI_mean.toFixed(4)} · TE ${otRow.TE_mean.toFixed(3)} · Act ${otRow.Active_Nodes_mean.toFixed(1)}/64` : '',
      fsRow ? `100-Shot Head: AgeMAE ${fsRow.MAE_Age_mean.toFixed(2)} · SexAUC ${fsRow.AUC_Sex_mean.toFixed(4)}` : '',
    ].filter(Boolean);

    document.getElementById('uda-pair-demo-wrap').innerHTML = chips
      .map(c => `<span class="demo-chip">${c}</span>`)
      .join('');
  }

  function renderCrossDatasetMatrix() {
    const wrap = document.getElementById('xds-matrix-wrap');
    wrap.innerHTML = '';
    const metric = uState.xdsMetric;
    const xdsSummary = uState.payload?.cross_dataset?.summary || [];
    const lookup = (s, t) => xdsSummary.find(r => r.source === s && r.target === t && r.regime === 'zero_shot');

    const vals = [];
    XDS_SOURCES.forEach(([sk]) => {
      XDS_TARGETS.forEach(([tk]) => {
        const r = lookup(sk, tk);
        if (r && r[`${metric}_mean`] != null) vals.push(r[`${metric}_mean`]);
      });
    });
    const ext = d3.extent(vals);
    const scale = LOWER_BETTER.has(metric)
      ? d3.scaleSequential(d3.interpolateYlOrRd).domain([ext[0] ?? 0, ext[1] ?? 1])
      : d3.scaleSequential(d3.interpolateYlGnBu).domain([ext[0] ?? 0, ext[1] ?? 1]);

    const M = { t: 38, r: 16, b: 14, l: 102 };
    const availW = Math.max((wrap.clientWidth || 460) - M.l - M.r, 270);
    const colW = Math.floor(availW / XDS_TARGETS.length);
    const rowH = 58;
    const svgW = M.l + XDS_TARGETS.length * colW + M.r;
    const svgH = M.t + XDS_SOURCES.length * rowH + M.b;

    const svg = d3.select(wrap).append('svg')
      .attr('viewBox', `0 0 ${svgW} ${svgH}`)
      .style('width', '100%').style('height', 'auto');
    const g = svg.append('g').attr('transform', `translate(${M.l},${M.t})`);

    XDS_TARGETS.forEach(([, tl], j) => {
      g.append('text')
        .attr('x', j * colW + colW / 2).attr('y', -12)
        .attr('text-anchor', 'middle')
        .style('font-size', '11.5px').style('font-weight', '700').style('fill', '#222222')
        .text(`→ ${tl}`);
    });

    XDS_SOURCES.forEach(([sk, sl], i) => {
      g.append('text')
        .attr('x', -10).attr('y', i * rowH + rowH / 2 + 4)
        .attr('text-anchor', 'end')
        .style('font-size', '11.5px').style('font-weight', '600').style('fill', '#222222')
        .text(sl);

      XDS_TARGETS.forEach(([tk], j) => {
        const r = lookup(sk, tk);
        const val = r ? r[`${metric}_mean`] : null;
        const std = r ? (r[`${metric}_std`] ?? 0) : 0;
        const isDiag = (sk.startsWith('ptbxl') && tk === 'ptbxl') || sk === tk;
        const fill = val == null ? '#fafafa' : scale(val);
        const cObj = d3.color(fill);
        const lum = cObj ? (0.299 * cObj.rgb().r + 0.587 * cObj.rgb().g + 0.114 * cObj.rgb().b) : 255;
        const txtCol = lum < 145 ? '#ffffff' : '#222222';

        const cg = g.append('g').attr('transform', `translate(${j * colW},${i * rowH})`);
        cg.append('rect')
          .attr('x', 2).attr('y', 2)
          .attr('width', colW - 4).attr('height', rowH - 4).attr('rx', 3)
          .attr('fill', fill)
          .attr('stroke', isDiag ? '#222222' : '#e2e2e2')
          .attr('stroke-width', isDiag ? 2.1 : 0.8);

        cg.append('text')
          .attr('x', colW / 2).attr('y', rowH / 2 - 2)
          .attr('text-anchor', 'middle')
          .style('font-family', 'var(--font-mono)').style('font-size', '12px')
          .style('font-weight', isDiag ? '700' : '600').style('fill', txtCol)
          .text(formatMetric(metric, val));

        cg.append('text')
          .attr('x', colW / 2).attr('y', rowH / 2 + 13)
          .attr('text-anchor', 'middle')
          .style('font-family', 'var(--font-mono)').style('font-size', '9.5px')
          .style('fill', txtCol).style('opacity', 0.85)
          .text(val == null ? '' : `±${std.toFixed(metric === 'MAE_Age' ? 2 : 3)}`);
      });
    });
  }

  function setupPlayground(catalog, pkls, isStatic = false) {
    const ckptSel = document.getElementById('uda-ckpt');
    const srcSel = document.getElementById('uda-src-data');
    const tgtSel = document.getElementById('uda-tgt-data');
    const methodSel = document.getElementById('uda-method');
    const statusEl = document.getElementById('uda-run-status');

    if (isStatic && statusEl) {
      statusEl.textContent = pkls.length
        ? 'Static Vercel preview: select a precomputed demo cohort & method, or pass ?api= for custom runs.'
        : 'Lightweight static build: run locally or pass ?api= to execute live UDA.';
    }

    ckptSel.innerHTML = '';
    (catalog.models || []).forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.value;
      opt.textContent = m.label;
      ckptSel.appendChild(opt);
    });
    const preferCkpt = (catalog.models || []).find(m => m.value.includes('gpu_fast_seed42_best') || m.value.includes('baseline'));
    if (preferCkpt) ckptSel.value = preferCkpt.value;

    srcSel.innerHTML = '';
    tgtSel.innerHTML = '';
    pkls.forEach(d => {
      srcSel.appendChild(new Option(d.label, d.value));
      tgtSel.appendChild(new Option(d.label, d.value));
    });
    const ptbOpt = pkls.find(d => d.value.includes('ptbxl'));
    const geoOpt = pkls.find(d => d.value.includes('georgia_transfer')) || pkls.find(d => d.value.includes('georgia'));
    const chpOpt = pkls.find(d => d.value.includes('chapman_transfer')) || pkls.find(d => d.value.includes('chapman'));
    if (ptbOpt) srcSel.value = ptbOpt.value;
    if (geoOpt) tgtSel.value = geoOpt.value;
    else if (chpOpt) tgtSel.value = chpOpt.value;

    const sliderMap = [
      ['uda-s-morph', 'val-s-morph', 2],
      ['uda-s-age',   'val-s-age',   2],
      ['uda-s-sex',   'val-s-sex',   2],
      ['uda-epochs',  'val-epochs',  0],
      ['uda-alpha-rec', 'val-alpha-rec', 2],
      ['uda-rho',     'val-rho',     2],
    ];
    sliderMap.forEach(([id, spanId, dec]) => {
      const el = document.getElementById(id);
      const sp = document.getElementById(spanId);
      el.addEventListener('input', () => { sp.textContent = Number(el.value).toFixed(dec); });
    });

    function updateMethodControls() {
      const m = methodSel.value;
      const needsSrc = (m === 'uda_bw_ot' || m === 'uda_ot_strong');
      srcSel.disabled = !needsSrc;
      document.querySelectorAll('.param-item').forEach(el => {
        const grp = el.dataset.param;
        let active = false;
        if (m === 'uda_bw_ot') active = (grp === 'ot-morph' || grp === 'ot-demo');
        else if (m === 'uda_sf_strong') active = (grp === 'dec' || grp === 'hyb');
        else if (m === 'uda_ot_strong') active = (grp === 'ot-demo' || grp === 'dec' || grp === 'hyb');
        else if (m === 'uda_vanilla') active = (grp === 'dec');
        el.classList.toggle('disabled', !active);
      });
    }
    methodSel.addEventListener('change', updateMethodControls);
    updateMethodControls();

    document.getElementById('uda-run-btn').addEventListener('click', runLiveUda);
    document.getElementById('uda-apply-btn').addEventListener('click', applyLiveUda);
  }

  async function runLiveUda() {
    const runBtn = document.getElementById('uda-run-btn');
    const applyBtn = document.getElementById('uda-apply-btn');
    const statusEl = document.getElementById('uda-run-status');
    const method = document.getElementById('uda-method').value;

    const payload = {
      ckpt:         document.getElementById('uda-ckpt').value,
      tgt_data:     document.getElementById('uda-tgt-data').value,
      src_data:     (method === 'uda_bw_ot' || method === 'uda_ot_strong') ? document.getElementById('uda-src-data').value : null,
      method,
      adapt_epochs: parseInt(document.getElementById('uda-epochs').value, 10),
      s_morph:      parseFloat(document.getElementById('uda-s-morph').value),
      s_age:        parseFloat(document.getElementById('uda-s-age').value),
      s_sex:        parseFloat(document.getElementById('uda-s-sex').value),
      alpha_rec:    parseFloat(document.getElementById('uda-alpha-rec').value),
      rho:          parseFloat(document.getElementById('uda-rho').value),
    };
    if (!payload.ckpt || !payload.tgt_data) {
      statusEl.textContent = 'Select a checkpoint and target cohort first.';
      return;
    }

    runBtn.disabled = true;
    applyBtn.disabled = true;
    statusEl.innerHTML = '<span class="spinner"></span>Running adaptation & target evaluation…';
    try {
      const data = await shared().apiFetch('/api/uda/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (data.error) {
        statusEl.textContent = 'Error: ' + data.error;
        return;
      }
      statusEl.textContent = `Completed ${data.method} in ${data.elapsed_s}s` + (data.note ? ` · ${data.note}` : '');
      renderLiveUdaResults(data);
      applyBtn.disabled = !data.can_apply;
    } catch (err) {
      statusEl.textContent = 'Error: ' + err.message;
    } finally {
      runBtn.disabled = false;
    }
  }

  function renderLiveUdaResults(data) {
    document.getElementById('uda-live-results').style.display = 'block';
    document.getElementById('uda-after-label').textContent = data.method;

    const specs = [
      ['Rec_AMI',      'Record AMI',    4, false],
      ['NMI',          'Beat NMI',      4, false],
      ['Rec_Purity',   'Record Purity', 4, false],
      ['AUC_Sex',      'Sex AUROC',     4, false],
      ['MAE_Age',      'Age MAE (yr)',  2, true],
      ['TE',           'Topo Error',    3, true],
      ['QE',           'Quant Error',   2, true],
      ['Active_Nodes', 'Active Nodes',  0, false],
    ];
    const b = data.before.metrics;
    const a = data.after.metrics;

    document.getElementById('uda-delta-grid').innerHTML = specs.map(([k, lbl, dec, lowerBetter]) => {
      const bv = Number(b[k]);
      const av = Number(a[k]);
      const dv = av - bv;
      const fmtVal = v => (k === 'Active_Nodes' ? `${Math.round(v)}/${data.som_dim[0] * data.som_dim[1]}` : v.toFixed(dec));
      const fmtDelta = d => (d >= 0 ? '+' : '') + (k === 'Active_Nodes' ? Math.round(d) : d.toFixed(dec));
      const cls = Math.abs(dv) < 1e-6 ? 'delta-flat'
        : ((lowerBetter ? dv < 0 : dv > 0) ? 'delta-pos' : 'delta-neg');
      return (
        `<div class="delta-card">
          <div class="kpi-label">${lbl}</div>
          <div class="delta-main">${fmtVal(bv)} → ${fmtVal(av)}</div>
          <span class="delta-badge ${cls}">${fmtDelta(dv)}</span>
        </div>`
      );
    }).join('');

    drawMiniSom('uda-mini-before', data.before);
    drawMiniSom('uda-mini-after',  data.after);
  }

  function drawMiniSom(containerId, evalObj) {
    const wrap = document.getElementById(containerId);
    wrap.innerHTML = '';
    const [H, W] = evalObj.som_dim;
    const cells = evalObj.cells || [];
    const classes = evalObj.class_names || [];
    const colors = shared().CLASS_COLORS;

    const avail = Math.max((wrap.clientWidth || 380) - 24, 240);
    const cell = Math.max(30, Math.min(48, Math.floor(avail / W)));
    const svgW = W * cell + 16;
    const svgH = H * cell + 16;

    const svg = d3.select(wrap).append('svg')
      .attr('viewBox', `0 0 ${svgW} ${svgH}`)
      .style('width', '100%').style('max-width', `${svgW}px`).style('height', 'auto');
    const g = svg.append('g').attr('transform', 'translate(8,8)');

    cells.forEach(c => {
      const fill = (c.count === 0 || c.majority < 0)
        ? '#fafafa'
        : softTint(colors[c.majority % colors.length]);
      const cg = g.append('g')
        .attr('transform', `translate(${c.col * cell},${c.row * cell})`)
        .on('mouseover', ev => {
          const parts = classes.map((cn, idx) => c.class_counts[idx] > 0 ? `${cn}:${c.class_counts[idx]}` : null).filter(Boolean);
          shared().showTooltipHtml(ev, `<b>Cell (${c.row}, ${c.col})</b> · n=${c.count}<br>${c.majority_label || 'empty'}<br><span style="color:#767676;font-size:11px">${parts.slice(0, 5).join('  ')}</span>`);
        })
        .on('mousemove', shared().moveTip)
        .on('mouseout', shared().hideTip);

      cg.append('rect')
        .attr('width', cell - 2).attr('height', cell - 2).attr('rx', 2.5)
        .attr('fill', fill).attr('stroke', '#dcdcdc').attr('stroke-width', 0.8);

      if (c.count > 0) {
        cg.append('text')
          .attr('x', (cell - 2) / 2).attr('y', (cell - 2) * 0.44)
          .attr('text-anchor', 'middle')
          .style('font-size', '8.5px').style('font-weight', '600').style('fill', '#222222')
          .text((c.majority_label || '').slice(0, 6));
        cg.append('text')
          .attr('x', (cell - 2) / 2).attr('y', (cell - 2) * 0.78)
          .attr('text-anchor', 'middle')
          .style('font-family', 'var(--font-mono)').style('font-size', '8px').style('fill', '#767676')
          .text(c.count);
      }
    });
  }

  async function applyLiveUda() {
    const applyBtn = document.getElementById('uda-apply-btn');
    const statusEl = document.getElementById('uda-run-status');
    applyBtn.disabled = true;
    statusEl.innerHTML = '<span class="spinner"></span>Loading adapted SOM into SOM Explorer…';
    try {
      const data = await shared().apiFetch('/api/uda/apply', { method: 'POST' });
      shared().applyAdaptedToSom(data);
      statusEl.textContent = 'Applied adapted model to SOM Explorer.';
    } catch (err) {
      statusEl.textContent = 'Error: ' + err.message;
    } finally {
      applyBtn.disabled = false;
    }
  }

  window.initUdaExplorer = initUdaExplorer;
})();
