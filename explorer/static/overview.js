'use strict';

(function () {
  const oState = {
    initialized: false,
    loading: false,
    data: null,
    selectedNode: 'input',
    benchCohort: 'ptbxl',
    benchMetric: 'Rec_AMI',
  };

  function shared() {
    return window.explorerShared;
  }

  async function initSystemOverview() {
    if (oState.initialized || oState.loading) return;
    oState.loading = true;
    try {
      const data = await shared().apiFetch('/api/overview');
      oState.data = data;
      oState.initialized = true;

      renderOverviewKpis(data);
      setupArchitectureDiagram(data.architecture || []);
      renderCohortCards(data.datasets || []);
      renderBenchChart();
      renderFiguresGallery(data.figures || []);

      document.getElementById('ov-bench-cohort').addEventListener('change', e => {
        oState.benchCohort = e.target.value;
        renderBenchChart();
      });
      document.getElementById('ov-bench-metric').addEventListener('change', e => {
        oState.benchMetric = e.target.value;
        renderBenchChart();
      });
      setupLightbox();
    } catch (err) {
      const el = document.getElementById('ov-kpi-grid');
      if (el) el.innerHTML = `<div class="kpi-card"><div class="kpi-label">Error</div><div class="kpi-foot">${err.message}</div></div>`;
    } finally {
      oState.loading = false;
    }
  }

  function renderOverviewKpis(data) {
    const pruned = (data.benchmarks || []).find(b => b.model === 'dpsom_pruned_8x8') || {};
    const bw = (data.uda_highlights || []).find(u => u.regime === 'uda_bw_ot') || {};
    const zs = (data.uda_highlights || []).find(u => u.regime === 'zero_shot') || {};

    const fmt = (v, d = 4) => (v == null ? '—' : Number(v).toFixed(d));
    const cards = [
      {
        label: 'Latent Factorization',
        val: '32D + 8D + 8D',
        accent: false,
        foot: 'Morphology z_res ⟂ (u_age, u_sex) · 8×8 Toroidal SOM',
      },
      {
        label: 'Curated Cohorts',
        val: '3 Continents',
        accent: false,
        foot: 'PTB-XL (17,374) · Georgia (2,412) · Chapman (2,330)',
      },
      {
        label: 'PTB-XL In-Domain (k=64)',
        val: `AMI ${fmt(pruned.ptbxl_Rec_AMI, 4)}`,
        accent: false,
        foot: `Age MAE ${fmt(pruned.ptbxl_MAE_Age, 2)} yr · Sex AUC ${fmt(pruned.ptbxl_AUC_Sex, 4)}`,
      },
      {
        label: 'External Rec AMI (Geo / Chp)',
        val: `${fmt(pruned.georgia_Rec_AMI, 3)} / ${fmt(pruned.chapman_Rec_AMI, 3)}`,
        accent: true,
        foot: 'Matches/beats 1D-ResNet on Chapman (0.1989 vs 0.1258)',
      },
      {
        label: 'BW-OT Zero-Label UDA',
        val: `Sex ${fmt(bw.AUC_Sex, 4)}`,
        accent: true,
        foot: `+${fmt((bw.AUC_Sex || 0) - (zs.AUC_Sex || 0), 4)} over Zero-Shot across 8/8 pairs (~0.1 s)`,
      },
    ];

    document.getElementById('ov-kpi-grid').innerHTML = cards.map(c => (
      `<div class="kpi-card">
        <div class="kpi-label">${c.label}</div>
        <div class="kpi-val ${c.accent ? 'accent' : ''}">${c.val}</div>
        <div class="kpi-foot">${c.foot}</div>
      </div>`
    )).join('');
  }

  function setupArchitectureDiagram(archNodes) {
    const byId = Object.fromEntries(archNodes.map(n => [n.id, n]));
    function showNode(id) {
      oState.selectedNode = id;
      document.querySelectorAll('#ov-arch-svg .arch-node').forEach(el => {
        el.classList.toggle('active', el.dataset.node === id);
      });
      const node = byId[id] || archNodes[0];
      if (!node) return;
      document.getElementById('ov-arch-detail').innerHTML = (
        `<span class="arch-detail-badge">${node.badge}</span>
         <h3>${node.title}</h3>
         <div class="arch-detail-code">${node.dims}</div>
         <p style="font-size:13px;color:var(--text)">${node.summary}</p>
         <div class="arch-detail-code">${node.formula}</div>
         <div class="arch-detail-finding"><strong>Key takeaway:</strong> ${node.finding}</div>
         <div style="font-family:var(--font-mono);font-size:11px;color:var(--muted)">Source: ${node.file}</div>`
      );
    }

    document.querySelectorAll('#ov-arch-svg .arch-node').forEach(el => {
      el.addEventListener('click', () => showNode(el.dataset.node));
    });
    showNode(oState.selectedNode);
  }

  function renderCohortCards(datasets) {
    document.getElementById('ov-cohorts').innerHTML = datasets.map(d => (
      `<div class="cohort-card">
        <div class="cohort-top">
          <strong>${d.name}</strong>
          <span class="cohort-region">${d.region}</span>
        </div>
        <div class="cohort-meta"><strong>Split:</strong> ${d.records}</div>
        <div class="cohort-meta"><strong>Demographics:</strong> ${d.age_sex}</div>
        <div class="cohort-meta"><strong>Labels:</strong> ${d.classes}</div>
        <div class="cohort-meta" style="color:var(--muted)">${d.notes}</div>
      </div>`
    )).join('');
  }

  function renderBenchChart() {
    const wrap = document.getElementById('ov-bench-chart');
    wrap.innerHTML = '';
    const benchmarks = oState.data?.benchmarks || [];
    const cohort = oState.benchCohort;
    const metric = oState.benchMetric;
    const key = `${cohort}_${metric}`;
    const lowerBetter = metric === 'MAE_Age';

    const labelMap = {
      vae_kmeans:              'VAE + k-means (k=64)',
      resnet1d_supervised:     '1D-ResNet + k-means',
      dpsom_pruned_4x4:        'DPSOM Pruned 4×4',
      dpsom_pruned_8x8:        'DPSOM Pruned 8×8 ★',
      dpsom_pruned_16x16:      'DPSOM Pruned 16×16',
      dpsom_pruned_8x8_no_res: 'DPSOM 8×8 (No Resid.)',
      dpsom_legacy_full_8x8:   'DPSOM Legacy Full 8×8',
    };

    const rows = benchmarks
      .map(b => ({ model: b.model, label: labelMap[b.model] || b.model, val: b[key] }))
      .filter(r => r.val != null && Number.isFinite(r.val));

    if (!rows.length) {
      wrap.innerHTML = '<p style="color:var(--muted)">No benchmark rows available.</p>';
      return;
    }

    const M = { t: 18, r: 44, b: 28, l: 158 };
    const svgW = Math.max(wrap.clientWidth || 500, 400);
    const svgH = Math.max(250, rows.length * 36 + M.t + M.b);
    const w = svgW - M.l - M.r;
    const h = svgH - M.t - M.b;

    const maxV = d3.max(rows, r => r.val) || 1;
    const minV = d3.min(rows, r => r.val) || 0;
    const domLo = metric === 'AUC_Sex' ? Math.max(0.5, minV - 0.04) : 0;
    const x = d3.scaleLinear().domain([domLo, maxV * 1.12]).range([0, w]);
    const y = d3.scaleBand().domain(rows.map(r => r.label)).range([0, h]).padding(0.28);

    const svg = d3.select(wrap).append('svg')
      .attr('viewBox', `0 0 ${svgW} ${svgH}`)
      .style('width', '100%').style('height', 'auto');
    const g = svg.append('g').attr('transform', `translate(${M.l},${M.t})`);

    g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(5))
      .call(ax => ax.select('.domain').attr('stroke', '#d0d0d0'));
    g.append('g').call(d3.axisLeft(y)).selectAll('text')
      .style('font-size', '11px')
      .style('font-weight', d => d.includes('★') ? '700' : '500')
      .style('fill', d => d.includes('★') ? '#b83b2a' : '#222222');

    rows.forEach(r => {
      const isHighlight = r.model === 'dpsom_pruned_8x8';
      const col = isHighlight ? '#b83b2a'
        : r.model.includes('resnet') || r.model.includes('vae') ? '#64748b' : '#2b6cb0';
      g.append('rect')
        .attr('x', 0)
        .attr('y', y(r.label))
        .attr('width', Math.max(2, x(r.val)))
        .attr('height', y.bandwidth())
        .attr('rx', 2.5)
        .attr('fill', col)
        .attr('opacity', isHighlight ? 0.94 : 0.78);

      const fmtV = lowerBetter ? r.val.toFixed(2) : r.val.toFixed(4);
      g.append('text')
        .attr('x', x(r.val) + 6)
        .attr('y', y(r.label) + y.bandwidth() / 2 + 4)
        .style('font-family', 'var(--font-mono)')
        .style('font-size', '11px')
        .style('font-weight', isHighlight ? '700' : '500')
        .style('fill', '#222222')
        .text(fmtV);
    });
  }

  function renderFiguresGallery(figures) {
    const grid = document.getElementById('ov-fig-grid');
    const apiBase = (shared().getApiBase && shared().getApiBase()) || '';
    grid.innerHTML = figures.map(f => {
      const figUrl = (apiBase && f.url && f.url.startsWith('/')) ? `${apiBase}${f.url}` : f.url;
      return `<div class="fig-card" data-url="${figUrl}" data-title="${f.title}" data-cap="${f.caption}">
        <img src="${figUrl}" alt="${f.title}" loading="lazy">
        <div class="fig-title">${f.title}</div>
        <div class="fig-cap">${f.caption}</div>
      </div>`;
    }).join('');

    grid.querySelectorAll('.fig-card').forEach(card => {
      card.addEventListener('click', () => {
        document.getElementById('lightbox-title').textContent = card.dataset.title;
        document.getElementById('lightbox-img').src = card.dataset.url;
        document.getElementById('lightbox-caption').textContent = card.dataset.cap;
        document.getElementById('fig-lightbox').style.display = 'flex';
      });
    });
  }

  function setupLightbox() {
    const lb = document.getElementById('fig-lightbox');
    document.getElementById('lightbox-close').addEventListener('click', () => {
      lb.style.display = 'none';
    });
    lb.addEventListener('click', e => {
      if (e.target === lb) lb.style.display = 'none';
    });
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape' && lb.style.display !== 'none') lb.style.display = 'none';
    });
  }

  window.initSystemOverview = initSystemOverview;
})();
