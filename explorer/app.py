"""Replaced by server.py (FastAPI + D3).  Run: python explorer/server.py"""
import os, sys, subprocess
subprocess.run(
    [sys.executable, os.path.join(os.path.dirname(__file__), "server.py")] + sys.argv[1:]
)
raise SystemExit

from __future__ import annotations

import argparse
import glob
import os
import sys

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path so `src` can be imported
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
import pickle

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html, no_update
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.model.dpsom import DPSOM_ECG
from src.training.trainer import _load_checkpoint_state

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
CLASS_NAMES = ["CD", "HYP", "MI", "NORM", "STTC"]
CLASS_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
MAX_SAMPLES = 20        # max individual beats shown in sample list
SAMPLE_COLS = 4         # columns in sample grid
FS = 100.0              # sampling rate
COLORSCALE = "YlGnBu"

# ---------------------------------------------------------------------------
# In-memory state (populated by the Load callback)
# ---------------------------------------------------------------------------
_STATE: dict = {}


# ---------------------------------------------------------------------------
# Helper: discover checkpoints
# ---------------------------------------------------------------------------

def _discover_ckpts(base_dir: str) -> list[str]:
    pattern = os.path.join(base_dir, "models", "**", "*.ckpt")
    return sorted(glob.glob(pattern, recursive=True))


def _default_data_path(base_dir: str) -> str:
    p = os.path.join(base_dir, "data", "ptbxl_100_T-12ms.pkl")
    return p if os.path.exists(p) else ""


# ---------------------------------------------------------------------------
# Helper: load dataset
# ---------------------------------------------------------------------------

def _load_dataset(data_path: str):
    with open(data_path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Helper: run inference and build per-cell index
# ---------------------------------------------------------------------------

def _run_inference(ckpt_path: str, dataset, split: str, device: torch.device):
    """
    Returns a dict with:
        model         – loaded DPSOM_ECG
        beats         – [N, T, C] float32 np array (beat_representations)
        labels        – [N]       int    np array
        ages          – [N]       float  np array
        sexes         – [N]       int    np array
        record_ids    – [N]       list   (record id per beat)
        beat_local_idx– [N]       list   (beat index within its record)
        k_all         – [N]       int    np array (SOM cell index, flat)
        prototype_signals – [H, W, T, C] decoded prototypes
        som_dim       – (H, W)
        cell_beats    – dict  flat_cell_idx -> list of global beat indices
        cell_counts   – [H, W] int
        cell_majority – [H, W] int  (-1 = empty)
        class_names   – list[str]
    """
    state, meta = _load_checkpoint_state(ckpt_path, device)

    latent_dim     = meta.get("latent_dim", 32)
    som_dim        = tuple(meta.get("som_dim", (8, 8)))
    input_length   = meta.get("input_length", 120)
    input_channels = meta.get("input_channels", 12)
    H, W = som_dim

    model = DPSOM_ECG(
        latent_dim=latent_dim, som_dim=som_dim,
        input_length=input_length, input_channels=input_channels,
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Gather beats
    records = dataset.get_data(split)
    class_names = getattr(dataset, "class_names", CLASS_NAMES)
    label_enc = dataset.label_encoder

    beats_list, labels_list, ages_list, sexes_list = [], [], [], []
    rec_ids, beat_local = [], []

    for rec in records:
        nb = len(rec.beat_representations)
        for i in range(nb):
            beats_list.append(rec.beat_representations[i])  # [T, C]
            labels_list.append(rec.label)
            ages_list.append(rec.age)
            sexes_list.append(rec.sex)
            rec_ids.append(rec.id)
            beat_local.append(i)

    beats_np = np.stack(beats_list).astype(np.float32)  # [N, T, C]
    labels_enc = label_enc.transform(labels_list)

    # Encode in batches
    BATCH = 512
    k_list = []
    x_t = torch.from_numpy(beats_np.transpose(0, 2, 1))  # [N, C, T]
    with torch.no_grad():
        for s in range(0, len(x_t), BATCH):
            xb = x_t[s:s + BATCH].to(device)
            mu, _, _, _ = model._encode(xb)
            k_list.append(model.k(mu).cpu().numpy())
    k_all = np.concatenate(k_list)  # [N]

    # Decode SOM prototypes
    with torch.no_grad():
        E_flat = model._embeddings.view(-1, latent_dim)  # [H*W, latent_dim]
        logits = model.decoder(E_flat)                   # [H*W, C*T]
        proto = logits.view(H, W, input_channels, input_length).cpu().numpy()
    # proto: [H, W, C, T]  →  [H, W, T, C]
    prototype_signals = proto.transpose(0, 1, 3, 2)

    # Per-cell index
    K = H * W
    num_classes = len(class_names)
    cell_beats: dict[int, list[int]] = {k: [] for k in range(K)}
    counts = np.zeros((K, num_classes), dtype=np.int64)
    for i, ki in enumerate(k_all):
        cell_beats[int(ki)].append(i)
        lbl = int(labels_enc[i])
        if 0 <= lbl < num_classes:
            counts[int(ki), lbl] += 1

    totals = counts.sum(axis=1)  # [K]
    cell_counts = totals.reshape(H, W)

    cell_majority = np.full(K, -1, dtype=np.int64)
    nonempty = totals > 0
    cell_majority[nonempty] = counts[nonempty].argmax(axis=1)
    cell_majority_grid = cell_majority.reshape(H, W)

    return {
        "model": model,
        "beats": beats_np,
        "labels": labels_enc,
        "ages": np.array(ages_list, dtype=np.float32),
        "sexes": np.array(sexes_list, dtype=np.int64),
        "record_ids": rec_ids,
        "beat_local_idx": beat_local,
        "k_all": k_all,
        "prototype_signals": prototype_signals,
        "som_dim": som_dim,
        "cell_beats": cell_beats,
        "cell_counts": cell_counts,
        "cell_majority": cell_majority_grid,
        "class_names": class_names,
        "counts_per_class": counts.reshape(H, W, num_classes),
        "input_length": input_length,
        "input_channels": input_channels,
    }


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def _make_heatmap_fig(state: dict, color_mode: str = "count") -> go.Figure:
    H, W = state["som_dim"]
    class_names = state["class_names"]
    num_classes = len(class_names)

    if color_mode == "count":
        z = state["cell_counts"].astype(float)
        colorbar_title = "# samples"
        colorscale = COLORSCALE
        text = np.array([[str(int(z[r, c])) if z[r, c] > 0 else ""
                          for c in range(W)] for r in range(H)])
    else:
        # majority class (use a discrete categorical colour)
        maj = state["cell_majority"].astype(float)
        maj_display = maj.copy()
        maj_display[maj_display < 0] = np.nan
        z = maj_display
        colorscale = [
            [i / (num_classes - 1), CLASS_COLORS[i % len(CLASS_COLORS)]]
            for i in range(num_classes)
        ]
        colorbar_title = "Majority class"
        counts_pc = state["counts_per_class"]
        totals = state["cell_counts"]
        text = np.full((H, W), "", dtype=object)
        for r in range(H):
            for c in range(W):
                tot = int(totals[r, c])
                if tot == 0:
                    continue
                mi = int(maj[r, c])
                lbl = class_names[mi] if 0 <= mi < num_classes else "?"
                n_maj = int(counts_pc[r, c, mi]) if 0 <= mi < num_classes else 0
                text[r, c] = f"{lbl}<br>{n_maj}/{tot}"

    hover = np.array([[
        f"Cell ({r},{c})<br>{text[r, c]}" for c in range(W)
    ] for r in range(H)])

    fig = go.Figure(go.Heatmap(
        z=z,
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(title=colorbar_title, thickness=15),
        text=text,
        texttemplate="%{text}",
        hovertext=hover,
        hovertemplate="%{hovertext}<extra></extra>",
        xgap=2,
        ygap=2,
    ))
    fig.update_layout(
        xaxis=dict(title="Column", tickmode="linear", tick0=0, dtick=1,
                   tickvals=list(range(W)), ticktext=[str(i) for i in range(W)]),
        yaxis=dict(title="Row", tickmode="linear", tick0=0, dtick=1,
                   tickvals=list(range(H)), ticktext=[str(i) for i in range(H)],
                   autorange="reversed"),
        margin=dict(l=60, r=40, t=40, b=60),
        height=500,
        plot_bgcolor="#1e1e2e",
        paper_bgcolor="#1e1e2e",
        font=dict(color="#cdd6f4"),
        clickmode="event",
    )
    return fig


def _make_prototype_fig(proto: np.ndarray, cell_rc: tuple, n_samples: int) -> go.Figure:
    """proto: [T, C]"""
    T, C = proto.shape
    t = np.arange(T) / FS
    n_leads = min(C, 12)
    cols = 4
    rows = -(-n_leads // cols)  # ceil

    fig = make_subplots(rows=rows, cols=cols,
                        shared_xaxes=True, shared_yaxes=False,
                        horizontal_spacing=0.04, vertical_spacing=0.10)

    for li in range(n_leads):
        r_sub = li // cols + 1
        c_sub = li % cols + 1
        lname = LEAD_NAMES[li] if li < len(LEAD_NAMES) else f"L{li}"
        fig.add_trace(go.Scatter(
            x=t, y=proto[:, li],
            mode="lines",
            line=dict(color="#89b4fa", width=1.5),
            name=lname,
            showlegend=False,
            hovertemplate=f"{lname}: %{{y:.3f}} mV<extra></extra>",
        ), row=r_sub, col=c_sub)
        fig.update_yaxes(title_text=lname, title_font=dict(size=9),
                         row=r_sub, col=c_sub)

    r, c = cell_rc
    fig.update_layout(
        title=dict(
            text=f"SOM Prototype — Cell ({r},{c})   [{n_samples} samples]",
            font=dict(size=13, color="#cdd6f4"),
        ),
        height=max(200, rows * 130),
        margin=dict(l=55, r=20, t=45, b=30),
        plot_bgcolor="#181825",
        paper_bgcolor="#181825",
        font=dict(color="#cdd6f4"),
    )
    return fig


def _make_percentile_fig(beats_in_cell: np.ndarray, cell_rc: tuple) -> go.Figure:
    """
    beats_in_cell: [N, T, C]
    Shows 5/25/50/75/95 percentile bands per lead.
    """
    N, T, C = beats_in_cell.shape
    t = np.arange(T) / FS
    n_leads = min(C, 12)
    cols = 4
    rows = -(-n_leads // cols)

    pcts = np.percentile(beats_in_cell, [5, 25, 50, 75, 95], axis=0)  # [5, T, C]

    fig = make_subplots(rows=rows, cols=cols,
                        shared_xaxes=True, shared_yaxes=False,
                        horizontal_spacing=0.04, vertical_spacing=0.10)

    band_fill  = "rgba(137,180,250,0.15)"
    band_fill2 = "rgba(137,180,250,0.25)"
    median_col = "#cba6f7"

    for li in range(n_leads):
        r_sub = li // cols + 1
        c_sub = li % cols + 1
        lname = LEAD_NAMES[li] if li < len(LEAD_NAMES) else f"L{li}"

        # 5–95 outer band
        fig.add_trace(go.Scatter(
            x=np.concatenate([t, t[::-1]]),
            y=np.concatenate([pcts[0, :, li], pcts[4, ::-1, li]]),
            fill="toself", fillcolor=band_fill,
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=(li == 0), name="5–95 pct",
            hoverinfo="skip",
        ), row=r_sub, col=c_sub)

        # 25–75 IQR band
        fig.add_trace(go.Scatter(
            x=np.concatenate([t, t[::-1]]),
            y=np.concatenate([pcts[1, :, li], pcts[3, ::-1, li]]),
            fill="toself", fillcolor=band_fill2,
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=(li == 0), name="IQR (25–75)",
            hoverinfo="skip",
        ), row=r_sub, col=c_sub)

        # Median
        fig.add_trace(go.Scatter(
            x=t, y=pcts[2, :, li],
            mode="lines",
            line=dict(color=median_col, width=2.0),
            showlegend=(li == 0), name="Median",
            hovertemplate=f"{lname} median: %{{y:.3f}} mV<extra></extra>",
        ), row=r_sub, col=c_sub)

        fig.update_yaxes(title_text=lname, title_font=dict(size=9),
                         row=r_sub, col=c_sub)

    r, c = cell_rc
    fig.update_layout(
        title=dict(
            text=f"Percentile Distribution — Cell ({r},{c})   [N={N}]",
            font=dict(size=13, color="#cdd6f4"),
        ),
        height=max(200, rows * 130),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0,
                    font=dict(size=9, color="#cdd6f4")),
        margin=dict(l=55, r=20, t=55, b=30),
        plot_bgcolor="#181825",
        paper_bgcolor="#181825",
        font=dict(color="#cdd6f4"),
    )
    return fig


def _make_sample_fig(beat: np.ndarray, title: str) -> go.Figure:
    """beat: [T, C] — compact 12-lead mini figure."""
    T, C = beat.shape
    t = np.arange(T) / FS
    n_leads = min(C, 12)
    cols = 4
    rows = -(-n_leads // cols)

    fig = make_subplots(rows=rows, cols=cols,
                        shared_xaxes=True, shared_yaxes=False,
                        horizontal_spacing=0.04, vertical_spacing=0.10)

    for li in range(n_leads):
        r_sub = li // cols + 1
        c_sub = li % cols + 1
        lname = LEAD_NAMES[li] if li < len(LEAD_NAMES) else f"L{li}"
        fig.add_trace(go.Scatter(
            x=t, y=beat[:, li],
            mode="lines",
            line=dict(color="#a6e3a1", width=1.2),
            name=lname,
            showlegend=False,
            hovertemplate=f"{lname}: %{{y:.3f}} mV<extra></extra>",
        ), row=r_sub, col=c_sub)
        fig.update_yaxes(title_text=lname, title_font=dict(size=8),
                         row=r_sub, col=c_sub)

    fig.update_layout(
        title=dict(text=title, font=dict(size=10, color="#cdd6f4")),
        height=max(160, rows * 120),
        margin=dict(l=50, r=10, t=35, b=20),
        plot_bgcolor="#181825",
        paper_bgcolor="#181825",
        font=dict(color="#cdd6f4"),
    )
    return fig


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

_BASE = _REPO_ROOT

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="ECG-SOM Explorer",
    suppress_callback_exceptions=True,
)


def _ckpt_options(base: str) -> list[dict]:
    ckpts = _discover_ckpts(base)
    return [{"label": os.path.relpath(p, base), "value": p} for p in ckpts]


def _build_layout(base: str) -> html.Div:
    ckpt_opts = _ckpt_options(base)
    default_ckpt = ckpt_opts[-1]["value"] if ckpt_opts else None
    default_data = _default_data_path(base)

    return html.Div([
        # ── Header ──────────────────────────────────────────────────────────
        dbc.Navbar(
            dbc.Container([
                html.Span("ECG-SOM Explorer", className="navbar-brand fs-4 fw-bold"),
            ], fluid=True),
            color="dark", dark=True, className="mb-3",
        ),

        dbc.Container([
            # ── Control row ─────────────────────────────────────────────────
            dbc.Card(dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Checkpoint", className="fw-bold"),
                        dcc.Dropdown(
                            id="ckpt-dropdown",
                            options=ckpt_opts,
                            value=default_ckpt,
                            placeholder="Select checkpoint…",
                            style={"color": "#000"},
                        ),
                    ], md=5),
                    dbc.Col([
                        dbc.Label("Dataset (pickle path)", className="fw-bold"),
                        dbc.Input(id="data-path", value=default_data, type="text",
                                  placeholder="path/to/ptbxl.pkl"),
                    ], md=4),
                    dbc.Col([
                        dbc.Label("Split", className="fw-bold"),
                        dbc.RadioItems(
                            id="split-radio",
                            options=[
                                {"label": "test",  "value": "test"},
                                {"label": "val",   "value": "val"},
                                {"label": "train", "value": "train"},
                                {"label": "all",   "value": "all"},
                            ],
                            value="test",
                            inline=True,
                        ),
                    ], md=2),
                    dbc.Col([
                        dbc.Label("\u00a0", className="fw-bold d-block"),
                        dbc.Button("Load & Run", id="load-btn", color="primary", n_clicks=0),
                    ], md=1, className="d-flex align-items-end"),
                ], align="end", className="g-2"),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Heatmap colour", className="fw-bold mt-2"),
                        dbc.RadioItems(
                            id="color-mode",
                            options=[
                                {"label": "Sample count", "value": "count"},
                                {"label": "Majority class", "value": "majority"},
                            ],
                            value="count",
                            inline=True,
                        ),
                    ]),
                ]),
            ]), className="mb-3"),

            # ── Status ──────────────────────────────────────────────────────
            html.Div(id="status-text", className="text-muted mb-2 small"),

            # ── Heatmap ─────────────────────────────────────────────────────
            dbc.Card(dbc.CardBody([
                dcc.Graph(id="som-heatmap", config={"displayModeBar": False},
                          style={"height": "520px"}),
            ]), className="mb-3"),

            # ── Cell detail panel ───────────────────────────────────────────
            html.Div(id="cell-detail-panel", children=[
                # hidden until a cell is clicked
            ]),

        ], fluid=True),

        # Hidden stores
        dcc.Store(id="inference-done", data=False),
        dcc.Store(id="clicked-cell", data=None),
    ])


app.layout = _build_layout(_BASE)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("status-text", "children"),
    Output("som-heatmap", "figure"),
    Output("inference-done", "data"),
    Input("load-btn", "n_clicks"),
    State("ckpt-dropdown", "value"),
    State("data-path", "value"),
    State("split-radio", "value"),
    State("color-mode", "value"),
    prevent_initial_call=True,
)
def load_and_run(n_clicks, ckpt_path, data_path, split, color_mode):
    if not ckpt_path or not data_path:
        return "Please select a checkpoint and data file.", go.Figure(), False

    try:
        device = torch.device("cpu")
        dataset = _load_dataset(data_path)
        state = _run_inference(ckpt_path, dataset, split, device)
        _STATE.clear()
        _STATE.update(state)
        fig = _make_heatmap_fig(state, color_mode)
        H, W = state["som_dim"]
        n = int(state["k_all"].shape[0])
        return (
            f"Loaded: {os.path.basename(ckpt_path)}  |  "
            f"Split: {split}  |  "
            f"Beats: {n}  |  "
            f"SOM: {H}×{W}",
            fig,
            True,
        )
    except Exception as exc:
        return f"Error: {exc}", go.Figure(), False


@callback(
    Output("som-heatmap", "figure", allow_duplicate=True),
    Input("color-mode", "value"),
    State("inference-done", "data"),
    prevent_initial_call=True,
)
def recolor_heatmap(color_mode, done):
    if not done or not _STATE:
        return no_update
    return _make_heatmap_fig(_STATE, color_mode)


@callback(
    Output("clicked-cell", "data"),
    Input("som-heatmap", "clickData"),
    State("inference-done", "data"),
    prevent_initial_call=True,
)
def store_click(click_data, done):
    if not done or click_data is None:
        return no_update
    pt = click_data["points"][0]
    row = int(pt["y"])
    col = int(pt["x"])
    return {"row": row, "col": col}


@callback(
    Output("cell-detail-panel", "children"),
    Input("clicked-cell", "data"),
    State("inference-done", "data"),
    prevent_initial_call=True,
)
def render_cell_detail(cell, done):
    if not done or cell is None or not _STATE:
        return []

    r, c = cell["row"], cell["col"]
    H, W = _STATE["som_dim"]
    flat_idx = r * W + c

    # ── Gather data for cell ──────────────────────────────────────────────
    beat_indices = _STATE["cell_beats"].get(flat_idx, [])
    n_total = len(beat_indices)
    class_names = _STATE["class_names"]

    # ── Prototype ─────────────────────────────────────────────────────────
    proto = _STATE["prototype_signals"][r, c]  # [T, C]
    fig_proto = _make_prototype_fig(proto, (r, c), n_total)

    # ── Percentile plot ───────────────────────────────────────────────────
    if n_total > 0:
        cell_beats_np = _STATE["beats"][beat_indices]  # [N, T, C]
        fig_pct = _make_percentile_fig(cell_beats_np, (r, c))
    else:
        fig_pct = go.Figure()
        fig_pct.update_layout(
            title="No samples in this cell",
            paper_bgcolor="#181825", plot_bgcolor="#181825",
            font=dict(color="#cdd6f4"), height=200,
        )

    # ── Sample list ───────────────────────────────────────────────────────
    sample_cards = []
    display_indices = beat_indices[:MAX_SAMPLES]
    for gi in display_indices:
        beat = _STATE["beats"][gi]          # [T, C]
        lbl_enc = int(_STATE["labels"][gi])
        lbl_name = (class_names[lbl_enc]
                    if 0 <= lbl_enc < len(class_names) else "?")
        age  = float(_STATE["ages"][gi])
        sex  = "F" if int(_STATE["sexes"][gi]) == 1 else "M"
        rid  = _STATE["record_ids"][gi]
        bi   = _STATE["beat_local_idx"][gi]
        title = f"Rec {rid} | Beat {bi} | {lbl_name} | {sex}, {int(age)}y"
        fig_s = _make_sample_fig(beat, title)
        sample_cards.append(
            dbc.Col(
                dbc.Card([
                    dbc.CardBody(dcc.Graph(
                        figure=fig_s,
                        config={"displayModeBar": False},
                        style={"height": f"{fig_s.layout.height}px"},
                    ), className="p-1"),
                ], className="mb-2"),
                md=12 // SAMPLE_COLS,
            )
        )

    more_label = ""
    if n_total > MAX_SAMPLES:
        more_label = (f"Showing {MAX_SAMPLES} of {n_total} samples. "
                      f"Remaining {n_total - MAX_SAMPLES} not displayed.")

    layout = dbc.Card(dbc.CardBody([
        html.H5(f"Cell ({r}, {c})  —  {n_total} samples", className="mb-3"),

        # Prototype + percentile side-by-side
        dbc.Row([
            dbc.Col(dcc.Graph(
                figure=fig_proto,
                config={"displayModeBar": False},
                style={"height": f"{fig_proto.layout.height}px"},
            ), md=6),
            dbc.Col(dcc.Graph(
                figure=fig_pct,
                config={"displayModeBar": False},
                style={"height": f"{fig_pct.layout.height}px"},
            ), md=6),
        ], className="mb-3"),

        # Sample list
        html.H6(f"Sample Beats (up to {MAX_SAMPLES})", className="mb-2"),
        html.P(more_label, className="text-muted small") if more_label else None,
        dbc.Row(sample_cards),
    ]), className="mb-4")

    return layout


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECG-SOM Explorer")
    parser.add_argument("--data", default=_default_data_path(_BASE),
                        help="Path to ECG_Dataset pickle (default: data/ptbxl_100_T-12ms.pkl)")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app.layout = _build_layout(_BASE)
    app.run(host=args.host, port=args.port, debug=args.debug)
