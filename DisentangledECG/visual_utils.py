import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from dpsom_ecg_model import DPSOM_ECG  # Ensure this import works


def log_som_visualizations(
    dataset,
    generator,
    ex_name,
    ckpt_path,
    device,
    single_label_only: bool = False,
    crop: int = 10,
):
    """
    Loads a model from a checkpoint onto CPU, runs inference on the test set,
    and logs SOM grid heatmap + centroid reconstructions to TensorBoard.
    """
    print("\nVisualizations...")

    # 1. Load Data (CPU)
    # NOTE: keep the generator call intact, but ignore age/sex outputs
    X_test, X_mask, y_test, _, _, sigmas_test, amaxes_test, _ = generator.get_all_beats_representation(split="test")

    if single_label_only:
        test_records = generator.ecg_dataset.get_data("test")
        keep_per_beat = []
        for rec in test_records:
            keep_rec = len(getattr(rec, "extra_labels", [])) == 0
            keep_per_beat.extend([keep_rec] * len(rec.beat_representations))
        keep_per_beat = np.asarray(keep_per_beat, dtype=bool)
        if keep_per_beat.shape[0] == X_test.shape[0]:
            X_test = X_test[keep_per_beat]
            y_test = np.asarray(y_test)[keep_per_beat]
        else:
            n = min(keep_per_beat.shape[0], X_test.shape[0], np.asarray(y_test).shape[0])
            X_test = X_test[:n][keep_per_beat[:n]]
            y_test = np.asarray(y_test)[:n][keep_per_beat[:n]]
        if X_test.shape[0] == 0:
            print("No single-label test beats available. Skipping SOM visualization logging.")
            return

    class_names = getattr(dataset, "class_names", ['CD', 'HYP', 'MI', 'NORM', 'STTC'])
    num_classes = len(class_names)

    # 2. Load Checkpoint & Infer Model Params
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f"Warning: Checkpoint not found at {ckpt_path}. Skipping visualization.")
        return

    meta = checkpoint.get("meta", {}) if isinstance(checkpoint, dict) else {}
    # Handle nesting (e.g. if saved as {"state_dict": ...} or {"model": ...})
    state = checkpoint
    if isinstance(state, dict):
        for k in ["state_dict", "model", "net", "weights"]:
            if k in state and isinstance(state[k], dict):
                state = state[k]
                break

    # Strip prefixes (e.g. "module." from DataParallel)
    def _strip_prefix(sd, prefix):
        if any(k.startswith(prefix) for k in sd.keys()):
            return {k[len(prefix):]: v for k, v in sd.items()}
        return sd

    state = _strip_prefix(state, "module.")
    state = _strip_prefix(state, "model.")

    # Deduce shapes from data (allow checkpoint override)
    input_length = int(meta.get("input_length", X_test.shape[1]))      # T
    input_channels = int(meta.get("input_channels", X_test.shape[2]))  # C

    som_dim = tuple(meta.get("som_dim", ()))
    latent_dim = int(meta.get("latent_dim", 0))

    if len(som_dim) != 2 or latent_dim <= 0:
        # Infer SOM/Latent dims from the embedding tensor shape in the state_dict
        embed_key = None
        for k, v in state.items():
            # Look for the embeddings tensor which is 3D: [H, W, D]
            if isinstance(v, torch.Tensor) and v.ndim == 3 and "embeddings" in k:
                embed_key = k
                break

        if embed_key is None:
            print("Error: Could not infer SOM dimensions from checkpoint. Skipping.")
            return

        H_size, W_size, D_size = state[embed_key].shape
        som_dim = (int(H_size), int(W_size))
        latent_dim = int(D_size)
    else:
        H_size, W_size = som_dim

    # 3. Instantiate & Load Model
    model = DPSOM_ECG(
        latent_dim=latent_dim,
        som_dim=som_dim,
        input_length=input_length,
        input_channels=input_channels
    ).to(device)

    model.load_state_dict(state, strict=False)
    model.eval()

    # 4. Prepare Input Tensors
    # [N, T, C] -> [N, C, T]
    x_t = torch.from_numpy(X_test.transpose(0, 2, 1)).contiguous().float().to(device)

    # 5. Inference (encode -> mu, compute BMU k(mu), reconstruct SOM centroids)
    with torch.no_grad():
        enc_out = model._encode(x_t)

        # Support both old and new _encode signatures:
        # old: (mu, logvar, z_age, z_sex), new: (mu, logvar) or similar
        mu = enc_out[0]
        # logvar = enc_out[1]  # not used here

        # Calculate BMU indices from mu
        E = model._embeddings
        k_all = model.k(mu).cpu().numpy()

        # Reconstruct Centroids from SOM embeddings
        E_flat = E.view(-1, latent_dim)

        # In the "no age/sex" setup, centroids should be decoded from morphology only.
        logits = model.decoder(E_flat)
        x_hat_all = logits.view(H_size, W_size, input_channels, input_length).cpu().numpy()

    # 6. Generate Heatmap Data
    k_flat = k_all.flatten()
    labels = y_test.flatten()
    K = H_size * W_size

    counts = np.zeros((K, num_classes), dtype=np.int64)
    valid_mask = labels < num_classes
    np.add.at(counts, (k_flat[valid_mask], labels[valid_mask].astype(np.int64)), 1)

    totals = counts.sum(axis=1)
    majority = np.full(K, np.nan, dtype=np.float32)
    nonempty = totals > 0
    majority[nonempty] = counts[nonempty].argmax(axis=1).astype(np.float32)

    clust_matr1 = majority.reshape(H_size, W_size)

    annot_labels = np.full((H_size, W_size), "", dtype=object)
    for i in range(H_size):
        for j in range(W_size):
            val = clust_matr1[i, j]
            if not np.isnan(val):
                cell_idx = i * W_size + j
                total = int(totals[cell_idx])
                if total == 0:
                    continue
                maj_label_idx = int(val)
                correct = int(counts[cell_idx, maj_label_idx])

                if 0 <= maj_label_idx < num_classes:
                    short_name = str(class_names[maj_label_idx])[:3]
                else:
                    short_name = "NA"
                annot_labels[i, j] = f"{short_name}\n{correct}/{total}"

    # 7. Plot & Log
    log_dir = f"logs/{ex_name}/test"
    os.makedirs(log_dir, exist_ok=True)

    # Crop centroids before visualization: remove `crop` points from both edges.
    crop = int(crop)
    if crop < 0:
        print(f"Warning: crop must be >= 0, got {crop}. Using crop=0.")
        crop = 0
    max_allowed_crop = max(0, (input_length - 1) // 2)
    if crop > max_allowed_crop:
        print(
            f"Warning: crop={crop} is too large for input_length={input_length}. "
            f"Using crop={max_allowed_crop}."
        )
        crop = max_allowed_crop

    if crop > 0:
        x_hat_all_plot = x_hat_all[:, :, :, crop:input_length - crop]
    else:
        x_hat_all_plot = x_hat_all

    # Heatmaps per class
    for c in range(num_classes):
        class_name = class_names[c] if c < len(class_names) else f"Class {c}"
        class_counts = counts[:, c].reshape(H_size, W_size)

        annot = class_counts.astype(str)
        annot[class_counts == 0] = ""

        fig_h = plt.figure(figsize=(10, 8))
        sns.heatmap(class_counts, cmap="YlGnBu", annot=annot, fmt="")
        plt.title(f"Heatmap: {class_name}")
        safe_name = class_name.replace("/", "_").replace(" ", "_")
        fig_h.savefig(os.path.join(log_dir, f"heatmap_{safe_name}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig_h)

    # Majority Class Heatmap
    fig_heatmap = plt.figure(figsize=(20, 15))
    sns.heatmap(clust_matr1, cmap="YlGnBu", annot=annot_labels, fmt="")
    plt.title("Majority Class per SOM Cell (CPU Eval)")
    label_set = 'All'
    fig_heatmap.savefig(os.path.join(log_dir, "heatmap_majority.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_heatmap)

    # Centroids grid
    fig_centroids, axes = plt.subplots(
        H_size, W_size,
        figsize=(W_size * 1.5, H_size * 1.5),
        sharex=True, sharey=True
    )

    # Make axes always 2D (robust for H_size==1 or W_size==1)
    if H_size == 1 and W_size == 1:
        axes = np.array([[axes]])
    elif H_size == 1 or W_size == 1:
        axes = axes.reshape(H_size, W_size)

    # Reduce free space at borders + between cells
    fig_centroids.subplots_adjust(
        left=0.02, right=0.995, bottom=0.02, top=0.995,
        wspace=0.05, hspace=0.05
    )

    # Consistent y-limits across all centroids
    ymin = float(np.min(x_hat_all_plot[:, :, 0, :]))
    ymax = float(np.max(x_hat_all_plot[:, :, 0, :]))
    plot_length = x_hat_all_plot.shape[-1]

    for i in range(H_size):
        for j in range(W_size):
            signal = x_hat_all_plot[i, j, 0, :]
            ax = axes[i, j]
            ax.plot(signal, linewidth=0.8)
            ax.set_xlim(0, plot_length - 1)
            ax.set_ylim(ymin, ymax)

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.6)

            ax.set_xticks([])
            ax.set_yticks([])

            if i == 0:
                ax.set_title(str(j), fontsize=10, pad=6)
            if j == 0:
                ax.set_ylabel(str(i), rotation=0, fontsize=10, labelpad=10, va="center")

    fig_centroids.supxlabel("Time", y=0.01)
    fig_centroids.savefig(os.path.join(log_dir, "som_centroids_lead0.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_centroids)

    # --- All-leads prototype grid ---
    # Each SOM cell shows all input_channels leads stacked with a vertical offset.
    lead_names = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
    n_leads = input_channels
    cell_h = max(1.8, n_leads * 0.28)   # height per cell in inches
    fig_proto, axes_proto = plt.subplots(
        H_size, W_size,
        figsize=(W_size * 2.0, H_size * cell_h),
        sharex=True,
    )
    if H_size == 1 and W_size == 1:
        axes_proto = np.array([[axes_proto]])
    elif H_size == 1 or W_size == 1:
        axes_proto = axes_proto.reshape(H_size, W_size)

    fig_proto.subplots_adjust(
        left=0.06, right=0.995, bottom=0.02, top=0.98,
        wspace=0.08, hspace=0.12
    )

    # Compute a global scale so offsets are consistent across cells
    global_amp = float(np.percentile(np.abs(x_hat_all_plot), 95))
    if global_amp < 1e-6:
        global_amp = 1.0
    lead_step = global_amp * 2.2  # vertical gap between leads

    for i in range(H_size):
        for j in range(W_size):
            ax = axes_proto[i, j]
            for lead_idx in range(n_leads - 1, -1, -1):   # plot lead 0 at top
                offset = (n_leads - 1 - lead_idx) * lead_step
                sig = x_hat_all_plot[i, j, lead_idx, :]
                ax.plot(sig + offset, linewidth=0.6, color="#2166ac")

            # Y-tick labels = lead names
            ax.set_yticks([(n_leads - 1 - k) * lead_step for k in range(n_leads)])
            if j == 0:
                ax.set_yticklabels(
                    [lead_names[k] if k < len(lead_names) else str(k) for k in range(n_leads)],
                    fontsize=5
                )
            else:
                ax.set_yticklabels([])
            ax.set_xticks([])

            # Cell label
            cell_idx = i * W_size + j
            total = int(totals[cell_idx])
            if total > 0:
                maj = int(counts[cell_idx].argmax())
                lbl = class_names[maj][:4] if maj < len(class_names) else ""
                ax.set_title(f"{i},{j} {lbl} ({total})", fontsize=5, pad=2)
            else:
                ax.set_title(f"{i},{j}", fontsize=5, pad=2)

    fig_proto.suptitle("SOM Cell Prototypes — all leads", fontsize=9, y=0.995)
    fig_proto.savefig(os.path.join(log_dir, "som_prototypes_all_leads.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_proto)
    print(f"Visualizations saved to {log_dir}/")
