"""Score every model on the shared folds and write the report.

Runs in the `sp-coral` env (CPU only — it reads stored features).

Two tables come out, and they answer different questions:

* **Linear probe** — each encoder's frozen features through the identical
  protocol. Comparable across encoders; this is the representation ranking.
* **Direct prediction** — DeepCell Types zero-shot and Spatium few-shot, each in
  its native usage mode. Comparable to the probe only as a *deployment* number,
  not as a representation number, because the supervision differs.

Both are reported. Conflating them is the easiest way to draw a wrong conclusion
from this repo, so the report labels them separately and says so.

Also emits a collapsed-label-space score (M1+M2 merged into `Macrophage`) to
quantify how much of DeepCell Types' zero-shot gap is its vocabulary being
coarser than the benchmark rather than its perception being worse.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import probe, protocol  # noqa: E402

FIGURES = protocol.RESULTS / "figures"

# One categorical slot per foundation model; the mean-marker baseline takes
# neutral ink because it is a reference line, not a fifth competitor.
#
# Validated all-pairs as CIEDE2000 under normal vision plus Machado severity-1.0
# protanopia, deuteranopia and tritanopia. Worst pair over the whole palette is
# dE 14.6 (KRONOS2 vs Spatium, tritanopia), which is a pre-existing limit of the
# original three; every pair involving VirTues is >= 16.5, so the fourth slot
# does not tighten the floor.
#
# VirTues' brick was chosen over the obvious purple: #8a5cd6 sits at dE 1.4 from
# the KRONOS2 blue under deuteranopia — the two bars would be one colour for the
# commonest form of CVD. Blue and purple collapse together on a deutan/protan
# axis, so the fourth categorical slot in a palette that already spends blue has
# to leave that hue family entirely.
#
# Spatium's aqua and DeepCell Types' orange sit below 3:1 on the light surface,
# so every chart using them ships value labels or a companion table (the relief
# rule). The brick is 8.1:1 and needs no relief of its own.
COLORS = {
    "KRONOS2": "#2a78d6",
    "DeepCell Types": "#eb6834",
    "Spatium": "#1baf7a",
    "VirTues": "#8f2d2d",
    "mean-marker": "#6b6a66",
}
METRICS = ["F1-Score", "Balanced Accuracy", "Average Precision", "ROC AUC"]

# Encoder display name -> stored feature file.
ENCODERS = {
    "mean-marker": "mean_marker.npz",
    "KRONOS2": "kronos2.npz",
    "DeepCell Types": "deepcell_types.npz",
    "Spatium": "spatium.npz",
    "VirTues": "virtues.npz",
}

# The collapse applied by `--collapse`: DeepCell Types has a single, undivided
# `Macrophage` class, so M1/M2 is unreachable for it by construction.
COLLAPSE = {"M1": "Macrophage", "M2": "Macrophage"}


CACHE = protocol.RESULTS / "probe_cache"


def _cache_key(path: Path, n_cells: int) -> str:
    """Fingerprint the inputs a probe result depends on.

    A cached result is only reusable if the features, the cell set and every
    protocol knob are unchanged — otherwise the cache would silently serve a
    score computed under different rules, which is worse than no cache.
    """
    import hashlib

    stat = path.stat()
    parts = (
        path.name, stat.st_size, int(stat.st_mtime), n_cells,
        protocol.N_TRIALS, protocol.C_RANGE, protocol.MAX_ITER,
        protocol.MAX_CELLS_PER_CLASS, protocol.SEED, tuple(protocol.CLASS_NAMES),
    )
    return hashlib.sha256(repr(parts).encode()).hexdigest()[:16]


def run_probes(
    cells: pd.DataFrame, only: str | None = None, *, write_summary: bool = True
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Probe every encoder whose features are on disk.

    Results are cached per encoder, keyed by :func:`_cache_key`, so adding an
    encoder re-probes only that one instead of redoing the others — each probe
    is 15 Optuna trials x 4 folds of multinomial logistic regression and takes
    the better part of an hour.

    Args:
        cells: The canonical cell table.
        only: Probe just this encoder. Used with ``--cache-only`` to warm one
            encoder's cache in a separate job while another probe is running.
        write_summary: Write ``probe_per_fold.csv`` / ``oof_probe.npz``. Off for
            a cache-warming run, which sees only a subset of the encoders and
            would otherwise overwrite a full run's tables with a partial one.

    Returns:
        ``(summary, oof)`` — a per-encoder metric table with mean and standard
        deviation across folds, and each encoder's out-of-fold predictions.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    rows, oof_by_model = {}, {}
    for name, filename in ENCODERS.items():
        if only is not None and name != only:
            continue
        path = protocol.RESULTS / "features" / filename
        if not path.exists():
            print(f"[skip] {name}: no features at {path}")
            continue

        stem = filename.replace(".npz", "")
        cached = CACHE / f"{stem}_{_cache_key(path, len(cells))}.npz"
        if cached.exists():
            blob = np.load(cached, allow_pickle=True)
            results = pd.DataFrame(
                blob["metrics"], columns=list(blob["columns"]),
                index=pd.Index(list(blob["folds"]), name="Fold"),
            )
            oof = blob["oof"]
            print(f"\n=== {name} — cached probe result ({cached.name}) ===")
        else:
            X = protocol.align_features(cells, path)
            print(f"\n=== {name} — probe on {X.shape[1]}-d features ===")
            results, oof, trials = probe.probe_encoder(X, cells, name=name)
            trials.to_csv(protocol.RESULTS / f"optuna_{stem}.csv", index=False)
            np.savez(
                cached,
                metrics=results.to_numpy(),
                columns=np.asarray(results.columns),
                folds=np.asarray(results.index),
                oof=oof,
            )

        summary = probe.summarize(results)
        rows[name] = summary
        oof_by_model[name] = oof
        print(summary.to_string())

    if not rows:
        raise RuntimeError(
            "No encoder features found under results/features/. Run the model "
            "scripts first (see CLAUDE.md)."
        )
    summary = pd.concat(rows, names=["encoder", "Fold"])
    if write_summary:
        summary.to_csv(protocol.RESULTS / "probe_per_fold.csv")
        np.savez(protocol.RESULTS / "oof_probe.npz", **oof_by_model)
    return summary, oof_by_model


def load_direct_predictions(cells: pd.DataFrame) -> dict[str, np.ndarray]:
    """Load native-mode predictions (zero-shot / few-shot) keyed by model."""
    preds = {}

    dct_csv = protocol.RESULTS / "predictions" / "dct_zeroshot.csv"
    if dct_csv.exists():
        df = pd.read_csv(dct_csv).set_index("cell_id")
        codes = df["code"].reindex(cells.index)
        if codes.isna().any():
            raise ValueError(
                f"{int(codes.isna().sum()):,} canonical cells have no DeepCell "
                "Types prediction; the mask ids may have drifted."
            )
        preds["DeepCell Types (zero-shot)"] = codes.to_numpy(dtype=int)

    for n_shot in protocol.SHOT_SIZES:
        path = protocol.RESULTS / "spatium" / f"fewshot_{n_shot}shot_oof.npy"
        if path.exists():
            preds[f"Spatium ({n_shot}-shot)"] = np.load(path)
    return preds


def score_direct(
    cells: pd.DataFrame, preds: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Score native-mode predictions on the same cells the probe uses."""
    y = cells["code"].to_numpy()
    rows = {
        name: probe.score_predictions(y, pred) for name, pred in preds.items()
    }
    table = pd.DataFrame(rows).T
    table.index.name = "model"
    table.to_csv(protocol.RESULTS / "direct_prediction.csv")
    return table


def collapse_codes(codes: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Remap class codes onto the collapsed label space.

    Applied identically to truth and prediction, so the result is an honest
    rescoring of the *same* calls under a coarser question — not a re-run of the
    models.

    Args:
        codes: Integer class codes in the 16-class space. ``-1`` (no call)
            passes through unchanged.
        names: The 16 class names, in code order.

    Returns:
        ``(collapsed_codes, collapsed_names)``.
    """
    collapsed_names = sorted({COLLAPSE.get(n, n) for n in names})
    lookup = {i: collapsed_names.index(COLLAPSE.get(n, n)) for i, n in enumerate(names)}
    out = np.array([lookup.get(int(c), -1) for c in codes], dtype=np.int64)
    return out, collapsed_names


def score_collapsed(
    cells: pd.DataFrame,
    oof_probe: dict[str, np.ndarray],
    direct: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Rescore every model with M1 and M2 merged into a single Macrophage class."""
    from sklearn.metrics import balanced_accuracy_score, f1_score

    y_c, names_c = collapse_codes(cells["code"].to_numpy(), protocol.CLASS_NAMES)
    labels = np.arange(len(names_c))

    rows = {}
    for source, preds in (("probe", oof_probe), ("native", direct)):
        for name, pred in preds.items():
            p_c, _ = collapse_codes(pred, protocol.CLASS_NAMES)
            rows[f"{name} [{source}]"] = {
                "F1-Score": f1_score(
                    y_c, p_c, average="macro", labels=labels, zero_division=0
                ),
                "Balanced Accuracy": balanced_accuracy_score(y_c, p_c),
            }
    table = pd.DataFrame(rows).T
    table.index.name = "model"
    table.to_csv(protocol.RESULTS / "collapsed_label_space.csv")
    return table


def per_class_f1(
    cells: pd.DataFrame, oof: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Per-class out-of-fold F1 for every model, with class abundances."""
    from sklearn.metrics import f1_score

    y = cells["code"].to_numpy()
    table = pd.DataFrame(
        {
            name: f1_score(
                y, pred, average=None, labels=protocol.LABELS, zero_division=0
            )
            for name, pred in oof.items()
        },
        index=protocol.CLASS_NAMES,
    )
    table["n_cells"] = (
        cells["label"].value_counts().reindex(protocol.CLASS_NAMES).to_numpy()
    )
    table = table.sort_values("n_cells", ascending=False)
    table.to_csv(protocol.RESULTS / "per_class_f1.csv")
    return table


def _style(ax: plt.Axes) -> None:
    """Recessive axes: the data carries the chart, not the furniture."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c8c7c2")
    ax.tick_params(colors="#52514e", length=3)
    ax.yaxis.grid(True, color="#e8e7e3", lw=0.8)
    ax.set_axisbelow(True)


def figure_summary(summary: pd.DataFrame) -> None:
    """One panel per metric; bars are encoders, with mean +/- std across folds.

    Small multiples rather than one grouped chart: the four metrics live on
    different scales and share no meaningful axis, so putting them side by side
    would invite reading across them.
    """
    means = summary.xs("Mean", level="Fold")
    stds = summary.xs("Std Dev", level="Fold")
    order = [n for n in ENCODERS if n in means.index]

    fig, axes = plt.subplots(1, len(METRICS), figsize=(15, 4.2))
    for ax, metric in zip(axes, METRICS):
        vals = [means.loc[n, metric] for n in order]
        errs = [stds.loc[n, metric] for n in order]
        bars = ax.bar(
            range(len(order)), vals, yerr=errs, capsize=3,
            color=[COLORS[n] for n in order], width=0.62,
            error_kw={"ecolor": "#52514e", "lw": 1},
        )
        # Direct value labels: the relief rule, and the numbers are the point.
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9,
                    color="#0b0b0b")
        ax.set(title=metric, ylim=(0, 1.08), xticks=range(len(order)))
        ax.set_xticklabels(order, rotation=20, ha="right", fontsize=9)
        _style(ax)
    axes[0].set_ylabel("out-of-fold score")
    fig.suptitle(
        "Linear probe on frozen features — identical cells, folds and probe",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "probe_summary.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def figure_per_class(table: pd.DataFrame) -> None:
    """Per-class F1, classes ordered by abundance."""
    models = [c for c in table.columns if c != "n_cells"]
    x = np.arange(len(table))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(14, 4.8))
    for i, name in enumerate(models):
        ax.bar(x + i * width - 0.4 + width / 2, table[name], width * 0.92,
               label=name, color=COLORS[name])
    ax.set(
        ylabel="out-of-fold F1", ylim=(0, 1),
        xticks=x, title="Per-class F1 (classes ordered by abundance)",
    )
    ax.set_xticklabels(
        [f"{c}\nn={n:,}" for c, n in zip(table.index, table["n_cells"])],
        fontsize=8,
    )
    ax.legend(frameon=False, ncol=len(models), loc="upper right", fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "per_class_f1.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def figure_confusion(cells: pd.DataFrame, oof: dict[str, np.ndarray]) -> None:
    """Row-normalised confusion, one panel per model.

    Row-normalised so each row reads as "where did the cells of this true class
    actually go" — the off-diagonal mass is the interesting part, and raw counts
    would be swamped by the abundant classes.
    """
    from sklearn.metrics import confusion_matrix

    y = cells["code"].to_numpy()
    models = list(oof)
    fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 6))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, models):
        cm = confusion_matrix(y, oof[name], labels=protocol.LABELS, normalize="true")
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)  # sequential, single hue
        ax.set(
            title=name, xlabel="predicted", ylabel="true",
            xticks=range(len(protocol.CLASS_NAMES)),
            yticks=range(len(protocol.CLASS_NAMES)),
            xticklabels=protocol.CLASS_NAMES, yticklabels=protocol.CLASS_NAMES,
        )
        ax.tick_params(axis="x", rotation=90, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, label="fraction of true class")
    fig.tight_layout()
    fig.savefig(FIGURES / "confusion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_spatial_errors(
    cells: pd.DataFrame,
    oof: dict[str, np.ndarray],
    *,
    bins: int = 60,
    min_cells: int = 20,
) -> None:
    """Map the local error *rate* onto the tissue, one panel per model.

    The question no aggregate metric answers: are the errors scattered, or do
    they cluster? Scattered errors are ordinary classifier noise; a whole region
    predicted wrong points at something structural — a staining gradient or a
    tissue compartment unlike anything in the training quadrants.

    Binned rather than a per-cell scatter, because a scatter cannot answer that
    question honestly at this density: with 139k cells and a ~30% error rate the
    wrong-coloured points overplot the right-coloured ones and the panel reads as
    near-total failure whatever the real rate is. A rate per spatial bin is
    immune to overplotting, is directly comparable between panels on a shared
    scale, and is the quantity the question is actually about.

    Args:
        cells: The canonical cell table.
        oof: Out-of-fold predictions per model.
        bins: Grid resolution across the slide's longer axis.
        min_cells: Bins with fewer cells are left blank — an error rate over
            three cells is noise, and rendering it invites reading tissue edges
            as hot spots.
    """
    y = cells["code"].to_numpy()
    x_um, y_um = cells["x"].to_numpy(), cells["y"].to_numpy()
    extent = [x_um.min(), x_um.max(), y_um.min(), y_um.max()]
    edges = (
        np.linspace(extent[0], extent[1], bins + 1),
        np.linspace(extent[2], extent[3], bins + 1),
    )
    counts, _, _ = np.histogram2d(x_um, y_um, bins=edges)

    rates = {}
    for name, pred in oof.items():
        wrong, _, _ = np.histogram2d(x_um, y_um, bins=edges, weights=(pred != y))
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = np.where(counts >= min_cells, wrong / counts, np.nan)
        rates[name] = rate.T  # histogram2d is (x, y); imshow wants (row, col)

    finite = np.concatenate([r[np.isfinite(r)] for r in rates.values()])
    vmin, vmax = np.percentile(finite, [2, 98])

    models = list(oof)
    fig, axes = plt.subplots(1, len(models), figsize=(5.4 * len(models), 5.8))
    axes = np.atleast_1d(axes)
    cmap = plt.get_cmap("magma_r").copy()  # sequential, single ramp
    cmap.set_bad("#f2f1ee")                # sparse bins read as background
    for ax, name in zip(axes, models):
        im = ax.imshow(
            rates[name], cmap=cmap, vmin=vmin, vmax=vmax,
            extent=extent, origin="upper", interpolation="nearest",
        )
        overall = (oof[name] != y).mean()
        ax.set_title(f"{name}\n{overall:.1%} of cells misclassified overall",
                     fontsize=11)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.colorbar(
        im, ax=axes, fraction=0.02, pad=0.01,
        label=f"local error rate (bins with <{min_cells} cells blank)",
    )
    fig.suptitle(
        "Out-of-fold error rate in space — shared colour scale", fontsize=13
    )
    fig.savefig(FIGURES / "spatial_errors.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-figures", action="store_true", help="Tables only."
    )
    parser.add_argument(
        "--only", choices=list(ENCODERS), help="Probe just this encoder."
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Populate the probe cache and stop, writing no tables or figures. "
        "Lets one encoder be probed in a separate job without clashing with a "
        "concurrent full evaluation run.",
    )
    args = parser.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)
    cells = protocol.load_cells()
    print(f"{len(cells):,} cells | {len(protocol.CLASS_NAMES)} classes | "
          f"4 spatial folds")

    summary, oof_probe = run_probes(
        cells, only=args.only, write_summary=not args.cache_only
    )
    if args.cache_only:
        print("\ncache populated; skipping report.")
        return
    direct_preds = load_direct_predictions(cells)

    print("\n=== Linear probe (mean across folds) ===")
    print(summary.xs("Mean", level="Fold")[METRICS].round(4).to_string())

    if direct_preds:
        direct = score_direct(cells, direct_preds)
        print("\n=== Native usage mode ===")
        print(direct.round(4).to_string())
        collapsed = score_collapsed(cells, oof_probe, direct_preds)
        print("\n=== Collapsed label space (M1+M2 -> Macrophage) ===")
        print(collapsed.round(4).to_string())

    per_class = per_class_f1(cells, oof_probe)
    print("\n=== Per-class F1 (probe) ===")
    print(per_class.round(3).to_string())

    if not args.no_figures:
        figure_summary(summary)
        figure_per_class(per_class)
        figure_confusion(cells, oof_probe)
        figure_spatial_errors(cells, oof_probe)
        print(f"\nwrote figures -> {FIGURES}")


if __name__ == "__main__":
    main()
