"""The shared scoring harness.

Two scoring modes, because the three models are not used the same way:

* **Linear probe** — freeze the encoder, fit multinomial logistic regression on
  the embeddings, report how well it separates the classes. This is the
  supervision-matched comparison: identical cells, folds, training budget and
  hyperparameter search for every encoder, so a score difference is a
  *representation* difference. Delegated wholesale to CORAL's own tutorial
  helpers (imported below) so the numbers are directly comparable to the
  published KRONOS benchmark rather than a lookalike reimplementation.

* **Zero-shot / direct prediction** — a model that emits cell-type calls with no
  training on this dataset (DeepCell Types) is scored on the same cells, but
  there is no fold structure to respect: it never saw any of them. Scored by
  :func:`score_predictions`.

The two are reported side by side and must not be read as one ranking; see
CLAUDE.md on what each mode does and does not establish.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from common import protocol

# CORAL keeps the probe protocol in its tutorials/, not in the installed
# package, so it joins the path rather than being imported from `coral`.
_TUTORIALS = protocol.REPO / "CORAL" / "tutorials"
if str(_TUTORIALS) not in sys.path:
    sys.path.insert(0, str(_TUTORIALS))

from utils.cell_phenotyping_utils import (  # noqa: E402
    cap_per_class,
    run_cross_validation,
    split_fold,
    summarize,
)

__all__ = [
    "cap_per_class",
    "draw_support_set",
    "probe_encoder",
    "run_cross_validation",
    "score_predictions",
    "split_fold",
    "summarize",
]


def probe_encoder(
    X: np.ndarray, cells: pd.DataFrame, *, name: str, verbose: bool = True
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Run the full four-fold linear probe on one encoder's features.

    Every protocol knob comes from :mod:`common.protocol`, so calling this for
    a second encoder is guaranteed to reuse the first one's settings.

    Args:
        X: Feature matrix aligned to ``cells``, shape ``(n_cells, n_features)``.
        cells: The canonical cell table from :func:`protocol.load_cells`.
        name: Encoder name, used in progress lines and result tables.
        verbose: Print per-fold progress.

    Returns:
        ``(results, oof, trials)`` — per-fold metrics, out-of-fold predictions
        aligned to ``cells``, and the Optuna trial history.
    """
    return run_cross_validation(
        X,
        cells["code"].to_numpy(),
        cells["quadrant"].to_numpy(),
        name=name,
        labels=protocol.LABELS,
        n_trials=protocol.N_TRIALS,
        c_range=protocol.C_RANGE,
        max_cells_per_class=protocol.MAX_CELLS_PER_CLASS,
        max_iter=protocol.MAX_ITER,
        seed=protocol.SEED,
        sampler_seed=protocol.SEED,
        verbose=verbose,
    )


def score_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Score hard cell-type calls against ground truth on the shared classes.

    Used for models that predict directly rather than through a probe. The
    metric set matches :func:`utils.cell_phenotyping_utils.evaluate` so the two
    modes appear in the same table, with the caveat that a model emitting no
    calibrated probabilities over :data:`protocol.CLASS_NAMES` cannot be given
    an average precision or ROC AUC — those come back as ``nan`` rather than
    being quietly computed off the hard labels, which would flatter the model.

    Args:
        y_true: Integer class codes, one per cell.
        y_pred: Predicted integer class codes. ``-1`` marks a cell the model
            declined to call (out-of-vocabulary or abstained); those cells are
            scored as errors, not dropped.
        y_prob: Optional ``(n_cells, n_classes)`` probabilities over
            :data:`protocol.CLASS_NAMES`.

    Returns:
        Dict of macro F1, balanced accuracy, average precision and ROC AUC.
    """
    labels = protocol.LABELS
    metrics = {
        "F1-Score": f1_score(
            y_true, y_pred, average="macro", labels=labels, zero_division=0
        ),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Average Precision": float("nan"),
        "ROC AUC": float("nan"),
    }
    if y_prob is not None:
        y_onehot = label_binarize(y_true, classes=labels)
        metrics["Average Precision"] = average_precision_score(
            y_onehot, y_prob, average="macro"
        )
        metrics["ROC AUC"] = roc_auc_score(
            y_true, y_prob, average="macro", multi_class="ovr", labels=labels
        )
    return metrics


def draw_support_set(
    cells: pd.DataFrame, fold_id: int, n_per_class: int, *, seed: int = protocol.SEED
) -> np.ndarray:
    """Draw a few-shot support set from a fold's training quadrants.

    The support cells come from the three quadrants held *in* for this fold, so
    no support cell is a spatial neighbour of a test cell — the same guarantee
    the linear probe gets. Classes with fewer than ``n_per_class`` cells
    available contribute everything they have rather than being dropped, which
    keeps the rare types in the label space.

    Args:
        cells: The canonical cell table.
        fold_id: Which quadrant (1..4) is held out as test.
        n_per_class: Support cells to draw per class.
        seed: Pins the draw.

    Returns:
        Row positions into ``cells`` for the support cells.
    """
    train_idx, _, _ = split_fold(
        cells["quadrant"].to_numpy(), fold_id, random_state=seed
    )
    return cap_per_class(
        train_idx, cells["code"].to_numpy(), n_per_class, random_state=seed
    )
