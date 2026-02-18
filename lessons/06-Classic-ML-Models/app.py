"""
How to run:
  pip install gradio numpy pandas matplotlib scikit-learn
  python app.py
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import warnings
from dataclasses import asdict, dataclass
from typing import Any

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TRAIN_CACHE: dict[str, dict[str, Any]] = {}
_DATASET_CACHE: dict[str, dict[str, Any]] = {}
FASHION_SUBSET_SIZE = 800  # fixed stratified subset for in-class speed


@dataclass
class RunConfig:
    dataset: str
    test_size: float
    random_seed: int
    standardize_features: bool
    use_pca_fashion: bool
    pca_components: int
    fast_mode: bool
    logreg_c: float
    logreg_max_iter: int
    rf_n_estimators: int
    rf_max_depth: int
    gb_n_estimators: int
    gb_learning_rate: float
    mlp_hidden_layer_sizes: str
    mlp_alpha: float
    mlp_max_iter: int
    mlp_early_stopping: bool


def _to_dataframe(X: Any, feature_names: list[str]) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()
    return pd.DataFrame(X, columns=feature_names)


def load_dataset(name: str) -> dict[str, Any]:
    key = (name or "Breast Cancer").strip().lower()
    if key in _DATASET_CACHE:
        return _DATASET_CACHE[key]

    if key == "breast cancer":
        data = load_breast_cancer(as_frame=True)
        X = data.data.copy()
       
        # y_fashion_sub = y_fashion[subset_indices]

        y = pd.Series(data.target, name="target").astype(int)
        bundle = {
            "name": "Breast Cancer",
            "X": X,
            "y": y,
            "feature_names": list(X.columns),
            "target_names": ["Malignant", "Benign"],
            "num_classes": int(y.nunique()),
            "task_type": "binary",
            "image_shape": None,
        }
        _DATASET_CACHE[key] = bundle
        return bundle

    if key == "fashion-mnist":
        data = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="auto")
        X_np = data.data.copy()
        y_np = np.asarray(data.target).astype(int)
        feature_names = [f"px_{i}" for i in range(X_np.shape[1])]

        rng = np.random.RandomState(42)
        subset_indices = []
        per_class = FASHION_SUBSET_SIZE // 10
        for label in range(10):
            label_idx = np.where(y_np == label)[0]
            if len(label_idx) < per_class:
                chosen = rng.choice(label_idx, size=per_class, replace=True)
            else:
                chosen = rng.choice(label_idx, size=per_class, replace=False)
            subset_indices.append(chosen)
        subset_indices = np.concatenate(subset_indices)
        rng.shuffle(subset_indices)

        X = _to_dataframe(X_np[subset_indices], feature_names)
        y_fashion_sub = y_np[subset_indices]
        y = pd.Series(y_fashion_sub, name="label")

        bundle = {
            "name": "Fashion-MNIST",
            "X": X,
            "y": y,
            "feature_names": feature_names,
            "target_names": [str(i) for i in sorted(y.unique())],
            "num_classes": int(y.nunique()),
            "task_type": "multiclass",
            "image_shape": (28, 28),
        }
        _DATASET_CACHE[key] = bundle
        return bundle

    raise ValueError(f"Unsupported dataset: {name}")





def _class_distribution_fig(y: pd.Series, title: str) -> plt.Figure:
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    plt.close(fig)
    return fig


def _breast_corr_heatmap(X: pd.DataFrame, y: pd.Series, top_k: int) -> plt.Figure:
    joined = X.copy()
    joined["target"] = y
    top = (
        joined.corr(numeric_only=True)["target"]
        .drop("target")
        .abs()
        .sort_values(ascending=False)
        .head(max(2, int(top_k)))
    )
    cols = list(top.index)
    corr = X[cols].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title(f"Breast Cancer Correlation Heatmap (Top {len(cols)})")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    fig.tight_layout()
    plt.close(fig)
    return fig


def _breast_hist_pair(
    X: pd.DataFrame,
    y: pd.Series,
    feature_1: str,
    feature_2: str,
    target_names: list[str],
) -> plt.Figure:
    f1 = feature_1 if feature_1 in X.columns else X.columns[0]
    f2 = feature_2 if feature_2 in X.columns else X.columns[min(1, len(X.columns) - 1)]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    for cls in sorted(y.unique()):
        mask = y == cls
        label = target_names[int(cls)] if int(cls) < len(target_names) else str(cls)
        axes[0].hist(X.loc[mask, f1], bins=30, alpha=0.65, label=label)
        axes[1].hist(X.loc[mask, f2], bins=30, alpha=0.65, label=label)

    axes[0].set_title(f"Histogram: {f1}")
    axes[1].set_title(f"Histogram: {f2}")
    for ax in axes:
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.grid(alpha=0.25)
    axes[1].legend(loc="best")
    fig.tight_layout()
    plt.close(fig)
    return fig


def _fashion_image_grid(
    X: pd.DataFrame,
    y: pd.Series,
    image_shape: tuple[int, int],
    n_rows: int = 2,
    n_cols: int = 5,
) -> plt.Figure:
    classes = sorted(y.unique())
    n_needed = n_rows * n_cols
    rng = np.random.default_rng(42)

    chosen_idx: list[int] = []
    per_class = max(1, n_needed // len(classes))
    for cls in classes:
        idx = np.where(y.values == cls)[0]
        take = min(len(idx), per_class)
        if take > 0:
            chosen_idx.extend(rng.choice(idx, size=take, replace=False).tolist())

    if len(chosen_idx) < n_needed:
        all_idx = np.arange(len(X))
        extras = rng.choice(all_idx, size=n_needed - len(chosen_idx), replace=False)
        chosen_idx.extend(extras.tolist())

    chosen_idx = chosen_idx[:n_needed]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10.0, 4.6))
    axes = np.asarray(axes).ravel()
    for i, idx in enumerate(chosen_idx):
        img = X.iloc[idx].to_numpy().reshape(image_shape)
        lbl = int(y.iloc[idx])
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"Label {lbl}")
        axes[i].axis("off")

    fig.suptitle("Fashion-MNIST Sample Image Grid", y=1.01)
    fig.tight_layout()
    plt.close(fig)
    return fig


def _fashion_tsne_scatter(X: pd.DataFrame, y: pd.Series, max_points: int = 2500) -> plt.Figure:
    if len(X) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=max_points, replace=False)
        X_sub = X.iloc[idx]
        y_sub = y.iloc[idx]
    else:
        X_sub = X
        y_sub = y

    # t-SNE is sensitive to scaling; standardize for a more stable visualization.
    X_scaled = StandardScaler().fit_transform(X_sub.to_numpy())

    n_samples = int(X_scaled.shape[0])
    # Perplexity must be < n_samples; keep it reasonable for speed.
    perplexity = min(30, max(5, (n_samples - 1) // 3))

    points = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=42,
    ).fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    for cls in sorted(y_sub.unique()):
        mask = y_sub == cls
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            s=7,
            alpha=0.65,
            label=str(int(cls)),
        )
    ax.set_title(f"Fashion-MNIST t-SNE (2D), n={len(X_sub)}")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.2)
    ax.legend(title="Class", ncol=2, fontsize=8)
    fig.tight_layout()
    plt.close(fig)
    return fig


def _fashion_pixel_hist(X: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.hist(X.to_numpy().ravel(), bins=50, alpha=0.85)
    ax.set_title("Fashion-MNIST Pixel Intensity Histogram")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    plt.close(fig)
    return fig


def make_eda_plots(data_bundle: dict[str, Any], eda_params: dict[str, Any]) -> dict[str, Any]:
    X = data_bundle["X"]
    y = data_bundle["y"]

    shape_df = pd.DataFrame(
        [
            {
                "dataset": data_bundle["name"],
                "samples": int(X.shape[0]),
                "features": int(X.shape[1]),
                "classes": int(y.nunique()),
            }
        ]
    )

    class_counts = y.value_counts().sort_index()
    class_df = pd.DataFrame({"class": class_counts.index.astype(str), "count": class_counts.values})

    class_fig = _class_distribution_fig(y, f"{data_bundle['name']} Class Distribution")

    if data_bundle["name"] == "Breast Cancer":
        top_k = int(eda_params.get("top_k", 10))
        top_n_stats = int(eda_params.get("top_n_stats", 10))
        feature_1 = str(eda_params.get("feature_1", X.columns[0]))
        feature_2 = str(eda_params.get("feature_2", X.columns[min(1, len(X.columns) - 1)]))

        relation_fig = _breast_corr_heatmap(X, y, top_k)
        extra_fig = _breast_hist_pair(X, y, feature_1, feature_2, data_bundle["target_names"])

        stats_df = pd.DataFrame({
            "feature": X.columns,
            "mean": X.mean(numeric_only=True).to_numpy(),
            "std": X.std(numeric_only=True).to_numpy(),
        })
        stats_df["abs_mean"] = stats_df["mean"].abs()
        stats_df = stats_df.sort_values("abs_mean", ascending=False).drop(columns=["abs_mean"]).head(top_n_stats)

        flat_preview_df = X.head(8).copy()

        return {
            "shape_df": shape_df,
            "class_df": class_df,
            "class_dist_fig": class_fig,
            "relation_fig": relation_fig,
            "extra_fig": extra_fig,
            "pixel_hist_fig": None,
            "summary_stats_df": stats_df,
            "flat_preview_df": flat_preview_df,
        }

    relation_fig = _fashion_image_grid(X, y, data_bundle["image_shape"])
    extra_fig = _fashion_tsne_scatter(X, y, max_points=2500)
    pixel_hist_fig = _fashion_pixel_hist(X)
    stats_df = class_df.rename(columns={"class": "label", "count": "sample_count"})
    flat_preview_df = X.head(8).copy()

    return {
        "shape_df": shape_df,
        "class_df": class_df,
        "class_dist_fig": class_fig,
        "relation_fig": relation_fig,
        "extra_fig": extra_fig,
        "pixel_hist_fig": pixel_hist_fig,
        "summary_stats_df": stats_df,
        "flat_preview_df": flat_preview_df,
    }


def _parse_hidden_layers(hidden_layer_sizes: str) -> tuple[int, ...]:
    text = str(hidden_layer_sizes).strip()
    if not text:
        raise ValueError("MLP hidden_layer_sizes cannot be empty.")
    parts = [x.strip() for x in text.split(",") if x.strip()]
    try:
        values = tuple(int(x) for x in parts)
    except ValueError as exc:
        raise ValueError("MLP hidden layers must be comma-separated integers, e.g., 64,32") from exc
    if any(v <= 0 for v in values):
        raise ValueError("MLP hidden layers must be positive integers.")
    return values


def build_models(config: dict[str, Any], dataset_name: str) -> dict[str, Pipeline]:
    standardize = bool(config["standardize_features"])
    use_pca = bool(config["use_pca_fashion"]) and dataset_name == "Fashion-MNIST"
    pca_components = int(config["pca_components"])
    random_seed = int(config["random_seed"])

    logreg_steps: list[tuple[str, Any]] = []
    mlp_steps: list[tuple[str, Any]] = []

    if standardize:
        logreg_steps.append(("scaler", StandardScaler()))
        mlp_steps.append(("scaler", StandardScaler()))

    if use_pca:
        logreg_steps.append(("pca", PCA(n_components=pca_components, random_state=random_seed)))
        mlp_steps.append(("pca", PCA(n_components=pca_components, random_state=random_seed)))

    logreg = LogisticRegression(
        C=float(config["logreg_c"]),
        max_iter=int(config["logreg_max_iter"]),
        random_state=random_seed,
        solver="lbfgs",
        multi_class="auto",
    )
    logreg_steps.append(("model", logreg))

    rf = RandomForestClassifier(
        n_estimators=int(config["rf_n_estimators"]),
        max_depth=None if int(config["rf_max_depth"]) <= 0 else int(config["rf_max_depth"]),
        random_state=random_seed,
        n_jobs=-1,
    )

    gb = GradientBoostingClassifier(
        n_estimators=int(config["gb_n_estimators"]),
        learning_rate=float(config["gb_learning_rate"]),
        random_state=random_seed,
    )

    mlp = MLPClassifier(
        hidden_layer_sizes=_parse_hidden_layers(str(config["mlp_hidden_layer_sizes"])),
        alpha=float(config["mlp_alpha"]),
        max_iter=int(config["mlp_max_iter"]),
        early_stopping=bool(config["mlp_early_stopping"]),
        random_state=random_seed,
    )
    mlp_steps.append(("model", mlp))

    return {
        "Logistic Regression": Pipeline(logreg_steps),
        "Random Forest": Pipeline([("model", rf)]),
        "Gradient Boosting": Pipeline([("model", gb)]),
        "MLP": Pipeline(mlp_steps),
    }


def train_and_evaluate(
    models: dict[str, Pipeline],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    dataset_name: str,
) -> dict[str, Any]:
    average = "binary" if dataset_name == "Breast Cancer" else "weighted"

    metrics_rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    warn_logs: list[str] = []

    for model_name, model in models.items():
        est = clone(model)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            est.fit(X_train, y_train)

        for warning_item in caught:
            warn_logs.append(f"[{model_name}] {warning_item.category.__name__}: {warning_item.message}")

        y_pred = est.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test,
            y_pred,
            average=average,
            zero_division=0,
        )

        row = {
            "Model": model_name,
            "Accuracy": float(acc),
            "Precision": float(prec),
            "Recall": float(rec),
            "F1": float(f1),
        }

        roc_payload = None
        if dataset_name == "Breast Cancer":
            score = None
            if hasattr(est, "predict_proba"):
                probs = est.predict_proba(X_test)
                if probs.ndim == 2 and probs.shape[1] >= 2:
                    score = probs[:, 1]
            elif hasattr(est, "decision_function"):
                score = est.decision_function(X_test)

            if score is not None:
                fpr, tpr, _ = roc_curve(y_test, score)
                row["ROC_AUC"] = float(roc_auc_score(y_test, score))
                roc_payload = {"fpr": fpr, "tpr": tpr, "auc": row["ROC_AUC"]}
            else:
                row["ROC_AUC"] = np.nan

        metrics_rows.append(row)

        sorted_labels = sorted(y_test.unique())
        target_names = [str(v) for v in sorted_labels]
        if dataset_name == "Breast Cancer":
            target_names = ["Malignant", "Benign"]

        details[model_name] = {
            "report": classification_report(
                y_test,
                y_pred,
                labels=sorted_labels,
                target_names=target_names,
                zero_division=0,
            ),
            "y_true": y_test.to_numpy(),
            "y_pred": np.asarray(y_pred),
            "confusion": confusion_matrix(y_test, y_pred, labels=sorted_labels),
            "labels": sorted_labels,
            "roc": roc_payload,
        }

    metrics_df = pd.DataFrame(metrics_rows)
    metric_cols = ["Accuracy", "Precision", "Recall", "F1"]
    metrics_df["Overall"] = metrics_df[metric_cols].mean(axis=1)
    metrics_df = metrics_df.sort_values("Overall", ascending=False).reset_index(drop=True)

    best_per_metric = {
        metric: str(metrics_df.loc[metrics_df[metric].idxmax(), "Model"])
        for metric in metric_cols
    }
    best_overall = str(metrics_df.loc[metrics_df["Overall"].idxmax(), "Model"])

    return {
        "metrics_df": metrics_df,
        "details": details,
        "warnings": warn_logs,
        "best_per_metric": best_per_metric,
        "best_overall": best_overall,
    }


def _confusion_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int],
    title: str,
    normalize: bool,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=labels,
        normalize="true" if normalize else None,
        ax=ax,
        cmap="Blues",
        colorbar=False,
    )
    ax.set_title(title)
    fig.tight_layout()
    plt.close(fig)
    return fig


def _roc_figure(details: dict[str, Any]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    for model_name, payload in details.items():
        roc_payload = payload.get("roc")
        if roc_payload is None:
            continue
        ax.plot(roc_payload["fpr"], roc_payload["tpr"], label=f"{model_name} (AUC={roc_payload['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.7)
    ax.set_title("ROC Curves (Breast Cancer)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    plt.close(fig)
    return fig


def render_results(result_bundle: dict[str, Any], selected_model: str, dataset_name: str) -> dict[str, Any]:
    details = result_bundle["details"]
    if not details:
        return {
            "metrics_df": pd.DataFrame(),
            "comparison_df": pd.DataFrame(),
            "conf_fig": None,
            "report": "",
            "roc_fig": None,
            "norm_conf_fig": None,
            "best_summary": "",
        }

    if selected_model not in details:
        selected_model = next(iter(details.keys()))

    payload = details[selected_model]

    conf_fig = _confusion_figure(
        payload["y_true"],
        payload["y_pred"],
        payload["labels"],
        f"Confusion Matrix - {selected_model}",
        normalize=False,
    )

    norm_conf_fig = None
    roc_fig = None
    if dataset_name == "Fashion-MNIST":
        norm_conf_fig = _confusion_figure(
            payload["y_true"],
            payload["y_pred"],
            payload["labels"],
            f"Normalized Confusion Matrix - {selected_model}",
            normalize=True,
        )
    else:
        roc_fig = _roc_figure(details)

    metric_cols = ["Accuracy", "Precision", "Recall", "F1", "Overall"]
    metrics_df = result_bundle["metrics_df"].copy()
    for col in metric_cols:
        metrics_df[col] = metrics_df[col].map(lambda v: round(float(v), 4))

    comparison_df = metrics_df.copy()
    for metric in ["Accuracy", "Precision", "Recall", "F1", "Overall"]:
        best_idx = comparison_df[metric].astype(float).idxmax()
        comparison_df[metric] = comparison_df[metric].astype(str)
        comparison_df.loc[best_idx, metric] = f"{comparison_df.loc[best_idx, metric]} ★"

    best = result_bundle["best_per_metric"]
    best_summary = (
        f"Best Accuracy: {best['Accuracy']} | "
        f"Best Precision: {best['Precision']} | "
        f"Best Recall: {best['Recall']} | "
        f"Best F1: {best['F1']} | "
        f"Best Overall: {result_bundle['best_overall']}"
    )

    return {
        "metrics_df": metrics_df,
        "comparison_df": comparison_df,
        "conf_fig": conf_fig,
        "report": payload["report"],
        "roc_fig": roc_fig,
        "norm_conf_fig": norm_conf_fig,
        "best_summary": best_summary,
    }


def _normalize_config(raw: dict[str, Any]) -> dict[str, Any]:
    normalized = asdict(RunConfig(**raw))
    normalized["dataset"] = str(normalized["dataset"])
    normalized["test_size"] = float(normalized["test_size"])
    normalized["random_seed"] = int(normalized["random_seed"])
    normalized["standardize_features"] = bool(normalized["standardize_features"])
    normalized["use_pca_fashion"] = bool(normalized["use_pca_fashion"])
    normalized["pca_components"] = int(normalized["pca_components"])
    normalized["fast_mode"] = bool(normalized["fast_mode"])
    normalized["logreg_c"] = float(normalized["logreg_c"])
    normalized["logreg_max_iter"] = int(normalized["logreg_max_iter"])
    normalized["rf_n_estimators"] = int(normalized["rf_n_estimators"])
    normalized["rf_max_depth"] = int(normalized["rf_max_depth"])
    normalized["gb_n_estimators"] = int(normalized["gb_n_estimators"])
    normalized["gb_learning_rate"] = float(normalized["gb_learning_rate"])
    normalized["mlp_hidden_layer_sizes"] = str(normalized["mlp_hidden_layer_sizes"])
    normalized["mlp_alpha"] = float(normalized["mlp_alpha"])
    normalized["mlp_max_iter"] = int(normalized["mlp_max_iter"])
    normalized["mlp_early_stopping"] = bool(normalized["mlp_early_stopping"])
    return normalized


def _apply_fast_mode(config: dict[str, Any]) -> dict[str, Any]:
    if not config["fast_mode"]:
        return config

    cfg = dict(config)
    cfg["logreg_max_iter"] = min(cfg["logreg_max_iter"], 250)
    cfg["rf_n_estimators"] = min(cfg["rf_n_estimators"], 120)
    cfg["gb_n_estimators"] = min(cfg["gb_n_estimators"], 120)
    cfg["mlp_max_iter"] = min(cfg["mlp_max_iter"], 180)

    if cfg["dataset"] == "Fashion-MNIST":
        cfg["use_pca_fashion"] = True
        cfg["pca_components"] = min(max(cfg["pca_components"], 10), 64)

    return cfg


def _validate_config(config: dict[str, Any]) -> None:
    if config["dataset"] not in {"Breast Cancer", "Fashion-MNIST"}:
        raise ValueError("Dataset must be Breast Cancer or Fashion-MNIST.")
    if not (0.1 <= config["test_size"] <= 0.4):
        raise ValueError("Test size must be between 0.1 and 0.4.")
    if config["dataset"] == "Breast Cancer" and config["use_pca_fashion"]:
        raise ValueError("Use PCA for Fashion-MNIST applies only to Fashion-MNIST.")
    if config["dataset"] == "Fashion-MNIST" and not (10 <= config["pca_components"] <= 256):
        raise ValueError("For Fashion-MNIST, PCA n_components must be between 10 and 256.")
    if config["logreg_c"] <= 0:
        raise ValueError("Logistic Regression C must be > 0.")
    if config["rf_n_estimators"] < 10:
        raise ValueError("Random Forest n_estimators must be at least 10.")
    if config["gb_learning_rate"] <= 0:
        raise ValueError("Gradient Boosting learning_rate must be > 0.")
    _parse_hidden_layers(config["mlp_hidden_layer_sizes"])


def _hash_config(config: dict[str, Any]) -> str:
    text = json.dumps(config, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _dataset_summary(data_bundle: dict[str, Any]) -> str:
    X = data_bundle["X"]
    y = data_bundle["y"]
    return (
        "### Dataset Summary\n"
        f"- Dataset: **{data_bundle['name']}**\n"
        f"- Shape: **{X.shape[0]} x {X.shape[1]}**\n"
        f"- Classes: **{int(y.nunique())}**\n"
        f"- Task: **{data_bundle['task_type']}**\n"
    )


def _export_files(metrics_df: pd.DataFrame, config_json: str) -> tuple[str, str]:
    metrics_file = tempfile.NamedTemporaryFile(mode="w", suffix="_metrics.csv", delete=False)
    config_file = tempfile.NamedTemporaryFile(mode="w", suffix="_config.json", delete=False)

    metrics_df.to_csv(metrics_file.name, index=False)
    metrics_file.close()

    with open(config_file.name, "w", encoding="utf-8") as handle:
        handle.write(config_json)

    return metrics_file.name, config_file.name


def refresh_dataset_and_eda(
    dataset_name: str,
    top_k: int,
    feature_1: str,
    feature_2: str,
    top_n_stats: int,
) -> tuple[Any, ...]:
    data_bundle = load_dataset(dataset_name)
    features = data_bundle["feature_names"]

    if data_bundle["name"] == "Breast Cancer":
        f1 = feature_1 if feature_1 in features else features[0]
        f2 = feature_2 if feature_2 in features else features[min(1, len(features) - 1)]
        f1_update = gr.update(choices=features, value=f1, interactive=True)
        f2_update = gr.update(choices=features, value=f2, interactive=True)
    else:
        f1_update = gr.update(choices=["N/A"], value="N/A", interactive=False)
        f2_update = gr.update(choices=["N/A"], value="N/A", interactive=False)

    eda = make_eda_plots(
        data_bundle,
        {
            "top_k": int(top_k),
            "feature_1": feature_1,
            "feature_2": feature_2,
            "top_n_stats": int(top_n_stats),
        },
    )

    return (
        _dataset_summary(data_bundle),
        eda["shape_df"],
        eda["class_df"],
        eda["class_dist_fig"],
        eda["relation_fig"],
        eda["extra_fig"],
        gr.update(visible=(data_bundle["name"] == "Fashion-MNIST"), value=eda["pixel_hist_fig"]),
        eda["summary_stats_df"],
        eda["flat_preview_df"],
        f1_update,
        f2_update,
    )


def run_training(
    dataset: str,
    test_size: float,
    random_seed: int,
    standardize_features: bool,
    use_pca_fashion: bool,
    pca_components: int,
    fast_mode: bool,
    logreg_c: float,
    logreg_max_iter: int,
    rf_n_estimators: int,
    rf_max_depth: int,
    gb_n_estimators: int,
    gb_learning_rate: float,
    mlp_hidden_layer_sizes: str,
    mlp_alpha: float,
    mlp_max_iter: int,
    mlp_early_stopping: bool,
    progress=gr.Progress(track_tqdm=False),
) -> tuple[Any, ...]:
    raw = {
        "dataset": dataset,
        "test_size": test_size,
        "random_seed": random_seed,
        "standardize_features": standardize_features,
        "use_pca_fashion": use_pca_fashion,
        "pca_components": pca_components,
        "fast_mode": fast_mode,
        "logreg_c": logreg_c,
        "logreg_max_iter": logreg_max_iter,
        "rf_n_estimators": rf_n_estimators,
        "rf_max_depth": rf_max_depth,
        "gb_n_estimators": gb_n_estimators,
        "gb_learning_rate": gb_learning_rate,
        "mlp_hidden_layer_sizes": mlp_hidden_layer_sizes,
        "mlp_alpha": mlp_alpha,
        "mlp_max_iter": mlp_max_iter,
        "mlp_early_stopping": mlp_early_stopping,
    }

    empty_metrics = pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall", "F1", "Overall"])
    empty_comp = pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall", "F1", "Overall"])

    try:
        config = _normalize_config(raw)
        config = _apply_fast_mode(config)
        _validate_config(config)
    except Exception as exc:
        msg = f"Validation error: {exc}"
        return (
            empty_metrics,
            empty_comp,
            "",
            msg,
            "",
            gr.update(choices=[], value=None),
            None,
            "",
            gr.update(visible=False, value=None),
            gr.update(visible=False, value=None),
            None,
            None,
            {},
        )

    config_json = json.dumps(config, indent=2, sort_keys=True)
    cache_key = _hash_config(config)

    progress(0.05, desc="Loading dataset")

    if cache_key in TRAIN_CACHE:
        cached = TRAIN_CACHE[cache_key]
        rendered = render_results(cached["result_bundle"], cached["selected_model"], config["dataset"])
        metrics_path, config_path = _export_files(rendered["metrics_df"], config_json)
        warnings_text = "\n".join(cached["result_bundle"]["warnings"]) if cached["result_bundle"]["warnings"] else "No warnings."
        progress(1.0, desc="Done (cached)")
        return (
            rendered["metrics_df"],
            rendered["comparison_df"],
            rendered["best_summary"],
            "Cache hit: reused previous training results.",
            warnings_text,
            gr.update(choices=list(cached["result_bundle"]["details"].keys()), value=cached["selected_model"]),
            rendered["conf_fig"],
            rendered["report"],
            gr.update(visible=(config["dataset"] == "Breast Cancer"), value=rendered["roc_fig"]),
            gr.update(visible=(config["dataset"] == "Fashion-MNIST"), value=rendered["norm_conf_fig"]),
            metrics_path,
            config_path,
            cached["result_bundle"],
        )

    data_bundle = load_dataset(config["dataset"])
    X = data_bundle["X"]
    y = data_bundle["y"]

    progress(0.2, desc="Splitting train/test")
    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["test_size"],
        random_state=config["random_seed"],
        stratify=stratify,
    )

    progress(0.35, desc="Building model pipelines")
    models = build_models(config, config["dataset"])

    progress(0.55, desc="Training and evaluating models")
    result_bundle = train_and_evaluate(models, X_train, X_test, y_train, y_test, config["dataset"])

    selected_model = result_bundle["best_overall"]
    rendered = render_results(result_bundle, selected_model, config["dataset"])
    metrics_path, config_path = _export_files(rendered["metrics_df"], config_json)

    TRAIN_CACHE[cache_key] = {
        "result_bundle": result_bundle,
        "selected_model": selected_model,
    }

    warnings_text = "\n".join(result_bundle["warnings"]) if result_bundle["warnings"] else "No warnings."
    progress(1.0, desc="Done")

    return (
        rendered["metrics_df"],
        rendered["comparison_df"],
        rendered["best_summary"],
        "Training completed and cached.",
        warnings_text,
        gr.update(choices=list(result_bundle["details"].keys()), value=selected_model),
        rendered["conf_fig"],
        rendered["report"],
        gr.update(visible=(config["dataset"] == "Breast Cancer"), value=rendered["roc_fig"]),
        gr.update(visible=(config["dataset"] == "Fashion-MNIST"), value=rendered["norm_conf_fig"]),
        metrics_path,
        config_path,
        result_bundle,
    )


def on_model_change(selected_model: str, dataset: str, result_state: dict[str, Any]) -> tuple[Any, ...]:
    if not result_state:
        return None, "", gr.update(visible=False, value=None), gr.update(visible=False, value=None)

    rendered = render_results(result_state, selected_model, dataset)
    return (
        rendered["conf_fig"],
        rendered["report"],
        gr.update(visible=(dataset == "Breast Cancer"), value=rendered["roc_fig"]),
        gr.update(visible=(dataset == "Fashion-MNIST"), value=rendered["norm_conf_fig"]),
    )


def _standardize_default_for_dataset(ds_name: str) -> gr.update:
    # default on for both datasets (for linear/MLP paths)
    return gr.update(value=True)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Classical ML Model Explorer") as demo:
        gr.Markdown("# Classical ML Model Explorer")

        result_state = gr.State({})

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Dataset")
                dataset = gr.Radio(
                    choices=["Breast Cancer", "Fashion-MNIST"],
                    value="Breast Cancer",
                    label="Choose dataset",
                )
                dataset_summary = gr.Markdown()
                shape_df = gr.Dataframe(label="Dataset Shape", interactive=False)
                class_df = gr.Dataframe(label="Class Counts", interactive=False)

                gr.Markdown("## EDA / Visualizations")
                top_k = gr.Slider(2, 20, value=10, step=1, label="Breast Cancer: Top-k correlated features")
                feature_1 = gr.Dropdown(choices=[], label="Breast Cancer: Histogram feature 1")
                feature_2 = gr.Dropdown(choices=[], label="Breast Cancer: Histogram feature 2")
                top_n_stats = gr.Slider(5, 25, value=10, step=1, label="Summary stats: Top N")

            with gr.Column(scale=2):
                class_dist_fig = gr.Plot(label="Class Distribution")
                relation_fig = gr.Plot(label="Relationship Plot")
                extra_fig = gr.Plot(label="Additional Plot")
                pixel_hist_fig = gr.Plot(label="Fashion-MNIST Pixel Histogram", visible=False)
                summary_stats_df = gr.Dataframe(label="Summary Statistics", interactive=False)
                flat_preview_df = gr.Dataframe(label="Flattened Feature Preview", interactive=False)

        gr.Markdown("## Training")
        with gr.Row():
            test_size = gr.Slider(0.1, 0.4, value=0.2, step=0.01, label="Test size")
            random_seed = gr.Number(value=42, precision=0, label="Random seed")
            standardize_features = gr.Checkbox(value=True, label="Standardize features")
            fast_mode = gr.Checkbox(value=False, label="Fast mode")

        with gr.Row():
            use_pca_fashion = gr.Checkbox(value=False, label="Use PCA for Fashion-MNIST (LogReg/MLP only)")
            pca_components = gr.Slider(10, 256, value=64, step=1, label="Fashion-MNIST PCA components")

        with gr.Row():
            logreg_c = gr.Slider(0.01, 10.0, value=1.0, step=0.01, label="Logistic Regression: C")
            logreg_max_iter = gr.Slider(100, 2000, value=500, step=50, label="Logistic Regression: max_iter")

        with gr.Row():
            rf_n_estimators = gr.Slider(50, 1000, value=300, step=10, label="Random Forest: n_estimators")
            rf_max_depth = gr.Slider(0, 50, value=0, step=1, label="Random Forest: max_depth (0=None)")

        with gr.Row():
            gb_n_estimators = gr.Slider(50, 1000, value=250, step=10, label="Gradient Boosting: n_estimators")
            gb_learning_rate = gr.Slider(0.01, 0.5, value=0.05, step=0.01, label="Gradient Boosting: learning_rate")

        with gr.Row():
            mlp_hidden_layer_sizes = gr.Textbox(value="64,32", label="MLP: hidden_layer_sizes")
            mlp_alpha = gr.Slider(1e-6, 1e-1, value=1e-4, step=1e-6, label="MLP: alpha")
            mlp_max_iter = gr.Slider(100, 2000, value=500, step=50, label="MLP: max_iter")
            mlp_early_stopping = gr.Checkbox(value=True, label="MLP: early_stopping")

        train_btn = gr.Button("Train & Evaluate", variant="primary")

        gr.Markdown("## Results")
        model_selector = gr.Dropdown(choices=[], label="Model for confusion matrix/report")
        conf_fig = gr.Plot(label="Confusion Matrix")
        report_box = gr.Textbox(label="Classification Report", lines=14)
        roc_fig = gr.Plot(label="Breast Cancer ROC + AUC", visible=False)
        norm_conf_fig = gr.Plot(label="Fashion-MNIST Normalized Confusion Matrix", visible=False)

        gr.Markdown("## Model Comparison")
        metrics_df = gr.Dataframe(label="Metrics Table", interactive=False)
        comparison_df = gr.Dataframe(label="Best-per-metric Highlighted", interactive=False)
        best_summary = gr.Textbox(label="Best Models Summary")

        gr.Markdown("## Downloads/Logs")
        config_box = gr.Textbox(label="Run Configuration (JSON)", lines=14)
        status_box = gr.Textbox(label="Status / Progress", lines=2)
        warnings_box = gr.Textbox(label="Warnings / Logs", lines=8)
        metrics_download = gr.File(label="Download Metrics CSV")
        config_download = gr.File(label="Download Config JSON")

        dataset.change(
            fn=_standardize_default_for_dataset,
            inputs=[dataset],
            outputs=[standardize_features],
        )

        refresh_inputs = [dataset, top_k, feature_1, feature_2, top_n_stats]
        refresh_outputs = [
            dataset_summary,
            shape_df,
            class_df,
            class_dist_fig,
            relation_fig,
            extra_fig,
            pixel_hist_fig,
            summary_stats_df,
            flat_preview_df,
            feature_1,
            feature_2,
        ]

        dataset.change(fn=refresh_dataset_and_eda, inputs=refresh_inputs, outputs=refresh_outputs)
        top_k.change(fn=refresh_dataset_and_eda, inputs=refresh_inputs, outputs=refresh_outputs)
        feature_1.change(fn=refresh_dataset_and_eda, inputs=refresh_inputs, outputs=refresh_outputs)
        feature_2.change(fn=refresh_dataset_and_eda, inputs=refresh_inputs, outputs=refresh_outputs)
        top_n_stats.change(fn=refresh_dataset_and_eda, inputs=refresh_inputs, outputs=refresh_outputs)

        train_btn.click(
            fn=run_training,
            inputs=[
                dataset,
                test_size,
                random_seed,
                standardize_features,
                use_pca_fashion,
                pca_components,
                fast_mode,
                logreg_c,
                logreg_max_iter,
                rf_n_estimators,
                rf_max_depth,
                gb_n_estimators,
                gb_learning_rate,
                mlp_hidden_layer_sizes,
                mlp_alpha,
                mlp_max_iter,
                mlp_early_stopping,
            ],
            outputs=[
                metrics_df,
                comparison_df,
                best_summary,
                status_box,
                warnings_box,
                model_selector,
                conf_fig,
                report_box,
                roc_fig,
                norm_conf_fig,
                metrics_download,
                config_download,
                result_state,
            ],
        )

        model_selector.change(
            fn=on_model_change,
            inputs=[model_selector, dataset, result_state],
            outputs=[conf_fig, report_box, roc_fig, norm_conf_fig],
        )

        demo.load(fn=refresh_dataset_and_eda, inputs=refresh_inputs, outputs=refresh_outputs)

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
