"""
Run:
  pip install gradio
  python app.py
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import warnings
from typing import Any, Dict, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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


TRAIN_CACHE: Dict[str, Dict[str, Any]] = {}


def load_dataset(name: str) -> Dict[str, Any]:
    dataset_name = (name or "Breast Cancer").strip().lower()

    if dataset_name == "breast cancer":
        data = load_breast_cancer(as_frame=True)
        X = data.data.copy()
        y = pd.Series(data.target, name="target")
        target_names = ["Malignant", "Benign"]
        return {
            "name": "Breast Cancer",
            "X": X,
            "y": y,
            "feature_names": list(X.columns),
            "target_names": target_names,
            "num_classes": int(y.nunique()),
            "task_type": "binary",
            "image_shape": None,
        }

    if dataset_name == "digits":
        data = load_digits()
        feature_names = [f"px_{idx}" for idx in range(data.data.shape[1])]
        X = pd.DataFrame(data.data, columns=feature_names)
        y = pd.Series(data.target, name="digit")
        target_names = [str(label) for label in sorted(y.unique())]
        return {
            "name": "Digits",
            "X": X,
            "y": y,
            "feature_names": feature_names,
            "target_names": target_names,
            "num_classes": int(y.nunique()),
            "task_type": "multiclass",
            "image_shape": (8, 8),
        }

    raise ValueError(f"Unsupported dataset: {name}")


def _make_class_distribution_plot(y: pd.Series, title: str) -> plt.Figure:
    class_counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(class_counts.index.astype(str), class_counts.values)
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def _make_breast_corr_heatmap(X: pd.DataFrame, y: pd.Series, top_k: int) -> plt.Figure:
    joined = X.copy()
    joined["target"] = y
    corrs = joined.corr(numeric_only=True)["target"].drop("target").abs().sort_values(ascending=False)
    selected = list(corrs.head(max(2, top_k)).index)
    corr_matrix = X[selected].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr_matrix.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title(f"Correlation Heatmap (Top {len(selected)} Features)")
    ax.set_xticks(np.arange(len(selected)))
    ax.set_xticklabels(selected, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(selected)))
    ax.set_yticklabels(selected, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def _make_breast_feature_histograms(
    X: pd.DataFrame,
    y: pd.Series,
    feature_1: str,
    feature_2: str,
    target_names: list[str],
) -> plt.Figure:
    f1 = feature_1 if feature_1 in X.columns else X.columns[0]
    f2 = feature_2 if feature_2 in X.columns else X.columns[min(1, len(X.columns) - 1)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    classes = sorted(y.unique())

    for cls in classes:
        cls_mask = y == cls
        label = target_names[int(cls)] if int(cls) < len(target_names) else str(cls)
        axes[0].hist(X.loc[cls_mask, f1], bins=30, alpha=0.6, label=label)
        axes[1].hist(X.loc[cls_mask, f2], bins=30, alpha=0.6, label=label)

    axes[0].set_title(f"Histogram: {f1}")
    axes[1].set_title(f"Histogram: {f2}")
    for axis in axes:
        axis.set_xlabel("Value")
        axis.set_ylabel("Frequency")
        axis.grid(alpha=0.25)
    axes[1].legend(loc="best")
    fig.tight_layout()
    return fig


def _make_digits_image_grid(X: pd.DataFrame, y: pd.Series, image_shape: Tuple[int, int]) -> plt.Figure:
    fig, axes = plt.subplots(2, 5, figsize=(9, 4.5))
    axes = axes.ravel()

    for digit in range(10):
        idx = int(np.where(y.values == digit)[0][0])
        image = X.iloc[idx].values.reshape(image_shape)
        axes[digit].imshow(image, cmap="gray")
        axes[digit].set_title(f"Digit {digit}")
        axes[digit].axis("off")

    fig.suptitle("Digits Sample Images", y=1.02)
    fig.tight_layout()
    return fig


def _make_digits_pca_plot(X: pd.DataFrame, y: pd.Series, max_points: int = 1000) -> plt.Figure:
    total = len(X)
    if total > max_points:
        rng = np.random.default_rng(42)
        subset_idx = rng.choice(total, size=max_points, replace=False)
        X_sub = X.iloc[subset_idx]
        y_sub = y.iloc[subset_idx]
    else:
        X_sub = X
        y_sub = y

    pca = PCA(n_components=2, random_state=42)
    points = pca.fit_transform(X_sub)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for cls in sorted(y_sub.unique()):
        cls_mask = y_sub == cls
        ax.scatter(
            points[cls_mask, 0],
            points[cls_mask, 1],
            s=14,
            alpha=0.75,
            label=str(cls),
        )

    ax.set_title(f"PCA Projection (2D) - {len(X_sub)} Samples")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(title="Class", ncol=2, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def _make_digits_pixel_hist(X: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.hist(X.values.ravel(), bins=40, alpha=0.85)
    ax.set_title("Pixel Intensity Distribution")
    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def make_eda_plots(data_bundle: Dict[str, Any], eda_params: Dict[str, Any]) -> Dict[str, Any]:
    X = data_bundle["X"]
    y = data_bundle["y"]
    dataset_name = data_bundle["name"]

    class_counts = y.value_counts().sort_index()
    class_df = pd.DataFrame({"class": class_counts.index.astype(str), "count": class_counts.values})
    shape_summary = pd.DataFrame(
        [
            {
                "samples": int(X.shape[0]),
                "features": int(X.shape[1]),
                "classes": int(y.nunique()),
                "dataset": dataset_name,
            }
        ]
    )

    class_dist_fig = _make_class_distribution_plot(y, f"{dataset_name}: Class Distribution")

    if dataset_name == "Breast Cancer":
        top_k = int(eda_params.get("top_k", 10))
        top_n_stats = int(eda_params.get("top_n_stats", 10))
        feature_1 = eda_params.get("feature_1", X.columns[0])
        feature_2 = eda_params.get("feature_2", X.columns[min(1, len(X.columns) - 1)])

        relation_fig = _make_breast_corr_heatmap(X, y, top_k)
        extra_fig = _make_breast_feature_histograms(X, y, feature_1, feature_2, data_bundle["target_names"])

        stats_df = pd.DataFrame(
            {
                "feature": X.columns,
                "mean": X.mean(numeric_only=True).values,
                "std": X.std(numeric_only=True).values,
            }
        )
        stats_df["abs_mean"] = stats_df["mean"].abs()
        stats_df = stats_df.sort_values("abs_mean", ascending=False).drop(columns=["abs_mean"]).head(top_n_stats)

        flat_preview = X.head(5).copy()
    else:
        relation_fig = _make_digits_image_grid(X, y, data_bundle["image_shape"])
        extra_fig = _make_digits_pca_plot(X, y, max_points=1000)

        pixel_hist_fig = _make_digits_pixel_hist(X)
        stats_df = class_df.copy()
        stats_df.columns = ["digit", "count"]

        flat_preview = X.head(5).copy()

        return {
            "shape_df": shape_summary,
            "class_df": class_df,
            "class_dist_fig": class_dist_fig,
            "relation_fig": relation_fig,
            "extra_fig": extra_fig,
            "summary_stats_df": stats_df,
            "flat_preview_df": flat_preview,
            "digits_pixel_hist_fig": pixel_hist_fig,
        }

    return {
        "shape_df": shape_summary,
        "class_df": class_df,
        "class_dist_fig": class_dist_fig,
        "relation_fig": relation_fig,
        "extra_fig": extra_fig,
        "summary_stats_df": stats_df,
        "flat_preview_df": flat_preview,
        "digits_pixel_hist_fig": None,
    }


def _parse_hidden_layers(hidden_layer_sizes: str) -> Tuple[int, ...]:
    text = str(hidden_layer_sizes).strip()
    if not text:
        raise ValueError("MLP hidden_layer_sizes cannot be empty.")
    values = [part.strip() for part in text.split(",") if part.strip()]
    try:
        parsed = tuple(int(part) for part in values)
    except ValueError as exc:
        raise ValueError("MLP hidden layers must be comma-separated integers, e.g. 64,32") from exc
    if any(v <= 0 for v in parsed):
        raise ValueError("MLP hidden layers must be positive integers.")
    return parsed


def build_models(config: Dict[str, Any], dataset_name: str) -> Dict[str, Pipeline]:
    standardize = bool(config["standardize_features"])
    use_pca_digits = bool(config["use_pca_digits"]) and dataset_name == "Digits"
    pca_components = int(config["pca_components"])

    logreg_steps = []
    mlp_steps = []

    if standardize:
        logreg_steps.append(("scaler", StandardScaler()))
        mlp_steps.append(("scaler", StandardScaler()))

    if use_pca_digits:
        logreg_steps.append(("pca", PCA(n_components=pca_components, random_state=int(config["random_seed"])) ))
        mlp_steps.append(("pca", PCA(n_components=pca_components, random_state=int(config["random_seed"])) ))

    logreg = LogisticRegression(
        C=float(config["logreg_c"]),
        max_iter=int(config["logreg_max_iter"]),
        random_state=int(config["random_seed"]),
        solver="lbfgs",
        n_jobs=-1,
    )
    logreg_steps.append(("model", logreg))

    rf = RandomForestClassifier(
        n_estimators=int(config["rf_n_estimators"]),
        max_depth=None if int(config["rf_max_depth"]) <= 0 else int(config["rf_max_depth"]),
        random_state=int(config["random_seed"]),
        n_jobs=-1,
    )

    gb = GradientBoostingClassifier(
        n_estimators=int(config["gb_n_estimators"]),
        learning_rate=float(config["gb_learning_rate"]),
        random_state=int(config["random_seed"]),
    )

    mlp = MLPClassifier(
        hidden_layer_sizes=_parse_hidden_layers(config["mlp_hidden_layer_sizes"]),
        alpha=float(config["mlp_alpha"]),
        max_iter=int(config["mlp_max_iter"]),
        early_stopping=bool(config["mlp_early_stopping"]),
        random_state=int(config["random_seed"]),
    )
    mlp_steps.append(("model", mlp))

    return {
        "Logistic Regression": Pipeline(logreg_steps),
        "Random Forest": Pipeline([("model", rf)]),
        "Gradient Boosting": Pipeline([("model", gb)]),
        "MLP": Pipeline(mlp_steps),
    }


def train_and_evaluate(
    models: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    dataset_name: str,
) -> Dict[str, Any]:
    average = "binary" if dataset_name == "Breast Cancer" else "weighted"
    rows = []
    details: Dict[str, Any] = {}
    warnings_log: list[str] = []

    for model_name, model in models.items():
        estimator = clone(model)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.fit(X_train, y_train)

        for item in caught:
            message = f"[{model_name}] {item.category.__name__}: {item.message}"
            warnings_log.append(message)

        y_pred = estimator.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            y_pred,
            average=average,
            zero_division=0,
        )

        row = {
            "Model": model_name,
            "Accuracy": float(accuracy),
            "Precision": float(precision),
            "Recall": float(recall),
            "F1": float(f1),
        }

        auc_value = np.nan
        roc_payload = None
        if dataset_name == "Breast Cancer":
            score_values = None
            if hasattr(estimator, "predict_proba"):
                probs = estimator.predict_proba(X_test)
                if probs.ndim == 2 and probs.shape[1] >= 2:
                    score_values = probs[:, 1]
            elif hasattr(estimator, "decision_function"):
                score_values = estimator.decision_function(X_test)

            if score_values is not None:
                fpr, tpr, _ = roc_curve(y_test, score_values)
                auc_value = float(roc_auc_score(y_test, score_values))
                roc_payload = {"fpr": fpr, "tpr": tpr, "auc": auc_value}
                row["ROC_AUC"] = auc_value
            else:
                row["ROC_AUC"] = np.nan

        rows.append(row)

        report = classification_report(
            y_test,
            y_pred,
            target_names=[str(t) for t in sorted(y_test.unique())] if dataset_name == "Digits" else ["Malignant", "Benign"],
            zero_division=0,
        )

        details[model_name] = {
            "y_true": y_test.to_numpy(),
            "y_pred": np.asarray(y_pred),
            "report": report,
            "confusion": confusion_matrix(y_test, y_pred),
            "roc": roc_payload,
        }

    metrics_df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)

    metric_cols = ["Accuracy", "Precision", "Recall", "F1"]
    metrics_for_rank = metrics_df[metric_cols].copy()
    metrics_df["Overall"] = metrics_for_rank.mean(axis=1)

    best_per_metric = {}
    for metric in metric_cols:
        best_per_metric[metric] = metrics_df.loc[metrics_df[metric].idxmax(), "Model"]
    best_overall = metrics_df.loc[metrics_df["Overall"].idxmax(), "Model"]

    return {
        "metrics_df": metrics_df,
        "details": details,
        "warnings": warnings_log,
        "best_per_metric": best_per_metric,
        "best_overall": best_overall,
    }


def _build_confusion_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[Any],
    title: str,
    normalize: bool = False,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4.2))
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
    return fig


def _build_roc_figure(details: Dict[str, Any]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for model_name, payload in details.items():
        roc = payload.get("roc")
        if roc is None:
            continue
        ax.plot(roc["fpr"], roc["tpr"], label=f"{model_name} (AUC={roc['auc']:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.7)
    ax.set_title("ROC Curves (Breast Cancer)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def render_results(result_bundle: Dict[str, Any], selected_model: str, dataset_name: str) -> Dict[str, Any]:
    details = result_bundle["details"]
    if selected_model not in details:
        selected_model = next(iter(details.keys()))

    payload = details[selected_model]
    labels = sorted(np.unique(payload["y_true"]))

    conf_fig = _build_confusion_figure(
        payload["y_true"],
        payload["y_pred"],
        labels=labels,
        title=f"Confusion Matrix - {selected_model}",
        normalize=False,
    )

    norm_conf_fig = None
    roc_fig = None
    if dataset_name == "Digits":
        norm_conf_fig = _build_confusion_figure(
            payload["y_true"],
            payload["y_pred"],
            labels=labels,
            title=f"Normalized Confusion Matrix - {selected_model}",
            normalize=True,
        )
    else:
        roc_fig = _build_roc_figure(details)

    best = result_bundle["best_per_metric"]
    best_summary = (
        f"Best Accuracy: {best['Accuracy']} | "
        f"Best Precision: {best['Precision']} | "
        f"Best Recall: {best['Recall']} | "
        f"Best F1: {best['F1']} | "
        f"Best Overall: {result_bundle['best_overall']}"
    )

    display_df = result_bundle["metrics_df"].copy()
    metric_cols = ["Accuracy", "Precision", "Recall", "F1", "Overall"]
    for column in metric_cols:
        display_df[column] = display_df[column].map(lambda value: round(float(value), 4))

    return {
        "metrics_df": display_df,
        "conf_fig": conf_fig,
        "report_text": payload["report"],
        "roc_fig": roc_fig,
        "norm_conf_fig": norm_conf_fig,
        "best_summary": best_summary,
    }


def _normalize_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(config)
    normalized["dataset"] = str(normalized["dataset"])
    normalized["random_seed"] = int(normalized["random_seed"])
    normalized["test_size"] = float(normalized["test_size"])
    normalized["standardize_features"] = bool(normalized["standardize_features"])
    normalized["use_pca_digits"] = bool(normalized["use_pca_digits"])
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


def _hash_config(config: Dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_config(config: Dict[str, Any]) -> None:
    if not (0.1 <= config["test_size"] <= 0.4):
        raise ValueError("Test size must be between 0.1 and 0.4.")
    if config["dataset"] == "Digits" and config["use_pca_digits"]:
        if not (10 <= config["pca_components"] <= 64):
            raise ValueError("For Digits PCA, n_components must be between 10 and 64.")
    if config["dataset"] == "Breast Cancer" and config["use_pca_digits"]:
        raise ValueError("Use PCA for Digits applies only to the Digits dataset.")
    _parse_hidden_layers(config["mlp_hidden_layer_sizes"])


def _apply_fast_mode(config: Dict[str, Any]) -> Dict[str, Any]:
    updated = dict(config)
    if not updated["fast_mode"]:
        return updated

    updated["logreg_max_iter"] = min(updated["logreg_max_iter"], 300)
    updated["rf_n_estimators"] = min(updated["rf_n_estimators"], 120)
    updated["gb_n_estimators"] = min(updated["gb_n_estimators"], 120)
    updated["mlp_max_iter"] = min(updated["mlp_max_iter"], 200)
    if updated["dataset"] == "Digits":
        updated["use_pca_digits"] = True
        updated["pca_components"] = min(max(updated["pca_components"], 10), 30)
    return updated


def _save_exports(metrics_df: pd.DataFrame, config_json: str) -> Tuple[str, str]:
    metrics_tmp = tempfile.NamedTemporaryFile(mode="w", suffix="_metrics.csv", delete=False)
    config_tmp = tempfile.NamedTemporaryFile(mode="w", suffix="_config.json", delete=False)

    metrics_df.to_csv(metrics_tmp.name, index=False)
    metrics_tmp.close()

    with open(config_tmp.name, "w", encoding="utf-8") as file:
        file.write(config_json)

    return metrics_tmp.name, config_tmp.name


def _dataset_summary_markdown(data_bundle: Dict[str, Any]) -> str:
    X = data_bundle["X"]
    y = data_bundle["y"]
    return (
        f"### Dataset Summary\n"
        f"- Dataset: **{data_bundle['name']}**\n"
        f"- Samples: **{len(X)}**\n"
        f"- Features: **{X.shape[1]}**\n"
        f"- Classes: **{y.nunique()}**\n"
    )


def refresh_dataset_and_eda(
    dataset_name: str,
    top_k: int,
    feature_1: str,
    feature_2: str,
    top_n_stats: int,
) -> Tuple[Any, ...]:
    data_bundle = load_dataset(dataset_name)
    feature_names = data_bundle["feature_names"]

    if data_bundle["name"] == "Breast Cancer":
        feature_1 = feature_1 if feature_1 in feature_names else feature_names[0]
        feature_2 = feature_2 if feature_2 in feature_names else feature_names[min(1, len(feature_names) - 1)]
        feature_1_update = gr.update(choices=feature_names, value=feature_1, interactive=True)
        feature_2_update = gr.update(choices=feature_names, value=feature_2, interactive=True)
    else:
        feature_1_update = gr.update(choices=["N/A"], value="N/A", interactive=False)
        feature_2_update = gr.update(choices=["N/A"], value="N/A", interactive=False)

    eda = make_eda_plots(
        data_bundle,
        {
            "top_k": top_k,
            "feature_1": feature_1,
            "feature_2": feature_2,
            "top_n_stats": top_n_stats,
        },
    )

    digits_pixel_fig = eda["digits_pixel_hist_fig"] if data_bundle["name"] == "Digits" else None

    return (
        _dataset_summary_markdown(data_bundle),
        eda["shape_df"],
        eda["class_df"],
        eda["class_dist_fig"],
        eda["relation_fig"],
        eda["extra_fig"],
        digits_pixel_fig,
        eda["summary_stats_df"],
        eda["flat_preview_df"],
        feature_1_update,
        feature_2_update,
    )


def run_training(
    dataset: str,
    test_size: float,
    random_seed: int,
    standardize_features: bool,
    use_pca_digits: bool,
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
) -> Tuple[Any, ...]:
    raw_config = {
        "dataset": dataset,
        "test_size": test_size,
        "random_seed": random_seed,
        "standardize_features": standardize_features,
        "use_pca_digits": use_pca_digits,
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

    try:
        config = _normalize_training_config(raw_config)
        config = _apply_fast_mode(config)
        _validate_config(config)
    except Exception as exc:
        empty_df = pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall", "F1", "Overall"])
        error_msg = f"Validation error: {exc}"
        return (
            empty_df,
            "",
            error_msg,
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

        metrics_path, config_path = _save_exports(rendered["metrics_df"], config_json)
        status_text = "Cache hit: reused previous training results."
        warnings_text = "\n".join(cached["result_bundle"]["warnings"]) if cached["result_bundle"]["warnings"] else "No warnings."

        progress(1.0, desc="Done (cached)")
        return (
            rendered["metrics_df"],
            rendered["best_summary"],
            status_text,
            config_json,
            gr.update(choices=list(cached["result_bundle"]["details"].keys()), value=cached["selected_model"]),
            rendered["conf_fig"],
            rendered["report_text"],
            gr.update(visible=(config["dataset"] == "Breast Cancer"), value=rendered["roc_fig"]),
            gr.update(visible=(config["dataset"] == "Digits"), value=rendered["norm_conf_fig"]),
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
    result_bundle = train_and_evaluate(
        models=models,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        dataset_name=config["dataset"],
    )

    selected_model = result_bundle["best_overall"]
    rendered = render_results(result_bundle, selected_model, config["dataset"])

    metrics_path, config_path = _save_exports(rendered["metrics_df"], config_json)

    TRAIN_CACHE[cache_key] = {
        "result_bundle": result_bundle,
        "selected_model": selected_model,
    }

    warnings_text = "\n".join(result_bundle["warnings"]) if result_bundle["warnings"] else "No warnings."
    status_text = "Training completed and cached."

    progress(1.0, desc="Done")
    return (
        rendered["metrics_df"],
        rendered["best_summary"],
        status_text,
        config_json,
        gr.update(choices=list(result_bundle["details"].keys()), value=selected_model),
        rendered["conf_fig"],
        rendered["report_text"],
        gr.update(visible=(config["dataset"] == "Breast Cancer"), value=rendered["roc_fig"]),
        gr.update(visible=(config["dataset"] == "Digits"), value=rendered["norm_conf_fig"]),
        metrics_path,
        config_path,
        result_bundle,
    )


def on_model_selection_change(
    selected_model: str,
    dataset_name: str,
    result_state: Dict[str, Any],
) -> Tuple[Any, ...]:
    if not result_state:
        return None, "", gr.update(visible=False, value=None), gr.update(visible=False, value=None)

    rendered = render_results(result_state, selected_model, dataset_name)

    return (
        rendered["conf_fig"],
        rendered["report_text"],
        gr.update(visible=(dataset_name == "Breast Cancer"), value=rendered["roc_fig"]),
        gr.update(visible=(dataset_name == "Digits"), value=rendered["norm_conf_fig"]),
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Classical ML Model Explorer") as demo:
        gr.Markdown("# Classical ML Model Explorer")

        result_state = gr.State({})

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Dataset")
                dataset = gr.Radio(
                    choices=["Breast Cancer", "Digits"],
                    value="Breast Cancer",
                    label="Select Dataset",
                )
                dataset_summary_md = gr.Markdown()
                shape_df = gr.Dataframe(label="Dataset Shape Summary", interactive=False)
                class_df = gr.Dataframe(label="Class Counts", interactive=False)

                gr.Markdown("## EDA / Visualizations")
                top_k = gr.Slider(2, 20, value=10, step=1, label="Breast Cancer: Top-k Correlated Features")
                feature_1 = gr.Dropdown(choices=[], label="Breast Cancer: Histogram Feature 1")
                feature_2 = gr.Dropdown(choices=[], label="Breast Cancer: Histogram Feature 2")
                top_n_stats = gr.Slider(5, 20, value=10, step=1, label="Summary Stats: Top N Features")

            with gr.Column(scale=2):
                class_dist_fig = gr.Plot(label="Class Distribution")
                relation_fig = gr.Plot(label="Relationship Plot")
                extra_fig = gr.Plot(label="Additional EDA Plot")
                digits_pixel_fig = gr.Plot(label="Digits: Pixel Intensity Histogram")
                summary_stats_df = gr.Dataframe(label="Summary Statistics", interactive=False)
                flat_preview_df = gr.Dataframe(label="Flattened Feature Preview", interactive=False)

        with gr.Row():
            gr.Markdown("## Training")

        with gr.Row():
            test_size = gr.Slider(0.1, 0.4, value=0.2, step=0.01, label="Test Size")
            random_seed = gr.Number(value=42, precision=0, label="Random Seed")
            standardize_features = gr.Checkbox(value=True, label="Standardize Features")
            fast_mode = gr.Checkbox(value=False, label="Fast Mode")

        with gr.Row():
            use_pca_digits = gr.Checkbox(value=False, label="Use PCA for Digits (LogReg/MLP only)")
            pca_components = gr.Slider(10, 64, value=32, step=1, label="Digits PCA Components")

        with gr.Row():
            logreg_c = gr.Slider(0.01, 10.0, value=1.0, step=0.01, label="Logistic Regression: C")
            logreg_max_iter = gr.Slider(100, 2000, value=500, step=50, label="Logistic Regression: max_iter")

        with gr.Row():
            rf_n_estimators = gr.Slider(50, 1000, value=300, step=10, label="Random Forest: n_estimators")
            rf_max_depth = gr.Slider(0, 40, value=0, step=1, label="Random Forest: max_depth (0=None)")

        with gr.Row():
            gb_n_estimators = gr.Slider(50, 1000, value=300, step=10, label="Gradient Boosting: n_estimators")
            gb_learning_rate = gr.Slider(0.01, 0.5, value=0.05, step=0.01, label="Gradient Boosting: learning_rate")

        with gr.Row():
            mlp_hidden_layer_sizes = gr.Textbox(value="64,32", label="MLP: hidden_layer_sizes")
            mlp_alpha = gr.Slider(1e-6, 1e-1, value=1e-4, step=1e-6, label="MLP: alpha")
            mlp_max_iter = gr.Slider(100, 2000, value=600, step=50, label="MLP: max_iter")
            mlp_early_stopping = gr.Checkbox(value=True, label="MLP: early_stopping")

        train_btn = gr.Button("Train & Evaluate", variant="primary")

        gr.Markdown("## Results")
        model_selector = gr.Dropdown(choices=[], label="Select Model for Detailed Results")
        conf_fig = gr.Plot(label="Confusion Matrix")
        classification_report_box = gr.Textbox(label="Classification Report", lines=14)

        roc_fig = gr.Plot(label="Breast Cancer: ROC Curves", visible=False)
        norm_conf_fig = gr.Plot(label="Digits: Normalized Confusion Matrix", visible=False)

        gr.Markdown("## Model Comparison")
        metrics_df = gr.Dataframe(label="Metrics Table", interactive=False)
        best_summary_box = gr.Textbox(label="Best Model Summary")

        gr.Markdown("## Downloads/Logs")
        status_box = gr.Textbox(label="Status / Progress", lines=2)
        config_box = gr.Textbox(label="Run Configuration (JSON)", lines=14)
        metrics_download = gr.File(label="Download Metrics CSV")
        config_download = gr.File(label="Download Config JSON")

        def _refresh_standardize_default(ds_name: str):
            default_value = True
            return gr.update(value=default_value)

        dataset.change(
            fn=_refresh_standardize_default,
            inputs=[dataset],
            outputs=[standardize_features],
        )

        dataset.change(
            fn=refresh_dataset_and_eda,
            inputs=[dataset, top_k, feature_1, feature_2, top_n_stats],
            outputs=[
                dataset_summary_md,
                shape_df,
                class_df,
                class_dist_fig,
                relation_fig,
                extra_fig,
                digits_pixel_fig,
                summary_stats_df,
                flat_preview_df,
                feature_1,
                feature_2,
            ],
        )

        top_k.change(
            fn=refresh_dataset_and_eda,
            inputs=[dataset, top_k, feature_1, feature_2, top_n_stats],
            outputs=[
                dataset_summary_md,
                shape_df,
                class_df,
                class_dist_fig,
                relation_fig,
                extra_fig,
                digits_pixel_fig,
                summary_stats_df,
                flat_preview_df,
                feature_1,
                feature_2,
            ],
        )

        feature_1.change(
            fn=refresh_dataset_and_eda,
            inputs=[dataset, top_k, feature_1, feature_2, top_n_stats],
            outputs=[
                dataset_summary_md,
                shape_df,
                class_df,
                class_dist_fig,
                relation_fig,
                extra_fig,
                digits_pixel_fig,
                summary_stats_df,
                flat_preview_df,
                feature_1,
                feature_2,
            ],
        )

        feature_2.change(
            fn=refresh_dataset_and_eda,
            inputs=[dataset, top_k, feature_1, feature_2, top_n_stats],
            outputs=[
                dataset_summary_md,
                shape_df,
                class_df,
                class_dist_fig,
                relation_fig,
                extra_fig,
                digits_pixel_fig,
                summary_stats_df,
                flat_preview_df,
                feature_1,
                feature_2,
            ],
        )

        top_n_stats.change(
            fn=refresh_dataset_and_eda,
            inputs=[dataset, top_k, feature_1, feature_2, top_n_stats],
            outputs=[
                dataset_summary_md,
                shape_df,
                class_df,
                class_dist_fig,
                relation_fig,
                extra_fig,
                digits_pixel_fig,
                summary_stats_df,
                flat_preview_df,
                feature_1,
                feature_2,
            ],
        )

        train_btn.click(
            fn=run_training,
            inputs=[
                dataset,
                test_size,
                random_seed,
                standardize_features,
                use_pca_digits,
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
                best_summary_box,
                status_box,
                config_box,
                model_selector,
                conf_fig,
                classification_report_box,
                roc_fig,
                norm_conf_fig,
                metrics_download,
                config_download,
                result_state,
            ],
        )

        model_selector.change(
            fn=on_model_selection_change,
            inputs=[model_selector, dataset, result_state],
            outputs=[conf_fig, classification_report_box, roc_fig, norm_conf_fig],
        )

        demo.load(
            fn=refresh_dataset_and_eda,
            inputs=[dataset, top_k, feature_1, feature_2, top_n_stats],
            outputs=[
                dataset_summary_md,
                shape_df,
                class_df,
                class_dist_fig,
                relation_fig,
                extra_fig,
                digits_pixel_fig,
                summary_stats_df,
                flat_preview_df,
                feature_1,
                feature_2,
            ],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
