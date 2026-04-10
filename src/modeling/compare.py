from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


def _slice_like(data, indices):
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    return data[indices]


def _clone_model(model):
    try:
        return clone(model)
    except RuntimeError:
        return copy.deepcopy(model)


def _safe_roc_auc(y_true, y_proba):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_proba)


def _safe_prc_auc(y_true, y_proba):
    if len(np.unique(y_true)) < 2:
        return np.nan

    # Keep the historical column name `prc_auc`, but align its value with
    # the project's tuning metric: sklearn average_precision.
    return average_precision_score(y_true, y_proba)


def _get_threshold_independent_metrics_from_proba(y_true, y_proba) -> dict:
    return {
        "roc_auc": _safe_roc_auc(y_true, y_proba),
        "prc_auc": _safe_prc_auc(y_true, y_proba),
    }


def _prefix_metrics(metrics: dict, prefix: str) -> dict:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def _attach_train_metrics(
    metrics: dict,
    model,
    X_train,
    y_train,
    threshold: float,
) -> dict:
    if X_train is None or y_train is None:
        return metrics

    train_metrics = evaluate_binary_classifier(
        model=model,
        X_eval=X_train,
        y_eval=y_train,
        threshold=threshold,
    )
    metrics.update(_prefix_metrics(train_metrics, "train_"))
    return metrics


def _build_threshold_unavailable_result(
    model_name: str,
    model,
    X_test,
    y_test,
    X_train=None,
    y_train=None,
) -> dict:
    y_test_proba = model.predict_proba(X_test)[:, 1]

    result = {
        "model": model_name,
        "best_threshold": None,
        "precision_1": None,
        "recall_1": None,
        "f1_1": None,
        "tn": None,
        "fp": None,
        "fn": None,
        "tp": None,
    }
    result.update(_get_threshold_independent_metrics_from_proba(y_test, y_test_proba))

    if X_train is not None and y_train is not None:
        y_train_proba = model.predict_proba(X_train)[:, 1]
        result.update(
            {
                "train_accuracy": None,
                "train_precision_1": None,
                "train_recall_1": None,
                "train_f1_1": None,
                "train_tn": None,
                "train_fp": None,
                "train_fn": None,
                "train_tp": None,
                "train_roc_auc": _safe_roc_auc(y_train, y_train_proba),
                "train_prc_auc": _safe_prc_auc(y_train, y_train_proba),
            }
        )

    return result


def evaluate_binary_classifier(model, X_eval, y_eval, threshold=0.5) -> dict:
    y_proba = model.predict_proba(X_eval)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_eval, y_pred, labels=[0, 1]).ravel()

    return {
        "accuracy": accuracy_score(y_eval, y_pred),
        "precision_1": precision_score(y_eval, y_pred, zero_division=0),
        "recall_1": recall_score(y_eval, y_pred, zero_division=0),
        "f1_1": f1_score(y_eval, y_pred, zero_division=0),
        "roc_auc": _safe_roc_auc(y_eval, y_proba),
        "prc_auc": _safe_prc_auc(y_eval, y_proba),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def find_best_threshold(model, X_eval, y_eval, metric="f1"):
    y_proba = model.predict_proba(X_eval)[:, 1]

    best_thresh = 0.5
    best_score = -1

    for threshold in np.linspace(0.01, 0.99, 999):
        y_pred = (y_proba >= threshold).astype(int)

        if metric == "f1":
            score = f1_score(y_eval, y_pred, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_eval, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_eval, y_pred, zero_division=0)
        else:
            raise ValueError(f"Metrique non supportee : {metric}")

        if score > best_score:
            best_score = score
            best_thresh = threshold

    return best_thresh, best_score


def get_precision_recall_thresholds_from_proba(y_true, y_proba) -> pd.DataFrame:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    precision = precision[:-1]
    recall = recall[:-1]
    f1 = 2 * (precision * recall) / (precision + recall + 1e-12)

    return pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


def get_precision_recall_thresholds(model, X_eval, y_eval) -> pd.DataFrame:
    y_proba = model.predict_proba(X_eval)[:, 1]
    return get_precision_recall_thresholds_from_proba(y_true=y_eval, y_proba=y_proba)


def find_best_threshold_from_pr_curve(model, X_eval, y_eval, metric="f1") -> dict:
    pr_df = get_precision_recall_thresholds(model, X_eval, y_eval)

    if metric not in pr_df.columns:
        raise ValueError(f"Metrique non supportee : {metric}")

    best_idx = pr_df[metric].idxmax()
    best_row = pr_df.loc[best_idx]

    return {
        "best_threshold": float(best_row["threshold"]),
        "best_precision": float(best_row["precision"]),
        "best_recall": float(best_row["recall"]),
        "best_f1": float(best_row["f1"]),
    }


def find_threshold_for_target_recall(model, X_eval, y_eval, target_recall=0.80):
    y_proba = model.predict_proba(X_eval)[:, 1]
    return find_threshold_for_target_recall_from_proba(
        y_true=y_eval,
        y_proba=y_proba,
        target_recall=target_recall,
    )


def find_best_threshold_from_proba(
    y_true,
    y_proba,
    metric="f1",
) -> dict:
    pr_df = get_precision_recall_thresholds_from_proba(y_true=y_true, y_proba=y_proba)

    if metric not in pr_df.columns:
        raise ValueError(f"Metrique non supportee : {metric}")

    best_idx = pr_df[metric].idxmax()
    best_row = pr_df.loc[best_idx]

    return {
        "best_threshold": float(best_row["threshold"]),
        "best_precision": float(best_row["precision"]),
        "best_recall": float(best_row["recall"]),
        "best_f1": float(best_row["f1"]),
    }


def find_threshold_for_target_recall_from_proba(
    y_true,
    y_proba,
    target_recall=0.80,
):
    pr_df = get_precision_recall_thresholds_from_proba(y_true=y_true, y_proba=y_proba)
    candidates = pr_df[pr_df["recall"] >= target_recall].copy()

    if candidates.empty:
        return None

    best_row = candidates.sort_values("threshold", ascending=False).iloc[0]

    return {
        "best_threshold": float(best_row["threshold"]),
        "precision": float(best_row["precision"]),
        "recall": float(best_row["recall"]),
        "f1": float(best_row["f1"]),
    }


def get_oof_predicted_proba(
    model,
    X,
    y,
    fit_kwargs: dict | None = None,
    cv: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    fit_kwargs = {} if fit_kwargs is None else dict(fit_kwargs)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    y_array = np.asarray(y)
    oof_proba = np.full(len(y_array), np.nan, dtype=float)

    for train_idx, valid_idx in skf.split(X, y_array):
        X_train_fold = _slice_like(X, train_idx)
        X_valid_fold = _slice_like(X, valid_idx)
        y_train_fold = _slice_like(y, train_idx)

        model_fold = _clone_model(model)
        model_fold.fit(X_train_fold, y_train_fold, **fit_kwargs)
        oof_proba[valid_idx] = model_fold.predict_proba(X_valid_fold)[:, 1]

    if np.isnan(oof_proba).any():
        raise ValueError("Certaines probabilites OOF n'ont pas ete calculees.")

    return oof_proba


def get_oof_predicted_proba_by_model_specs(
    model_specs: dict,
    X,
    y,
    cv: int = 5,
    random_state: int = 42,
) -> dict[str, np.ndarray]:
    return {
        model_name: get_oof_predicted_proba(
            model=spec["model"],
            X=X,
            y=y,
            fit_kwargs=spec.get("fit_kwargs", {}),
            cv=cv,
            random_state=random_state,
        )
        for model_name, spec in model_specs.items()
    }


def compare_models(
    trained_models,
    X_test,
    y_test,
    threshold=0.5,
    sort_by="roc_auc",
    X_train=None,
    y_train=None,
):
    results = []

    for model_name, model in trained_models.items():
        metrics = evaluate_binary_classifier(
            model=model,
            X_eval=X_test,
            y_eval=y_test,
            threshold=threshold,
        )
        metrics = _attach_train_metrics(
            metrics=metrics,
            model=model,
            X_train=X_train,
            y_train=y_train,
            threshold=threshold,
        )
        metrics["model"] = model_name
        metrics["threshold"] = threshold
        results.append(metrics)

    results_df = pd.DataFrame(results)

    if "model" in results_df.columns:
        results_df.insert(0, "model", results_df.pop("model"))

    if sort_by in results_df.columns:
        results_df = results_df.sort_values(sort_by, ascending=False)

    return results_df.reset_index(drop=True)


def compare_models_with_optimal_threshold(
    trained_models,
    X_test,
    y_test,
    metric="f1",
    sort_by="f1_1",
    X_train=None,
    y_train=None,
):
    results = []

    for model_name, model in trained_models.items():
        best_thresh, best_score = find_best_threshold(
            model=model,
            X_eval=X_test,
            y_eval=y_test,
            metric=metric,
        )

        metrics = evaluate_binary_classifier(
            model=model,
            X_eval=X_test,
            y_eval=y_test,
            threshold=best_thresh,
        )
        metrics = _attach_train_metrics(
            metrics=metrics,
            model=model,
            X_train=X_train,
            y_train=y_train,
            threshold=best_thresh,
        )

        metrics["model"] = model_name
        metrics["best_threshold"] = best_thresh
        metrics[f"{metric}_optimized"] = best_score
        results.append(metrics)

    results_df = pd.DataFrame(results)

    if sort_by in results_df.columns:
        results_df = results_df.sort_values(sort_by, ascending=False)

    if "model" in results_df.columns:
        results_df.insert(0, "model", results_df.pop("model"))

    return results_df.reset_index(drop=True)


def compare_models_with_pr_optimal_threshold(
    trained_models,
    X_test,
    y_test,
    metric="f1",
    sort_by="f1_1",
    X_train=None,
    y_train=None,
):
    results = []

    for model_name, model in trained_models.items():
        best_info = find_best_threshold_from_pr_curve(
            model=model,
            X_eval=X_test,
            y_eval=y_test,
            metric=metric,
        )

        metrics = evaluate_binary_classifier(
            model=model,
            X_eval=X_test,
            y_eval=y_test,
            threshold=best_info["best_threshold"],
        )
        metrics = _attach_train_metrics(
            metrics=metrics,
            model=model,
            X_train=X_train,
            y_train=y_train,
            threshold=best_info["best_threshold"],
        )

        metrics["model"] = model_name
        metrics["best_threshold"] = best_info["best_threshold"]
        metrics["best_precision_curve"] = best_info["best_precision"]
        metrics["best_recall_curve"] = best_info["best_recall"]
        metrics["best_f1_curve"] = best_info["best_f1"]
        results.append(metrics)

    results_df = pd.DataFrame(results)

    if sort_by in results_df.columns:
        results_df = results_df.sort_values(sort_by, ascending=False)

    if "model" in results_df.columns:
        results_df.insert(0, "model", results_df.pop("model"))

    return results_df.reset_index(drop=True)


def compare_models_with_target_recall(
    trained_models,
    X_test,
    y_test,
    target_recall=0.80,
    sort_by="precision_1",
    X_train=None,
    y_train=None,
):
    results = []

    for model_name, model in trained_models.items():
        threshold_info = find_threshold_for_target_recall(
            model=model,
            X_eval=X_test,
            y_eval=y_test,
            target_recall=target_recall,
        )

        if threshold_info is None:
            result = _build_threshold_unavailable_result(
                model_name=model_name,
                model=model,
                X_test=X_test,
                y_test=y_test,
                X_train=X_train,
                y_train=y_train,
            )
            result["target_recall"] = target_recall
            results.append(result)
            continue

        metrics = evaluate_binary_classifier(
            model=model,
            X_eval=X_test,
            y_eval=y_test,
            threshold=threshold_info["best_threshold"],
        )
        metrics = _attach_train_metrics(
            metrics=metrics,
            model=model,
            X_train=X_train,
            y_train=y_train,
            threshold=threshold_info["best_threshold"],
        )

        metrics["model"] = model_name
        metrics["target_recall"] = target_recall
        metrics["best_threshold"] = threshold_info["best_threshold"]
        results.append(metrics)

    results_df = pd.DataFrame(results)

    if sort_by in results_df.columns:
        results_df = results_df.sort_values(sort_by, ascending=False)

    if "model" in results_df.columns:
        results_df.insert(0, "model", results_df.pop("model"))

    return results_df.reset_index(drop=True)


def compare_models_with_cv_pr_optimal_threshold(
    trained_models,
    X_train,
    y_train,
    X_test,
    y_test,
    metric="f1",
    sort_by="f1_1",
    model_specs: dict | None = None,
    oof_proba_by_model: dict[str, np.ndarray] | None = None,
    cv: int = 5,
    random_state: int = 42,
):
    if oof_proba_by_model is None:
        if model_specs is None:
            raise ValueError("model_specs est requis si oof_proba_by_model n'est pas fourni.")
        oof_proba_by_model = get_oof_predicted_proba_by_model_specs(
            model_specs=model_specs,
            X=X_train,
            y=y_train,
            cv=cv,
            random_state=random_state,
        )

    results = []

    for model_name, model in trained_models.items():
        if model_name not in oof_proba_by_model:
            raise ValueError(f"Probabilites OOF manquantes pour le modele : {model_name}")

        best_info = find_best_threshold_from_proba(
            y_true=y_train,
            y_proba=oof_proba_by_model[model_name],
            metric=metric,
        )

        metrics = evaluate_binary_classifier(
            model=model,
            X_eval=X_test,
            y_eval=y_test,
            threshold=best_info["best_threshold"],
        )
        metrics = _attach_train_metrics(
            metrics=metrics,
            model=model,
            X_train=X_train,
            y_train=y_train,
            threshold=best_info["best_threshold"],
        )

        metrics["model"] = model_name
        metrics["best_threshold"] = best_info["best_threshold"]
        metrics["cv_best_precision_curve"] = best_info["best_precision"]
        metrics["cv_best_recall_curve"] = best_info["best_recall"]
        metrics["cv_best_f1_curve"] = best_info["best_f1"]
        results.append(metrics)

    results_df = pd.DataFrame(results)

    if sort_by in results_df.columns:
        results_df = results_df.sort_values(sort_by, ascending=False)

    if "model" in results_df.columns:
        results_df.insert(0, "model", results_df.pop("model"))

    return results_df.reset_index(drop=True)


def compare_models_with_cv_target_recall(
    trained_models,
    X_train,
    y_train,
    X_test,
    y_test,
    target_recall=0.80,
    sort_by="precision_1",
    model_specs: dict | None = None,
    oof_proba_by_model: dict[str, np.ndarray] | None = None,
    cv: int = 5,
    random_state: int = 42,
):
    if oof_proba_by_model is None:
        if model_specs is None:
            raise ValueError("model_specs est requis si oof_proba_by_model n'est pas fourni.")
        oof_proba_by_model = get_oof_predicted_proba_by_model_specs(
            model_specs=model_specs,
            X=X_train,
            y=y_train,
            cv=cv,
            random_state=random_state,
        )

    results = []

    for model_name, model in trained_models.items():
        if model_name not in oof_proba_by_model:
            raise ValueError(f"Probabilites OOF manquantes pour le modele : {model_name}")

        threshold_info = find_threshold_for_target_recall_from_proba(
            y_true=y_train,
            y_proba=oof_proba_by_model[model_name],
            target_recall=target_recall,
        )

        if threshold_info is None:
            result = _build_threshold_unavailable_result(
                model_name=model_name,
                model=model,
                X_test=X_test,
                y_test=y_test,
                X_train=X_train,
                y_train=y_train,
            )
            result["target_recall"] = target_recall
            results.append(result)
            continue

        metrics = evaluate_binary_classifier(
            model=model,
            X_eval=X_test,
            y_eval=y_test,
            threshold=threshold_info["best_threshold"],
        )
        metrics = _attach_train_metrics(
            metrics=metrics,
            model=model,
            X_train=X_train,
            y_train=y_train,
            threshold=threshold_info["best_threshold"],
        )

        metrics["model"] = model_name
        metrics["target_recall"] = target_recall
        metrics["best_threshold"] = threshold_info["best_threshold"]
        metrics["cv_precision_curve"] = threshold_info["precision"]
        metrics["cv_recall_curve"] = threshold_info["recall"]
        metrics["cv_f1_curve"] = threshold_info["f1"]
        results.append(metrics)

    results_df = pd.DataFrame(results)

    if sort_by in results_df.columns:
        results_df = results_df.sort_values(sort_by, ascending=False)

    if "model" in results_df.columns:
        results_df.insert(0, "model", results_df.pop("model"))

    return results_df.reset_index(drop=True)


def cross_validate_model_specs(
    model_specs: dict,
    X,
    y,
    cv: int = 5,
    threshold: float = 0.5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    y_array = np.asarray(y)
    fold_rows = []

    for model_name, spec in model_specs.items():
        fit_kwargs = dict(spec.get("fit_kwargs", {}))

        for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y_array), start=1):
            X_train_fold = _slice_like(X, train_idx)
            X_valid_fold = _slice_like(X, valid_idx)
            y_train_fold = _slice_like(y, train_idx)
            y_valid_fold = _slice_like(y, valid_idx)

            model = _clone_model(spec["model"])
            model.fit(X_train_fold, y_train_fold, **fit_kwargs)

            train_metrics = evaluate_binary_classifier(
                model=model,
                X_eval=X_train_fold,
                y_eval=y_train_fold,
                threshold=threshold,
            )
            valid_metrics = evaluate_binary_classifier(
                model=model,
                X_eval=X_valid_fold,
                y_eval=y_valid_fold,
                threshold=threshold,
            )

            row = {
                "model": model_name,
                "fold": fold_idx,
                "threshold": threshold,
            }
            row.update(_prefix_metrics(train_metrics, "train_"))
            row.update(_prefix_metrics(valid_metrics, "valid_"))
            fold_rows.append(row)

    folds_df = pd.DataFrame(fold_rows)
    if folds_df.empty:
        return folds_df, pd.DataFrame()

    metric_columns = [
        column
        for column in folds_df.columns
        if column not in {"model", "fold", "threshold"}
    ]

    summary_rows = []
    for model_name, group in folds_df.groupby("model", sort=False):
        summary_row = {
            "model": model_name,
            "cv_folds": cv,
            "threshold": threshold,
        }
        for column in metric_columns:
            summary_row[f"{column}_mean"] = group[column].mean()
            summary_row[f"{column}_std"] = group[column].std(ddof=0)
        summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows)
    return folds_df, summary_df
