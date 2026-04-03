import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    auc
)


def evaluate_binary_classifier(model, X_test, y_test, threshold=0.5) -> dict:

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    precision_test, recall_test, thresholds_test = precision_recall_curve(y_test, y_proba)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_1": precision_score(y_test, y_pred, zero_division=0),
        "recall_1": recall_score(y_test, y_pred, zero_division=0),
        "f1_1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "prc_auc": auc(recall_test, precision_test),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def find_best_threshold(model, X_test, y_test, metric="f1"):
    """
    Recherche du meilleur seuil par balayage simple.
    """
    y_proba = model.predict_proba(X_test)[:, 1]

    best_thresh = 0.5
    best_score = -1

    for t in np.linspace(0.01, 0.99, 999):
        y_pred = (y_proba >= t).astype(int)

        if metric == "f1":
            score = f1_score(y_test, y_pred, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_test, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_test, y_pred, zero_division=0)
        else:
            raise ValueError(f"Métrique non supportée : {metric}")

        if score > best_score:
            best_score = score
            best_thresh = t

    return best_thresh, best_score


def get_precision_recall_thresholds(model, X_test, y_test) -> pd.DataFrame:
    """
    Retourne un DataFrame avec threshold / precision / recall / f1
    à partir de precision_recall_curve.
    """
    y_proba = model.predict_proba(X_test)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

    # precision et recall ont une longueur = len(thresholds) + 1
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


def find_best_threshold_from_pr_curve(model, X_test, y_test, metric="f1") -> dict:
    """
    Trouve le meilleur seuil à partir de la precision-recall curve.

    metric supportée :
    - 'f1'
    - 'precision'
    - 'recall'
    """
    pr_df = get_precision_recall_thresholds(model, X_test, y_test)

    if metric not in pr_df.columns:
        raise ValueError(f"Métrique non supportée : {metric}")

    best_idx = pr_df[metric].idxmax()
    best_row = pr_df.loc[best_idx]

    return {
        "best_threshold": float(best_row["threshold"]),
        "best_precision": float(best_row["precision"]),
        "best_recall": float(best_row["recall"]),
        "best_f1": float(best_row["f1"]),
    }


def find_threshold_for_target_recall(model, X_test, y_test, target_recall=0.80):
    """
    Trouve le plus grand seuil permettant d'atteindre au moins le recall cible.
    Retourne None si aucun seuil ne satisfait la contrainte.
    """
    pr_df = get_precision_recall_thresholds(model, X_test, y_test)

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


def compare_models(trained_models, X_test, y_test, threshold=0.5, sort_by="roc_auc"):
    """
    Compare plusieurs modèles à seuil fixe.
    """
    results = []

    for model_name, model in trained_models.items():
        metrics = evaluate_binary_classifier(
            model=model,
            X_test=X_test,
            y_test=y_test,
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
):
    """
    Compare plusieurs modèles avec seuil optimisé par balayage simple.
    """
    results = []

    for model_name, model in trained_models.items():
        best_thresh, best_score = find_best_threshold(
            model=model,
            X_test=X_test,
            y_test=y_test,
            metric=metric,
        )

        metrics = evaluate_binary_classifier(
            model=model,
            X_test=X_test,
            y_test=y_test,
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
):
    """
    Compare plusieurs modèles avec seuil optimal trouvé via precision-recall curve.
    """
    results = []

    for model_name, model in trained_models.items():
        best_info = find_best_threshold_from_pr_curve(
            model=model,
            X_test=X_test,
            y_test=y_test,
            metric=metric,
        )

        metrics = evaluate_binary_classifier(
            model=model,
            X_test=X_test,
            y_test=y_test,
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
):
    """
    Compare plusieurs modèles avec un seuil choisi pour atteindre un recall cible.
    """
    results = []

    for model_name, model in trained_models.items():
        threshold_info = find_threshold_for_target_recall(
            model=model,
            X_test=X_test,
            y_test=y_test,
            target_recall=target_recall,
        )

        if threshold_info is None:
            results.append(
                {
                    "model": model_name,
                    "target_recall": target_recall,
                    "best_threshold": None,
                    "precision_1": None,
                    "recall_1": None,
                    "f1_1": None,
                    "roc_auc": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
                    "tn": None,
                    "fp": None,
                    "fn": None,
                    "tp": None,
                }
            )
            continue

        metrics = evaluate_binary_classifier(
            model=model,
            X_test=X_test,
            y_test=y_test,
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


