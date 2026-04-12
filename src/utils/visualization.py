import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
from pathlib import Path


def get_prediction_type(pred, target):
    if pred == 1 and target == 1:
        return "true_positive"
    elif pred == 0 and target == 0:
        return "true_negative"
    elif pred == 1 and target == 0:
        return "false_positive"
    elif pred == 0 and target == 1:
        return "false_negative"
    else:
        return "unknown"


color_palette = {
    "true_positive": "green",
    "true_negative": "red",
    "false_positive": "blue",
    "false_negative": "orange",
}


def plot_probability_distrib_per_pred_type(
    model,
    X,
    y,
    threshold=0.5,
    categories_to_exclude=["true_negative"],
    save_path=None,
    show=True,
):
    """
    Affiche la distribution des probabilit?s par type de pr?diction.
    """

    if categories_to_exclude is None:
        categories_to_exclude = []

    df = X.copy()
    df["probability_score"] = model.predict_proba(X)[:, 1]
    df["prediction"] = (df["probability_score"] >= threshold).astype(int)
    df["target"] = y.values
    df["prediction_type"] = df.apply(
        lambda row: get_prediction_type(row["prediction"], row["target"]),
        axis=1,
    )

    df_filtered = df[~df["prediction_type"].isin(categories_to_exclude)]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        data=df_filtered,
        x="probability_score",
        hue="prediction_type",
        bins=30,
        kde=True,
        palette=color_palette,
        stat="density",
        common_norm=False,
        ax=ax,
    )

    ax.set_title("Distribution des probabilit?s par type de pr?diction")
    ax.set_xlabel("Probability score")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)
    return df_filtered


def plot_numeric_distributions_by_prediction_type(
    model,
    X,
    y,
    num_features,
    threshold=0.5,
    features_per_row=2,
):
    """
    Analyse des distributions des variables numériques selon :
    - True Positive (TP)
    - False Negative (FN)
    - True Negative (TN)

    Parameters
    ----------
    model : fitted model
    X : DataFrame
    y : Series
    num_features : list
    threshold : float
    features_per_row : int
    """

    # ===== Préparation dataset =====
    df = X.copy()

    df["y_true"] = y.values
    df["y_proba"] = model.predict_proba(X)[:, 1]
    df["y_pred"] = (df["y_proba"] >= threshold).astype(int)

    # ===== Segmentation =====
    fn = df[(df["y_true"] == 1) & (df["y_pred"] == 0)]
    tp = df[(df["y_true"] == 1) & (df["y_pred"] == 1)]
    tn = df[(df["y_true"] == 0) & (df["y_pred"] == 0)]

    print(f"FN: {fn.shape[0]} | TP: {tp.shape[0]} | TN: {tn.shape[0]}")

    # ===== Moyennes comparatives =====
    summary = pd.DataFrame({
        "FN_mean": fn[num_features].mean(),
        "TP_mean": tp[num_features].mean(),
        "TN_mean": tn[num_features].mean(),
    })

    print("\n===== Moyennes par groupe =====")
    print(summary.sort_values("FN_mean", ascending=False).head(10))

    # ===== Plots =====
    n_features = len(num_features)
    n_rows = (n_features + features_per_row - 1) // features_per_row

    fig, axes = plt.subplots(
        n_rows,
        features_per_row,
        figsize=(6 * features_per_row, 3 * n_rows),
    )

    axes = axes.flatten() if n_features > 1 else [axes]

    for i, col in enumerate(num_features):
        ax = axes[i]

        if col not in df.columns:
            continue

        sns.kdeplot(fn[col], label="FN", fill=True, ax=ax, color="orange")
        sns.kdeplot(tp[col], label="TP", fill=True, ax=ax, color="green")
        sns.kdeplot(tn[col], label="TN", fill=True, ax=ax, color="red")

        ax.set_title(col)
        ax.legend()

    # supprimer axes vides
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_numeric_feature_diagnostics(
    model,
    X,
    y,
    num_features,
    threshold=0.5,
    kind="kde",
    ncols=2,
    output_dir=None,
    filename_prefix="",
    show=True,
):
    """
    Pour chaque variable num?rique, trace 2 graphiques :
    1. distribution classe 1 vs classe 0 : on regarde si la variable s?pare globalement les classes
        - distributions tr?s diff?rentes ? variable potentiellement utile
        - distributions tr?s proches ? variable peu discriminante
    2. distribution FN vs TP : on regarde si la variable aide ? comprendre les d?parts rat?s.
        - distributions diff?rentes ? la variable semble li?e au fait d??tre d?tect? ou rat?
        - distributions proches ? la variable n?explique pas bien les FN
    """
    df = X.copy()
    df["y_true"] = y.values
    df["y_proba"] = model.predict_proba(X)[:, 1]
    df["y_pred"] = (df["y_proba"] >= threshold).astype(int)

    fn = df[(df["y_true"] == 1) & (df["y_pred"] == 0)].copy()
    tp = df[(df["y_true"] == 1) & (df["y_pred"] == 1)].copy()
    class_1 = df[df["y_true"] == 1].copy()
    class_0 = df[df["y_true"] == 0].copy()

    print(
        f"Classe 1: {len(class_1)} | Classe 0: {len(class_0)} | "
        f"TP: {len(tp)} | FN: {len(fn)}"
    )

    valid_features = [col for col in num_features if col in df.columns]
    if len(valid_features) == 0:
        print("Aucune variable num?rique valide ? tracer.")
        return []

    output_dir = Path(output_dir) if output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for col in valid_features:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        if kind == "kde":
            if class_1[col].notna().sum() > 1:
                sns.kdeplot(class_1[col], fill=True, label="Classe 1", ax=axes[0])
            if class_0[col].notna().sum() > 1:
                sns.kdeplot(class_0[col], fill=True, label="Classe 0", ax=axes[0])
        elif kind == "hist":
            sns.histplot(class_1[col], label="Classe 1", stat="density", kde=True, ax=axes[0], alpha=0.4)
            sns.histplot(class_0[col], label="Classe 0", stat="density", kde=True, ax=axes[0], alpha=0.4)
        else:
            raise ValueError("kind doit ?tre 'kde' ou 'hist'")

        axes[0].set_title(f"{col} ? Classe 1 vs Classe 0")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        if kind == "kde":
            if fn[col].notna().sum() > 1:
                sns.kdeplot(fn[col], fill=True, label="FN", ax=axes[1])
            if tp[col].notna().sum() > 1:
                sns.kdeplot(tp[col], fill=True, label="TP", ax=axes[1])
        elif kind == "hist":
            sns.histplot(fn[col], label="FN", stat="density", kde=True, ax=axes[1], alpha=0.4)
            sns.histplot(tp[col], label="TP", stat="density", kde=True, ax=axes[1], alpha=0.4)

        axes[1].set_title(f"{col} ? FN vs TP")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()

        if output_dir is not None:
            safe_col = str(col).replace('/', '_').replace(' ', '_')
            save_path = output_dir / f"{filename_prefix}{safe_col}.png"
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
            saved_paths.append(save_path)

        if show:
            plt.show()

        plt.close(fig)

    return saved_paths


def summarize_numeric_feature_diagnostics(model, X, y, num_features, threshold=0.5):
    """
    Résumé tabulaire des moyennes par variable pour :
    - classe 1
    - classe 0
    - TP
    - FN
    """
    df = X.copy()
    df["y_true"] = y.values
    df["y_proba"] = model.predict_proba(X)[:, 1]
    df["y_pred"] = (df["y_proba"] >= threshold).astype(int)

    fn = df[(df["y_true"] == 1) & (df["y_pred"] == 0)].copy()
    tp = df[(df["y_true"] == 1) & (df["y_pred"] == 1)].copy()
    class_1 = df[df["y_true"] == 1].copy()
    class_0 = df[df["y_true"] == 0].copy()

    valid_features = [col for col in num_features if col in df.columns]

    summary = pd.DataFrame(
        {
            "mean_class_1": class_1[valid_features].mean(),
            "mean_class_0": class_0[valid_features].mean(),
            "mean_tp": tp[valid_features].mean(),
            "mean_fn": fn[valid_features].mean(),
        }
    )

    summary["abs_diff_class1_class0"] = (
        summary["mean_class_1"] - summary["mean_class_0"]
    ).abs()

    summary["abs_diff_fn_tp"] = (
        summary["mean_fn"] - summary["mean_tp"]
    ).abs()

    return summary.sort_values(
        ["abs_diff_fn_tp", "abs_diff_class1_class0"],
        ascending=False,
    )


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# *****************************************
# *       MODALITES CATEGORIELLES         *
# *****************************************

def plot_categorical_feature_diagnostics(
    model,
    X,
    y,
    cat_features,
    threshold=0.5,
    normalize=True,
    top_n=None,
    output_dir=None,
    filename_prefix="",
    show=True,
):
    """
    Pour chaque variable cat?gorielle, trace 2 graphiques :
    1. distribution des modalit?s pour Classe 1 vs Classe 0
    2. distribution des modalit?s pour FN vs TP
    """
    df = X.copy()
    df["y_true"] = y.values
    df["y_proba"] = model.predict_proba(X)[:, 1]
    df["y_pred"] = (df["y_proba"] >= threshold).astype(int)

    fn = df[(df["y_true"] == 1) & (df["y_pred"] == 0)].copy()
    tp = df[(df["y_true"] == 1) & (df["y_pred"] == 1)].copy()
    class_1 = df[df["y_true"] == 1].copy()
    class_0 = df[df["y_true"] == 0].copy()

    print(
        f"Classe 1: {len(class_1)} | Classe 0: {len(class_0)} | "
        f"TP: {len(tp)} | FN: {len(fn)}"
    )

    valid_features = [col for col in cat_features if col in df.columns]
    if not valid_features:
        print("Aucune variable cat?gorielle valide ? tracer.")
        return []

    output_dir = Path(output_dir) if output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for col in valid_features:
        tmp = df[[col, "y_true", "y_pred"]].copy()
        tmp[col] = tmp[col].astype("object").fillna("MISSING")

        if top_n is not None:
            top_modalities = tmp[col].value_counts().head(top_n).index
            tmp[col] = tmp[col].where(tmp[col].isin(top_modalities), other="AUTRES")

        class_1_col = tmp[tmp["y_true"] == 1].copy()
        class_0_col = tmp[tmp["y_true"] == 0].copy()
        fn_col = tmp[(tmp["y_true"] == 1) & (tmp["y_pred"] == 0)].copy()
        tp_col = tmp[(tmp["y_true"] == 1) & (tmp["y_pred"] == 1)].copy()

        def _dist(group_df, col_name):
            s = group_df[col_name].value_counts(normalize=normalize, dropna=False)
            return s.rename_axis(col_name).reset_index(name="value")

        dist_class_1 = _dist(class_1_col, col).rename(columns={"value": "classe_1"})
        dist_class_0 = _dist(class_0_col, col).rename(columns={"value": "classe_0"})
        dist_fn = _dist(fn_col, col).rename(columns={"value": "fn"})
        dist_tp = _dist(tp_col, col).rename(columns={"value": "tp"})

        left_df = dist_class_1.merge(dist_class_0, on=col, how="outer").fillna(0)
        right_df = dist_fn.merge(dist_tp, on=col, how="outer").fillna(0)
        left_plot = left_df.melt(id_vars=col, var_name="group", value_name="value")
        right_plot = right_df.melt(id_vars=col, var_name="group", value_name="value")
        modality_order = tmp[col].value_counts().index.tolist()

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        sns.barplot(data=left_plot, x=col, y="value", hue="group", order=modality_order, ax=axes[0])
        axes[0].set_title(f"{col} ? Classe 1 vs Classe 0")
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].grid(alpha=0.3)

        sns.barplot(data=right_plot, x=col, y="value", hue="group", order=modality_order, ax=axes[1])
        axes[1].set_title(f"{col} ? FN vs TP")
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].grid(alpha=0.3)

        ylabel = "Proportion" if normalize else "Count"
        axes[0].set_ylabel(ylabel)
        axes[1].set_ylabel(ylabel)
        plt.tight_layout()

        if output_dir is not None:
            safe_col = str(col).replace('/', '_').replace(' ', '_')
            save_path = output_dir / f"{filename_prefix}{safe_col}.png"
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
            saved_paths.append(save_path)

        if show:
            plt.show()

        plt.close(fig)

    return saved_paths



