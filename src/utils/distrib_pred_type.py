import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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
):
    """
    Affiche la distribution des probabilités par type de prédiction.
    """

    if categories_to_exclude is None:
        categories_to_exclude = []

    df = X.copy()

    # Probabilités
    df["probability_score"] = model.predict_proba(X)[:, 1]

    # Prédictions
    df["prediction"] = (df["probability_score"] >= threshold).astype(int)

    # Target
    df["target"] = y.values

    # Type de prédiction
    df["prediction_type"] = df.apply(
        lambda row: get_prediction_type(row["prediction"], row["target"]),
        axis=1,
    )

    # Filtrage
    df_filtered = df[~df["prediction_type"].isin(categories_to_exclude)]

    # Plot
    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df_filtered,
        x="probability_score",
        hue="prediction_type",
        bins=30,
        kde=True,
        palette=color_palette,
        stat="density",
        common_norm=False,
    )

    plt.title("Distribution des probabilités par type de prédiction")
    plt.xlabel("Probability score")
    plt.ylabel("Density")
    plt.grid(alpha=0.3)

    plt.show()

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