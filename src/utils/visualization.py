import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
from pathlib import Path
from matplotlib.ticker import PercentFormatter, FuncFormatter, NullFormatter


#*********************
#* Analyse bivariée  *
#*********************

# ? ==== CORRELATION AVEC LA CIBLE ====

def plot_top_spearman_corr(
    df: pd.DataFrame,
    target: str = "a_quitte_l_entreprise",
    top_n: int = 10,
    figsize: tuple = (9, 6),
    pos_color: str = "#FFADA6",
    neg_color: str = "#A7DEB7",
):
    # Corrélations de Spearman avec la cible
    corr = (
        df.corr(method="spearman", numeric_only=True)[target]
        .drop(labels=[target], errors="ignore")
        .dropna()
    )

    # On garde les variables les plus corrélées en valeur absolue
    corr = corr.reindex(corr.abs().sort_values(ascending=False).head(top_n).index)

    # Tri visuel pour le plot horizontal
    corr = corr.sort_values(ascending=True)

    plot_df = corr.reset_index()
    plot_df.columns = ["feature", "correlation"]

    # Labels plus propres
    plot_df["feature_label"] = (
        plot_df["feature"]
        .str.replace("_", " ", regex=False)
        .str.capitalize()
    )

    # Couleurs selon le signe
    colors = [
        pos_color if val > 0 else neg_color
        for val in plot_df["correlation"]
    ]

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.barh(
        plot_df["feature_label"],
        plot_df["correlation"],
        color=colors,
        edgecolor="#444444",
        linewidth=0.8,
        height=0.7,
    )

    # Ligne de référence
    ax.axvline(0, color="#555555", linestyle="--", linewidth=1.2)

    # Titre et labels
    ax.set_title(
        "Variables les plus corrélées à la cible",
        fontsize=16,
        weight="bold",
        pad=12,
    )
    ax.set_xlabel("Corrélation de Spearman", fontsize=11)
    ax.set_ylabel("Features")

    # Grille légère
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)

    # Nettoyage visuel
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Limites un peu aérées
    max_abs = max(abs(plot_df["correlation"].min()), abs(plot_df["correlation"].max()))
    ax.set_xlim(-max_abs * 1.20, max_abs * 1.20)

    # Annotations
    for bar, val in zip(bars, plot_df["correlation"]):
        x = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2

        if val >= 0:
            ax.text(
                x + max_abs * 0.03,
                y,
                f"{val:.3f}",
                va="center",
                ha="left",
                fontsize=10,
                color="#333333",
            )
        else:
            ax.text(
                x - max_abs * 0.03,
                y,
                f"{val:.3f}",
                va="center",
                ha="right",
                fontsize=10,
                color="#333333",
            )

    plt.tight_layout()
    plt.show()

    return corr

# ? ===== CORRELATION FORTE ENTRE LES FEATURES =====

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_strong_feature_correlations(
    strong_corr: pd.DataFrame,
    figsize: tuple = (10, 6),
    pos_color: str = "#FFADA6",
    neg_color: str = "#A7DEB7",
):
    df_plot = strong_corr.copy()

    df_plot["pair_label"] = (
        df_plot["level_0"].str.replace("_", " ", regex=False).str.capitalize()
        + "  ↔  " +
        df_plot["level_1"].str.replace("_", " ", regex=False).str.capitalize()
    )

    df_plot = df_plot.sort_values("corr", ascending=True)

    colors = [
        pos_color if val > 0 else neg_color
        for val in df_plot["corr"]
    ]

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.barh(
        df_plot["pair_label"],
        df_plot["corr"],
        color=colors,
        edgecolor="#444444",
        linewidth=0.8,
        height=0.7,
    )

    ax.axvline(0, color="#555555", linestyle="--", linewidth=1.2)

    ax.set_title(
        "Paires de variables fortement corrélées",
        fontsize=16,
        weight="bold",
        pad=12,
    )
    ax.set_xlabel("Corrélation de Spearman", fontsize=11)
    ax.set_ylabel("")

    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    max_abs = df_plot["corr"].abs().max()
    ax.set_xlim(-max_abs * 1.15, max_abs * 1.15)

    for bar, val in zip(bars, df_plot["corr"]):
        x = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2

        ax.text(
            x + (max_abs * 0.03 if val >= 0 else -max_abs * 0.03),
            y,
            f"{val:.2f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=10,
            color="#333333",
        )

    plt.tight_layout()
    plt.show()

# ? ======  MODALITES NUMERIQUES ======

def plot_attrition_by_num(
    df: pd.DataFrame,
    col: str,
    target: str = "a_quitte_l_entreprise",
    q: int = 5,
    figsize: tuple = (8, 4.8),
    color: str = "#d5b5f5",
    # edgecolor: str = "#2F4B6E",
    annotate: bool = True,
    show_range: bool = True
):
    # Copie de travail
    tmp_df = df[[col, target]].dropna().copy()

    # Binning quantiles
    bins = pd.qcut(tmp_df[col], q=q, duplicates="drop")

    # Agrégation
    agg = (
        tmp_df
        .groupby(bins, observed=False)[target]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "attrition_rate", "count": "n"})
    )

    # Labels courts pour le PPT
    agg["bin_label"] = [f"Q{i+1}" for i in range(len(agg))]

    # Bornes lisibles pour éventuel sous-titre / debug
    agg["left"] = agg[col].apply(lambda x: x.left)
    agg["right"] = agg[col].apply(lambda x: x.right)

    # Figure
    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(
        agg["bin_label"],
        agg["attrition_rate"],
        color=color,
        # edgecolor=edgecolor,
        linewidth=1.2,
        width=0.65,
    )

    # Titre propre
    pretty_col = col.replace("_", " ").capitalize()
    ax.set_title(f"Taux d’attrition selon {pretty_col}", fontsize=16, weight="bold", pad=15)

    # Axes
    ax.set_ylabel("Taux d’attrition", fontsize=11)
    ax.set_xlabel(f"{pretty_col}")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=10)

    # Grille légère horizontale seulement
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    # Nettoyage des bordures
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Laisser de l'air au-dessus
    ymax = agg["attrition_rate"].max()
    ax.set_ylim(0, ymax * 1.20)

    # Annotations sur les barres
    if annotate:
        for bar, rate, n in zip(bars, agg["attrition_rate"], agg["n"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ymax * 0.03,
                f"{rate:.1%}\n(n={n})",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Sous-titre discret avec bornes réelles
    if show_range:
        ranges_txt = "   ".join(
            [
                f"{lbl}: [{left:.0f} ; {right:.0f}]"
                for lbl, left, right in zip(agg["bin_label"], agg["left"], agg["right"])
            ]
        )
        fig.text(
            0.5, -0.02,
            f"Classes en quantiles — {ranges_txt}",
            ha="center",
            fontsize=9,
            color="dimgray"
        )

    plt.tight_layout()
    plt.show()

    return agg

# ? ======  MODALITES CATEGORIELLES ======

def plot_attrition_by_cat(
    df: pd.DataFrame,
    col: str,
    target: str = "a_quitte_l_entreprise",
    figsize: tuple = (8, 4.8),
    color: str = "#a7deb7",
    highlight_color: str = "#d5b5f5",
    min_count: int | None = None,
    annotate: bool = True,
    highlight_max: bool = True,
):
    tmp_df = df[[col, target]].dropna().copy()

    agg = (
        tmp_df
        .groupby(col, dropna=False)[target]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={col: "category", "mean": "attrition_rate", "count": "n"})
        .sort_values("attrition_rate", ascending=True)
    )

    # Optionnel : filtrer les catégories trop rares
    if min_count is not None:
        agg = agg.loc[agg["n"] >= min_count].copy()

    if agg.empty:
        print(f"Aucune catégorie exploitable pour {col}")
        return None

    colors = [color] * len(agg)
    if highlight_max:
        max_idx = agg["attrition_rate"].idxmax()
        pos = agg.index.get_loc(max_idx)
        colors[pos] = highlight_color

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.barh(
        y=agg["category"],
        width=agg["attrition_rate"],
        color=colors,
        # edgecolor="#2F4B6E",
        # linewidth=1.0,
        # height=0.75,
    )

    pretty_col = col.replace("_", " ").capitalize()
    ax.set_title(f"Taux d’attrition selon {pretty_col}", fontsize=16, weight="bold", pad=12)
    ax.set_xlabel("Taux d’attrition", fontsize=11)
    ax.set_ylabel(f"{pretty_col}")
    ax.xaxis.set_major_formatter(PercentFormatter(1))

    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    xmax = agg["attrition_rate"].max()
    ax.set_xlim(0, xmax * 1.20)

    if annotate:
        for bar, rate, n in zip(bars, agg["attrition_rate"], agg["n"]):
            ax.text(
                x=bar.get_width() + xmax * 0.02,
                y=bar.get_y() + bar.get_height() / 2,
                s=f"{rate:.1%}  (n={n})",
                va="center",
                fontsize=9,
            )

    plt.tight_layout()
    plt.show()

    return agg

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

#************************
#* Analyse multivariée  *
#************************

#? PROFILS COMBINES

def plot_attrition_heatmaps(df_multivar, target="a_quitte_l_entreprise"):
    sns.set_theme(style="white")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # -------- Heatmap 1 : revenu x satisfaction --------
    revenu_order = ["revenu_bas", "revenu_moyen", "revenu_haut"]
    satisfaction_order = [
        "satisfaction_basse",
        "satisfaction_moyenne",
        "satisfaction_haute",
    ]

    revenu_satisfaction_heatmap = (
        df_multivar
        .groupby(["revenu_bin_eda", "satisfaction_bin_eda"], observed=True)[target]
        .mean()
        .mul(100)
        .unstack()
        .reindex(index=revenu_order, columns=satisfaction_order)
    )

    sns.heatmap(
        revenu_satisfaction_heatmap,
        annot=True,
        fmt=".1f",
        cmap=sns.light_palette("#d5b5f5", as_cmap=True),
        linewidths=1,
        linecolor="white",
        cbar_kws={"label": "Taux d'attrition (%)", "shrink": 0.85},
        annot_kws={"fontsize": 11, "weight": "bold", "color": "#333333"},
        ax=axes[0],
    )

    axes[0].set_title(
        "Attrition selon le revenu et la satisfaction",
        fontsize=15,
        weight="bold",
        pad=14,
    )
    axes[0].set_xlabel("Satisfaction globale", fontsize=11)
    axes[0].set_ylabel("Revenu mensuel", fontsize=11)
    axes[0].set_xticklabels(
        ["Basse", "Moyenne", "Haute"],
        rotation=0,
        fontsize=10,
    )
    axes[0].set_yticklabels(
        ["Bas", "Moyen", "Haut"],
        rotation=0,
        fontsize=10,
    )

    # -------- Heatmap 2 : poste x déplacements --------
    poste_travel_heatmap = (
        df_multivar
        .groupby(["poste", "frequence_deplacement"], observed=True)[target]
        .mean()
        .mul(100)
        .unstack()
    )

    # Trier les postes par attrition moyenne pour une lecture plus claire
    poste_order = poste_travel_heatmap.mean(axis=1).sort_values().index
    poste_travel_heatmap = poste_travel_heatmap.loc[poste_order]

    sns.heatmap(
        poste_travel_heatmap,
        annot=True,
        fmt=".1f",
        cmap=sns.light_palette("#a7deb7", as_cmap=True),
        linewidths=1,
        linecolor="white",
        cbar_kws={"label": "Taux d'attrition (%)", "shrink": 0.85},
        annot_kws={"fontsize": 10, "weight": "bold", "color": "#333333"},
        ax=axes[1],
    )

    axes[1].set_title(
        "Attrition selon le poste et les déplacements",
        fontsize=15,
        weight="bold",
        pad=14,
    )
    axes[1].set_xlabel("Fréquence de déplacement", fontsize=11)
    axes[1].set_ylabel("Poste", fontsize=11)
    axes[1].tick_params(axis="x", rotation=0, labelsize=10)
    axes[1].tick_params(axis="y", labelsize=9)

    plt.tight_layout()
    plt.show()

#? Analyse des odds ratios

def plot_odds_ratios(or_plot):
    plot_df = or_plot.sort_values("odds_ratio", ascending=True).copy()

    palette = {
        "OR > 1 : Association positive avec l'attrition": "#d5b5f5",
        "OR < 1 : Association négative avec l'attrition": "#a7deb7",
    }

    fig, ax = plt.subplots(figsize=(10, 7))

    sns.barplot(
        data=plot_df,
        x="odds_ratio",
        y="feature_label",
        hue="direction",
        dodge=False,
        palette=palette,
        ax=ax,
        edgecolor="0.25",
        linewidth=0.8,
    )

    ax.axvline(1, color="#444444", linestyle="--", linewidth=1.3)
    ax.set_xscale("log")

    ax.set_title(
        "Régression logistique exploratoire — Odds ratios",
        fontsize=16,
        weight="bold",
        pad=14,
    )
    ax.set_xlabel("Odds ratio (échelle logarithmique)", fontsize=11)
    ax.set_ylabel("Variables", fontsize=11)

    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())

    # annotations
    for patch, val in zip(ax.patches, plot_df["odds_ratio"]):
        y = patch.get_y() + patch.get_height() / 2
        x = patch.get_width()

        if val >= 1:
            xpos = val * 1.03
            ha = "left"
        else:
            xpos = val * 1.03
            ha = "left"

        ax.text(
            xpos,
            y,
            f"{val:.2f}",
            va="center",
            ha=ha,
            fontsize=9,
            color="#333333"
        )

    handles, labels = ax.get_legend_handles_labels()
    label_map = {
        "hausse_risque": "Odds ratio > 1",
        "baisse_risque": "Odds ratio < 1",
    }
    clean_labels = [label_map.get(l, l) for l in labels]
    ax.legend(handles, clean_labels, title="", frameon=False, loc=0)

    plt.tight_layout()
    plt.show()

#*******************************
#* Analyse distribution modèle *
#*******************************

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

# ? ======  MODALITES NUMERIQUES ======

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
        print("Aucune variable numérique valide à tracer.")
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
            raise ValueError("kind doit être 'kde' ou 'hist'")

        axes[0].set_title(f"{col} - Classe 1 vs Classe 0")
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

        axes[1].set_title(f"{col} - FN vs TP")
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


# ? ======  MODALITES CATEGORIELLES ======


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
        print("Aucune variable catégorielle valide à tracer.")
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
        axes[0].set_title(f"{col} - Classe 1 vs Classe 0")
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].grid(alpha=0.3)

        sns.barplot(data=right_plot, x=col, y="value", hue="group", order=modality_order, ax=axes[1])
        axes[1].set_title(f"{col} - FN vs TP")
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
