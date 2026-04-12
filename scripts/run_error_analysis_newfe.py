from __future__ import annotations

import warnings
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from models.logistic_regression_model import build_elastic_net_logistic_regression_pipeline
from models.xgboost_model import build_xgboost_pipeline
from src.data.preprocessing import build_preprocessor, filter_existing_features
from src.data.split_data import make_train_test_split
from src.features.feature_engineering import make_feature_engineering
from src.features.features_selection import DROP_COLUMNS, TARGET, get_feature_set

warnings.filterwarnings(
    "ignore",
    message=r".*mismatched devices.*",
    category=UserWarning,
)

SEED = 51
TEST_SIZE = 0.2
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
TOP_K_CONTRIBUTORS = 3

CANDIDATES = [
    {
        "short_name": "en_silent_v1",
        "candidate_name": "elastic_net_fe_silent_attrition_v1_smote",
        "feature_set": "fe_silent_attrition_v1",
        "model_key": "elastic_net",
        "sampling_method": "smote",
        "threshold": 0.318754,
        "threshold_label": "seuil_cv_opt_prc",
        "use_gpu": False,
        "params": {
            "model__C": 0.1,
            "model__class_weight": None,
            "model__l1_ratio": 0.85,
            "model__max_iter": 5000,
        },
    },
]

ANOMALY_NUMERIC_COLUMNS = [
    "age",
    "revenu_mensuel",
    "annee_experience_totale",
    "annees_dans_l_entreprise",
    "annees_dans_le_poste_actuel",
    "annees_depuis_la_derniere_promotion",
    "distance_domicile_travail",
    "satisfaction_global",
    "note_evaluation_precedente",
    "note_evaluation_actuelle",
    "niveau_hierarchique_poste",
    "salary_vs_level",
    "experience_mismatch",
    "promotion_speed",
    "promotion_delay",
    "nb_formations_suivies",
    "nombre_participation_pee",
]
NUMERIC_OUTLIER_IQR_MULTIPLIER = 3.0

IDENTIFIER_COLUMNS = ["id_employee", "eval_number", "code_sondage"]

def prepare_dataset(data_path: Path, feature_set_name: str):
    source_df = pd.read_csv(data_path)
    model_df = source_df.drop(columns=DROP_COLUMNS, errors="ignore").copy()
    model_df = make_feature_engineering(model_df)

    feature_config = get_feature_set(feature_set_name)
    num_features = feature_config["num"]
    cat_features = feature_config["cat"]
    feature_columns = num_features + cat_features

    X_train, X_test, y_train, y_test = make_train_test_split(
        df=model_df,
        feature_columns=feature_columns,
        target_column=TARGET,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=True,
    )

    num_features = filter_existing_features(num_features, X_train.columns.tolist())
    cat_features = filter_existing_features(cat_features, X_train.columns.tolist())

    preprocessor = build_preprocessor(
        num_features=num_features,
        cat_features=cat_features,
        X_reference=X_train,
    )

    return {
        "source_df": source_df,
        "model_df": model_df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "num_features": num_features,
        "cat_features": cat_features,
        "preprocessor": preprocessor,
        "model_train_rows": model_df.loc[X_train.index].copy(),
        "model_test_rows": model_df.loc[X_test.index].copy(),
        "source_test_rows": source_df.loc[X_test.index].copy(),
    }


def build_candidate_model(candidate: dict, preprocessor):
    params = candidate["params"]

    if candidate["model_key"] == "elastic_net":
        return build_elastic_net_logistic_regression_pipeline(
            preprocessor=preprocessor,
            random_state=SEED,
            sampling_method=candidate["sampling_method"],
            C=params["model__C"],
            l1_ratio=params["model__l1_ratio"],
            class_weight=params["model__class_weight"],
            max_iter=params["model__max_iter"],
        )

    if candidate["model_key"] == "xgboost":
        return build_xgboost_pipeline(
            preprocessor=preprocessor,
            random_state=SEED,
            sampling_method=candidate["sampling_method"],
            n_estimators=params["model__n_estimators"],
            max_depth=params["model__max_depth"],
            learning_rate=params["model__learning_rate"],
            subsample=params["model__subsample"],
            colsample_bytree=params["model__colsample_bytree"],
            min_child_weight=params["model__min_child_weight"],
            gamma=params["model__gamma"],
            reg_alpha=params["model__reg_alpha"],
            reg_lambda=params["model__reg_lambda"],
            scale_pos_weight=params["model__scale_pos_weight"],
            use_gpu=candidate["use_gpu"],
            n_jobs=-1,
        )

    raise ValueError(f"model_key non supporte: {candidate['model_key']}")


def build_numeric_bounds(df: pd.DataFrame) -> dict:
    bounds = {}
    numeric_columns = [
        col for col in ANOMALY_NUMERIC_COLUMNS
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]

    for column in numeric_columns:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty or series.nunique() <= 2:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue

        bounds[column] = {
            "lower": q1 - NUMERIC_OUTLIER_IQR_MULTIPLIER * iqr,
            "upper": q3 + NUMERIC_OUTLIER_IQR_MULTIPLIER * iqr,
        }

    return bounds


def build_rare_category_reference(df: pd.DataFrame, min_count: int = 5, min_freq: float = 0.02) -> dict:
    """
    Pour chaque colonne catégorielle, on compte les modalités.
    Une modalité est marquée comme “rare” si :
        - elle apparaît moins de 5 fois
        - ou si sa fréquence est inférieure à 2%
    """
    reference = {}
    categorical_columns = [
        col for col in df.columns
        if col != TARGET and not pd.api.types.is_numeric_dtype(df[col])
    ]

    for column in categorical_columns:
        series = df[column].astype("string").fillna("<MISSING>")
        counts = series.value_counts(dropna=False)
        total = max(len(series), 1)
        rare_values = counts[(counts < min_count) | ((counts / total) < min_freq)].index.tolist()
        reference[column] = set(str(value) for value in rare_values)

    return reference


def get_numeric_outlier_flags(row: pd.Series, bounds: dict) -> list[str]:
    flags = []
    for column, info in bounds.items():
        if column not in row.index:
            continue
        value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
        if pd.isna(value):
            continue
        if value < info["lower"] or value > info["upper"]:
            flags.append(column)
    return flags


def get_rare_category_flags(row: pd.Series, reference: dict) -> list[str]:
    """ 
    prend une ligne
    regarde la valeur de chaque variable catégorielle
    si cette valeur fait partie des modalités rares apprises sur le train, elle ajoute un flag du type :
        - poste=Directeur Technique
        - domaine_etude=Data
    Objectif : 
        - repérer les cas où le modèle voit un profil catégoriel peu fréquent
        - vérifier si certaines erreurs extrêmes viennent de modalités très rares ou peu représentées
    """
    flags = []
    for column, rare_values in reference.items():
        if column not in row.index or not rare_values:
            continue
        value = "<MISSING>" if pd.isna(row[column]) else str(row[column])
        if value in rare_values:
            flags.append(f"{column}={value}")
    return flags


def get_rule_issue_flags(row: pd.Series) -> list[str]:
    """
    La fonction lit certaines colonnes numériques d'une ligne, puis déclenche des flags si elle voit une incohérence ou quelque chose de très douteux.
    Les règles actuelles sont :

        - experience_gt_age_minus_15
            si annee_experience_totale > age - 15 ==> “trop d’expérience pour l’âge”
        - tenure_gt_total_experience
            si annees_dans_l_entreprise > annee_experience_totale ==> ancienneté entreprise supérieure à l’expérience totale
        - role_years_gt_tenure
            si annees_dans_le_poste_actuel > annees_dans_l_entreprise
        - last_promo_gt_tenure
            si annees_depuis_la_derniere_promotion > annees_dans_l_entreprise
        - large_eval_jump
            si l’écart entre note_evaluation_actuelle et note_evaluation_precedente est supérieur ou égal à 3
        - non_positive_salary
            si revenu_mensuel <= 0
        - senior_level_low_salary
            si niveau_hierarchique_poste >= 4 et revenu_mensuel < 3000
    """
    flags = []

    age = row.get("age")
    experience = row.get("annee_experience_totale")
    tenure = row.get("annees_dans_l_entreprise")
    role_years = row.get("annees_dans_le_poste_actuel")
    last_promo = row.get("annees_depuis_la_derniere_promotion")
    current_eval = row.get("note_evaluation_actuelle")
    previous_eval = row.get("note_evaluation_precedente")
    salary = row.get("revenu_mensuel")
    level = row.get("niveau_hierarchique_poste")

    if pd.notna(age) and pd.notna(experience) and experience > max(age - 15, 0):
        flags.append("experience_gt_age_minus_15")
    if pd.notna(tenure) and pd.notna(experience) and tenure > experience:
        flags.append("tenure_gt_total_experience")
    if pd.notna(role_years) and pd.notna(tenure) and role_years > tenure:
        flags.append("role_years_gt_tenure")
    if pd.notna(last_promo) and pd.notna(tenure) and last_promo > tenure:
        flags.append("last_promo_gt_tenure")
    if pd.notna(current_eval) and pd.notna(previous_eval) and abs(current_eval - previous_eval) >= 3:
        flags.append("large_eval_jump")
    if pd.notna(salary) and salary <= 0:
        flags.append("non_positive_salary")
    if pd.notna(level) and pd.notna(salary) and level >= 4 and salary < 3000:
        flags.append("senior_level_low_salary")

    return flags


def classify_error_type(y_true: int, y_pred: int) -> str:
    if y_true == 1 and y_pred == 1:
        return "TP"
    if y_true == 0 and y_pred == 0:
        return "TN"
    if y_true == 0 and y_pred == 1:
        return "FP"
    return "FN"


def _format_top_contributors(values: np.ndarray, feature_names: np.ndarray, largest: bool = True) -> str:
    if values.size == 0:
        return ""

    order = np.argsort(values)
    if largest:
        indices = [idx for idx in order[::-1] if values[idx] > 0][:TOP_K_CONTRIBUTORS]
    else:
        indices = [idx for idx in order if values[idx] < 0][:TOP_K_CONTRIBUTORS]

    if not indices:
        return ""

    return " | ".join(
        f"{feature_names[idx]}={values[idx]:+.3f}" for idx in indices
    )


def attach_linear_contributions(rows_df: pd.DataFrame, model, X_eval: pd.DataFrame) -> pd.DataFrame:
    prep = model.named_steps["prep"]
    linear_model = model.named_steps["model"]

    X_trans = prep.transform(X_eval)
    X_trans = np.asarray(X_trans)
    feature_names = np.asarray(prep.get_feature_names_out())
    coef = linear_model.coef_[0]
    intercept = float(linear_model.intercept_[0])
    contributions = X_trans * coef
    logits = contributions.sum(axis=1) + intercept

    rows_df = rows_df.copy()
    rows_df["logit_score"] = logits
    rows_df["top_positive_contributors"] = [
        _format_top_contributors(row, feature_names, largest=True)
        for row in contributions
    ]
    rows_df["top_negative_contributors"] = [
        _format_top_contributors(row, feature_names, largest=False)
        for row in contributions
    ]
    return rows_df


def build_rows_analysis(candidate: dict, prepared: dict) -> pd.DataFrame:
    """
    Expliquer localement pourquoi Elastic Net a donné une proba élevée ou faible à une ligne.

    Concrètement :
        - Elle récupère le préprocesseur prep (smote, borerline) et le modèle linéaire model dans le pipeline.
        - Elle transforme X_eval avec prep.transform(...). Important : on travaille donc sur les variables après preprocessing
            variables numériques imputées / scalées
            catégories one-hot encodées
            variables ordinales encodées
        - Elle récupère les noms des features transformées avec prep.get_feature_names_out().
        - Elle récupère les coefficients du modèle linéaire : coef_ et intercept_
        - Elle calcule, pour chaque ligne et pour chaque feature transformée :
            - contribution = valeur_transformee * coefficient
        - Elle somme toutes les contributions et ajoute l’intercept : ça donne le logit_score (donc pas directement une probabilité, mais le score avant la sigmoïde)
        - Elle extrait ensuite :
            - les top_positive_contributors (Ce sont les features qui poussent le plus la prédiction vers la classe positive, donc vers “départ”.)
            - les top_negative_contributors (Ce sont les features qui poussent le plus vers la classe négative, donc vers “reste”.)
    """
    X_train = prepared["X_train"]
    X_test = prepared["X_test"]
    y_train = prepared["y_train"]
    y_test = prepared["y_test"]
    model_train_rows = prepared["model_train_rows"]
    model_test_rows = prepared["model_test_rows"]
    source_test_rows = prepared["source_test_rows"]

    model = build_candidate_model(candidate, prepared["preprocessor"])
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= candidate["threshold"]).astype(int)
    proba_rank_pct = pd.Series(y_proba, index=X_test.index).rank(pct=True, method="average")

    numeric_bounds = build_numeric_bounds(model_train_rows)
    rare_reference = build_rare_category_reference(model_train_rows)

    analysis_rows = []
    for idx in X_test.index:
        engineered_row = model_test_rows.loc[idx]
        source_row = source_test_rows.loc[idx]
        true_value = int(y_test.loc[idx])
        pred_value = int(y_pred[list(X_test.index).index(idx)])
        proba_value = float(y_proba[list(X_test.index).index(idx)])
        error_type = classify_error_type(true_value, pred_value)
        numeric_flags = get_numeric_outlier_flags(engineered_row, numeric_bounds)
        rare_flags = get_rare_category_flags(engineered_row, rare_reference)
        rule_flags = get_rule_issue_flags(engineered_row)
        feature_missing_count = int(X_test.loc[idx].isna().sum())
        abs_margin = abs(proba_value - candidate["threshold"])
        tail_error = (
            (error_type == "FP" and proba_rank_pct.loc[idx] >= 0.90)
            or (error_type == "FN" and proba_rank_pct.loc[idx] <= 0.10)
        )
        confident_error = (error_type in {"FP", "FN"}) and (
            tail_error or abs_margin >= 0.20
        )

        row_dict = {
            "row_index": idx,
            "candidate_name": candidate["candidate_name"],
            "feature_set": candidate["feature_set"],
            "threshold_label": candidate["threshold_label"],
            "threshold": candidate["threshold"],
            "y_true": true_value,
            "y_pred": pred_value,
            "y_proba": proba_value,
            "proba_rank_pct": float(proba_rank_pct.loc[idx]),
            "margin_vs_threshold": proba_value - candidate["threshold"],
            "abs_margin_vs_threshold": abs_margin,
            "error_type": error_type,
            "is_error": int(error_type in {"FP", "FN"}),
            "is_confident_error": int(confident_error),
            "is_tail_extreme_error": int(tail_error),
            "numeric_outlier_count": len(numeric_flags),
            "rare_category_count": len(rare_flags),
            "rule_issue_count": len(rule_flags),
            "missing_feature_count": feature_missing_count,
            "anomaly_score": len(numeric_flags) + len(rare_flags) + len(rule_flags) + feature_missing_count,
            "possible_data_issue": int(
                bool(numeric_flags or rare_flags or rule_flags or feature_missing_count)
            ),
            "numeric_outlier_features": ", ".join(numeric_flags),
            "rare_category_flags": ", ".join(rare_flags),
            "rule_issue_flags": ", ".join(rule_flags),
        }

        for identifier in IDENTIFIER_COLUMNS:
            if identifier in source_row.index:
                row_dict[identifier] = source_row[identifier]

        for column in model_test_rows.columns:
            if column == TARGET:
                continue
            row_dict[column] = engineered_row[column]

        analysis_rows.append(row_dict)

    rows_df = pd.DataFrame(analysis_rows).sort_values("y_proba", ascending=False).reset_index(drop=True)

    if candidate["model_key"] == "elastic_net":
        x_eval_sorted = X_test.loc[rows_df["row_index"]]
        rows_df = attach_linear_contributions(rows_df, model=model, X_eval=x_eval_sorted)

    return rows_df


def build_candidate_summary(candidate: dict, rows_df: pd.DataFrame) -> pd.DataFrame:
    y_true = rows_df["y_true"].to_numpy()
    y_pred = rows_df["y_pred"].to_numpy()
    y_proba = rows_df["y_proba"].to_numpy()

    summary = {
        "candidate_name": candidate["candidate_name"],
        "feature_set": candidate["feature_set"],
        "threshold_label": candidate["threshold_label"],
        "threshold": candidate["threshold"],
        "n_test": len(rows_df),
        "precision_1": precision_score(y_true, y_pred, zero_division=0),
        "recall_1": recall_score(y_true, y_pred, zero_division=0),
        "f1_1": f1_score(y_true, y_pred, zero_division=0),
        "average_precision": average_precision_score(y_true, y_proba),
        "n_tp": int((rows_df["error_type"] == "TP").sum()),
        "n_tn": int((rows_df["error_type"] == "TN").sum()),
        "n_fp": int((rows_df["error_type"] == "FP").sum()),
        "n_fn": int((rows_df["error_type"] == "FN").sum()),
        "n_confident_errors": int(rows_df["is_confident_error"].sum()),
        "n_rows_with_possible_issue": int(rows_df["possible_data_issue"].sum()),
        "n_confident_errors_with_issue": int(
            rows_df.loc[rows_df["is_confident_error"] == 1, "possible_data_issue"].sum()
        ),
        "avg_anomaly_score_all": float(rows_df["anomaly_score"].mean()),
        "avg_anomaly_score_errors": float(
            rows_df.loc[rows_df["is_error"] == 1, "anomaly_score"].mean()
        ) if (rows_df["is_error"] == 1).any() else 0.0,
    }
    return pd.DataFrame([summary])


def build_flag_summary(candidate: dict, rows_df: pd.DataFrame) -> pd.DataFrame:
    groups = {
        "all_rows": rows_df,
        "all_errors": rows_df[rows_df["is_error"] == 1],
        "confident_errors": rows_df[rows_df["is_confident_error"] == 1],
        "false_positives": rows_df[rows_df["error_type"] == "FP"],
        "false_negatives": rows_df[rows_df["error_type"] == "FN"],
    }

    records = []
    for group_name, group in groups.items():
        if group.empty:
            records.append(
                {
                    "candidate_name": candidate["candidate_name"],
                    "group": group_name,
                    "n_rows": 0,
                    "rows_with_possible_issue": 0,
                    "rows_with_numeric_outlier": 0,
                    "rows_with_rare_category": 0,
                    "rows_with_rule_issue": 0,
                    "avg_anomaly_score": 0.0,
                }
            )
            continue

        records.append(
            {
                "candidate_name": candidate["candidate_name"],
                "group": group_name,
                "n_rows": len(group),
                "rows_with_possible_issue": int(group["possible_data_issue"].sum()),
                "rows_with_numeric_outlier": int((group["numeric_outlier_count"] > 0).sum()),
                "rows_with_rare_category": int((group["rare_category_count"] > 0).sum()),
                "rows_with_rule_issue": int((group["rule_issue_count"] > 0).sum()),
                "avg_anomaly_score": float(group["anomaly_score"].mean()),
            }
        )

    return pd.DataFrame(records)


def select_export_columns(rows_df: pd.DataFrame) -> list[str]:
    priority_columns = [
        "row_index",
        "candidate_name",
        "feature_set",
        "threshold_label",
        "threshold",
        "y_true",
        "y_pred",
        "y_proba",
        "proba_rank_pct",
        "margin_vs_threshold",
        "abs_margin_vs_threshold",
        "error_type",
        "is_error",
        "is_confident_error",
        "is_tail_extreme_error",
        "possible_data_issue",
        "anomaly_score",
        "missing_feature_count",
        "numeric_outlier_count",
        "rare_category_count",
        "rule_issue_count",
        "numeric_outlier_features",
        "rare_category_flags",
        "rule_issue_flags",
        "id_employee",
        "eval_number",
        "code_sondage",
        "top_positive_contributors",
        "top_negative_contributors",
        "logit_score",
    ]
    remaining_columns = [column for column in rows_df.columns if column not in priority_columns]
    return [column for column in priority_columns if column in rows_df.columns] + remaining_columns


def export_error_analysis(
    overview_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    flag_df: pd.DataFrame,
    candidate_results: dict[str, dict[str, pd.DataFrame]],
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{pd.Timestamp.now().strftime('%y%m%d')}-error_analysis_newfe.xlsx"

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        overview_df.to_excel(writer, sheet_name="overview", index=False)
        summary_df.to_excel(writer, sheet_name="candidate_summary", index=False)
        flag_df.to_excel(writer, sheet_name="flag_summary", index=False)

        for short_name, blocks in candidate_results.items():
            blocks["rows"].to_excel(writer, sheet_name=f"{short_name}_rows", index=False)
            blocks["extreme"].to_excel(writer, sheet_name=f"{short_name}_extreme", index=False)
            blocks["fp"].to_excel(writer, sheet_name=f"{short_name}_fp", index=False)
            blocks["fn"].to_excel(writer, sheet_name=f"{short_name}_fn", index=False)

    return output_path


def main():
    data_path = PROJECT_ROOT / "data" / "interim" / "data_eda.csv"

    overview_rows = []
    summary_blocks = []
    flag_blocks = []
    candidate_results: dict[str, dict[str, pd.DataFrame]] = {}
    prepared_cache: dict[str, dict] = {}

    print("\n=== ERROR ANALYSIS CONFIG ===")
    print(f"CANDIDATES : {[candidate['candidate_name'] for candidate in CANDIDATES]}")
    print()

    for candidate in CANDIDATES:
        feature_set = candidate["feature_set"]
        if feature_set not in prepared_cache:
            prepared_cache[feature_set] = prepare_dataset(
                data_path=data_path,
                feature_set_name=feature_set,
            )

        print("=" * 80)
        print(f"ERROR ANALYSIS: {candidate['candidate_name']}")
        print(f"FEATURE SET   : {candidate['feature_set']}")
        print(f"THRESHOLD     : {candidate['threshold']} ({candidate['threshold_label']})")
        print("=" * 80)

        rows_df = build_rows_analysis(candidate=candidate, prepared=prepared_cache[feature_set])
        rows_df = rows_df[select_export_columns(rows_df)]

        summary_df = build_candidate_summary(candidate=candidate, rows_df=rows_df)
        flags_df = build_flag_summary(candidate=candidate, rows_df=rows_df)

        extreme_df = rows_df[
            (rows_df["is_confident_error"] == 1)
            | ((rows_df["is_error"] == 1) & (rows_df["possible_data_issue"] == 1))
        ].copy()
        extreme_df = extreme_df.sort_values(
            ["is_confident_error", "possible_data_issue", "abs_margin_vs_threshold", "anomaly_score"],
            ascending=[False, False, False, False],
        )
        fp_df = rows_df[rows_df["error_type"] == "FP"].copy().sort_values(
            ["y_proba", "anomaly_score"],
            ascending=[False, False],
        )
        fn_df = rows_df[rows_df["error_type"] == "FN"].copy().sort_values(
            ["y_proba", "anomaly_score"],
            ascending=[True, False],
        )

        summary_blocks.append(summary_df)
        flag_blocks.append(flags_df)
        candidate_results[candidate["short_name"]] = {
            "rows": rows_df,
            "extreme": extreme_df,
            "fp": fp_df,
            "fn": fn_df,
        }

        overview_rows.append(
            {
                "candidate_name": candidate["candidate_name"],
                "feature_set": candidate["feature_set"],
                "threshold_label": candidate["threshold_label"],
                "threshold": candidate["threshold"],
                "model_key": candidate["model_key"],
                "sampling_method": candidate["sampling_method"],
                "uses_gpu": candidate["use_gpu"],
            }
        )

        print(summary_df.to_string(index=False))

    overview_df = pd.DataFrame(overview_rows)
    summary_df = pd.concat(summary_blocks, ignore_index=True)
    flag_df = pd.concat(flag_blocks, ignore_index=True)

    output_path = export_error_analysis(
        overview_df=overview_df,
        summary_df=summary_df,
        flag_df=flag_df,
        candidate_results=candidate_results,
    )

    print("\n=== ERROR ANALYSIS SUMMARY ===")
    print(summary_df.to_string(index=False))
    print(f"\nError analysis exported: {output_path}")


if __name__ == "__main__":
    main()
