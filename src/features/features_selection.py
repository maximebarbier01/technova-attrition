
# *-----------------------------
# * 1) Features numériques
# *-----------------------------

NUM_FEATURES_BASELINE = [
    "age",
    "revenu_mensuel",
    "annee_experience_totale",
    "annees_dans_l_entreprise",
    "annees_dans_le_poste_actuel",
    "distance_domicile_travail",
    "note_evaluation_precedente",
    "niveau_hierarchique_poste",
]

NUM_FEATURES_INTERMEDIATE = NUM_FEATURES_BASELINE + [
    "salary_vs_level",
    "experience_mismatch",
    "promotion_speed",
    "delta_evaluation",
]

NUM_FEATURES_ADVANCED = NUM_FEATURES_INTERMEDIATE + [
    "low_salary",
    "mid_salary",
    "high_salary",
    "salary_underpaid",
    "salary_fair",
    "low_tenure",
    "mid_tenure",
    "high_tenure",
    "stagnation_flag",
    "long_commute",
    "low_satisfaction",
    "mid_satisfaction",
    "high_satisfaction",
    "slow_promotion",
]

NUM_FEATURES_BEHAVIORAL = [
    "low_satisfaction",
    "mid_satisfaction",
    "high_satisfaction",
    "stagnation_flag",
    "promotion_speed",
    "delta_evaluation",
    "experience_mismatch",
]

NUM_FEATURES_SALARY = [
    "revenu_mensuel",
    "salary_vs_level",
    "salary_underpaid",
    "salary_fair",
    "low_salary",
    "mid_salary",
    "high_salary",
]

NUM_FEATURES_RISK = [
    "grey_zone_employee",
    "career_frustration",
    "underpaid_senior",
    "fn_risk_profile",
]

# *-----------------------------
# * 2) Features catégorielles
# *-----------------------------

CAT_FEATURES_BASELINE = [
    "genre",
    "statut_marital",
    "departement",
    "poste",
    "niveau_education",
    "domaine_etude",
    "frequence_deplacement",
]

FEATURE_SETS = {
    "baseline": {
        "num": NUM_FEATURES_BASELINE,
        "cat": CAT_FEATURES_BASELINE,
    },
    "intermediate": {
        "num": NUM_FEATURES_INTERMEDIATE,
        "cat": CAT_FEATURES_BASELINE,
    },
    "advanced": {
        "num": NUM_FEATURES_ADVANCED,
        "cat": CAT_FEATURES_BASELINE,
    },
    "behavioral": {
        "num": NUM_FEATURES_BEHAVIORAL,
        "cat": CAT_FEATURES_BASELINE,
    },
    "salary": {
        "num": NUM_FEATURES_SALARY,
        "cat": CAT_FEATURES_BASELINE,
    },
    "risk": {
        "num": NUM_FEATURES_RISK,
        "cat": CAT_FEATURES_BASELINE,
    },
}

# *-----------------------------
# * 3) Target
# *-----------------------------

TARGET = "a_quitte_l_entreprise"

# *-----------------------------
# * 3) Colonne à supprimer
# *-----------------------------

DROP_COLUMNS = [
    "Unnamed: 0",
    "id_employee",
    "eval_number",
    "code_sondage",
]