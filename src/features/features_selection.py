from __future__ import annotations

TARGET = "a_quitte_l_entreprise"

DROP_COLUMNS = [
    "Unnamed: 0",
    "id_employee",
    "eval_number",
    "code_sondage",
]

#* =========================================================
#* RAW FEATURES
#* =========================================================

RAW_NUM_FEATURES = [
    "age",
    "revenu_mensuel",
    "annee_experience_totale",
    "annees_dans_l_entreprise",
    "annees_dans_le_poste_actuel",
    "annees_depuis_la_derniere_promotion",
    "distance_domicile_travail",
    "satisfaction_employee_environnement",
    "satisfaction_employee_equipe",
    "satisfaction_employee_equilibre_pro_perso",
    "satisfaction_global",
    "note_evaluation_precedente",
    "note_evaluation_actuelle",
    "niveau_hierarchique_poste",
]

RAW_CAT_FEATURES = [
    "genre",
    "statut_marital",
    "departement",
    "poste",
    "niveau_education",
    "domaine_etude",
    "frequence_deplacement",
]

#* =========================================================
#* BLOC DE BASE
#* =========================================================

# Catégorielles créées
FE_CAT_BUCKETS = [
    "age_bucket",
    "revenu_bin",
]

# FE continues / peu redondantes / lisibles
FE_NUM_CORE = [
    "salary_vs_level",
    "experience_mismatch",
    "promotion_speed",
    "promotion_delay",
    "delta_evaluation",
]

# FE binaires métier utiles
FE_NUM_FLAGS = [
    "is_low_salary_for_job",
    "stagnation_flag",
    "long_commute",
    "mid_level",
]

# FE risques / interprétation
FE_NUM_RISK = [
    "grey_zone_employee",
    "career_frustration",
    "underpaid_senior",
    "fn_risk_profile",
]

# FE issues des catégorielles
FE_NUM_CAT_DRIVEN = [
    "consulting_risk",
    "married_fn_risk",
    "tech_fn_risk",
    "travel_fn_risk",
    "categorical_fn_profile",
    "senior_frustration",
    "tech_stagnation",
    "consulting_travel_risk",
]


#* =========================================================
#* BLOCS POUR ABLATION A PARTIR DE fe_compact
#* =========================================================

FE_COMPACT_NUM_BASE = [
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
]

FE_COMPACT_NUM_ENGINEERED = [
    "salary_vs_level",
    "experience_mismatch",
    "promotion_speed",
    "promotion_delay",
    "delta_evaluation",
]

FE_COMPACT_NUM_FLAGS = [
    "is_low_salary_for_job",
    "stagnation_flag",
    "long_commute",
    "mid_level",
]

FE_COMPACT_NUM_FN = [
    "career_frustration",
    "fn_risk_profile",
]

FE_COMPACT_CAT_BASE = [
    "statut_marital",
    "departement",
    "poste",
    "niveau_education",
    "domaine_etude",
    "frequence_deplacement",
]

FE_COMPACT_CAT_BUCKETS = [
    "age_bucket",
    "revenu_bin",
]

#* =========================================================
#* FEATURE SETS
#* =========================================================

FEATURE_SETS = {
    #? -----------------------------------------------------
    #? 1. Référence brute
    #? utile pour benchmark
    #? -----------------------------------------------------
    "raw_baseline": {
        "num": RAW_NUM_FEATURES,
        "cat": RAW_CAT_FEATURES,
    },

    #? -----------------------------------------------------
    #? 2. Brut + buckets simples
    #? robuste pour logreg / arbres
    #? -----------------------------------------------------
    "raw_plus_buckets": {
        "num": RAW_NUM_FEATURES,
        "cat": RAW_CAT_FEATURES + FE_CAT_BUCKETS,
    },

    #? -----------------------------------------------------
    #? 3. FE coeur de métier
    #? meilleur compromis simplicité / robustesse
    #? -----------------------------------------------------
    "fe_core": {
        "num": RAW_NUM_FEATURES + FE_NUM_CORE + FE_NUM_FLAGS,
        "cat": RAW_CAT_FEATURES + FE_CAT_BUCKETS,
    },

    #? -----------------------------------------------------
    #? 4. FE compacte et robuste
    #? peu de bruit, peu de redondance
    #? -----------------------------------------------------
    "fe_compact": {
        "num": (
            FE_COMPACT_NUM_BASE
            + FE_COMPACT_NUM_ENGINEERED
            + FE_COMPACT_NUM_FLAGS
            + FE_COMPACT_NUM_FN
        ),
        "cat": FE_COMPACT_CAT_BASE + FE_COMPACT_CAT_BUCKETS,
    },
    "fe_compact_minus_fn": {
        "num": (
            FE_COMPACT_NUM_BASE
            + FE_COMPACT_NUM_ENGINEERED
            + FE_COMPACT_NUM_FLAGS
        ),
        "cat": FE_COMPACT_CAT_BASE + FE_COMPACT_CAT_BUCKETS,
    },
    "fe_compact_minus_flags": {
        "num": (
            FE_COMPACT_NUM_BASE
            + FE_COMPACT_NUM_ENGINEERED
            + FE_COMPACT_NUM_FN
        ),
        "cat": FE_COMPACT_CAT_BASE + FE_COMPACT_CAT_BUCKETS,
    },
    "fe_compact_minus_buckets": {
        "num": (
            FE_COMPACT_NUM_BASE
            + FE_COMPACT_NUM_ENGINEERED
            + FE_COMPACT_NUM_FLAGS
            + FE_COMPACT_NUM_FN
        ),
        "cat": FE_COMPACT_CAT_BASE,
    },
    "fe_compact_minus_engineered": {
        "num": (
            FE_COMPACT_NUM_BASE
            + FE_COMPACT_NUM_FLAGS
            + FE_COMPACT_NUM_FN
        ),
        "cat": FE_COMPACT_CAT_BASE + FE_COMPACT_CAT_BUCKETS,
    },
    "fe_compact_plus_cat_driven": {
        "num": (
            FE_COMPACT_NUM_BASE
            + FE_COMPACT_NUM_ENGINEERED
            + FE_COMPACT_NUM_FLAGS
            + FE_COMPACT_NUM_FN
            + FE_NUM_CAT_DRIVEN
        ),
        "cat": FE_COMPACT_CAT_BASE + FE_COMPACT_CAT_BUCKETS,
    },
    "fe_compact_plus_risk": {
        "num": (
            FE_COMPACT_NUM_BASE
            + FE_COMPACT_NUM_ENGINEERED
            + FE_COMPACT_NUM_FLAGS
            + FE_COMPACT_NUM_FN
            + [
                "grey_zone_employee",
                "underpaid_senior",
            ]
        ),
        "cat": FE_COMPACT_CAT_BASE + FE_COMPACT_CAT_BUCKETS,
    },

    #? -----------------------------------------------------
    #? 5. FE orienté recall / faux négatifs
    #? pour capter les profils "discrets"
    #? -----------------------------------------------------
    "fe_fn_focus": {
        "num": [
            "age",
            "revenu_mensuel",
            "annees_dans_l_entreprise",
            "annees_dans_le_poste_actuel",
            "distance_domicile_travail",
            "satisfaction_global",
            "niveau_hierarchique_poste",
            "salary_vs_level",
            "experience_mismatch",
            "promotion_speed",
            "delta_evaluation",
            "is_low_salary_for_job",
            "stagnation_flag",
            "career_frustration",
            "grey_zone_employee",
            "fn_risk_profile",
            "consulting_risk",
            "married_fn_risk",
            "tech_fn_risk",
            "travel_fn_risk",
            "categorical_fn_profile",
            "senior_frustration",
            "tech_stagnation",
            "consulting_travel_risk",
        ],
        "cat": [
            "statut_marital",
            "departement",
            "poste",
            "domaine_etude",
            "frequence_deplacement",
            "age_bucket",
            "revenu_bin",
        ],
    },

    #? -----------------------------------------------------
    #? 6. FE orienté business / explicabilité
    #? très bien pour présentation RH
    #? -----------------------------------------------------
    "fe_business": {
        "num": [
            "age",
            "revenu_mensuel",
            "annees_dans_l_entreprise",
            "annees_dans_le_poste_actuel",
            "distance_domicile_travail",
            "satisfaction_global",
            "niveau_hierarchique_poste",
            "salary_vs_level",
            "experience_mismatch",
            "promotion_speed",
            "promotion_delay",
            "delta_evaluation",
            "is_low_salary_for_job",
            "stagnation_flag",
            "long_commute",
            "career_frustration",
            "underpaid_senior",
        ],
        "cat": [
            "statut_marital",
            "departement",
            "poste",
            "niveau_education",
            "domaine_etude",
            "frequence_deplacement",
            "age_bucket",
        ],
    },

    #? -----------------------------------------------------
    #? 7. FE complet mais encore propre
    #? plus riche, sans toutes les binaires redondantes
    #? -----------------------------------------------------
    "fe_full_robust": {
        "num": (
            RAW_NUM_FEATURES
            + FE_NUM_CORE
            + FE_NUM_FLAGS
            + FE_NUM_RISK
            + FE_NUM_CAT_DRIVEN
        ),
        "cat": RAW_CAT_FEATURES + FE_CAT_BUCKETS,
    },

    #? -----------------------------------------------------
    #? 8. Set minimaliste pour modèles linéaires
    #? -----------------------------------------------------
    "fe_linear_clean": {
        "num": [
            "age",
            "revenu_mensuel",
            "annee_experience_totale",
            "annees_dans_l_entreprise",
            "annees_dans_le_poste_actuel",
            "distance_domicile_travail",
            "satisfaction_global",
            "note_evaluation_precedente",
            "note_evaluation_actuelle",
            "niveau_hierarchique_poste",
            "salary_vs_level",
            "experience_mismatch",
            "promotion_speed",
            "delta_evaluation",
            "is_low_salary_for_job",
            "stagnation_flag",
            "long_commute",
        ],
        "cat": [
            "statut_marital",
            "departement",
            "poste",
            "domaine_etude",
            "frequence_deplacement",
            "age_bucket",
        ],
    },
}

#* =========================================================
#* FONCTION D'UTILISATION 
#* =========================================================

def get_feature_set(name: str) -> dict:
    if name not in FEATURE_SETS:
        available = ", ".join(FEATURE_SETS.keys())
        raise ValueError(f"Unknown feature set: {name}. Available: {available}")
    return FEATURE_SETS[name]