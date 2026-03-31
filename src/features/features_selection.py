NUM_FEATURES = [
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
#    "delta_evaluation",
    "salary_vs_level",
 #   "salary_vs_tenure",
    "promotion_speed",
    "experience_mismatch",
    "salary_gap_level",
    "career_stagnation",
    "niveau_hierarchique_poste",
]

CAT_FEATURES = [
    "genre",
    "statut_marital",
    "departement",
    "poste",
    "niveau_education",
    "domaine_etude",
    "frequence_deplacement",
]

TARGET = "a_quitte_l_entreprise"

DROP_COLUMNS = [
    "Unnamed: 0",
    "id_employee",
    "eval_number",
    "code_sondage",
]