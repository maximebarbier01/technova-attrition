from __future__ import annotations

import pandas as pd

SATISFACTION_COLUMNS = [
    'satisfaction_employee_environnement',
    'satisfaction_employee_nature_travail',
    'satisfaction_employee_equipe',
    'satisfaction_employee_equilibre_pro_perso',
]

SENIOR_ROLES = {
    'Senior Manager',
    'Directeur Technique',
    'Manager',
}

TECH_PROFILES = {
    'Infra Cloud',
    'Transformation Digitale',
}


def _as_int(series: pd.Series) -> pd.Series:
    return series.astype(int)


def _ensure_satisfaction_global(df: pd.DataFrame) -> pd.DataFrame:
    if 'satisfaction_global' in df.columns:
        return df

    available_columns = [
        column for column in SATISFACTION_COLUMNS if column in df.columns
    ]
    if available_columns:
        df['satisfaction_global'] = df[available_columns].mean(axis=1)

    return df


def make_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _ensure_satisfaction_global(df)

    # Categorical buckets used in several feature sets.
    df['age_bucket'] = pd.cut(
        df['age'],
        bins=[0, 30, 45, 100],
        labels=['young', 'mid', 'senior'],
        include_lowest=True,
    ).astype('string')

    df['revenu_bin'] = pd.qcut(
        df['revenu_mensuel'],
        q=4,
        duplicates='drop',
    ).astype('string')

    # Shared aggregates used to contextualize salary, satisfaction, promotion and training.
    poste_salary_median = df.groupby('poste')['revenu_mensuel'].transform('median')
    poste_satisfaction_mean = df.groupby('poste')['satisfaction_global'].transform('mean')
    poste_promotion_median = df.groupby('poste')[
        'annees_depuis_la_derniere_promotion'
    ].transform('median')
    department_training_median = df.groupby('departement')[
        'nb_formations_suivies'
    ].transform('median')

    # Keep helper masks local: they are useful to build business features but are not
    # part of the final feature sets themselves.
    salary_vs_level = df['revenu_mensuel'] / (df['niveau_hierarchique_poste'] + 1)
    low_salary_for_job = df['revenu_mensuel'] < poste_salary_median
    salary_underpaid = salary_vs_level < 1500

    low_satisfaction = df['satisfaction_global'] <= 2
    mid_satisfaction = (
        (df['satisfaction_global'] > 2)
        & (df['satisfaction_global'] <= 3)
    )
    mid_salary = df['revenu_mensuel'].between(4000, 8000, inclusive='both')
    mid_tenure = df['annees_dans_l_entreprise'].between(3, 8, inclusive='both')

    is_married = df['statut_marital'].isin(['Marié(e)', 'Marie(e)'])
    is_consulting = df['departement'].eq('Consulting')
    is_senior_role = df['poste'].isin(SENIOR_ROLES)
    is_tech_profile = df['domaine_etude'].isin(TECH_PROFILES)
    frequent_travel = df['frequence_deplacement'].eq('Frequent')

    # Core engineered signals.
    df['salary_vs_level'] = salary_vs_level
    df['experience_mismatch'] = (
        df['annee_experience_totale'] - df['annees_dans_le_poste_actuel']
    )
    df['promotion_speed'] = (
        df['annees_depuis_la_derniere_promotion'] / (df['annees_dans_l_entreprise'] + 1)
    )
    # Historical naming preserved to avoid changing downstream feature sets.
    df['promotion_delay'] = (
        df['annees_dans_l_entreprise']
        / (df['annees_depuis_la_derniere_promotion'] + 1)
    )
    df['delta_evaluation'] = (
        df['note_evaluation_actuelle'] - df['note_evaluation_precedente']
    )

    # Business flags used in compact / robust / business-oriented sets.
    df['is_low_salary_for_job'] = _as_int(low_salary_for_job)
    df['stagnation_flag'] = _as_int(df['annees_dans_le_poste_actuel'] > 4)
    df['long_commute'] = _as_int(df['distance_domicile_travail'] > 20)
    df['mid_level'] = _as_int(
        df['niveau_hierarchique_poste'].between(2, 3, inclusive='both')
    )

    # Interaction features oriented toward silent attrition / false negatives.
    df['grey_zone_employee'] = _as_int(mid_satisfaction & mid_salary)
    df['career_frustration'] = _as_int(df['stagnation_flag'].eq(1) & low_satisfaction)
    df['underpaid_senior'] = _as_int(
        salary_underpaid & (df['annees_dans_l_entreprise'] > 5)
    )
    df['fn_risk_profile'] = _as_int(mid_satisfaction & mid_tenure & mid_salary)

    df['married_fn_risk'] = _as_int(is_married & mid_satisfaction)
    df['consulting_risk'] = _as_int(is_consulting & mid_satisfaction)
    df['tech_fn_risk'] = _as_int(is_tech_profile & mid_satisfaction)
    df['travel_fn_risk'] = _as_int(frequent_travel & mid_satisfaction)
    df['categorical_fn_profile'] = _as_int(
        is_consulting & frequent_travel & mid_satisfaction
    )
    df['senior_frustration'] = _as_int(is_senior_role & low_satisfaction)
    df['tech_stagnation'] = _as_int(is_tech_profile & df['stagnation_flag'].eq(1))
    df['consulting_travel_risk'] = _as_int(is_consulting & frequent_travel)

    # Features created after analysing extreme false positives / false negatives.
    df['salary_gap_vs_poste_median'] = df['revenu_mensuel'] - poste_salary_median
    df['satisfaction_gap_vs_poste_mean'] = (
        df['satisfaction_global'] - poste_satisfaction_mean
    )
    df['promo_delay_vs_poste_median'] = (
        df['annees_depuis_la_derniere_promotion'] - poste_promotion_median
    )
    df['role_stagnation_ratio'] = (
        df['annees_dans_le_poste_actuel'] / (df['annees_dans_l_entreprise'] + 1)
    )
    df['senior_plateau_flag'] = _as_int(
        (df['niveau_hierarchique_poste'] >= 4)
        & (df['annees_depuis_la_derniere_promotion'] >= 3)
        & (df['satisfaction_global'] <= 3.5)
    )
    df['consulting_hidden_attrition'] = _as_int(
        is_consulting
        & df['satisfaction_global'].between(2.0, 3.5, inclusive='both')
        & (df['annees_dans_le_poste_actuel'] >= 3)
        & df['frequence_deplacement'].ne('Aucun')
    )
    df['assistant_direction_hidden_attrition'] = _as_int(
        df['poste'].eq('Assistant de Direction')
        & (df['satisfaction_global'] <= 3.5)
        & (df['annees_dans_le_poste_actuel'] >= 2)
    )
    df['techlead_hidden_attrition'] = _as_int(
        df['poste'].eq('Tech Lead')
        & (df['satisfaction_global'] <= 3.0)
        & (df['annees_depuis_la_derniere_promotion'] >= 2)
    )
    df['training_gap_vs_department_median'] = (
        df['nb_formations_suivies'] - department_training_median
    )
    df['low_training_vs_department'] = _as_int(
        df['training_gap_vs_department_median'] < 0
    )
    df['pee_disengagement_flag'] = _as_int(
        (df['nombre_participation_pee'] == 0)
        & (df['annees_dans_l_entreprise'] >= 2)
    )

    return df
