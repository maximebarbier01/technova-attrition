from __future__ import annotations

import numpy as np
import pandas as pd

def make_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()

    #? CATEGORIAL FAEATURES

    #******* AGE *******   
    #df["is_young"] = (df["age"] < 30).astype(int)

    #df["is_senior_risk"] = (df["age"] > 45).astype(int)

    df["age_bucket"] = pd.cut(
        df["age"],
        bins=[0, 30, 45, 100],
        labels=["young", "mid", "senior"],
        include_lowest=True
    ).astype("string")

    #******* REVENU *******   
    df["revenu_bin"] = pd.qcut(
        df["revenu_mensuel"],
        q=4,
        duplicates="drop"
    ).astype("string")
    df["low_salary"] = (df["revenu_mensuel"] < 4000).astype(int)

    df["mid_salary"] = (
        (df["revenu_mensuel"] >= 4000) &
        (df["revenu_mensuel"] <= 8000)
    ).astype(int)

    df["high_salary"] = (df["revenu_mensuel"] > 8000).astype(int)

    #df["salary_vs_age"] = df["revenu_mensuel"] / (df["age"] + 1)
    #df["salary_vs_tenure"] = df["revenu_mensuel"] / (df["annees_dans_l_entreprise"] + 1)
    df["salary_vs_level"] = df["revenu_mensuel"] / (df["niveau_hierarchique_poste"] + 1)
    
    df["is_low_salary_for_job"] = (
        df["revenu_mensuel"] < df.groupby("poste")["revenu_mensuel"].transform("median")
        ).astype(int)
    
    df["salary_underpaid"] = (df["salary_vs_level"] < 1500).astype(int)

    df["salary_fair"] = (
        (df["salary_vs_level"] >= 1500) &
        (df["salary_vs_level"] <= 2500)
    ).astype(int)
    
    #******* EXPERIENCE / ANCIENNETE *******  
    df["low_tenure"] = (df["annees_dans_l_entreprise"] < 3).astype(int)

    df["mid_tenure"] = (
        (df["annees_dans_l_entreprise"] >= 3) &
        (df["annees_dans_l_entreprise"] <= 8)
    ).astype(int)

    df["high_tenure"] = (df["annees_dans_l_entreprise"] > 8).astype(int)

    #******* ANNEES DANS LE POSTE *******  
    df["experience_mismatch"] = df["annee_experience_totale"] - df["annees_dans_le_poste_actuel"]
    
    df["stagnation_flag"] = (
        df["annees_dans_le_poste_actuel"] > 4
    ).astype(int)

    #******* DISTANCE *******   
    df["long_commute"] = (df["distance_domicile_travail"] > 20).astype(int)

    #******* SATISFACTION *******
    df["low_satisfaction"] = (df["satisfaction_global"] <= 2).astype(int)

    df["mid_satisfaction"] = (
        (df["satisfaction_global"] > 2) &
        (df["satisfaction_global"] <= 3)
    ).astype(int)

    df["high_satisfaction"] = (df["satisfaction_global"] > 3).astype(int)

    #******* PROMOTION *******   
    df["promotion_speed"] = df["annees_depuis_la_derniere_promotion"] / (df["annees_dans_l_entreprise"] + 1)
    df["promotion_delay"] = (
        df["annees_dans_l_entreprise"] /
        (df["annees_depuis_la_derniere_promotion"] + 1)
    )
    #df["fast_promotion"] = (df["annees_dans_le_poste_actuel"] < 2) & (df["niveau_hierarchique_poste"] >= 3)
    df["slow_promotion"] = (df["promotion_speed"] < 0.2).astype(int)

    #******* NIVEAU HIERARCHIQUE *******  
    df["mid_level"] = (
        (df["niveau_hierarchique_poste"] >= 2) &
        (df["niveau_hierarchique_poste"] <= 3)
    ).astype(int)
  
    #******* EVALUATION *******
    df["delta_evaluation"] = (
        df["note_evaluation_actuelle"] - df["note_evaluation_precedente"]
    )

    #******* FEATURES COMBINÉES *******
    df["grey_zone_employee"] = (
        (df["mid_satisfaction"] == 1) &
        (df["mid_salary"] == 1)
    ).astype(int)
    df["career_frustration"] = (
        (df["stagnation_flag"] == 1) &
        (df["low_satisfaction"] == 1)
    ).astype(int)
    df["underpaid_senior"] = (
        (df["salary_underpaid"] == 1) &
        (df["annees_dans_l_entreprise"] > 5)
    ).astype(int)
    df["fn_risk_profile"] = (
        (df["mid_satisfaction"] == 1) &
        (df["mid_tenure"] == 1) &
        (df["mid_salary"] == 1)
    ).astype(int)

    #? NUMERIC FAEATURES

    #******* STATUT MATRIMONIAL *******  
    df["is_single"] = (df["statut_marital"] == "Célibataire").astype(int)

    df["is_married"] = (df["statut_marital"] == "Marié(e)").astype(int) 

    df["married_fn_risk"] = (
        (df["statut_marital"] == "Marié(e)") &
        (df["mid_satisfaction"] == 1)
    ).astype(int)

    #******* DEPARTEMENT *******    

    df["is_consulting"] = (df["departement"] == "Consulting").astype(int)

    df["is_commercial"] = (df["departement"] == "Commercial").astype(int)  

    df["consulting_risk"] = (
        (df["departement"] == "Consulting") &
        (df["mid_satisfaction"] == 1)
    ).astype(int)

    #******* POSTE *******    

    df["is_senior_role"] = df["poste"].isin([
        "Senior Manager",
        "Directeur Technique",
        "Manager"
    ]).astype(int)

    df["is_senior_role"] = df["poste"].isin([
        "Senior Manager",
        "Directeur Technique",
        "Manager"
    ]).astype(int)

    #******* DOMAINE D'ETUDE *******    
    df["is_tech_profile"] = df["domaine_etude"].isin([
        "Infra Cloud",
        "Transformation Digitale"
    ]).astype(int)

    df["is_business_profile"] = df["domaine_etude"].isin([
        "Marketing",
        "Entrepreneuriat"
    ]).astype(int)

    df["tech_fn_risk"] = (
        (df["is_tech_profile"] == 1) &
        (df["mid_satisfaction"] == 1)
    ).astype(int)

    #******* FREQUENCE DEPLACEMENT *******    
    df["frequent_travel"] = (df["frequence_deplacement"] == "Frequent").astype(int)

    df["no_travel"] = (df["frequence_deplacement"] == "Aucun").astype(int)

    df["travel_fn_risk"] = (
        (df["frequent_travel"] == 1) &
        (df["mid_satisfaction"] == 1)
    ).astype(int)

    #? FEATURES COMBINÉES

    df["categorical_fn_profile"] = (
        (df["is_consulting"] == 1) &
        (df["frequent_travel"] == 1) &
        (df["mid_satisfaction"] == 1)
    ).astype(int)

    df["senior_frustration"] = (
        (df["is_senior_role"] == 1) &
        (df["low_satisfaction"] == 1)
    ).astype(int)

    df["tech_stagnation"] = (
        (df["is_tech_profile"] == 1) &
        (df["stagnation_flag"] == 1)
    ).astype(int)

    df["consulting_travel_risk"] = (
        (df["is_consulting"] == 1) &
        (df["frequent_travel"] == 1)
    ).astype(int)

    return df 
