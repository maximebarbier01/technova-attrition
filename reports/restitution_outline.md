# Trame De Restitution - Projet Attrition TechNova

## Slide 1 - Contexte et objectif

**Titre suggere**
Predire et comprendre l'attrition chez TechNova Partners

**Messages cles**
- TechNova observe un turnover plus eleve que d'habitude.
- L'objectif est double :
  identifier les facteurs associes aux departs et construire un score de risque d'attrition.
- La finalite metier n'est pas de "predire parfaitement", mais d'aider les RH a prioriser des actions de retention.

**A dire a l'oral**
Nous avons traite le sujet comme un probleme a la fois analytique et metier : comprendre les causes probables de depart, puis proposer un modele capable de scorer le risque pour orienter des actions RH.

---

## Slide 2 - Donnees mobilisees

**Titre suggere**
Trois sources de donnees rapprochees pour construire une vue employee 360

**Messages cles**
- Fichier 1 : extrait SIRH
  variables socio-demographiques, salaire, anciennete, poste, departement.
- Fichier 2 : evaluations annuelles
  notes d'evaluation, heures supplementaires, dimensions de satisfaction au travail.
- Fichier 3 : sondage employee
  engagement, formations, distance domicile-travail, cible d'attrition.
- Les trois tables contiennent 1 470 lignes et ont ete rapprochees par identifiants equivalents :
  `id_employee`, `eval_number`, `code_sondage`.

**Point de methode**
- Le dataframe central a ete construit par jointures `inner`, apres nettoyage des identifiants.
- Dans les notebooks, cela correspond a un rapprochement :
  `id_employee -> eval_number -> code_sondage`.

**A montrer**
- Un schema simple des 3 tables et de leur jointure.

---

## Slide 3 - Hypotheses de preparation des donnees

**Titre suggere**
Preparation des donnees : hypotheses et choix structurants

**Messages cles**
- Nettoyage des types et harmonisation des colonnes quantitatives / qualitatives.
- Suppression des colonnes d'identifiants du modele :
  `id_employee`, `eval_number`, `code_sondage`.
- Decoupage train/test avec stratification pour respecter le desequilibre de classes.
- Pipeline de preprocessing differenciee :
  `SimpleImputer`, `OneHotEncoder`, `OrdinalEncoder`, `StandardScaler`, `RobustScaler`.
- Les variables numeriques a outliers ont ete traitees avec `RobustScaler`.

**A dire a l'oral**
Le point important ici est que le pipeline n'est pas uniforme : l'encodage et le scaling ont ete adaptes a la nature des variables pour limiter le bruit et mieux exploiter la structure des donnees.

---

## Slide 4 - Insights cles de l'EDA

**Titre suggere**
L'EDA fait ressortir des profils plus exposes au risque de depart

**Messages cles**
- Les employes qui quittent l'entreprise presentent plus souvent :
  une satisfaction globale plus faible,
  une remuneration plus faible,
  des frequences de deplacement plus elevees,
  une anciennete / progression de carriere moins favorables.
- Des differences apparaissent aussi selon certains segments metiers :
  `Commercial`, `Consulting`, certains postes comme `Consultant` ou `Representant Commercial`.
- L'attrition n'est pas due a une seule variable :
  elle semble multifactorielle.

**A montrer**
- 2 a 3 graphiques maximum, parmi les plus parlants :
  satisfaction globale vs attrition,
  revenu mensuel vs attrition,
  distribution par departement ou frequence de deplacement.

**Conseil**
- Ne pas montrer trop de graphes.
- Garde uniquement ceux qui racontent un message simple et actionnable.

---

## Slide 5 - Methodologie de modelisation

**Titre suggere**
Une demarche iterative, alignee avec le contexte metier

**Messages cles**
- Benchmark de plusieurs familles de modeles :
  Dummy, modeles lineaires, modeles non lineaires a base d'arbres.
- Metrique principale de tuning :
  `average_precision`, plus adaptee a une classe positive minoritaire.
- Choix du seuil fait sur le train uniquement, via validation croisee et probabilites `out-of-fold`.
- Comparaison de plusieurs strategies :
  seuil `0.5`, seuil optimisant le `F1`, seuil sous contrainte de rappel cible.

**A dire a l'oral**
Le point methodologique le plus important est que le seuil n'a pas ete choisi sur le jeu de test. Cela permet de garder une evaluation finale plus propre et d'eviter un score artificiellement optimiste.

---

## Slide 6 - Resultats de modelisation

**Titre suggere**
Un signal utile, mais une performance globalement moderee

**Messages cles**
- Les performances des baselines sont restees modestes.
- Le meilleur modele final est un `Elastic Net` enrichi :
  `elastic_net_fe_silent_attrition_v1_smote`.
- Resultats finaux au seuil optimise :
  `precision = 0.423`,
  `recall = 0.702`,
  `f1 = 0.528`,
  `average_precision / prc_auc = 0.536`.
- Les modeles plus complexes n'ont pas depasse clairement ce modele lineaire regularise.

**Lecture metier**
- Le modele est utile pour prioriser un risque d'attrition.
- En revanche, il ne permet pas une prediction "forte" ou exhaustive de tous les departs.

**A montrer**
- Un tableau simple :
  meilleur baseline,
  meilleur tuning,
  meilleur modele final.

---

## Slide 7 - Ce que l'error analysis a appris

**Titre suggere**
Les erreurs du modele sont surtout des limites de signal, pas de qualite de donnees

**Messages cles**
- Sur l'analyse finale :
  `59` erreurs au total, `45` faux positifs et `14` faux negatifs.
- Aucune erreur n'est expliquee par un vrai signal de donnee suspecte.
- Les faux positifs sont souvent des profils RH plausibles a risque :
  `Commercial`, `Consulting`, satisfaction basse a moyenne, revenu modere, mobilite.
- Les faux negatifs restants sont surtout des departs "silencieux" dans le `Consulting` :
  profils plus seniors, plus remuneres, ou plus stables en apparence.

**Conclusion slide**
- Le modele capte bien les profils fragiles "classiques".
- Il capte moins bien des departs plus complexes, probablement lies a des variables absentes du dataset.

---

## Slide 8 - Apport du feature engineering

**Titre suggere**
Le feature engineering cible ameliore legerement le modele

**Messages cles**
- De nouvelles features ont ete construites apres analyse des erreurs :
  ecart de salaire par rapport au poste,
  ecart de satisfaction par rapport au poste,
  indicateurs de stagnation,
  signaux de progression relative.
- Ces nouvelles features ont permis un gain reel mais modeste.
- Le gain vient surtout de variables continues relatives au groupe de reference.
- Les flags binaires tres specifiques n'ont pas ete retenus par l'Elastic Net.

**A dire a l'oral**
L'apport du feature engineering existe, mais il ne change pas la nature du probleme. On gagne un peu de signal, sans casser le plafond du dataset.

---

## Slide 9 - Feature importance globale

**Titre suggere**
Quels facteurs expliquent globalement le risque d'attrition ?

**Visuel a inserer**
- `reports/figure/260412-final_model_permutation_importance.png`

**Annexe technique possible**
- `reports/figure/260412-final_model_shap_beeswarm.png`

**Messages cles a presenter**
- Le risque d'attrition est principalement explique par un faisceau de signaux RH, et non par une seule variable.
- Les facteurs les plus structurants sont :
  `statut_marital`, `satisfaction_global`, `frequence_deplacement`,
  `promotion_delay`, `note_evaluation_precedente`, `role_stagnation_ratio`,
  `departement`, `revenu_bin`, `niveau_hierarchique_poste`, `age`.
- Les sorties SHAP confirment cette lecture et mettent aussi en avant :
  `training_gap_vs_department_median` et `salary_gap_vs_poste_median`.
- Le modele confirme donc une logique metier coherente :
  l'attrition est davantage associee a une combinaison d'insatisfaction,
  de mobilite, de progression limitee et de position relative moins favorable.

**Phrase orale possible**
Le point cle ici est que le modele ne raconte pas une cause unique du depart.
Il montre plutot qu'un salarie devient a risque lorsqu'on observe plusieurs
signaux faibles qui s'additionnent : moins de satisfaction, plus de mobilite,
une progression de carriere moins favorable et un positionnement relatif moins avantageux.

---

## Slide 10 - Feature importance locale

**Titre suggere**
Expliquer localement un risque de depart : deux cas concrets

**Visuels a inserer**
- `reports/figure/260412-final_model_tp_high_risk_waterfall.png`
- `reports/figure/260412-final_model_fn_silent_attrition_waterfall.png`

**Messages cles**
- Cas 1, depart correctement detecte :
  salarie `Commercial`, `Representant Commercial`, `celibataire`,
  avec deplacements frequents et faible revenu.
  Le modele identifie ici un profil RH classiquement fragile
  et attribue un risque eleve de depart.
- Cas 2, depart silencieux non detecte :
  salarie `Consulting`, `Senior Manager`, marie, bien remunere,
  avec une satisfaction encore correcte.
  Le modele est ici "rassure" par la stabilite apparente du profil
  et sous-estime le risque reel de depart.
- Ces deux exemples montrent la force et la limite du modele :
  il capte bien les profils a risque classiques,
  mais il manque encore une partie des departs plus discrets,
  notamment chez des profils seniors ou plus stables en apparence.

**Phrase orale possible**
Le premier cas illustre un depart que le modele comprend bien.
Le second est plus interessant, car il montre exactement la limite du dataset :
certains departs ne sont pas lies a une forte insatisfaction visible,
mais a des facteurs plus subtils que nos variables actuelles ne captent pas completement.

---

## Slide 11 - Limites et recommandations RH

**Titre suggere**
Ce que l'on peut deja faire, et ce qu'il manque pour aller plus loin

**Messages cles**
- Le modele est utile comme outil d'aide a la priorisation, pas comme outil de certitude.
- Les profils les plus a risque combinent souvent :
  satisfaction plus basse,
  mobilite,
  remuneration relative moins favorable,
  sentiment de stagnation.
- Recommandations RH possibles :
  cibler certains segments `Commercial` et `Consulting`,
  renforcer les points d'alerte autour de la satisfaction et de la progression,
  approfondir les cas de stagnation percue.
- Pour ameliorer le modele, il faudrait enrichir les donnees avec :
  contexte de mission,
  management,
  charge de travail,
  trajectoire de carriere plus fine.

---

## Slide 12 - Conclusion

**Titre suggere**
Conclusion

**Messages cles**
- Le projet a permis de construire une vue employee consolidee a partir de trois sources.
- Les analyses exploratoires et le modele mettent en evidence plusieurs facteurs associes a l'attrition.
- Le meilleur modele obtenu est utile pour scorer et prioriser le risque, mais reste limite par la richesse du dataset.
- La principale valeur du travail est double :
  objectiver les signaux de risque et proposer une base exploitable pour des actions RH ciblees.

---

## Annexes suggerees

- Tableau comparatif des principaux modeles testes.
- Details sur la selection du seuil.
- Resultats de l'error analysis.
- Details techniques du preprocessing.

---

## Ce qu'il manque encore pour coller parfaitement au brief

- Une vraie slide de feature importance globale avec visuel final.
- Une vraie slide d'explication locale au format SHAP / waterfall si tu veux etre parfaitement aligne avec l'etape 5 et l'etape 6.

En l'etat, la narration est prete. Il reste surtout a transformer l'interpretabilite en 2 ou 3 visuels lisibles pour l'audience.
