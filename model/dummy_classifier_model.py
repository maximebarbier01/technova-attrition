from __future__ import annotations

from sklearn.dummy import DummyClassifier

def get_dummy_model(
    strategy: str = "most_frequent",
    random_state: int = 42,
) -> DummyClassifier:
    """
    Retourne un modèle de classification de base (DummyClassifier).

    Parameters
    ----------
    strategy : str
        Stratégie de prédiction ('most_frequent', 'stratified', etc.).
    random_state : int
        Graine aléatoire.

    Returns
    -------
    DummyClassifier
        Modèle sklearn initialisé.
    """
    return DummyClassifier(strategy=strategy, random_state=random_state)