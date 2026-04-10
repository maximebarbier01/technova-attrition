from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
from sqlalchemy import create_engine, text


# *=============================================================================
# * PARAMETRES
# *=============================================================================

DB_USER = "maxime"
DB_PASSWORD = "@udrey29Le"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "technova-attrition"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

FILES = {
    "extrait_eval": DATA_DIR / "extrait_eval.csv",
    "extrait_sirh": DATA_DIR / "extrait_sirh.csv",
    "extrait_sondage": DATA_DIR / "extrait_sondage.csv",
}


# =============================================================================
# OUTILS
# =============================================================================

def clean_column_name(col: str) -> str:
    """
    Nettoie un nom de colonne pour PostgreSQL / SQLAlchemy.
    """
    col = col.strip().lower()
    col = col.replace("é", "e").replace("è", "e").replace("ê", "e")
    col = col.replace("à", "a").replace("â", "a")
    col = col.replace("î", "i").replace("ï", "i")
    col = col.replace("ô", "o")
    col = col.replace("ù", "u").replace("û", "u").replace("ü", "u")
    col = col.replace("ç", "c")

    col = re.sub(r"[ \-\/]+", "_", col)
    col = re.sub(r"[()'\"]+", "", col)
    col = re.sub(r"[^a-z0-9_]", "", col)
    col = re.sub(r"_+", "_", col)
    col = col.strip("_")

    if not col:
        col = "colonne_sans_nom"

    if col[0].isdigit():
        col = f"col_{col}"

    return col


def load_one_csv(
    engine,
    file_path: Path,
    table_name: str,
    schema: str = "raw",
    if_exists: str = "replace",
    sep: str = ",",
    encoding: str = "utf-8",
) -> None:
    """
    Charge un CSV dans PostgreSQL.
    Toutes les colonnes sont lues en texte pour éviter les erreurs de typage.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    print(f"\nLecture du fichier : {file_path}")

    df = pd.read_csv(
        file_path,
        dtype=str,
        sep=sep,
        encoding=encoding,
    )

    df.columns = [clean_column_name(col) for col in df.columns]

    # Remplace NaN pandas par None pour un insert SQL propre
    df = df.where(pd.notna(df), None)

    print(f"Chargement dans {schema}.{table_name} ...")
    print(f"Shape : {df.shape}")

    df.to_sql(
        name=table_name,
        con=engine,
        schema=schema,
        if_exists=if_exists,
        index=False,
        method="multi",
        chunksize=1000,
    )

    print(f"Table chargée : {schema}.{table_name}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    connection_url = (
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
        f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    engine = create_engine(connection_url)

    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS raw;"))

    for table_name, file_path in FILES.items():
        load_one_csv(
            engine=engine,
            file_path=file_path,
            table_name=table_name,
            schema="raw",
            if_exists="replace",   # ou "fail" si tu veux éviter d'écraser
            sep=",",
            encoding="utf-8",
        )

    print("\nChargement terminé avec succès.")


if __name__ == "__main__":
    main()