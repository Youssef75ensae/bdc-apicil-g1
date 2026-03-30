"""
Transformation des données externes issues de l'INSEE.

Les fichiers bruts doivent être téléchargés manuellement depuis l'INSEE
et placés dans data/external/raw/ avant de lancer ce script.

Sources :
- Indice des prix — complémentaires santé (idBank 001762477) :
  https://www.insee.fr/fr/statistiques/serie/001762477
- Indice des prix — ensemble hors tabac (idBank 011814056) :
  https://www.insee.fr/fr/statistiques/serie/011814056
- Indicateur de confiance des ménages (idBank 8730744) :
  https://www.insee.fr/fr/statistiques/8730744
- Taux de chômage localisé par département :
  https://www.insee.fr/fr/statistiques/2012804#tableau-TCRD_025_tab1_departements

Usage
-----
    python src/transform_external.py
"""

import os
import pandas as pd


RAW_DIR = "data/external/raw"
OUT_DIR = "data/external"

# Correspondance trimestre -> mois
TRIMESTRE_TO_MONTHS = {
    "T1": ["01", "02", "03"],
    "T2": ["04", "05", "06"],
    "T3": ["07", "08", "09"],
    "T4": ["10", "11", "12"],
}


# ---------------------------------------------------------------------------
# Fonctions de transformation
# ---------------------------------------------------------------------------

def transform_indice_csv(raw_filename: str, output_filename: str, description: str) -> None:
    """
    Transforme un fichier CSV INSEE (format brut avec métadonnées)
    vers le format date;valeur.

    Parameters
    ----------
    raw_filename : str
        Nom du fichier brut dans data/external/raw/.
    output_filename : str
        Nom du fichier transformé dans data/external/.
    description : str
        Description de la série pour les logs.
    """
    raw_path = os.path.join(RAW_DIR, raw_filename)
    output_path = os.path.join(OUT_DIR, output_filename)

    print(f"\n{description}")
    print(f"Lecture de {raw_path} ...")

    df = pd.read_csv(
        raw_path, sep=";", skiprows=4, header=None, usecols=[0, 1], dtype=str
    )
    df.columns = ["date", "valeur"]
    df["date"] = df["date"].str.strip('"')
    df["valeur"] = df["valeur"].str.strip('"')

    df.to_csv(output_path, sep=";", index=False)
    print(f"Sauvegardé dans {output_path} ({len(df)} lignes)")


def transform_indice_xlsx(
    raw_filename: str,
    output_filename: str,
    description: str,
    sheet: str,
    skiprows: int,
    date_min: str,
    date_max: str,
) -> None:
    """
    Transforme un fichier XLSX INSEE vers le format date;valeur,
    filtré sur une plage de dates.

    Parameters
    ----------
    raw_filename : str
        Nom du fichier brut dans data/external/raw/.
    output_filename : str
        Nom du fichier transformé dans data/external/.
    description : str
        Description de la série pour les logs.
    sheet : str
        Nom de la feuille Excel à lire.
    skiprows : int
        Nombre de lignes d'en-tête à ignorer.
    date_min : str
        Date de début du filtre (format 'YYYY-MM-DD', incluse).
    date_max : str
        Date de fin du filtre (format 'YYYY-MM-DD', incluse).
    """
    raw_path = os.path.join(RAW_DIR, raw_filename)
    output_path = os.path.join(OUT_DIR, output_filename)

    print(f"\n{description}")
    print(f"Lecture de {raw_path} ...")

    df = pd.read_excel(
        raw_path,
        sheet_name=sheet,
        skiprows=skiprows,
        header=None,
        usecols=[0, 1],
        dtype=str,
    )
    df.columns = ["date", "valeur"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[(df["date"] >= date_min) & (df["date"] <= date_max)]
    df["date"] = df["date"].dt.strftime("%Y-%m")

    df.to_csv(output_path, sep=";", index=False)
    print(f"Sauvegardé dans {output_path} ({len(df)} lignes)")


def transform_chomage_dept(
    raw_filename: str,
    output_filename: str,
    description: str,
) -> None:
    """
    Transforme le fichier XLS du taux de chômage localisé par département
    vers le format long code_dept;date;valeur.

    Chaque valeur trimestrielle est répétée sur les trois mois du trimestre
    (ex : T1-2015 → 2015-01, 2015-02, 2015-03 avec la même valeur).

    Parameters
    ----------
    raw_filename : str
        Nom du fichier brut dans data/external/raw/.
    output_filename : str
        Nom du fichier transformé dans data/external/.
    description : str
        Description de la série pour les logs.
    """
    raw_path = os.path.join(RAW_DIR, raw_filename)
    output_path = os.path.join(OUT_DIR, output_filename)

    print(f"\n{description}")
    print(f"Lecture de {raw_path} ...")

    # Colonnes 0 (code) + 134 à 177 (T1_2015 à T4_2025)
    # Lignes 4 à 103 (départements, hors en-têtes et notes de bas de page)
    col_code = 0
    col_start = 134  # T1_2015
    col_end = 177    # T4_2025

    df_raw = pd.read_excel(
        raw_path,
        sheet_name="Département",
        header=None,
    )

    headers = df_raw.iloc[3].tolist()
    trimestre_cols = {
        headers[i]: i for i in range(col_start, col_end + 1)
    }

    df = df_raw.iloc[4:104, [col_code] + list(range(col_start, col_end + 1))].copy()
    df.columns = ["code_dept"] + list(trimestre_cols.keys())
    df = df.dropna(subset=["code_dept"])
    df["code_dept"] = df["code_dept"].astype(str).str.strip().str.zfill(2)

    # Passage en format long (une ligne par trimestre)
    df_long = df.melt(id_vars="code_dept", var_name="trimestre", value_name="valeur")

    # Expansion : une ligne par mois
    rows = []
    for _, row in df_long.iterrows():
        # trimestre au format "T1_2015"
        parts = str(row["trimestre"]).split("_")
        trimestre = parts[0]   # ex: "T1"
        annee = parts[1]       # ex: "2015"
        for mois in TRIMESTRE_TO_MONTHS[trimestre]:
            rows.append({
                "code_dept": row["code_dept"],
                "date": f"{annee}-{mois}",
                "valeur": row["valeur"],
            })

    dataset_final = pd.DataFrame(rows)
    dataset_final = dataset_final.sort_values(["code_dept", "date"]).reset_index(drop=True)

    dataset_final.to_csv(output_path, sep=";", index=False)
    print(f"Sauvegardé dans {output_path} ({len(dataset_final)} lignes)")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    transform_indice_csv(
        raw_filename="indice_prix_conso_complementaires_sante.csv",
        output_filename="indice_prix_conso_complementaires_sante.csv",
        description="Indice des prix à la consommation — complémentaires santé (idBank 001762477)",
    )

    transform_indice_csv(
        raw_filename="indice_prix_conso_general.csv",
        output_filename="indice_prix_conso.csv",
        description="Indice des prix à la consommation — ensemble hors tabac (idBank 011814056)",
    )

    transform_indice_xlsx(
        raw_filename="indice_confiance_des_menages.xlsx",
        output_filename="indice_confiance_menages.csv",
        description="Indicateur synthétique de confiance des ménages (idBank 8730744)",
        sheet="C.A.M.",
        skiprows=6,
        date_min="2015-01-01",
        date_max="2025-12-31",
    )

    transform_chomage_dept(
        raw_filename="taux_de_chomage_departement.xls",
        output_filename="taux_chomage.csv",
        description="Taux de chômage localisé par département (T1-2015 à T4-2025)",
    )

    print("\nToutes les transformations ont été effectuées.")