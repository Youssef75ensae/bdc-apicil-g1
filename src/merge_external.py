"""
Fusion du panel avec les données externes issues de l'INSEE.

Lit data/processed/panel.csv et les quatre fichiers de data/external/,
et produit data/processed/panel_ext.csv enrichi des indicateurs macroéconomiques.

Usage
-----
    python src/merge_external.py
"""

import os
import pandas as pd


PANEL_PATH = os.environ.get("PANEL_LOCAL_PATH", "data/processed/panel.csv")
OUT_PATH   = os.environ.get("PANEL_EXT_LOCAL_PATH", "data/processed/panel_ext.csv")

EXTERNAL = {
    "indice_prix_conso_general": {
        "path": "data/external/transformed/indice_prix_conso_general.csv",
        "value_col": "indice_prix_conso_general",
    },
    "indice_prix_conso_complementaires_sante": {
        "path": "data/external/transformed/indice_prix_conso_complementaires_sante.csv",
        "value_col": "indice_prix_conso_complementaires_sante",
    },
    "indice_confiance_des_menages": {
        "path": "data/external/transformed/indice_confiance_des_menages.csv",
        "value_col": "indice_confiance_des_menages",
    },
    "taux_chomage_departement": {
        "path": "data/external/transformed/taux_chomage_departement.csv",
        "value_col": "taux_chomage",
        "key_col": "code_dept",  # jointure sur deux clés
    },
}


def load_external(path: str, value_col: str, key_col: str = None) -> pd.DataFrame:
    """
    Charge un fichier externe INSEE et renomme la colonne valeur.

    Parameters
    ----------
    path : str
        Chemin vers le fichier CSV externe.
    value_col : str
        Nom à donner à la colonne valeur dans le DataFrame résultat.
    key_col : str, optional
        Colonne de clé supplémentaire (ex: code_dept). None si absent.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(path, sep=";", dtype=str)
    df = df.rename(columns={"valeur": value_col})
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    return df


if __name__ == "__main__":
    print("Chargement du panel...")
    panel = pd.read_csv(PANEL_PATH)
    print(f"  {len(panel):,} lignes chargées")

    # Clé de jointure temporelle : extraire YYYY-MM depuis date_reference (YYYY-MM-DD)
    panel["date_ym"] = panel["date_reference"].str[:7]

    # --- Fusion des trois indicateurs mensuels (clé : date_ym) ---
    for name, meta in EXTERNAL.items():
        if "key_col" in meta:
            continue  # traité séparément

        print(f"Fusion avec {meta['path']} ...")
        df_ext = load_external(meta["path"], meta["value_col"])
        df_ext = df_ext.rename(columns={"date": "date_ym"})

        panel = panel.merge(df_ext[["date_ym", meta["value_col"]]], on="date_ym", how="left")
        print(f"  OK — {panel[meta['value_col']].isna().sum()} valeurs manquantes")

    # --- Fusion du taux de chômage (clé : date_ym + code_dept) ---
    meta_chomage = EXTERNAL["taux_chomage_departement"]
    print(f"Fusion avec {meta_chomage['path']} ...")
    df_chomage = load_external(
        meta_chomage["path"],
        meta_chomage["value_col"],
        key_col=meta_chomage["key_col"],
    )
    df_chomage = df_chomage.rename(columns={"date": "date_ym"})
    df_chomage["code_dept"] = df_chomage["code_dept"].astype(str).str.zfill(2)

    # Adapter le format du code département dans le panel si nécessaire
    # Remplacez "Contrat : Code département" par le nom exact de la colonne dans vos données
    COL_DEPT = "Département"
    panel[COL_DEPT] = panel[COL_DEPT].astype(str).str.zfill(2)

    panel = panel.merge(
        df_chomage[["code_dept", "date_ym", meta_chomage["value_col"]]],
        left_on=[COL_DEPT, "date_ym"],
        right_on=["code_dept", "date_ym"],
        how="left",
    ).drop(columns=["code_dept"])

    print(f"  OK — {panel[meta_chomage['value_col']].isna().sum()} valeurs manquantes")

    # Supprimer la clé temporaire
    panel = panel.drop(columns=["date_ym"])

    # Sauvegarde
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    panel.to_csv(OUT_PATH, index=False)
    print(f"\npanel_ext.csv sauvegardé dans {OUT_PATH} ({len(panel):,} lignes)")