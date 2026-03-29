"""
Construction d'un dataset en panel par fenêtre glissante mensuelle
pour la prédiction de résiliation de contrats d'assurance.
"""

import os
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta


def create_sliding_window_dataset(
    df, start_date="2015-01-31", end_date="2025-12-31"
):
    """
    Construit un dataset en panel à partir d'un DataFrame de contrats d'assurance.

    Pour chaque mois entre start_date et end_date, la fonction sélectionne les
    contrats actifs à cette date et crée une observation par contrat, enrichie
    de variables temporelles (ancienneté, mois, année, augmentations tarifaires).
    La variable cible 'resilie' vaut 1 si le contrat est résilié dans le mois
    suivant la date de référence, 0 sinon.

    Les contrats de moins de 12 mois d'ancienneté sont exclus.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame brut issu de data/raw/apicil.csv.
    start_date : str
        Date de début de la fenêtre glissante (format 'YYYY-MM-DD').
    end_date : str
        Date de fin de la fenêtre glissante (format 'YYYY-MM-DD').

    Renvoie
    -------
    pd.DataFrame
        Dataset en panel avec une ligne par (contrat, mois).
    """
    rows = []

    df["Contrat : Date de début d'effet"] = pd.to_datetime(
        df["Contrat : Date de début d'effet"]
    )
    df["Contrat : Date de fin d'effet"] = pd.to_datetime(
        df["Contrat : Date de fin d'effet"]
    )

    df = df.rename(columns={
        "Augmentation 2025 (au 01/01)": "Augmentation 2025",
        "A 2021": "annee_2021"
    })

    reference_dates = []
    current_date = pd.to_datetime(start_date)
    final_date = pd.to_datetime(end_date)

    while current_date <= final_date:
        reference_dates.append(current_date)
        current_date += relativedelta(months=1, day=31)

    print(f"Nombre de fenêtres : {len(reference_dates)}")

    cols_augmentation = [col for col in df.columns if col.startswith("Augmenta")]

    for date_ref in reference_dates:
        print(f"Traitement date : {date_ref.strftime('%Y-%m-%d')}")

        contrats_actifs = df[
            (df["Contrat : Date de début d'effet"] <= date_ref) &
            (
                (df["Contrat : Date de fin d'effet"] > date_ref) |
                (df["Contrat : Date de fin d'effet"].isna())
            )
        ].copy()

        contrats_actifs["anciennete_jours"] = (
            date_ref - contrats_actifs["Contrat : Date de début d'effet"]
        ).dt.days
        contrats_actifs["anciennete_mois"] = (
            contrats_actifs["anciennete_jours"] / 30.44
        )

        contrats_actifs["mois"] = date_ref.month
        contrats_actifs["année"] = date_ref.year

        if date_ref > pd.to_datetime("2023-04-30"):
            contrats_actifs["nb_reclamations"] = (
                contrats_actifs["Nb de récla depuis mai 2023"]
            )
        else:
            contrats_actifs["nb_reclamations"] = np.nan

        contrats_actifs = contrats_actifs[
            contrats_actifs["anciennete_mois"] >= 12
        ].copy()

        contrats_actifs["Compte personnel\xa0: Âge"] = (
            contrats_actifs["Compte personnel\xa0: Âge"]
            - ((final_date - date_ref).days / 365.25)
        )

        annee_ref = date_ref.year
        for col in cols_augmentation:
            annee_col = int(col.split(" ")[1])
            contrats_actifs[col] = np.where(annee_ref < annee_col, np.nan, contrats_actifs[col])

        if len(contrats_actifs) == 0:
            continue

        date_fin_fenetre = date_ref + relativedelta(months=1)
        contrats_actifs["resilie"] = (
            (contrats_actifs["Contrat : Date de fin d'effet"] > date_ref) &
            (contrats_actifs["Contrat : Date de fin d'effet"] <= date_fin_fenetre)
        ).astype(int)

        contrats_actifs["date_reference"] = date_ref

        rows.append(contrats_actifs)

    dataset_final = pd.concat(rows, ignore_index=True)

    print(f"\nNotre dataset final : {len(dataset_final):,} observations")
    print(f"Taux de résiliation : {dataset_final['resilie'].mean():.2%}")

    return dataset_final


if __name__ == "__main__":
    output_path = os.environ.get("PANEL_LOCAL_PATH", "data/processed/panel.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Chargement des données brutes...")
    df_raw = pd.read_csv("data/raw/apicil.csv")

    print("Construction du panel...")
    df_panel = create_sliding_window_dataset(df_raw)

    df_panel.to_csv(output_path, index=False)
    print(f"Panel sauvegardé dans {output_path} ({len(df_panel):,} lignes)")