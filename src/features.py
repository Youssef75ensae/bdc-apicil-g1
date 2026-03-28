"""
Feature engineering à partir du panel enrichi (panel_ext.csv).

Usage
-----
    from src.features import build_features

    df = pd.read_csv("data/processed/panel_ext.csv")
    df = build_features(df)
"""

import os
import numpy as np
import pandas as pd


PANEL_EXT_PATH = os.environ.get("PANEL_EXT_LOCAL_PATH", "data/processed/panel_ext.csv")
OUT_PATH = os.environ.get("FEATURES_LOCAL_PATH", "data/processed/panel_final.csv")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit toutes les variables à partir du panel enrichi.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame issu de data/processed/panel_ext.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame enrichi avec les nouvelles variables.
    """
    df = df.copy()

    # -------------------------------------------------------------------------
    # Âge : discrétisation en quintiles et encodage one-hot
    # pd.qcut découpe l'âge en 5 classes de taille égale (Q1 = les plus jeunes,
    # Q5 = les plus âgés), ce qui évite les classes déséquilibrées qu'on
    # obtiendrait avec des intervalles fixes.
    # -------------------------------------------------------------------------
    df["age_bin"] = pd.qcut(
        df["Compte personnel\xa0: Âge"],
        q=5,
        labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
    )
    age_dummies = pd.get_dummies(df["age_bin"], prefix="age").astype(int)
    df = pd.concat([df, age_dummies], axis=1)
    df = df.drop(columns=["age_bin"])

    # -------------------------------------------------------------------------
    # Score digital : moyenne de trois indicateurs binaires de maturité numérique
    # (email connu, téléphone connu, espace digital ouvert).
    # Vaut 0 si aucun canal numérique n'est renseigné, 1 si tous le sont.
    # -------------------------------------------------------------------------
    df["score_digital"] = (1 / 3) * (
        df["Connaissance email ?"]
        + df["Connaissance tel ?"]
        + df["Compte personnel\xa0: Espace digital ouvert ?"]
    )

    # -------------------------------------------------------------------------
    # Délai depuis la dernière réclamation
    # Exprimé en jours entre la date de référence et la date de la dernière
    # réclamation. NaN si le client n'a jamais réclamé.
    # -------------------------------------------------------------------------
    df["Date dernière récla"] = pd.to_datetime(df["Date dernière récla"], errors="coerce")
    df["date_reference"] = pd.to_datetime(df["date_reference"], errors="coerce")
    df["delai_depuis_derniere_recla"] = (
        df["date_reference"] - df["Date dernière récla"]
    ).dt.days

    # -------------------------------------------------------------------------
    # Variables sur les augmentations tarifaires annuelles
    # -------------------------------------------------------------------------
    aug_cols = [
        "Augmentation 2018", "Augmentation 2019", "Augmentation 2020",
        "Augmentation 2021", "Augmentation 2022", "Augmentation 2023",
        "Augmentation 2024", "Augmentation 2025"
    ]

    # Augmentation cumulée : produit des (1 + taux) sur toutes les années connues.
    # Donne la hausse totale de la prime depuis 2018, en proportion.
    df["aug_cumulee"] = (1 + df[aug_cols] / 100).prod(axis=1) - 1

    # Augmentation moyenne : taux d'augmentation annuel moyen sur les années connues.
    df["aug_moyenne"] = df[aug_cols].mean(axis=1)

    # Dernière et avant-dernière augmentation connue.
    # ffill(axis=1) propage la dernière valeur non-NaN vers la droite,
    # ce qui permet de récupérer la valeur la plus récente disponible
    # quelle que soit la date de référence de l'observation.
    df["aug_derniere"] = df[aug_cols].ffill(axis=1).iloc[:, -1]
    df["aug_avant_derniere"] = df[aug_cols].ffill(axis=1).iloc[:, -2]

    # Accélération : différence entre la dernière et l'avant-dernière augmentation.
    # Une valeur positive signale une hausse qui s'accélère, potentiellement
    # un facteur déclencheur de résiliation.
    df["aug_acceleration"] = df["aug_derniere"] - df["aug_avant_derniere"]

    # Volatilité des augmentations : écart-type sur les années connues.
    # Mesure l'instabilité tarifaire perçue par le client.
    df["aug_volatilite"] = df[aug_cols].std(axis=1)

    # -------------------------------------------------------------------------
    # Score de satisfaction globale : moyenne des quatre indicateurs de satisfaction
    # (CSAT, NPS, CES, COS). NaN si aucun indicateur n'est renseigné.
    # -------------------------------------------------------------------------
    df["satisfaction_globale"] = df[["CSAT", "NPS", "CES", "COS"]].mean(axis=1)

    # -------------------------------------------------------------------------
    # Encodage temporel : mois de la date de référence
    # Trois représentations complémentaires :
    # - One-hot (mois_1 à mois_12) : capture les effets mensuels non linéaires
    # - Cyclique sin/cos : préserve la continuité entre décembre et janvier,
    #   utile pour les modèles qui exploitent la distance entre observations
    # -------------------------------------------------------------------------
    df["mois_reference"] = pd.to_datetime(df["date_reference"]).dt.month
    for m in range(1, 13):
        df[f"mois_{m}"] = (df["mois_reference"] == m).astype(int)
    df["mois_sin"] = np.sin(2 * np.pi * df["mois_reference"] / 12)
    df["mois_cos"] = np.cos(2 * np.pi * df["mois_reference"] / 12)

    # -------------------------------------------------------------------------
    # Encodage temporel : année de la date de référence (one-hot)
    # Permet de capturer des effets conjoncturels propres à chaque année
    # (ex : réforme réglementaire, crise sanitaire).
    # -------------------------------------------------------------------------
    df["annee_reference"] = pd.to_datetime(df["date_reference"]).dt.year
    for a in range(2015, 2027):
        df[f"annee_{a}"] = (df["annee_reference"] == a).astype(int)

    # -------------------------------------------------------------------------
    # Encodage one-hot des variables catégorielles
    # -------------------------------------------------------------------------

    # Niveau de garanties : capture l'effet du niveau de couverture sur la résiliation
    garanties_dummies = pd.get_dummies(df["Niveau garanties"], prefix="garantie").astype(int)
    df = pd.concat([df, garanties_dummies], axis=1)
    df = df.drop(columns=["Niveau garanties"])

    # Situation familiale : célibataire, marié, etc.
    sit_fam_dummies = pd.get_dummies(df["Situation familiale"], prefix="sit_fam").astype(int)
    df = pd.concat([df, sit_fam_dummies], axis=1)
    df = df.drop(columns=["Situation familiale"])

    # -------------------------------------------------------------------------
    # Suppression des colonnes de dates brutes, devenues inutiles après
    # la construction des variables temporelles et d'ancienneté dans panel.py
    # -------------------------------------------------------------------------
    df = df.drop(columns=["Contrat : Date de début d'effet", "Contrat : Date de fin d'effet"])

    # Encodage binaire du sexe : 1 = Masculin, 0 = Féminin
    df["Sexe"] = (df["Sexe"] == "Masculin").astype(int)

    # Zone critique (12-15 mois)
    df['zone_critique'] = ((df['anciennete_mois'] >= 12) & (df['anciennete_mois'] <= 15)).astype(int)

    return df

if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    print("Chargement du panel enrichi...")
    df = pd.read_csv(PANEL_EXT_PATH)
    print(f"  {len(df):,} lignes chargées")

    print("Construction des features...")
    df = build_features(df)

    df.to_csv(OUT_PATH, index=False)
    print(f"Sauvegardé dans {OUT_PATH} ({len(df):,} lignes)")