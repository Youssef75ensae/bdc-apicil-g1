"""
Fonctions utilitaires pour la modélisation du churn.

Contient le split train/test temporel, l'entraînement, l'évaluation
et la sauvegarde des modèles.

Usage
-----
    from src.model import FEATURES, split_data, train_model, evaluate, save_model
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    brier_score_loss,
    average_precision_score,
    classification_report,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)


TARGET = "resilie"
SPLIT_YEAR = 2025
MODELS_DIR = "models/auc_roc"

FEATURES = [
    'Sexe',
    'Compte personnel\xa0: Âge',
    'Est salarié ?',
    "Nb d'ayant droit",
    'Nb de récla depuis mai 2023',
    'Délai moyen traitement récla (en J)',
    "Volume d'appel pris",
    'Mois gratuit',
    'Volume demande \nGestion De la Personne',
    'Délai de traitement (GDP)',
    'Volume demande \nConnexion Noémie',
    'Délai de traitement (Noémie)',
    'Volume demande \nCotisations',
    'Délai de traitement (Cotisation)',
    'Volume demande \nDevis médical',
    'Délai de traitement (Devis)',
    'Volume demande \nDe prestations',
    'Délai de traitement (Prestations)',
    'Volume demande \nSuivi Client',
    'Délai de traitement (Suivi client)',
    'Augmentation 2018',
    'Augmentation 2019',
    'Augmentation 2020',
    'Augmentation 2021',
    'Augmentation 2022',
    'Augmentation 2023',
    'Augmentation 2024',
    'Augmentation 2025',
    'CSAT',
    'NPS',
    'CES',
    'COS',
    'anciennete_jours',
    'indice_prix_conso',
    'indice_prix_conso_complementaires_sante',
    'indice_confiance_menages',
    'taux_chomage',
    "delta_chomage",
    "ecart_chomage_national",
    'age_Q1',
    'age_Q2',
    'age_Q3',
    'age_Q4',
    'age_Q5',
    'score_digital',
    'delai_depuis_derniere_recla',
    'aug_cumulee',
    'aug_moyenne',
    'aug_derniere',
    'aug_avant_derniere',
    'aug_acceleration',
    'aug_volatilite',
    'satisfaction_globale',
    'mois_reference',
    'mois_1', 'mois_2', 'mois_3', 'mois_4', 'mois_5', 'mois_6',
    'mois_7', 'mois_8', 'mois_9', 'mois_10', 'mois_11', 'mois_12',
    'mois_sin',
    'mois_cos',
    'garantie_Elevé',
    'garantie_Faible',
    'garantie_Modéré',
    'garantie_Non Renseigné',
    'sit_fam_Concubin(e)',
    'sit_fam_Célibataire',
    'sit_fam_Divorcé(e)',
    'sit_fam_Inconnue',
    'sit_fam_Marié(e)',
    'sit_fam_PACS',
    'sit_fam_Séparé(e)',
    'sit_fam_Veuf(ve)',
    'zone_critique',
    'annee_2016', 'annee_2017', 'annee_2018', 'annee_2019', 'annee_2020',
    'annee_2021', 'annee_2022', 'annee_2023', 'annee_2024',
    'reg_Auvergne-Rhône-Alpes',
    'reg_Bourgogne-Franche-Comté',
    'reg_Bretagne',
    'reg_Centre-Val de Loire',
    'reg_Grand Est',
    'reg_Guadeloupe',
    'reg_Guyane',
    'reg_Hauts-de-France',
    'reg_Hors France',
    'reg_Inconnu',
    'reg_La Réunion',
    'reg_Martinique',
    'reg_Mayotte',
    'reg_Normandie',
    'reg_Nouvelle-Aquitaine',
    'reg_Occitanie',
    'reg_Pays de la Loire',
    "reg_Provence-Alpes-Côte d'Azur",
    'reg_Île-de-France'
]


# ---------------------------------------------------------------------------
# Split train / test temporel
# ---------------------------------------------------------------------------

def split_data(
    df: pd.DataFrame,
    features: list = None,
    target: str = TARGET,
    split_year: int = SPLIT_YEAR,
) -> tuple:
    """
    Découpe le panel en train/test selon une coupure temporelle.

    Toutes les observations avant split_year sont dans le train,
    les observations de split_year et après sont dans le test.
    Ce split temporel évite tout data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Panel enrichi avec features.
    features : list[str]
        Liste des colonnes à utiliser comme variables explicatives.
        Si None, utilise la liste FEATURES par défaut.
    target : str
        Nom de la colonne cible.
    split_year : int
        Année de coupure (défaut : 2025).

    Returns
    -------
    x_train, x_test, y_train, y_test : tuple de DataFrames/Series
    """
    if features is None:
        features = FEATURES

    df["date_reference"] = pd.to_datetime(df["date_reference"])

    train = df[df["date_reference"].dt.year < split_year]
    test  = df[df["date_reference"].dt.year >= split_year]

    x_train = train[features]
    x_test  = test[features]
    y_train = train[target]
    y_test  = test[target]

    print(f"Train : {len(x_train):,} observations ({y_train.mean():.2%} de résiliations)")
    print(f"Test  : {len(x_test):,} observations ({y_test.mean():.2%} de résiliations)")

    return x_train, x_test, y_train, y_test


# ---------------------------------------------------------------------------
# Entraînement
# ---------------------------------------------------------------------------

def train_model(model, x_train: pd.DataFrame, y_train: pd.Series):
    """
    Entraîne un modèle sklearn-compatible.

    Parameters
    ----------
    model : estimateur sklearn
        Modèle à entraîner (LogisticRegression, RandomForest, XGBoost...).
    x_train : pd.DataFrame
    y_train : pd.Series

    Returns
    -------
    model : estimateur entraîné
    """
    print(f"Entraînement de {model.__class__.__name__}...")
    model.fit(x_train, y_train)
    print("Entraînement terminé.")
    return model


# ---------------------------------------------------------------------------
# Évaluation
# ---------------------------------------------------------------------------

def evaluate(
    model,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    plot_roc: bool = True,
) -> dict:
    """
    Évalue un modèle entraîné sur le jeu de test.

    Calcule un ensemble de métriques adaptées aux classes déséquilibrées :
    AUC-ROC, AUC-PR, F1, F2, recall, précision, MCC et Brier score.

    Parameters
    ----------
    model : estimateur sklearn entraîné
    x_test : pd.DataFrame
    y_test : pd.Series
    plot_roc : bool
        Si True, affiche la courbe ROC et la courbe Précision-Recall.

    Returns
    -------
    dict : dictionnaire des métriques
    """
    y_proba = model.predict_proba(x_test)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)

    auc       = roc_auc_score(y_test, y_proba)
    auc_pr    = average_precision_score(y_test, y_proba)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    f2        = fbeta_score(y_test, y_pred, beta=2, zero_division=0)
    mcc       = matthews_corrcoef(y_test, y_pred)
    brier     = brier_score_loss(y_test, y_proba)

    print(f"\n{'='*40}")
    print(f"Modèle : {model.__class__.__name__}")
    print(f"{'='*40}")
    print(f"AUC-ROC   : {auc:.4f}  — aire sous la courbe ROC")
    print(
        f"AUC-PR    : {auc_pr:.4f}"
        f"  — aire sous la courbe Précision-Recall"
        f" (plus fiable avec classes déséquilibrées)"
    )
    print(f"Recall    : {recall:.4f}  — taux de résiliants détectés")
    print(f"Précision : {precision:.4f}  — part de vrais résiliants parmi les alertes")
    print(f"F1        : {f1:.4f}  — équilibre recall/précision")
    print(f"F2        : {f2:.4f}  — pénalise davantage les faux négatifs")
    print(f"MCC       : {mcc:.4f}  — robuste au déséquilibre des classes")
    print(f"Brier     : {brier:.4f}  — calibration des probabilités (plus bas = mieux)")
    print(f"\n{classification_report(y_test, y_pred, zero_division=0)}")

    if plot_roc:
        _, axes = plt.subplots(1, 2, figsize=(12, 5))
        RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[0])
        axes[0].set_title(f"Courbe ROC — {model.__class__.__name__}")
        PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=axes[1])
        axes[1].set_title(f"Courbe Précision-Recall — {model.__class__.__name__}")
        plt.tight_layout()
        plt.show()

    return {
        "auc":       auc,
        "auc_pr":    auc_pr,
        "recall":    recall,
        "precision": precision,
        "f1":        f1,
        "f2":        f2,
        "mcc":       mcc,
        "brier":     brier,
    }


# ---------------------------------------------------------------------------
# Sauvegarde et chargement
# ---------------------------------------------------------------------------

def save_model(model, name: str) -> str:
    """
    Sauvegarde un modèle entraîné dans le dossier models/.

    Parameters
    ----------
    model : estimateur sklearn entraîné
    name : str
        Nom du fichier sans extension (ex: 'logistic_regression').

    Returns
    -------
    str : chemin du fichier sauvegardé
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"Modèle sauvegardé dans {path}")
    return path


def load_model(name: str):
    """
    Charge un modèle sauvegardé depuis le dossier models/.

    Parameters
    ----------
    name : str
        Nom du fichier sans extension (ex: 'logistic_regression').

    Returns
    -------
    estimateur sklearn
    """
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    model = joblib.load(path)
    print(f"Modèle chargé depuis {path}")
    return model