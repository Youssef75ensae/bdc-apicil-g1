"""
Téléchargement et mise en forme des données externes issues de l'INSEE.

Ce script est à lancer une fois pour alimenter data/external/.
Il peut être relancé à tout moment pour mettre à jour les données.

Usage
-----
    python src/download_external.py
"""

import os
import io
import requests
import pandas as pd


# ---------------------------------------------------------------------------
# URLs de téléchargement direct depuis l'INSEE
# Pour retrouver une URL :
#   1. Aller sur https://www.insee.fr/fr/statistiques/serie/<idBank>
#   2. Cliquer sur "Télécharger" > "Format CSV (point-virgule)"
#   3. Copier le lien du bouton de téléchargement
# ---------------------------------------------------------------------------

SOURCES_CSV = {
    "indice_prix_conso_complementaires_sante": {
        "url": "https://www.insee.fr/fr/statistiques/telecharger/csv/001762477/valeurs_mensuelles.csv",
        "output": "data/external/indice_prix_conso_complementaires_sante.csv",
        "description": "Indice des prix à la consommation harmonisé — complémentaires santé (base 2015, idBank 001762477)",
    },
    "indice_prix_conso_general": {
        "url": "https://www.insee.fr/fr/statistiques/telecharger/csv/011814056/valeurs_mensuelles.csv",
        "output": "data/external/indice_prix_conso_general.csv",
        "description": "Indice des prix à la consommation — ensemble hors tabac (base 2025, idBank 011814056)",
    },
}

SOURCES_XLSX = {
    "indice_confiance_des_menages": {
        "url": "https://www.insee.fr/fr/statistiques/fichier/8730744/19_IR_Camme.xlsx",
        "output": "data/external/indice_confiance_des_menages.csv",
        "description": "Indicateur synthétique de confiance des ménages (idBank 8730744)",
        "sheet": "C.A.M.",
        "skiprows": 6,
        "date_min": "2015-01-01",
        "date_max": "2025-12-31",
    },
}


# ---------------------------------------------------------------------------
# Fonctions de téléchargement et transformation
# ---------------------------------------------------------------------------

def download_indice_csv(url: str, output_path: str, description: str) -> None:
    """
    Télécharge et transforme un indice INSEE au format CSV (point-virgule)
    vers le format date;valeur.

    Parameters
    ----------
    url : str
        URL de téléchargement du fichier INSEE.
    output_path : str
        Chemin de sauvegarde du fichier transformé.
    description : str
        Description de la série pour les logs.
    """
    print(f"\n{description}")
    print(f"Téléchargement depuis {url} ...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    raw_path = output_path.replace(".csv", "_raw.csv")
    with open(raw_path, "wb") as f:
        f.write(response.content)

    df = pd.read_csv(
        raw_path, sep=";", skiprows=4, header=None, usecols=[0, 1], dtype=str
    )
    df.columns = ["date", "valeur"]
    df["date"] = df["date"].str.strip('"')
    df["valeur"] = df["valeur"].str.strip('"')

    df.to_csv(output_path, sep=";", index=False)
    print(f"Sauvegardé dans {output_path} ({len(df)} lignes)")

    os.remove(raw_path)


def download_indice_xlsx(
    url: str,
    output_path: str,
    description: str,
    sheet: str,
    skiprows: int,
    date_min: str,
    date_max: str,
) -> None:
    """
    Télécharge et transforme un indice INSEE au format XLSX
    vers le format date;valeur, filtré sur une plage de dates.

    Parameters
    ----------
    url : str
        URL de téléchargement du fichier INSEE.
    output_path : str
        Chemin de sauvegarde du fichier transformé.
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
    print(f"\n{description}")
    print(f"Téléchargement depuis {url} ...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    df = pd.read_excel(
        io.BytesIO(response.content),
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


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("data/external", exist_ok=True)

    for name, source in SOURCES_CSV.items():
        download_indice_csv(
            url=source["url"],
            output_path=source["output"],
            description=source["description"],
        )

    for name, source in SOURCES_XLSX.items():
        download_indice_xlsx(
            url=source["url"],
            output_path=source["output"],
            description=source["description"],
            sheet=source["sheet"],
            skiprows=source["skiprows"],
            date_min=source["date_min"],
            date_max=source["date_max"],
        )

    print("\nToutes les sources externes ont été téléchargées.")