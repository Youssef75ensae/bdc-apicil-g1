import os
import s3fs
import pandas as pd

def load_data() -> pd.DataFrame:
    """
    Fonction permettant de charger les données au format .xlxs
    depuis Onyxia et de les convertir en un DataFrame

    """
    endpoint = os.environ["AWS_S3_ENDPOINT"]
    if not endpoint.startswith('http'):
        endpoint = "https://" + endpoint
    
    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url" : endpoint})

    path = ("projet-bdc-data/apicil/"
            "Copie de V5_ADENA_extract client indiv direct et courté.xlsx"
    )

    with fs.open(path, "rb") as f:
        df = pd.read_excel(f)

    df["id_client"] = range(1, len(df) + 1)
    
    return df

if __name__ == "__main__":
    output_path = os.environ.get("DATA_LOCAL_PATH", "data/raw/apicil.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
 
    print("Chargement des données depuis S3...")
    df = load_data()
 
    df.to_csv(output_path, index=False)
    print(f"Données sauvegardées dans {output_path} ({len(df)} lignes)")