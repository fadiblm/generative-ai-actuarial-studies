from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


def csv_to_parquet(input_rel_path: str,
                   output_rel_path: str | None = None,
                   sep: str = ';') -> Path:
    """
    Charge un CSV et le sauvegarde en Parquet si le fichier Parquet n'existe pas encore.

    Parameters
    ----------
    input_rel_path : str
        Chemin relatif vers le fichier CSV.
    output_rel_path : str | None
        Chemin relatif vers le fichier Parquet. Si None, remplace .csv par .parquet.
    sep : str
        Séparateur de colonnes du CSV (par défaut ';').

    Returns
    -------
    Path
        Chemin du fichier Parquet.
    """
    csv_path = Path(input_rel_path)

    if output_rel_path is None:
        output_rel_path = csv_path.with_suffix('.parquet')

    parquet_path = Path(output_rel_path)

    if not parquet_path.exists():
        df = pd.read_csv(csv_path, sep=sep)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, engine="pyarrow", index=False)
        print(f"CSV converti en Parquet : {parquet_path}")
    else:
        print(f"Le fichier Parquet existe déjà : {parquet_path}")

    return parquet_path




def monthly_df_to_gpkg(
    df,
    output_gpkg_path: str,
    layer_name: str = "datagouv_swimonthly",
    x_col: str = "LAMBX",
    y_col: str = "LAMBY",
    crs: str = "EPSG:27572",
    units: str = "hm",          # "hm" (hectomètres) ou "m"
) -> Path:
    """
    Convertit un DataFrame mensuel (LAMBX/LAMBY) en GeoPackage.

    Si units="hm", les coordonnées sont supposées en hectomètres
    et sont converties en mètres avant création de la géométrie.
    """
    gpkg_path = Path(output_gpkg_path)

    if not gpkg_path.exists():
        gpkg_path.parent.mkdir(parents=True, exist_ok=True)

        # conversion éventuelle hm -> m
        if units == "hm":
            x = df[x_col] * 100
            y = df[y_col] * 100
        else:
            x = df[x_col]
            y = df[y_col]

        gdf = gpd.GeoDataFrame(
            df,
            geometry=[Point(xy) for xy in zip(x, y)],
            crs=crs
        )

        gdf.to_file(gpkg_path, driver="GPKG", layer=layer_name)
        print(f"Fichier GPKG créé : {gpkg_path.resolve()}")
    else:
        print(f"Le fichier GPKG existe déjà : {gpkg_path.resolve()}")

    return gpkg_path