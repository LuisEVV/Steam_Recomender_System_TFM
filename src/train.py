"""Training script for the SteamRec hybrid recommender."""

import os

import pandas as pd

from model import SteamRecommender


def filter_sparse_interactions(
    df_inter: pd.DataFrame,
    min_user_interactions: int = 5,
    min_item_interactions: int = 10,
) -> pd.DataFrame:
    """Filter sparse users and items before model training.

    Args:
        df_inter: Raw interaction dataframe.
        min_user_interactions: Minimum number of interactions required per user.
        min_item_interactions: Minimum number of interactions required per item.

    Returns:
        Filtered interaction dataframe ready for training.
    """
    user_counts = df_inter['steam_id'].value_counts()
    item_counts = df_inter['app_id'].value_counts()
    return df_inter[
        (df_inter['steam_id'].isin(user_counts[user_counts >= min_user_interactions].index))
        & (df_inter['app_id'].isin(item_counts[item_counts >= min_item_interactions].index))
    ]


def main() -> None:
    """Load data, train the recommender and persist the fitted artifact."""
    print("Cargando datos...")
    df_i = pd.read_csv('data/matriz_interacciones_test.csv')
    df_m = pd.read_csv('data/juegos_metadata.csv')

    df_i = filter_sparse_interactions(df_i)

    rec = SteamRecommender()
    rec.train(df_i, df_m)
    os.makedirs('models', exist_ok=True)
    rec.save_model()
    print("¡Modelo generado correctamente para la API!")

if __name__ == "__main__":
    main()