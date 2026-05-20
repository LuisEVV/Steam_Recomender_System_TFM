"""Hybrid recommender model used by the SteamRec TFM project.

The class combines collaborative filtering based on truncated SVD and a
content-based fallback based on TF-IDF over genre metadata.
"""

import warnings
from typing import List, Union

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

class SteamRecommender:
    """Train and serve a hybrid recommender system.

    Attributes:
        k_factors: Number of latent factors used by the SVD factorization.
        user_encoder: Label encoder that maps Steam IDs to matrix rows.
        item_encoder: Label encoder that maps app IDs to matrix columns.
        is_trained: Flag indicating whether the model has been fitted.
    """

    def __init__(self, k_factors: int = 50) -> None:
        self.k_factors = k_factors
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.is_trained = False

    @staticmethod
    def _prepare_metacritic(df_meta: pd.DataFrame) -> pd.DataFrame:
        """Normalize Metacritic scores to floats in [0, 100], defaulting missing values to 50."""
        meta = df_meta.copy()
        scores = pd.to_numeric(meta["metacritic"], errors="coerce")
        scores = scores.where((scores > 0) & scores.notna(), 50.0)
        meta["metacritic"] = scores.astype(np.float64)
        return meta

    @staticmethod
    def _adjust_scores_by_metacritic(
        base_scores: np.ndarray, metacritic_scores: np.ndarray
    ) -> np.ndarray:
        """Apply quality-aware re-ranking: score * (1 + 0.2 * ((nota/100) - 0.5))."""
        return base_scores * (1.0 + 0.2 * ((metacritic_scores / 100.0) - 0.5))

    def _transform_ratings(self, df_inter: pd.DataFrame) -> pd.DataFrame:
        """Convert playtime into implicit ratings in the range [1, 5].

        The transformation applies log1p to reduce the impact of long-tail
        playtimes and then performs a per-user MinMax normalization.

        Args:
            df_inter: Interaction dataframe with at least the columns
                ``steam_id`` and ``playtime_forever_min``.

        Returns:
            A copy of the input dataframe with the engineered columns
            ``log_playtime`` and ``rating``.
        """
        df = df_inter.copy()
        df['log_playtime'] = np.log1p(df['playtime_forever_min'])
        
        def minmax(g):
            if g.max() == g.min(): return pd.Series(3.0, index=g.index)
            return 1.0 + 4.0 * (g - g.min()) / (g.max() - g.min())
            
        df['rating'] = df.groupby('steam_id')['log_playtime'].transform(minmax)
        return df

    def train(self, df_inter: pd.DataFrame, df_meta: pd.DataFrame) -> None:
        """Fit the collaborative and content-based components.

        Args:
            df_inter: Interaction matrix in long format, including Steam IDs,
                app IDs and playtime in minutes.
            df_meta: Game metadata dataframe with at least the columns
                ``app_id``, ``nombre`` and ``generos``.
        """
        print("Preprocesando datos...")
        self.df_meta = self._prepare_metacritic(df_meta)
        self.metacritic_scores = self.df_meta["metacritic"].to_numpy(dtype=np.float64)
        df_ratings = self._transform_ratings(df_inter)
        
        # 1. Modelo Colaborativo (SVD)
        print("Entrenando Modelo Colaborativo (SVD)...")
        df_ratings['uid'] = self.user_encoder.fit_transform(df_ratings['steam_id'])
        df_ratings['iid'] = self.item_encoder.fit_transform(df_ratings['app_id'])
        
        self.n_users = len(self.user_encoder.classes_)
        self.n_items = len(self.item_encoder.classes_)
        
        R = csr_matrix((df_ratings['rating'], (df_ratings['uid'], df_ratings['iid'])), 
                       shape=(self.n_users, self.n_items), dtype=np.float64)
        
        U, sigma, Vt = svds(R, k=self.k_factors, which='LM')
        self.R_hat = U @ np.diag(sigma) @ Vt
        self.R_hat = np.clip(self.R_hat, 1.0, 5.0)
        self.df_encoded = df_ratings # Guardar historial

        meta_by_app = self.df_meta.set_index("app_id")["metacritic"]
        self.metacritic_by_item = (
            meta_by_app.reindex(self.item_encoder.classes_).fillna(50.0).to_numpy(dtype=np.float64)
        )

        # 2. Modelo Basado en Contenido (Cold Start)
        print("Entrenando Modelo de Contenido (TF-IDF)...")
        self.df_meta['generos'] = self.df_meta['generos'].fillna('')
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.df_meta['generos'])
        
        self.is_trained = True
        print("Entrenamiento completado.")

    def recommend_collaborative(self, steam_id: Union[int, str], n: int = 5) -> pd.DataFrame:
        """Recommend games using the collaborative SVD score matrix.

        Args:
            steam_id: Steam user identifier to score.
            n: Number of recommendations to return.

        Returns:
            Dataframe with the top-N recommendations merged with metadata.

        Raises:
            ValueError: If the user was not seen during training.
        """
        if steam_id not in self.user_encoder.classes_:
            raise ValueError(f"Usuario {steam_id} no encontrado. Usa la recomendación por contenido.")
        
        u_idx = self.user_encoder.transform([steam_id])[0]
        base_scores = self.R_hat[u_idx].copy()

        jugados = self.df_encoded[self.df_encoded["uid"] == u_idx]["iid"].to_numpy(dtype=int)
        base_scores[jugados] = -np.inf

        adjusted_scores = self._adjust_scores_by_metacritic(base_scores, self.metacritic_by_item)
        top_idx = np.argsort(adjusted_scores)[::-1][:n]
        top_appids = self.item_encoder.inverse_transform(top_idx)

        recs = pd.DataFrame({"app_id": top_appids, "score": adjusted_scores[top_idx]})
        return recs.merge(self.df_meta, on="app_id", how="left")

    def recommend_content(self, game_names: List[str], n: int = 5) -> pd.DataFrame:
        """Recommend games based on TF-IDF similarity over genre metadata.

        Args:
            game_names: Seed titles provided by the user.
            n: Number of recommendations to return.

        Returns:
            Dataframe with the most similar games and their content score.

        Raises:
            ValueError: If none of the provided titles exist in the metadata.
        """
        seed_mask = self.df_meta["nombre"].isin(game_names).to_numpy()
        indices = np.flatnonzero(seed_mask)
        if indices.size == 0:
            raise ValueError("Ninguno de los juegos proporcionados existe en la base de datos.")

        perfil = self.tfidf_matrix[indices].mean(axis=0)
        similitud = linear_kernel(np.asarray(perfil), self.tfidf_matrix).flatten()
        base_scores = similitud * 5.0
        base_scores[indices] = -np.inf

        adjusted_scores = self._adjust_scores_by_metacritic(base_scores, self.metacritic_scores)
        top_idx = np.argsort(adjusted_scores)[::-1][:n]

        recs = self.df_meta.iloc[top_idx].copy()
        recs["score"] = adjusted_scores[top_idx]
        return recs

    def save_model(self, path: str = 'models/recommender.pkl') -> None:
        """Persist the trained recommender to disk as a pickle artifact.

        Args:
            path: Destination file path for the serialized model.
        """
        joblib.dump(self, path)
        print(f"Modelo guardado en {path}")
