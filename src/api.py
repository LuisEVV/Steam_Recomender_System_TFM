"""FastAPI service that exposes the SteamRec hybrid recommender."""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src import model  
from src.model import SteamRecommender

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# Steam Web API Key: https://steamcommunity.com/dev/apikey  →  .env como STEAM_API_KEY=...
STEAM_WEB_API_KEY = (os.getenv("STEAM_API_KEY") or "").strip()
STEAM_OWNED_GAMES_URL = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/"
STEAM_API_TIMEOUT_SECONDS = 10
COLD_START_MIN_SEED_GAMES = 3
COLD_START_MAX_SEED_GAMES = 5

# Make pickle loading compatible with artifacts trained from the local module name.
sys.modules['model'] = model

app = FastAPI(title="SteamRec API - TFM Big Data")


def _load_recommender() -> Optional[SteamRecommender]:
    """Load the persisted recommender model from disk.

    Returns:
        The loaded recommender instance, or ``None`` when loading fails.
    """
    try:
        recommender_model = joblib.load('models/recommender.pkl')
        print("Modelo cargado correctamente.")
        return recommender_model
    except Exception as exc:
        print(f"ERROR CRÍTICO AL CARGAR EL MODELO: {exc}")
        return None


recommender: Optional[SteamRecommender] = _load_recommender()

class ContentRequest(BaseModel):
    """Payload for content-based recommendation requests."""

    games: List[str]


def _recommendations_to_records(recs) -> List[Dict[str, object]]:
    """Serialize a recommendation dataframe to API records."""
    return recs[['app_id', 'nombre', 'generos', 'score']].to_dict(orient='records')


def _request_steam_owned_games(steam_id: int) -> Dict[str, object]:
    """Call Steam GetOwnedGames and return the ``response`` object.

    Raises:
        HTTPException: On missing API key, auth errors, or transport failures.
    """
    if not STEAM_WEB_API_KEY:
        raise HTTPException(
            status_code=500,
            detail=(
                "STEAM_API_KEY no configurada. Añádela al archivo .env en la raíz del proyecto "
                "(STEAM_API_KEY=tu_clave) y reinicia uvicorn."
            ),
        )

    try:
        response = requests.get(
            STEAM_OWNED_GAMES_URL,
            params={
                "key": STEAM_WEB_API_KEY,
                "steamid": steam_id,
                "include_appinfo": 1,
            },
            timeout=STEAM_API_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 401:
            raise HTTPException(
                status_code=502,
                detail=(
                    "Steam rechazó la API key (401 Unauthorized). "
                    "Verifica que STEAM_API_KEY en .env sea válida en "
                    "https://steamcommunity.com/dev/apikey"
                ),
            ) from exc
        raise HTTPException(
            status_code=502,
            detail=f"No se pudo consultar la biblioteca de Steam: {exc}",
        ) from exc
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"No se pudo consultar la biblioteca de Steam: {exc}",
        ) from exc

    return payload.get("response") or {}


def _get_steam_owned_game_count(steam_id: int) -> Optional[int]:
    """Return the real owned-game count from Steam, or ``None`` if unavailable."""
    try:
        steam_response = _request_steam_owned_games(steam_id)
    except HTTPException:
        return None

    game_count = steam_response.get("game_count")
    if game_count is not None:
        return int(game_count)

    games = steam_response.get("games") or []
    return len(games) if games else None


def _fetch_steam_top_played_games(steam_id: int) -> List[Tuple[int, str]]:
    """Fetch the user's public library from Steam and return top-played titles.

    Args:
        steam_id: Steam user identifier.

    Returns:
        A list of ``(app_id, name)`` tuples sorted by ``playtime_forever`` descending.

    Raises:
        HTTPException: If the Steam API call fails or the library is empty/private.
    """
    steam_response = _request_steam_owned_games(steam_id)
    games = steam_response.get("games") or []
    if not games:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Usuario {steam_id} no encontrado en el modelo y su biblioteca "
                "de Steam no está disponible (perfil privado o sin juegos públicos)."
            ),
        )

    ranked = sorted(games, key=lambda game: game.get("playtime_forever", 0), reverse=True)
    n_seeds = min(COLD_START_MAX_SEED_GAMES, len(ranked))
    if n_seeds < COLD_START_MIN_SEED_GAMES:
        n_seeds = len(ranked)

    seeds: List[Tuple[int, str]] = []
    for game in ranked[:n_seeds]:
        app_id = int(game["appid"])
        name = str(game.get("name") or "").strip()
        seeds.append((app_id, name))
    return seeds


def _resolve_seed_game_names(seed_games: List[Tuple[int, str]]) -> List[str]:
    """Map Steam app IDs to catalog titles when possible, else use Steam names."""
    meta_names = recommender.df_meta.set_index("app_id")["nombre"]
    resolved: List[str] = []
    for app_id, steam_name in seed_games:
        if app_id in meta_names.index:
            resolved.append(str(meta_names.loc[app_id]))
        elif steam_name:
            resolved.append(steam_name)
    return resolved


def _cold_start_recommendations(steam_id: int, n: int) -> List[Dict[str, object]]:
    """Build content-based recommendations from a user's public Steam library."""
    seed_games = _fetch_steam_top_played_games(steam_id)
    seed_titles = _resolve_seed_game_names(seed_games)
    if not seed_titles:
        raise HTTPException(
            status_code=404,
            detail=f"No se pudieron resolver títulos de juego para el usuario {steam_id}.",
        )

    try:
        recs = recommender.recommend_content(seed_titles, n)
        return _recommendations_to_records(recs)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/")
def read_root() -> Dict[str, object]:
    """Return a lightweight health-check payload for the API."""
    return {"status": "API Activa", "modelo_cargado": recommender is not None}

@app.get("/recommend/collaborative/{steam_id}")
def get_collaborative_recs(steam_id: int, n: int = 5) -> List[Dict[str, object]]:
    """Generate collaborative recommendations for a Steam user.

    Args:
        steam_id: Steam user identifier validated by FastAPI.
        n: Number of recommendations to return.

    Returns:
        A JSON-serializable list of recommendation records.
    """
    if recommender is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado.")
    try:
        recs = recommender.recommend_collaborative(steam_id, n)
        return _recommendations_to_records(recs)
    except ValueError:
        return _cold_start_recommendations(steam_id, n)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error interno al generar recomendaciones: {exc}")


@app.get("/profile/{steam_id}")
def get_user_profile(steam_id: int) -> Dict[str, object]:
    """Return basic user statistics derived from the training interactions.

    Args:
        steam_id: Steam user identifier validated by FastAPI.

    Returns:
        A JSON-serializable dictionary with aggregate profile statistics.

    Raises:
        HTTPException: If the model is not loaded or the user does not exist.
    """
    if recommender is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado.")
    try:
        if steam_id not in recommender.user_encoder.classes_:
            raise ValueError(f"Usuario {steam_id} no encontrado.")

        u_idx = recommender.user_encoder.transform([steam_id])[0]
        user_rows = recommender.df_encoded[recommender.df_encoded['uid'] == u_idx]
        user_rows = user_rows.merge(
            recommender.df_meta[['app_id', 'nombre', 'tipo', 'generos']],
            on='app_id',
            how='left',
        )

        genre_series = user_rows['generos'].fillna('').str.lower()
        name_series = user_rows['nombre'].fillna('').str.lower()
        type_series = user_rows['tipo'].fillna('').str.lower()
        valid_mask = (
            type_series.eq('game')
            & ~name_series.str.contains(
                'wallpaper engine|tool|utility|utilities|benchmark|driver|sdk|software',
                regex=True,
            )
            & ~genre_series.str.contains('utilities|utility|animation & modeling|photo editing', regex=True)
        )
        filtered_rows = user_rows[valid_mask].copy()
        if filtered_rows.empty:
            filtered_rows = user_rows.copy()

        top_games = (
            filtered_rows.sort_values('playtime_forever_min', ascending=False)
            .head(3)[['app_id', 'nombre', 'playtime_forever_min']]
            .assign(playtime_hours=lambda frame: (frame['playtime_forever_min'] / 60.0).round(2))
            .to_dict(orient='records')
        )

        total_minutes = float(user_rows['playtime_forever_min'].sum())
        owned_games_training = int(user_rows['app_id'].nunique())
        owned_games = _get_steam_owned_game_count(steam_id) or owned_games_training
        avg_minutes = float(user_rows['playtime_forever_min'].mean()) if not user_rows.empty else 0.0
        most_played_row = user_rows.sort_values('playtime_forever_min', ascending=False).iloc[0]

        return {
            "steam_id": int(steam_id),
            "total_hours_played": round(total_minutes / 60.0, 2),
            "owned_games": owned_games,
            "avg_minutes_per_game": round(avg_minutes, 2),
            "most_played_app_id": int(most_played_row['app_id']),
            "most_played_name": str(most_played_row.get('nombre', '')),
            "most_played_minutes": round(float(most_played_row['playtime_forever_min']), 2),
            "top_games": top_games,
        }
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error interno al construir el perfil: {exc}")

@app.post("/recommend/content/")
def get_content_recs(req: ContentRequest, n: int = 5) -> List[Dict[str, object]]:
    """Generate content-based recommendations from a list of seed games.

    Args:
        req: Request body containing the selected seed games.
        n: Number of recommendations to return.

    Returns:
        A JSON-serializable list of recommendation records.
    """
    if recommender is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado.")
    if len(req.games) < 1:
        raise HTTPException(status_code=400, detail="Debes proporcionar al menos 1 juego.")
    try:
        recs = recommender.recommend_content(req.games, n)
        return _recommendations_to_records(recs)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error interno al generar recomendaciones: {exc}")