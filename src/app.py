"""Dark-mode Streamlit dashboard for the SteamRec hybrid recommender."""

from __future__ import annotations

import html
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import xml.etree.ElementTree as ET
import streamlit as st

API_URL = "http://localhost:8000"
API_TIMEOUT_SECONDS = 10
STEAM_TIMEOUT_SECONDS = 8

PAGE_TITLE = "SteamRec TFM"
PAGE_ICON = "🎮"
TIMEOUT_ERROR_MESSAGE = (
    "La API tardó demasiado en responder. Verifica que FastAPI esté activo y "
    "vuelve a intentarlo."
)


st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(76, 125, 255, 0.18), transparent 30%),
                radial-gradient(circle at top right, rgba(26, 202, 165, 0.14), transparent 28%),
                linear-gradient(180deg, #07111f 0%, #0b1324 55%, #08101b 100%);
            color: #edf3ff;
        }

        header, footer, #MainMenu {
            visibility: hidden;
            height: 0;
        }

        section[data-testid="stSidebar"] {
            display: none;
        }

        .hero-shell {
            max-width: 1080px;
            margin: 1.5rem auto 1rem auto;
            padding: 2rem 2.2rem;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 28px;
            background: rgba(7, 13, 24, 0.72);
            backdrop-filter: blur(18px);
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
        }

        .eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.22em;
            font-size: 0.75rem;
            color: #88a8ff;
            margin-bottom: 0.65rem;
        }

        .hero-title {
            font-size: clamp(2.1rem, 5vw, 4.2rem);
            line-height: 1.02;
            font-weight: 800;
            margin-bottom: 0.8rem;
            color: #f6f9ff;
        }

        .hero-copy {
            max-width: 760px;
            font-size: 1.02rem;
            color: #b7c4dc;
            line-height: 1.7;
            margin-bottom: 0;
        }

        .mode-shell {
            max-width: 760px;
            margin: 1rem auto 1.4rem auto;
            padding: 0.9rem 1rem;
            border-radius: 28px;
            background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
            display: flex;
            justify-content: center;
            align-items: center;
            position: relative;
            box-shadow: 0 10px 30px rgba(2,6,23,0.45);
        }

        .mode-label {
            position: absolute;
            top: -0.7rem;
            left: 50%;
            transform: translateX(-50%);
            font-size: 0.92rem;
            color: #ffffff;
            font-weight: 700;
            padding: 0.28rem 0.85rem;
            border-radius: 999px;
            background: linear-gradient(90deg, #2b6cff 0%, #6ab7ff 100%);
            box-shadow: 0 8px 20px rgba(43,108,255,0.12);
        }

        .app-footer {
            max-width: 1080px;
            margin: 1.25rem auto 2.2rem auto;
            text-align: center;
            color: #9fb0cc;
            font-size: 0.92rem;
            opacity: 0.95;
        }

        /* Center Streamlit widgets placed inside the mode-shell */
        .mode-shell div[data-testid="stRadio"] {
            display: flex;
            justify-content: center;
            width: 100%;
        }

        .mode-shell div[data-testid="stTextInput"],
        .mode-shell div[data-testid="stMultiselect"],
        .mode-shell div[role="form"] {
            max-width: 560px;
            margin: 0.35rem auto 0 auto;
            width: 100%;
        }

        .mode-shell .stButton>button {
            display: block;
            margin: 0.4rem auto 0 auto;
        }

        div[data-baseweb="radio"] {
            justify-content: center;
        }

        div[data-testid="stRadio"] > div {
            gap: 0.6rem;
        }

        div[data-testid="stRadio"] label {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 999px;
            padding: 0.45rem 0.9rem;
        }

        .section-card {
            max-width: 1080px;
            margin: 1rem auto;
            padding: 1.4rem 1.4rem 1.7rem 1.4rem;
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            background: rgba(10, 15, 28, 0.82);
            box-shadow: 0 18px 44px rgba(0, 0, 0, 0.22);
        }

        .section-title {
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 0.45rem;
            color: #f4f7ff;
        }

        .section-subtitle {
            color: #aebbd4;
            margin-bottom: 1rem;
        }

        .profile-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 1rem;
        }

        .stat-card {
            background: linear-gradient(180deg, rgba(18, 28, 53, 0.95), rgba(12, 18, 34, 0.98));
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 18px;
            padding: 1rem 1.1rem;
        }

        .stat-label {
            color: #93a4c5;
            font-size: 0.8rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .stat-value {
            color: #ffffff;
            font-size: 1.45rem;
            font-weight: 800;
            margin-top: 0.3rem;
        }

        .profile-hint {
            margin-top: 1rem;
            color: #9fb0cc;
            font-size: 0.92rem;
        }

        .cards-shell {
            max-width: 1080px;
            margin: 1rem auto 2.2rem auto;
            padding: 1.4rem 1.4rem 1.8rem 1.4rem;
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            background: rgba(9, 14, 26, 0.8);
        }

        /* Grid-based compact layout for recommendation cards */
        .cards-grid {
            display: grid;
            /* Increase min width to avoid label overlap on smaller cards */
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 15px;
            align-items: start;
        }

        .game-card {
            width: 100%;
            min-width: 220px; /* ensure room for footer elements */
            border-radius: 12px;
            overflow: hidden;
            background: linear-gradient(180deg, rgba(21, 32, 58, 0.98), rgba(9, 13, 24, 0.98));
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 18px 36px rgba(0, 0, 0, 0.26);
            scroll-snap-align: start;
            flex: 0 0 auto;
        }

        .game-banner {
            width: 100%;
            max-height: 110px;
            height: auto;
            object-fit: cover;
            display: block;
        }

        .game-body {
            padding: 0.6rem 0.7rem 0.7rem 0.7rem;
            font-size: 12px;
        }

        .game-title {
            font-size: 0.98rem;
            font-weight: 800;
            line-height: 1.25;
            color: #f8fbff;
            margin-bottom: 0.35rem;
        }

        .game-genre {
            color: #8ea2c9;
            font-size: 0.76rem;
            margin-bottom: 0.55rem;
        }

        .game-desc {
            color: #d5deee;
            font-size: 12px;
            line-height: 1.3;
            min-height: 0;
            display: -webkit-box;
            -webkit-line-clamp: 3;
            -webkit-box-orient: vertical;
            overflow: hidden;
            max-height: 3.9em; /* fallback for non-webkit browsers */
        }

        .game-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.6rem;
            margin-top: 0.6rem;
        }

        .game-score {
            display: inline-flex;
            align-items: center;
            color: #93ffa9;
            font-size: 0.72rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            font-weight: 700;
            white-space: nowrap; /* prevent label wrapping */
            gap: 0.35rem;
        }

        .steam-button {
            display: inline-block;
            background: linear-gradient(135deg, #2a5fff 0%, #4cb4ff 100%);
            color: #fff !important;
            padding: 0.4rem 0.6rem;
            border-radius: 999px;
            text-decoration: none;
            font-size: 12px;
            font-weight: 700;
            white-space: nowrap;
            flex-shrink: 0; /* keep button from shrinking when space is tight */
        }


        .top-game-card {
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 0.85rem 0.95rem;
            background: rgba(255, 255, 255, 0.03);
        }

        .top-game-rank {
            color: #88a8ff;
            font-size: 0.72rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .top-game-title {
            color: #f7f9ff;
            font-size: 0.95rem;
            font-weight: 800;
            line-height: 1.25;
        }

        .top-game-meta {
            color: #aab8d3;
            font-size: 0.8rem;
            margin-top: 0.35rem;
        }

        .empty-state {
            text-align: center;
            padding: 2.2rem 1rem;
            color: #a9b9d3;
        }

        @media (max-width: 900px) {
            .profile-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }

        @media (max-width: 640px) {
            .hero-shell, .mode-shell, .section-card, .cards-shell {
                margin-left: 0.5rem;
                margin-right: 0.5rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .profile-grid {
                grid-template-columns: 1fr;
            }

            .game-card {
                min-width: 82vw;
                max-width: 82vw;
            }

            .top-games-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def safe_text(value: object, default: str = "-", max_length: Optional[int] = None) -> str:
    """Return a sanitised string for HTML rendering."""
    if value is None:
        text = default
    else:
        text = str(value)
        if not text.strip():
            text = default
    if max_length is not None and len(text) > max_length:
        text = text[: max_length - 1].rstrip() + "…"
    return html.escape(text)


def get_json_error_message(response: requests.Response, fallback: str) -> str:
    """Extract a friendly error message from an HTTP response."""
    try:
        payload = response.json()
        if isinstance(payload, dict):
            return str(payload.get("detail", fallback))
    except ValueError:
        pass
    return response.text or fallback


@st.cache_data(show_spinner=False)
def load_game_names() -> List[str]:
    """Load available game names from the metadata CSV."""
    df = pd.read_csv("data/juegos_metadata.csv")
    return df["nombre"].dropna().unique().tolist()


@st.cache_data(show_spinner=False)
def fetch_steam_game_details(app_id: int) -> Dict[str, str]:
    """Fetch Steam store metadata for a given app identifier.

    Returns a resilient dictionary with banner, description and store URL.
    """
    fallback = {
        "header_image": "https://via.placeholder.com/460x215?text=Steam+Game",
        "short_description": "Sin descripción disponible en Steam.",
        "store_url": f"https://store.steampowered.com/app/{app_id}",
        "steam_name": "",
    }
    try:
        response = requests.get(
            f"https://store.steampowered.com/api/appdetails?appids={app_id}",
            timeout=STEAM_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json().get(str(app_id), {})
        if not payload.get("success"):
            return fallback
        data = payload.get("data", {})
        return {
            "header_image": data.get("header_image", fallback["header_image"]),
            "short_description": data.get("short_description", fallback["short_description"]),
            "store_url": f"https://store.steampowered.com/app/{app_id}",
            "steam_name": data.get("name", ""),
        }
    except Exception:
        return fallback


def render_stat_cards(profile: Dict[str, object]) -> None:
    """Render a compact dashboard with the main user stats."""
    st.markdown(
        f"""
        <div class="profile-grid">
            <div class="stat-card"><div class="stat-label">Steam ID</div><div class="stat-value">{safe_text(profile.get('steam_id'))}</div></div>
            <div class="stat-card"><div class="stat-label">Horas totales</div><div class="stat-value">{safe_text(profile.get('total_hours_played'))} h</div></div>
            <div class="stat-card"><div class="stat-label">Juegos en propiedad</div><div class="stat-value">{safe_text(profile.get('owned_games'))}</div></div>
            <div class="stat-card"><div class="stat-label">Media por juego</div><div class="stat-value">{safe_text(profile.get('avg_minutes_per_game'))} min</div></div>
        </div>
        <div class="profile-hint">Juego más jugado: {safe_text(profile.get('most_played_app_id'))} · {safe_text(profile.get('most_played_minutes'))} min</div>
        """,
        unsafe_allow_html=True,
    )

    top_games = profile.get('top_games') or []
    if top_games:
        top_games_html = []
        for index, game in enumerate(top_games, start=1):
            title = safe_text(game.get('nombre', game.get('app_id')))
            hours = float(game.get('playtime_hours', 0.0))
            top_games_html.append(
                f"""
                <div class="top-game-card">
                    <div class="top-game-rank">Top {index}</div>
                    <div class="top-game-title">{title}</div>
                    <div class="top-game-meta">{hours:.2f} h · {safe_text(game.get('app_id'))}</div>
                </div>
                """
            )



@st.cache_data(show_spinner=False)
def fetch_steam_user_profile(steam_id: int) -> Dict[str, str]:
    """Fetch a minimal Steam profile without requiring an API key using the public XML endpoint.

    Returns a dict with `nickname`, `avatar` and `profile_url` keys. Falls back to placeholders.
    """
    fallback = {
        "nickname": f"Usuario {steam_id}",
        "avatar": "https://via.placeholder.com/128?text=Avatar",
        "profile_url": f"https://steamcommunity.com/profiles/{steam_id}",
    }
    try:
        resp = requests.get(f"https://steamcommunity.com/profiles/{steam_id}?xml=1", timeout=STEAM_TIMEOUT_SECONDS)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        persona = root.findtext('steamID') or root.findtext('steamID64') or fallback['nickname']
        avatar = root.findtext('avatarFull') or root.findtext('avatar') or fallback['avatar']
        return {"nickname": persona, "avatar": avatar, "profile_url": f"https://steamcommunity.com/profiles/{steam_id}"}
    except Exception:
        return fallback


def build_user_game_card(game: Dict[str, object]) -> str:
    """Build an HTML card for a user's played game (includes playtime)."""
    details = fetch_steam_game_details(int(game["app_id"]))
    title = safe_text(game.get("nombre") or details.get("steam_name") or game.get("app_id"), max_length=80)
    genres = safe_text(game.get("generos", "Sin género"), max_length=90)
    description = safe_text(details.get("short_description", "Sin descripción disponible."), max_length=190)
    image = safe_text(details.get("header_image"), default="https://via.placeholder.com/460x215?text=Steam+Game")
    steam_url = safe_text(details.get("store_url", f"https://store.steampowered.com/app/{game['app_id']}"))
    play_hours = float(game.get('playtime_hours', 0.0))

    return f"""
    <article class="game-card">
        <img class="game-banner" src="{image}" alt="Portada de {title}" />
        <div class="game-body">
            <div class="game-title">{title}</div>
            <div class="game-genre">{genres}</div>
            <div class="game-desc">{description}</div>
            <div class="game-footer">
                <div class="game-score">{play_hours:.2f} h</div>
                <a class="steam-button" href="{steam_url}" target="_blank" rel="noopener noreferrer">Abrir en Steam</a>
            </div>
        </div>
    </article>
    """


def render_user_profile_section(profile: Dict[str, object], steam_profile: Optional[Dict[str, str]]) -> None:
    """Render the user header (avatar + nickname), stats and the user's games as cards.

    This visually separates the player zone from recommendations.
    """
    avatar = steam_profile.get('avatar') if steam_profile else "https://via.placeholder.com/128?text=Avatar"
    nickname = steam_profile.get('nickname') if steam_profile else safe_text(profile.get('steam_id'))

    # Header with avatar + nickname
    header_html = f"""
    <div style="display:flex;gap:1rem;align-items:center;margin-bottom:1rem;">
        <img src="{avatar}" alt="Avatar" style="width:72px;height:72px;border-radius:999px;object-fit:cover;border:2px solid rgba(255,255,255,0.06);" />
        <div>
            <div style="font-size:1.05rem;font-weight:800;color:#f4f7ff;">{safe_text(nickname)}</div>
            <div style="color:#aebbd4;font-size:0.9rem;">Steam ID: {safe_text(profile.get('steam_id'))}</div>
        </div>
    </div>
    """

    st.markdown(header_html, unsafe_allow_html=True)

    # Stats
    render_stat_cards(profile)

    # User's top games rendered as enriched cards
    top_games = profile.get('top_games') or []
    if top_games:
        fragments = [build_user_game_card(g).strip() for g in top_games]
        container_html = f"<div class=\"cards-grid\">{''.join(fragments)}</div>"
        st.markdown('<div style="margin-top:0.8rem"></div>' + container_html, unsafe_allow_html=True)


def build_game_card(game: Dict[str, object], rank: int) -> str:
    """Build the HTML for a Steam recommendation card and inject an affinity label based on rank.

    Labels:
    - 1-2: Imprescindible (orange/red)
    - 3-4: Muy Recomendado (yellow/gold)
    - 5+: Recomendado (blue/green)
    """
    details = fetch_steam_game_details(int(game["app_id"]))
    title = safe_text(game.get("nombre") or details.get("steam_name") or game.get("app_id"), max_length=80)
    genres = safe_text(game.get("generos", "Sin género"), max_length=90)
    description = safe_text(details.get("short_description", "Sin descripción disponible."), max_length=190)
    image = safe_text(details.get("header_image"), default="https://via.placeholder.com/460x215?text=Steam+Game")
    steam_url = safe_text(details.get("store_url", f"https://store.steampowered.com/app/{game['app_id']}"))

    # Determine label by ranking (prefers rank over raw score for clarity)
    if rank <= 2:
        label_text = "Imprescindible"
        label_style = "background:#ff6b1a;color:#fff;padding:0.18rem 0.5rem;border-radius:8px;font-weight:800;"
    elif rank <= 4:
        label_text = "Muy Recomendado"
        label_style = "background:#ffd54f;color:#222;padding:0.18rem 0.5rem;border-radius:8px;font-weight:800;"
    else:
        label_text = "Recomendado"
        label_style = "background:#2aa76a;color:#fff;padding:0.18rem 0.5rem;border-radius:8px;font-weight:700;"

    label_html = f"<span style=\"{label_style}\">{label_text}</span>"

    return f"""
    <article class="game-card">
        <img class="game-banner" src="{image}" alt="Portada de {title}" />
        <div class="game-body">
            <div class="game-title">{title}</div>
            <div class="game-genre">{genres}</div>
            <div class="game-desc">{description}</div>
            <div class="game-footer">
                <div class="game-score">{label_html}</div>
                <a class="steam-button" href="{steam_url}" target="_blank" rel="noopener noreferrer">Abrir en Steam</a>
            </div>
        </div>
    </article>
    """


def render_recommendation_cards(recommendations: List[Dict[str, object]]) -> None:
    """Render a horizontal carousel of recommendation cards.

    Pass the ranking position to each card so labels can be generated.
    """
    fragments = [build_game_card(game, rank).strip() for rank, game in enumerate(recommendations, start=1)]
    container_html = f"<div class=\"cards-grid\">{''.join(fragments)}</div>"
    st.markdown(container_html, unsafe_allow_html=True)


def fetch_api_json(method: str, url: str, payload: Optional[Dict[str, object]] = None) -> Optional[Dict[str, object]]:
    """Call the FastAPI backend and return a JSON payload when successful."""
    try:
        if method == "get":
            response = requests.get(url, timeout=API_TIMEOUT_SECONDS)
        else:
            response = requests.post(url, json=payload or {}, timeout=API_TIMEOUT_SECONDS)

        if response.status_code == 200:
            return response.json()

        st.error(get_json_error_message(response, "Error al consultar la API."))
        return None
    except requests.Timeout:
        st.error(TIMEOUT_ERROR_MESSAGE)
        return None
    except requests.RequestException as exc:
        st.error(f"No se pudo conectar con la API: {exc}")
        return None


def main() -> None:
    """Render the dashboard and drive the recommendation workflows."""
    st.markdown(
        """
        <div class="hero-shell">
            <div class="eyebrow">TFM · Sistema de recomendación híbrido</div>
            <div class="hero-title">SteamRec</div>
            <p class="hero-copy">
                Explora recomendaciones por filtrado colaborativo o por contenido con una interfaz centrada,
                oscura y orientada a tarjetas. El perfil del usuario se calcula antes de recomendar y cada
                juego se enriquece en tiempo real con la API pública de Steam.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="mode-shell">', unsafe_allow_html=True)
    st.markdown('<div class="mode-label">¿Cómo quieres tus recomendaciones?</div>', unsafe_allow_html=True)
    mode = st.radio(
        "",
        ["Por Steam ID", "Seleccionar mis juegos"],
        horizontal=True,
        label_visibility="collapsed",
        key="recommendation_mode",
    )
    

    game_names = load_game_names()

    if "profile_data" not in st.session_state:
        st.session_state.profile_data = None
    if "player_info" not in st.session_state:
        st.session_state.player_info = None
    if "recommendations" not in st.session_state:
        st.session_state.recommendations = []
    if "recommendation_title" not in st.session_state:
        st.session_state.recommendation_title = ""

    # Compact input block: form widgets appear directly under the mode selector

    if mode == "Por Steam ID":
        with st.form("steam_id_form", clear_on_submit=False):
            steam_id_raw = st.text_input("Steam ID", placeholder="76561198273227245")
            submit = st.form_submit_button("Generar recomendaciones")
            if submit:
                steam_id_raw = steam_id_raw.strip()
                if not steam_id_raw:
                    st.warning("Debes introducir un Steam ID.")
                elif not steam_id_raw.isdigit():
                    st.error("El Steam ID debe contener solo dígitos.")
                else:
                    steam_id = int(steam_id_raw)
                    with st.spinner("Cargando perfil y recomendaciones..."):
                        profile = fetch_api_json("get", f"{API_URL}/profile/{steam_id}")
                        player_info = fetch_steam_user_profile(steam_id)
                        recs = fetch_api_json(
                            "get",
                            f"{API_URL}/recommend/collaborative/{steam_id}?n=8",
                        )
                        if profile and recs is not None:
                            st.session_state.profile_data = profile
                            st.session_state.player_info = player_info
                            st.session_state.recommendations = recs
                            st.session_state.recommendation_title = "Recomendaciones colaborativas"

        if st.session_state.profile_data:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Perfil de usuario</div><div class="section-subtitle">Resumen calculado a partir de la matriz de interacciones.</div>', unsafe_allow_html=True)
            render_user_profile_section(st.session_state.profile_data, st.session_state.get('player_info'))
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        with st.form("content_form", clear_on_submit=False):
            selected_games = st.multiselect(
                "Selecciona tus juegos de referencia",
                game_names,
                placeholder="Puedes añadir tantos juegos como quieras",
            )
            submit = st.form_submit_button("Generar recomendaciones")
            if submit:
                if len(selected_games) < 3:
                    st.warning("Por favor, selecciona al menos 3 juegos.")
                else:
                    with st.spinner("Buscando recomendaciones por contenido..."):
                        recs = fetch_api_json(
                            "post",
                            f"{API_URL}/recommend/content/?n=8",
                            payload={"games": selected_games},
                        )
                        if recs is not None:
                            st.session_state.profile_data = None
                            st.session_state.recommendations = recs
                            st.session_state.recommendation_title = "Recomendaciones por contenido"

    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown(
        """
        <div class="app-footer">
            Desarrollado por <a href="https://github.com/LuisEVV" target="_blank" style="color:#88a8ff;text-decoration:none;font-weight:700;">LuisEVV</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="cards-shell">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-title">{safe_text(st.session_state.recommendation_title or "Resultados")}</div>'
        '<div class="section-subtitle">Tarjetas enriquecidas con portada, sinopsis y enlace directo a Steam.</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.recommendations:
        with st.spinner("Enriqueciendo juegos con datos de Steam..."):
            render_recommendation_cards(st.session_state.recommendations)
    else:
        st.markdown(
            """
            <div class="empty-state">
                Selecciona un modo, completa la entrada y pulsa generar para ver aquí tus resultados.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()