
# -*- coding: utf-8 -*-
# Désactive la copie de build/index.html (contournement du PermissionError)
import streamlit_timeline
streamlit_timeline._import_styles = lambda *args, **kwargs: None


import asyncio
import json
import math
import os
import re
import time
import logging
import random
import sqlite3
import hashlib
import pandas as pd
import numpy as np
import requests
import feedparser
import aiohttp
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import matplotlib.pyplot as plt
import html

from io import BytesIO
from urllib.parse import urlparse
from string import punctuation
from PIL import Image, UnidentifiedImageError # Added UnidentifiedImageError
from bs4 import BeautifulSoup
from streamlit_echarts import st_echarts
from streamlit_timeline import st_timeline
from wordcloud import WordCloud

# Apply nest_asyncio if needed (often for Streamlit/Jupyter)
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    logging.info("nest_asyncio not installed or not needed in this environment.")

# === Configuration ===

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Constants
PASSWORD_REGEX = r'^(?=.*[A-Z])(?=.*[a-z])(?=.*\d).{8,}$'
# Headers par défaut
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/110.0.0.0 Safari/537.36"
    ),
    "Connection": "close"  # force fermeture, évite les "Connection closed"
}

# ---- Liste de secours pour la rotation UA ----
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:105.0) Gecko/20100101 Firefox/105.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
]

DB_FILE = 'rss.db'
DEFAULT_FEEDS = [
    {"url": "https://www.lemonde.fr/rss/une.xml"},
    {"url": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml"},
    {"url": "http://feeds.bbci.co.uk/news/world/rss.xml"}
]
USER_DATA_BASE_DIR = "user_data" # Base directory for user-specific files

# === Database Management ===

#####  BEGIN mod DB  #####
@st.cache_resource
def get_db_connection():
    """Connexion & auto-migration légère (ajout id / user_id)."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False, timeout=10)
    conn.execute("PRAGMA foreign_keys = ON")

    with conn:
        # ---------- USERS ----------
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
            """
        )

        # ---------- FEEDS ----------
        conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS feeds (
                        id      INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER,
                        username TEXT,              -- pour l’ancien schéma
                        url     TEXT NOT NULL,
                        folder  TEXT,              -- New column for folder
                        UNIQUE (user_id, url),
                        FOREIGN KEY (user_id)
                            REFERENCES users(id) ON DELETE CASCADE
                    )
                    """
                )

        # ---------- ARTICLES ----------
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS articles (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                title    TEXT,
                summary  TEXT,
                url      TEXT,
                published_timestamp INTEGER,
                UNIQUE(username, url)
            )
            """
        )
        try:
            conn.execute("SELECT feed_url FROM articles LIMIT 1")
        except sqlite3.OperationalError:
            logging.info("Ajout de la colonne 'feed_url' à la table articles.")
            conn.execute("ALTER TABLE articles ADD COLUMN feed_url TEXT")

        # ---------- MIGRATION LÉGÈRE ----------
        # 1) remplit users.id manquant
        conn.execute("UPDATE users SET id = rowid WHERE id IS NULL")
        # 2) copie id dans feeds.user_id si vide
        conn.execute(
            """
            UPDATE feeds
               SET user_id = (SELECT id FROM users WHERE users.username = feeds.username)
             WHERE user_id IS NULL
            """
        )
        # 3) Add folder column if it doesn't exist
        try:
            conn.execute("SELECT folder FROM feeds LIMIT 1")
        except sqlite3.OperationalError:
            logging.info("Adding 'folder' column to 'feeds' table.")
            conn.execute("ALTER TABLE feeds ADD COLUMN folder TEXT")
            # Optional: Assign a default folder to existing feeds if you want
            # conn.execute("UPDATE feeds SET folder = 'Sans dossier' WHERE folder IS NULL")
    return conn

@st.cache_data(ttl=3600, show_spinner="Chargement des articles depuis la base...")
def load_articles_from_db_cached(username: str) -> tuple[list[tuple[int, str, str]], dict[str, str]]:
    """
    Charge depuis la DB jusqu'à 200 articles pour un utilisateur,
    renvoie une liste de tuples (id, summary, feed_url) et 
    un mapping post_id -> url pour ouvrir les posts dans le graphe.
    """
    if not username:
        return [], {}

    conn = get_db_connection()
    with conn:
        cursor = conn.execute(
            """
            SELECT id, summary, url, feed_url
              FROM articles
             WHERE username = ?
          ORDER BY published_timestamp DESC
             LIMIT 200
            """,
            (username,)
        )
        rows = cursor.fetchall()

    # Mapping post_id → url
    post_id_to_url_map = { str(r[0]): (r[2] or "#") for r in rows }

    # On ne garde que l’id, le résumé et la source (feed_url)
    articles_data = [
        (r[0], r[1] or "", r[3] or "")
        for r in rows
    ]

    return articles_data, post_id_to_url_map

#####  END    mod DB  #####

# === User Authentication & Management ===

#####  BEGIN mod AUTH  #####
# ---------- AUTH ----------
@st.cache_data(show_spinner=False)
def fetch_user_hash(username: str) -> str | None:
    row = get_db_connection().execute(
        "SELECT password FROM users WHERE username = ?",
        (username,),
    ).fetchone()
    return row[0] if row else None


def hash_pw(pwd: str) -> str:
    return hashlib.sha256(pwd.encode()).hexdigest()


def validate_pw(pwd: str) -> bool:
    return bool(re.match(PASSWORD_REGEX, pwd))


def get_user_id(username: str) -> int | None:
    row = get_db_connection().execute(
        "SELECT id FROM users WHERE username = ?",
        (username,),
    ).fetchone()
    return row[0] if row else None


def register_user(username: str, pwd: str) -> tuple[bool, str]:
    if fetch_user_hash(username):
        return False, "✖ Nom d’utilisateur déjà utilisé."
    if not validate_pw(pwd):
        return False, "✖ Mot de passe trop faible."
    get_db_connection().execute(
        "INSERT INTO users (username, password) VALUES (?, ?)",
        (username, hash_pw(pwd)),
    )
    st.cache_data.clear()
    return True, "✅ Inscription réussie !"


def authenticate_user(username: str, pwd: str) -> bool:
    if fetch_user_hash(username) == hash_pw(pwd):
        st.session_state.user = username
        st.session_state.user_id = get_user_id(username)
        return True
    return False

#####  END    mod AUTH  #####

# === Article & Feed Data Handling ===

def load_articles_from_db() -> list[tuple[int, str]]:
    """
    Retourne la liste des articles enregistrés (id, résumé)
    en filtrant selon les flux actifs, sauf si seuls les DEFAULT_FEEDS sont présents
    (cas du fallback automatique), auquel cas on renvoie tout.
    """
    user = st.session_state.get("user")
    if not user:
        return []

    # articles_raw est une liste de tuples (id, summary, feed_url)
    articles_raw, post_id_map = load_articles_from_db_cached(user)
    # on stocke la map pour le graph
    st.session_state['post_id_to_url'] = post_id_map

    # Récupère les URL de flux actuels
    user_feeds       = load_feeds()
    current_feed_urls = [f["url"] for f in user_feeds]
    # Liste des URLs par défaut (fallback)
    default_urls     = [d["url"] for d in DEFAULT_FEEDS]

    # Si on est en plein fallback (seuls les DEFAULT_FEEDS sont chargés),
    # on renvoie tous les articles ; sinon on ne garde que ceux des flux actifs
    if set(current_feed_urls) == set(default_urls):
        filtered = [
            (aid, summary)
            for aid, summary, _ in articles_raw
        ]
    else:
        filtered = [
            (aid, summary)
            for aid, summary, feed_url in articles_raw
            if feed_url in current_feed_urls
        ]

    return filtered




def store_articles_in_db(articles_to_store: list[dict]):
    """Stores fetched articles in the database for the logged-in user, including feed_url."""
    if 'user' not in st.session_state:
        st.error("Utilisateur non connecté. Impossible d'enregistrer les articles.")
        return 0
    user = st.session_state.user
    if not articles_to_store:
        logging.info("No articles provided to store_articles_in_db.")
        return 0

    conn = get_db_connection()
    new_count = 0

    # Préparer les tuples à insérer
    articles_processed = []
    for art in articles_to_store:
        title   = art.get("title", "Titre non disponible")
        summary = str(art.get("summary", "") or "")
        url     = art.get("link")
        feed_url = art.get("feed_url")

        # Calcul du timestamp comme avant
        published_ts = int(time.time())
        published_parsed = art.get("published_parsed")
        if published_parsed:
            try:
                published_ts = int(time.mktime(published_parsed))
            except (OverflowError, TypeError, ValueError):
                logging.warning(f"Could not parse timestamp for article '{title}'. Using current time.")

        articles_processed.append((
            user,
            title,
            summary,
            url,
            published_ts,
            feed_url
        ))

    try:
        with conn:
            cursor = conn.cursor()
            # On a maintenant 6 colonnes : username, title, summary, url, published_timestamp, feed_url
            cursor.executemany(
                """INSERT OR IGNORE INTO articles 
                   (username, title, summary, url, published_timestamp, feed_url)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                articles_processed
            )
            new_count = cursor.rowcount if cursor.rowcount > 0 else 0
            conn.commit()

        # Clear cache so qu'on recharge bien après modif
        st.cache_data.clear()
        return new_count

    except sqlite3.Error as e:
        logging.error(f"SQLite error inserting articles for user {user}: {e}", exc_info=True)
        st.error(f"Erreur lors de l'enregistrement des articles : {e}")
        return 0


#####  BEGIN mod FEEDS  #####
# ---------- FEEDS ----------
@st.cache_data(ttl=600, show_spinner=False)
def load_feeds_cached(user_id: int, username: str) -> list[dict]:
    """Charge les flux en gérant l’ancien ou le nouveau schéma (avec dossier)."""
    cols = {
        c[1] for c in get_db_connection().execute("PRAGMA table_info(feeds)")
    }
    conn = get_db_connection()
    feeds_list = []

    # Check if both user_id and folder columns exist (newest schema)
    if "user_id" in cols and "folder" in cols:
        rows = conn.execute(
            "SELECT url, folder FROM feeds WHERE user_id = ?", (user_id,)
        ).fetchall()
        # Assign "Sans dossier" if folder is NULL
        feeds_list = [{"url": r[0], "folder": r[1] if r[1] is not None else "Sans dossier"} for r in rows]
    # Check if only user_id exists (new schema, but missing folder - needs migration check)
    elif "user_id" in cols:
         rows = conn.execute(
            "SELECT url FROM feeds WHERE user_id = ?", (user_id,)
        ).fetchall()
         # For existing entries without folder, assign "Sans dossier"
         feeds_list = [{"url": r[0], "folder": "Sans dossier"} for r in rows]
    # Handle old schema (only username)
    else:
        rows = conn.execute(
            "SELECT url FROM feeds WHERE username = ?", (username,)
        ).fetchall()
        feeds_list = [{"url": r[0], "folder": "Sans dossier"} for r in rows] # Old feeds get "Sans dossier"

    # If no feeds are loaded for the user AND the user is logged in, add defaults with folders
    if not feeds_list and (user_id is not None or username is not None):
        logging.info(f"No feeds found for user {username}, adding defaults.")
        save_feeds(DEFAULT_FEEDS) # Save defaults with their folders
        # Reload after saving defaults. The recursive call is safe here as it won't loop if save succeeds.
        return load_feeds_cached(user_id, username)

    return feeds_list


def save_feeds(feeds_to_save: list[dict]):
    """Sauvegarde la liste des flux pour l'utilisateur connecté, incluant les dossiers."""
    uid, uname = st.session_state.get("user_id"), st.session_state.get("user")
    if uid is None:
        st.error("Utilisateur non connecté."); return

    conn = get_db_connection()
    cols = {c[1] for c in conn.execute("PRAGMA table_info(feeds)")}
    # --- suppression des anciens feeds ---
    if "user_id" in cols:
        conn.execute("DELETE FROM feeds WHERE user_id = ?", (uid,))
    else:
        conn.execute("DELETE FROM feeds WHERE username = ?", (uname,))

    # --- insertion des nouveaux feeds (votre code existant) ---
    if "user_id" in cols and "folder" in cols:
        data = [(uid, f["url"], f.get("folder", "Sans dossier")) for f in feeds_to_save if "url" in f]
        conn.executemany("INSERT OR IGNORE INTO feeds (user_id, url, folder) VALUES (?, ?, ?)", data)
    elif "user_id" in cols:
        data = [(uid, f["url"]) for f in feeds_to_save if "url" in f]
        conn.executemany("INSERT OR IGNORE INTO feeds (user_id, url) VALUES (?, ?)", data)
    else:
        data = [(uname, f["url"]) for f in feeds_to_save if "url" in f]
        conn.executemany("INSERT OR IGNORE INTO feeds (username, url) VALUES (?, ?)", data)

    # --- NOUVEAU : on nettoie la table articles pour ce user ---
    # on ne garde que les articles dont le feed_url est dans la liste active
    active_urls = [f["url"] for f in feeds_to_save if "url" in f]
    if active_urls:
        placeholders = ",".join("?" for _ in active_urls)
        params = [st.session_state.user] + active_urls
        conn.execute(
            f"DELETE FROM articles "
            f"WHERE username = ? "
            f"AND (feed_url NOT IN ({placeholders}) OR feed_url IS NULL)",
            params
        )
    else:
        # s’il n’y a plus aucun feed, on peut aussi tout supprimer
        conn.execute("DELETE FROM articles WHERE username = ?", (st.session_state.user,))

    conn.commit()
    # on vide le cache des feeds
    load_feeds_cached.clear()



def load_feeds() -> list[dict]:
    """Renvoie simplement les flux enregistrés — ne remet plus les defaults."""
    uid = st.session_state.get("user_id")
    uname = st.session_state.get("user")
    return load_feeds_cached(uid, uname)


#####  END    mod FEEDS  #####

# === Feed Fetching & Parsing ===

def extract_image_from_summary(summary_html: str) -> str | None:
    """Extracts the first image URL from summary HTML, safely."""
    if not summary_html or not isinstance(summary_html, str):
        return None
    try:
        soup = BeautifulSoup(summary_html, "html.parser")
        img_tag = soup.find("img")
        if img_tag and img_tag.has_attr("src"):
             # Basic check for potentially valid image URLs (crude)
             src = img_tag["src"]
             if isinstance(src, str) and (src.startswith('http://') or src.startswith('https://')):
                 return src
    except Exception as e:
         logging.warning(f"Error parsing summary HTML for image: {e}")
    return None

def extraire_image(entry: dict) -> str | None:
    """Extracts an image URL from a feed entry (media content, thumbnail, or summary)."""
    if not isinstance(entry, dict): return None

    # Prioritize media_content
    media_content = entry.get("media_content")
    if isinstance(media_content, list) and media_content:
        for item in media_content:
            if isinstance(item, dict) and item.get("medium") == "image" and isinstance(item.get("url"), str):
                return item["url"]
        for item in media_content: # Fallback if medium!=image
             if isinstance(item, dict) and isinstance(item.get("url"), str):
                 return item["url"]

    # Check media_thumbnail
    media_thumbnail = entry.get("media_thumbnail")
    if isinstance(media_thumbnail, list) and media_thumbnail:
         item = media_thumbnail[0]
         if isinstance(item, dict) and isinstance(item.get("url"), str):
            return item["url"]

    # Fallback to parsing the summary
    return extract_image_from_summary(entry.get("summary"))

async def fetch_single_feed_async(session: aiohttp.ClientSession, url: str) -> dict:
    """Télécharge et parse un flux RSS/Atom avec retry + rotation UA."""
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        logging.error(f"URL invalide : {url}")
        return {"bozo": True, "articles": [], "error": "Invalid URL"}

    response = None
    content  = b""

    for attempt in range(3):  # 0,1,2
        try:
            headers = HEADERS.copy()
            if attempt > 0:                    # dès le 1ᵉʳ retry
                headers["User-Agent"] = random.choice(USER_AGENTS)
                headers["Referer"]     = url

            timeout = aiohttp.ClientTimeout(total=20 + attempt * 10)
            async with session.get(url, headers=headers, timeout=timeout, ssl=False) as resp:
                response = resp
                if resp.status == 200:
                    content = await resp.read()
                    break                      # ✔️ succès → on sort
                if resp.status == 403 and attempt < 2:
                    logging.warning(f"{url} → 403 (essai {attempt}), on ré‑essaie…")
                    await asyncio.sleep(2 ** attempt)
                    continue
                error_msg = f"HTTP Status {resp.status}"
                logging.error(f"{url} → {error_msg}")
                return {"bozo": True, "articles": [], "error": error_msg}

        except asyncio.TimeoutError:
            logging.warning(f"Timeout {url} (essai {attempt})")
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
                continue
            return {"bozo": True, "articles": [], "error": "Timeout"}

        except aiohttp.ClientError as e:
            logging.warning(f"Erreur réseau {url} (essai {attempt}) : {e}")
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
                continue
            return {"bozo": True, "articles": [], "error": f"Network: {e}"}

    if response is None or response.status != 200:
        return {"bozo": True, "articles": [], "error": "Failed after retries"}

    # ---------- Parsing avec feedparser ----------
    loop = asyncio.get_event_loop()
    raw_feed = await loop.run_in_executor(None, feedparser.parse, content)

    articles = []
    for entry in raw_feed.entries:
        summary_html = entry.get("summary", "")
        clean_summary = BeautifulSoup(summary_html, "html.parser").get_text(" ", strip=True) if summary_html else ""
        articles.append({
            "title":   str(entry.get("title", "Titre non disponible")).strip(),
            "link":    str(entry.get("link", "#")),
            "summary": clean_summary,
            "published_parsed": entry.get("published_parsed"),
            "published": entry.get("published"),
            "media_content": entry.get("media_content"),
            "media_thumbnail": entry.get("media_thumbnail")
        })

    return {"bozo": raw_feed.bozo, "articles": articles, "error": None}


async def fetch_all_feeds_async(urls: list[str]) -> list[dict]:
    """Fetches multiple RSS feeds asynchronously."""
    if not urls: return []
    # Create session within the async function context
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_single_feed_async(session, url) for url in urls if isinstance(url, str)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    processed_results = []
    for i, res in enumerate(results):
        original_url = urls[i] # Assume order is preserved
        if isinstance(res, Exception):
            logging.error(f"Gathered exception for feed {original_url}: {res}", exc_info=res)
            processed_results.append({"bozo": True, "articles": [], "error": f"Exception during gather: {res}", "feed_url": original_url})
        elif isinstance(res, dict):
            res["feed_url"] = original_url # Ensure feed URL is in the result dict
            processed_results.append(res)
        else:
             logging.error(f"Unexpected result type from gather for {original_url}: {type(res)}")
             processed_results.append({"bozo": True, "articles": [], "error": "Unexpected result type", "feed_url": original_url})

    return processed_results


def get_timestamp(entry: dict) -> float:
    """Returns the timestamp (float) of an article for sorting, robustly."""
    if not isinstance(entry, dict): return time.time()
    published_parsed = entry.get("published_parsed")
    if published_parsed:
        try:
            return time.mktime(published_parsed)
        except (OverflowError, TypeError, ValueError):
            pass
    # Fallback: try parsing 'published' string if parsed struct failed/missing
    published_str = entry.get("published")
    if isinstance(published_str, str):
        try:
            # Attempt common formats (add more if needed)
            ts = pd.to_datetime(published_str, errors='coerce')
            if pd.notna(ts):
                 return ts.timestamp()
        except Exception:
             pass # Ignore parsing errors here
    return time.time() # Default to current time

@st.cache_data(ttl=86400, show_spinner=False) # Cache favicon color for a day
def get_dominant_color(favicon_url: str) -> str:
    """Calculates the dominant color of a favicon (cached), robustly."""
    default_color = "#444444" # Default grey
    if not isinstance(favicon_url, str): return default_color
    try:
        response = requests.get(favicon_url, timeout=7, stream=True) # Slightly longer timeout
        response.raise_for_status()
        content_type = response.headers.get('content-type', '').lower()
        # Basic check for image types
        if not content_type.startswith('image/'):
             logging.warning(f"Favicon URL {favicon_url} did not return an image content-type ({content_type}). Skipping.")
             return default_color

        content_length = int(response.headers.get('content-length', 0))
        if content_length > 100 * 1024: # Limit to 100KB
             logging.warning(f"Favicon {favicon_url} too large ({content_length} bytes), skipping dominant color.")
             return default_color

        img_content = response.content
        if not img_content:
             logging.warning(f"Favicon {favicon_url} returned empty content.")
             return default_color

        img = Image.open(BytesIO(img_content)).resize((16, 16), Image.Resampling.LANCZOS).convert('RGB')
        img_array = np.array(img)
        colors, counts = np.unique(img_array.reshape(-1, 3), axis=0, return_counts=True)

        if colors.size == 0: return default_color # Handle empty image case

        # Filter out pure white/black unless they are overwhelmingly dominant
        non_bw_mask = ~np.all((colors == [255, 255, 255]) | (colors == [0, 0, 0]), axis=1)
        if np.any(non_bw_mask) and counts[non_bw_mask].sum() > len(img_array.reshape(-1, 3)) * 0.1:
             dominant_color_rgb = colors[non_bw_mask][np.argmax(counts[non_bw_mask])]
        else:
             dominant_color_rgb = colors[np.argmax(counts)]

        return f"rgb({dominant_color_rgb[0]}, {dominant_color_rgb[1]}, {dominant_color_rgb[2]})"

    except requests.exceptions.RequestException as e:
        logging.warning(f"Could not fetch favicon {favicon_url}: {e}")
    except UnidentifiedImageError:
         logging.warning(f"Could not identify image file from favicon {favicon_url}. Content likely not an image.")
    except Exception as e:
        # Catch potential PIL errors (decompression bomb, etc.)
        logging.warning(f"Could not process favicon {favicon_url} for dominant color: {e}", exc_info=True)
    return default_color

# === Dashboard Data Fetching ===

@st.cache_data(ttl=1800, show_spinner="Chargement données tableau de bord...")
def fetch_dashboard_data(rss_url: str, flux_name: str) -> pd.DataFrame | None:
    """Fetches and parses RSS data specifically for dashboard use (cached). Returns None on failure."""
    if not isinstance(rss_url, str) or not rss_url.startswith(('http://', 'https://')):
        logging.error(f"Invalid URL for dashboard: {rss_url}")
        return None
    try:
        response = requests.get(rss_url, headers=HEADERS, timeout=20) # Increased timeout
        response.raise_for_status()
        content = response.content
        # Check if content is empty
        if not content:
            logging.warning(f"Empty content received from dashboard feed {flux_name} ({rss_url}).")
            return pd.DataFrame() # Return empty DF, not None

        feed = feedparser.parse(content)
        logging.info(f"Fetched and parsed dashboard data for {flux_name} (Entries: {len(feed.entries)})")

        if not feed.entries:
            return pd.DataFrame()

        entries_list = []
        for entry in feed.entries:
            published_dt = pd.to_datetime(entry.get('published', entry.get('updated')), errors='coerce')
            if pd.isna(published_dt):
                logging.warning(f"Skipping entry with unparsable date in {flux_name}: {entry.get('title')}")
                continue

            # Ensure required fields have defaults and correct types
            title = str(entry.get('title', 'Sans titre')).strip()
            link = str(entry.get('link', '#'))
            category = str(entry.get('category', 'Inconnu'))

            entries_list.append({
                'Titre': title,
                'Date': published_dt.date(),
                'Timestamp': published_dt,
                'Commentaires': int(entry.get('slash_comments', 0)) if 'slash_comments' in entry else 0,
                'Lien': link,
                'Flux': flux_name,
                'Catégorie': category,
                'Image': extraire_image(entry)
            })

        if not entries_list:
             return pd.DataFrame()

        return pd.DataFrame(entries_list)

    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching dashboard feed {flux_name} ({rss_url}): {e}", exc_info=True)
        st.error(f"Erreur réseau lors du chargement du flux '{flux_name}'.")
        return None # Indicate failure with None
    except Exception as e:
         logging.error(f"Unexpected error processing dashboard feed {flux_name} ({rss_url}): {e}", exc_info=True)
         st.error(f"Erreur inattendue lors du traitement du flux '{flux_name}'.")
         return None # Indicate failure with None

# === UI Components & Styling ===

def apply_custom_css():
    """Applies custom CSS styles."""
    # (CSS code remains largely the same as previous version - omitted for brevity, assume it's correct)
    st.markdown("""
        <style>
            /* General Styles */
            html, body, .stApp { background-color: #1E1E1E; color: #E0E0E0; font-family: "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
            h1, h2, h3, h4, h5, h6 { color: #00AACC; }
            a { color: #00CFFF; } a:hover { color: #80DFFF; }
            hr { border-top: 1px solid #444444; margin: 0.5rem 0; }
            /* Sidebar Styles */
            [data-testid="stSidebar"] { background-color: #252526; padding-top: 1rem; }
            .st-emotion-cache-1 PADDING { padding-top: 5rem; } /* Adjust selector if needed */
            [data-testid="stSidebar"] .stButton button { background-color: transparent; color: #E0E0E0; border: 1px solid #555555; padding: 0.6rem 1rem; margin-bottom: 0.5rem; border-radius: 4px; width: 100%; text-align: left; transition: background-color 0.2s ease, border-color 0.2s ease; }
            [data-testid="stSidebar"] .stButton button:hover { background-color: #333333; border-color: #777777; }
            [data-testid="stSidebar"] .stButton button:disabled { background-color: #00AACC; color: #1E1E1E; border-color: #00AACC; cursor: default; opacity: 1; }
            [data-testid="stSidebar"] .stImage > img { display: block; margin-left: auto; margin-right: auto; margin-bottom: 2rem; }
            /* Main Content Styles */
            .stTextInput input, .stTextArea textarea, .stSelectbox select, .stNumberInput input { background-color: #333333; color: #E0E0E0; border: 1px solid #555555; }
            .stButton button:not([data-testid="stSidebar"] .stButton button) { background-color: #007ACC; color: white; border: none; padding: 0.5rem 1rem; border-radius: 4px; transition: background-color 0.2s ease; }
            .stButton button:not([data-testid="stSidebar"] .stButton button):hover { background-color: #009FFF; }
            .stForm { border: 1px solid #444444; padding: 1rem; border-radius: 5px; background-color: #252526; margin-bottom: 1rem; }
            .stExpander { border: 1px solid #444444; border-radius: 5px; background-color: #252526; }
            .stExpander header { font-weight: bold; color: #00AACC; }
            /* RSS Reader Specific */
            .rss-article-detailed { border-bottom: 1px solid #444; padding-bottom: 1rem; margin-bottom: 1rem; }
            .rss-article-detailed h3 a { color: #00CFFF; text-decoration: none; }
            .rss-article-detailed h3 a:hover { text-decoration: underline; }
            .rss-article-detailed p { line-height: 1.6; }
            .rss-article-meta { font-size: 0.9rem; color: #AAAAAA; margin-bottom: 0.5rem; }
            .rss-article-summary { line-height: 1.6; }
            .rss-favicon { vertical-align: middle; margin-right: 5px; width: 16px; height: 16px; }
            .rss-compact-item { margin-bottom: 0.3rem; line-height: 1.3; }
            .rss-compact-item strong a { color: #00CFFF; text-decoration: none; }
            .rss-cube { background-color: #252526; padding: 0.8rem; border-radius: 5px; height: 350px; display: flex; flex-direction: column; justify-content: space-between; overflow: hidden; border: 1px solid #444; }
            .rss-cube h4 { font-size: 1rem; margin: 0 0 0.3rem 0; line-height: 1.3; max-height: 3.9em; overflow: hidden; text-overflow: ellipsis; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; }
            .rss-cube h4 a { color: #00CFFF; text-decoration: none; }
            .rss-cube .cube-image-container { height: 180px; width: 100%; display: flex; align-items: center; justify-content: center; margin-bottom: 0.5rem; border-radius: 3px; overflow: hidden; }
            .rss-cube .cube-image { width: 100%; height: 100%; object-fit: cover; }
            .rss-cube .cube-favicon-placeholder { width: 30px; height: 30px; }
            .rss-cube-meta { font-size: 0.8rem; color: #AAAAAA; margin-top: auto; }
            .rss-cube-meta img { width: 14px; height: 14px; vertical-align: middle; margin-right: 3px; }
            .rss-cube-meta p { margin: 0.1rem 0; }
            .rss-cube-meta hr { margin: 0.3rem 0 0 0; border-top: 1px solid #555; }
            /* Other Specific Styles */
            .stAlert { border-radius: 4px; }
            [data-testid="stSidebar"] [data-testid="stImage"] { text-align: center; }
            /* D3 Tooltip */
            .tooltip { position: absolute; background-color: rgba(40, 40, 40, 0.9); color: #E0E0E0; padding: 8px 12px; border-radius: 4px; font-size: 12px; pointer-events: none; opacity: 0; transition: opacity 0.2s ease; max-width: 350px; white-space: normal; z-index: 10; }
        </style>
    """, unsafe_allow_html=True)

def setup_sidebar(pages: dict):
    """Configures the sidebar navigation."""
    with st.sidebar:
        try:
            kyrria_logo = "Kyrria_logo.png" # Ensure this file exists
            if os.path.exists(kyrria_logo):
                st.image(kyrria_logo, width=200)
            else:
                logging.warning("Logo image 'Kyrria_logo.png' not found.")
                st.markdown("### KYRRIA Demo") # Fallback text
        except Exception as e:
            logging.error(f"Error loading logo: {e}", exc_info=True)
            st.warning(f"Impossible de charger le logo.")

        st.markdown("---")
        st.subheader("Navigation")
        if "selected_page" not in st.session_state:
            st.session_state.selected_page = list(pages.keys())[0]

        for page_name in pages.keys():
            is_selected = (page_name == st.session_state.selected_page)
            if st.button(page_name, key=f"nav_{page_name}", disabled=is_selected, use_container_width=True):
                st.session_state.selected_page = page_name
                st.rerun()

        st.markdown("---")
        if 'user' in st.session_state:
            st.subheader("Compte")
            st.caption(f"Connecté: **{st.session_state.user}**") # Use caption for less emphasis
            if st.button('🔒 Déconnexion', key="logout_sidebar", use_container_width=True):
                user_logout = st.session_state.user
                # Clear session state selectively
                keys_to_keep = {'page'} # Add any other keys that should persist across logout/login
                for key in list(st.session_state.keys()):
                     if key not in keys_to_keep:
                         del st.session_state[key]
                st.session_state.page = 'login'
                st.cache_data.clear()
                st.cache_resource.clear() # Clears DB connection too, will reconnect on next access
                logging.info(f"User {user_logout} logged out.")
                st.success('🔒 Déconnecté avec succès.')
                st.rerun()

# === Application Pages ===

def display_home_page():
    """Displays the Home page content."""
    st.title("Bienvenue sur KYRRIA App Demo")
    st.markdown("""
        **KYRRIA App Demo** est une application de démonstration présentant diverses visualisations interactives de données agrégées et analysées.

        **Premiers pas :**
        1. Allez dans `Gestionnaire de flux` pour ajouter des URLs de flux RSS ou Atom.
        2. Explorez le `Lecteur RSS` pour lire les derniers articles.
        3. Cliquez sur `Enregistrer ces articles en DB` dans le Lecteur RSS pour sauvegarder les articles affichés dans votre base de données.
        4. Une fois des articles enregistrés, visitez `Entities & Relations` pour lancer l'analyse (peut nécessiter une clé API Google).
    """)
    st.info("Cette application utilise une base de données locale (`rss.db`) pour stocker vos informations utilisateur, vos flux et les articles que vous choisissez d'enregistrer.")

def display_login_signup_forms():
    """Handles the display of login and signup forms."""
    if 'page' not in st.session_state:
        st.session_state.page = 'login'

    st.title("Authentification Requise")
    st.markdown("Veuillez vous connecter ou créer un compte pour accéder à l'application.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        with st.form('login_form'):
            st.subheader('🔑 Connexion')
            login_user = st.text_input('Nom d’utilisateur', key='login_user', autocomplete="username")
            login_pwd = st.text_input('Mot de passe', type='password', key='login_pwd', autocomplete="current-password")
            login_btn = st.form_submit_button('Se connecter')

        if login_btn:
            if not login_user or not login_pwd:
                st.error('Veuillez remplir tous les champs.')
            elif authenticate_user(login_user, login_pwd):
                st.session_state.user = login_user
                st.session_state.page = 'app'
                # Clear potentially stale state from previous user/session
                keys_to_clear = ['selected_page', 'all_entities', 'all_relations', 'processed_post_ids',
                                 'results_by_post', 'next_article_index', 'processing_active', 'ner_data_loaded',
                                 'selected_feeds', 'view_mode', 'post_id_to_url']
                for key in keys_to_clear:
                    st.session_state.pop(key, None)

                st.success(f'Connexion réussie ! Bienvenue, **{login_user}**.')
                time.sleep(1.5)
                st.rerun()
            else:
                st.error('❌ Identifiants incorrects.')

    with col2:
        with st.form('signup_form'):
            st.subheader('📝 Inscription')
            signup_user = st.text_input('Choisissez un nom d’utilisateur', key='signup_user_reg', autocomplete="username")
            signup_pwd = st.text_input('Choisissez un mot de passe', type='password', key='signup_pwd_reg', autocomplete="new-password")
            signup_cpwd = st.text_input('Confirmez le mot de passe', type='password', key='signup_cpwd_reg', autocomplete="new-password")
            signup_btn = st.form_submit_button('S’inscrire')

        if signup_btn:
            if not signup_user or not signup_pwd or not signup_cpwd:
                st.error('Veuillez remplir tous les champs.')
            elif signup_pwd != signup_cpwd:
                st.error('✖ Les mots de passe ne correspondent pas.')
            else:
                success, msg = register_user(signup_user, signup_pwd)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)

def display_feed_manager_page():
    """Displays the Feed Manager page."""
    st.title("📡 Gestionnaire de flux RSS/Atom")
    st.markdown("Ajoutez, visualisez et supprimez les flux RSS/Atom que vous souhaitez suivre.")

    feeds = load_feeds() # Uses the wrapper function


    # --- IMPORT OPML FILE UPLOADER ---
    uploaded_opml = st.file_uploader("📂 Importer un fichier OPML", type=["xml", "opml"])
    if uploaded_opml:
        import xml.etree.ElementTree as ET
        try:
            tree = ET.parse(uploaded_opml)
            root = tree.getroot()
            # Cherche tous les <outline> avec xmlUrl
            urls = [
                o.attrib["xmlUrl"]
                for o in root.findall(".//outline")
                if "xmlUrl" in o.attrib
            ]
            # Ne garde que les URLs absentes de vos feeds actuels
            existing = {f["url"] for f in feeds}
            new_urls = [u for u in urls if u not in existing]
            if new_urls:
                for u in new_urls:
                    feeds.append({"url": u})
                save_feeds(feeds)
                load_feeds_cached.clear()
                st.success(f"✅ {len(new_urls)} flux importés depuis OPML")
                #st.experimental_rerun()
            else:
                st.info("Aucun nouveau flux à ajouter depuis l’OPML.")
        except ET.ParseError as e:
            st.error(f"Erreur de parsing OPML : {e}")

    with st.form("flux_form"):
        new_flux_url = st.text_input("URL du nouveau flux", placeholder="https://exemple.com/rss", key="new_feed_url")
        add_button = st.form_submit_button("➕ Ajouter le flux")

        if add_button:
            if new_flux_url:
                try:
                    parsed_url = urlparse(new_flux_url)
                    if not all([parsed_url.scheme in ['http', 'https'], parsed_url.netloc]):
                        st.error("URL invalide. Doit commencer par http:// ou https:// et contenir un domaine.")
                    elif new_flux_url in [feed["url"] for feed in feeds]:
                        st.warning("Ce flux est déjà dans votre liste.")
                    else:
                        feeds.append({"url": new_flux_url})
                        save_feeds(feeds)
                        st.success(f"✅ Flux ajouté : {new_flux_url}")
                        # Clear feed cache explicitly after saving and rerun
                        load_feeds_cached.clear()
                        st.rerun()
                except Exception as e:
                     st.error(f"Erreur lors de l'ajout de l'URL : {e}")
                     logging.error(f"Error parsing/adding URL {new_flux_url}: {e}", exc_info=True)
            else:
                st.error("Veuillez entrer une URL.")

    st.markdown("---")
    st.subheader("Vos Flux Enregistrés")

    if not feeds:
        st.info("Vous n'avez aucun flux enregistré. Ajoutez une URL ci-dessus pour commencer.")
    else:
        for index, feed in enumerate(feeds):
            feed_url = feed.get("url", "URL Manquante")
            domain = "Invalide"
            favicon_url = ""
            try:
                parsed = urlparse(feed_url)
                if parsed.netloc:
                    domain = parsed.netloc.replace("www.", "")
                    favicon_url = f"https://www.google.com/s2/favicons?sz=32&domain_url={parsed.netloc}"
            except Exception as parse_err:
                logging.warning(f"Could not parse feed URL {feed_url}: {parse_err}")

            col1, col2, col3 = st.columns([0.5, 3, 0.8]) # Adjusted ratios

            with col1:
                if favicon_url:
                    st.image(favicon_url, width=24)
                else:
                    st.markdown("❔") # Placeholder

            with col2:
                st.markdown(f"**{domain}**", unsafe_allow_html=True)
                st.caption(f"[{feed_url}]({feed_url})") # Link the caption

            with col3:
                if st.button("Suppr.", key=f"remove_{index}_{feed_url}", help=f"Supprimer {domain}"):
                    removed_feed = feeds.pop(index)
                    save_feeds(feeds)
                    st.success(f"🗑️ Flux '{domain}' supprimé.")
                    load_feeds_cached.clear() # Clear cache
                    st.rerun()
            st.markdown("---", unsafe_allow_html=True) # Separator between feeds

def display_rss_reader_page():
    # === Pagination “Charger plus” ===
    PAGE_SIZE = 20
    if "rss_offset" not in st.session_state:
        st.session_state["rss_offset"] = PAGE_SIZE

    """Displays the RSS Reader page."""
    st.title("📰 Lecteur de flux RSS/Atom")
    st.markdown("Agrégation de vos flux enregistrés. Choisissez les flux et le mode d'affichage.")

    feeds = load_feeds()
    if not feeds:
        st.warning("Aucun flux n'est disponible. Veuillez en ajouter dans le 'Gestionnaire de flux'.")
        if st.button("Aller au Gestionnaire de flux"):
            st.session_state.selected_page = "📡Gestionnaire de flux"
            st.rerun()
        return

    # == Configuration d'affichage compacte ==
    st.markdown("**⚙️ Configuration d'affichage**")  # titre minimal

    # une seule ligne : 2/3 pour le mode, 1/3 pour le filtre
    col_mode, col_filter = st.columns([2, 1], gap="small")

    with col_mode:
        # on cache le label natif, et on affiche la légende en markdown
        st.markdown("Mode")
        view_mode = st.selectbox(
            "", 
            ["Liste détaillée", "Liste raccourcie", "Vue en cubes"],
            key="view_mode_radio",
            label_visibility="collapsed"
        )
        if view_mode != st.session_state.get("view_mode"):
            st.session_state.view_mode = view_mode

    with col_filter:
        st.markdown("Filtre")
        search_keyword = st.text_input(
            "", 
            key="search_reader", 
            placeholder="Ex : technologie",
            label_visibility="collapsed"
        ).lower()

    feed_urls = [feed.get("url") for feed in feeds if isinstance(feed, dict) and feed.get("url")]
    feed_display_names = {url: urlparse(url).netloc.replace("www.", "") for url in feed_urls if urlparse(url).netloc}

    # Use domain names for selection, default to all URLs if names fail
    options = [feed_display_names.get(url, url) for url in feed_urls]
    # Default selection: Use URLs stored in state, map back to display names
    default_selection_urls = st.session_state.get('selected_feeds', feed_urls)
    default_display_selection = [feed_display_names.get(url, url) for url in default_selection_urls if url in feed_display_names]

    selected_display_names = st.multiselect(
        "Sélectionner les flux à afficher",
        options=options,
        default=default_display_selection,
        key="feed_selector_reader"
    )

    # Map selected display names back to URLs
    selected_feeds_urls = [url for url, name in feed_display_names.items() if name in selected_display_names]

    # Réinitialise la pagination quand on change de sélection
    if selected_feeds_urls != st.session_state.get('selected_feeds'):
         st.session_state.selected_feeds = selected_feeds_urls
         st.session_state["rss_offset"] = PAGE_SIZE
         st.rerun()

    if not selected_feeds_urls:
        st.info("Veuillez sélectionner au moins un flux à afficher.")
        return

    # --- Fetching and Processing Articles ---
    articles = []
    flux_metadata = {}
    fetch_errors = []
    all_fetched_articles = [] # Keep all before filtering/sorting

    # Use a placeholder while fetching
    fetch_placeholder = st.empty()
    fetch_placeholder.info(f"📡 Chargement des articles depuis {len(selected_feeds_urls)} flux...")

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    feed_results = loop.run_until_complete(fetch_all_feeds_async(selected_feeds_urls))
    fetch_placeholder.empty() # Remove placeholder

    for result in feed_results:
        feed_url = result.get("feed_url")
        if not feed_url: continue # Skip if URL somehow missing

        if result.get("error"):
            err_msg = result.get("error", "Erreur inconnue")
            domain_name = urlparse(feed_url).netloc
            logging.warning(f"Problem loading feed {feed_url}: {err_msg}")
            fetch_errors.append(f"⚠️ {domain_name}: {err_msg}")
        else:
            try:
                parsed = urlparse(feed_url)
                domain = parsed.netloc.replace("www.", "") if parsed.netloc else "Source Inconnue"
                favicon_url = f"https://www.google.com/s2/favicons?sz=32&domain_url={parsed.netloc}" if parsed.netloc else ""
                dominant_color = get_dominant_color(favicon_url) if favicon_url else "#444444"
                flux_metadata[feed_url] = (favicon_url, dominant_color, domain)

                for item in result.get("articles", []):
                    item["feed_url"] = feed_url
                    item["image_url"] = extraire_image(item)
                    all_fetched_articles.append(item)
            except Exception as e:
                 logging.error(f"Error processing data for feed {feed_url} after fetch: {e}", exc_info=True)
                 fetch_errors.append(f"⚠️ Erreur interne traitant {urlparse(feed_url).netloc}")

    if fetch_errors:
        st.warning("Certains flux n'ont pas pu être chargés:\n" + "\n".join(fetch_errors))

    # --- Filtering and Sorting ---
    articles = all_fetched_articles
    if search_keyword:
        articles = [
            entry for entry in articles
            if search_keyword in str(entry.get("title", "")).lower() or \
               search_keyword in str(entry.get("summary", "")).lower()
        ]

    articles = sorted(articles, key=get_timestamp, reverse=True)
    articles_to_display = articles[: st.session_state["rss_offset"]]


    st.markdown("---")
    st.subheader(f"📰 Articles Agrégés ({len(articles_to_display)} affichés / {len(articles)} total chargés)")


    if articles_to_display:
        st.info(
            "Cliquez sur 'Enregistrer ces articles affichés en DB' pour les sauvegarder "
            "et pouvoir les utiliser dans la section 'Entities & Relations'."
        )
        if st.button("💾 Enregistrer tous les articles des flux sélectionnés"):
            with st.spinner("Sauvegarde en cours…"):
                inserted_count = store_articles_in_db(articles)
            if inserted_count > 0:
                st.success(f"{inserted_count} articles enregistrés.")
            else:
                st.info("Aucun nouvel article (probablement déjà en base).")


    else:
        st.info("Aucun article à afficher correspondant aux flux/filtres sélectionnés.")
        return

    # --- Display Modes ---
    view_mode = st.session_state.get("view_mode", "Liste détaillée") # Get current mode

    # (Detailed, Compact, Cube view rendering code remains the same as previous version - omitted for brevity)
    # Ensure the corrected HTML comment `<!-- ... -->` is used in the Cube view.
# ------------------------------------------------------------
# Vue détaillée
# ------------------------------------------------------------
    if view_mode == "Liste détaillée":
        for entry in articles_to_display:
# ... (dans la boucle for entry in articles_to_display:) ...

            title      = entry.get("title",    "Titre non disponible")
            link       = entry.get("link",     "#")
            published  = entry.get("published","Date inconnue")
            # --- MODIFICATION ICI : Essayer 'content' d'abord ---
            raw_content_list = entry.get("content")
            if raw_content_list and isinstance(raw_content_list, list) and len(raw_content_list) > 0:
                # feedparser met souvent le contenu dans une liste, on prend le premier élément
                # et on essaie d'accéder à sa valeur (.value)
                raw_text_source = raw_content_list[0].get("value", "")
                # Si .value est vide, on prend le summary comme fallback
                if not raw_text_source:
                    raw_text_source = entry.get("summary", "")
            else:
                # Si pas de 'content', on utilise 'summary'
                raw_text_source = entry.get("summary", "")
            # --- FIN MODIFICATION ---

            feed_url   = entry.get("feed_url", "")
            image_url  = entry.get("image_url")

            # --- favicon / domaine ---
            favicon_url, _, domain = flux_metadata.get(
                feed_url, ("", "", "Source inconnue")
            )

            # ---------- nettoyage du résumé/contenu ----------
            # 1) D'ABORD, déséchapper les entités HTML potentielles (ex: &lt;p&gt; -> <p>)
            unescaped_text = html.unescape(str(raw_text_source)) # Utiliser raw_text_source

            # 2) ENSUITE, analyser le HTML (maintenant correct) et extraire le texte brut
            soup = BeautifulSoup(unescaped_text, "html.parser")
            clean = soup.get_text(" ", strip=True)

            # 3) Effectuer les nettoyages spécifiques (comme pour Reddit)
            # (On peut affiner ça plus tard si nécessaire)
            if domain == "news.ycombinator.com" and clean.strip() == "Comments":
                 clean = "[Lien vers les commentaires HN]" # Remplacer "Comments" seul par qqc de plus clair
            elif "reddit.com" in domain: # Garder le nettoyage spécifique Reddit si besoin
                clean = re.sub(r"submitted by /u/.*", "", clean, flags=re.I).strip()
                clean = re.sub(r"\[comments\]", "", clean, flags=re.I).strip() # Enlever aussi [comments]

            # --- Affichage ---
            #clean_escaped = html.escape(clean) if clean else 'Aucun résumé disponible.'
            clean_text = clean

            html_fragment = f"""
            <div class="rss-article-detailed">
            <h3>
                <a href="{link}" target="_blank">{html.escape(title)}</a>
            </h3>
            <p class="rss-article-meta">
                📅 {published} |
                <img src="{favicon_url}" class="rss-favicon" alt=""> {domain}
            </p>
            {f"<img src='{image_url}' alt='Article image' style='max-width: 300px; height: auto; margin-bottom: 0.5rem; border-radius: 4px;'><br>" if image_url else ""}
            <p class="rss-article-summary">
                {clean_text}
            </p>
            </div>
            """
            st.markdown(html_fragment, unsafe_allow_html=True)

    elif view_mode == "Liste raccourcie":
        for entry in articles_to_display:
            title = entry.get("title", "Titre non disponible")
            link = entry.get("link", "#")
            published = entry.get("published", "Date inconnue")
            feed_url = entry.get("feed_url", "")
            favicon_url, _, domain = flux_metadata.get(feed_url, ("", "", "Source inconnue"))

            st.markdown(
                f"<div class='rss-compact-item'>"
                f"<img src='{favicon_url}' class='rss-favicon' alt='favicon'> "
                f"<strong><a href='{link}' target='_blank'>{title}</a></strong> - "
                f"<em>{published} ({domain})</em>"
                f"</div>",
                unsafe_allow_html=True
            )
        st.markdown("<hr style='margin-top: 1rem;'>", unsafe_allow_html=True) # Add separator at the end

    elif view_mode == "Vue en cubes":
        num_cols = 4 # Fixed number of columns for cube view
        total_articles = len(articles_to_display)
        rows = (total_articles + num_cols - 1) // num_cols

        for i in range(rows):
            cols = st.columns(num_cols, gap="medium") # Use medium gap
            for j in range(num_cols):
                index = i * num_cols + j
                if index < total_articles:
                    entry = articles_to_display[index]
                    with cols[j]:
                        title = entry.get("title", "Titre non disponible")
                        link = entry.get("link", "#")
                        published = entry.get("published", "Date inconnue")
                        feed_url = entry.get("feed_url", "")
                        image_url = entry.get("image_url")

                        favicon_url, dominant_color, domain = flux_metadata.get(feed_url, ("", "#444444", "Source inconnue"))

                        # --- Image/Placeholder ---
                        image_html = ""
                        if image_url:
                            image_html = f"<img src='{image_url}' class='cube-image' alt='Article image'>"
                        else:
                            # Use dominant color and favicon as placeholder
                            image_html = (f"<div style='background-color:{dominant_color}; display:flex; align-items:center; justify-content:center; width:100%; height:100%; border-radius: 3px;'>"
                                          f"<img src='{favicon_url}' class='cube-favicon-placeholder' alt='favicon'></div>")

                        # --- Cube HTML --- Corrected Comment
                        article_html = f"""
                        <div class="rss-cube">
                            <div> <!-- Top part for title and image -->
                                <h4><a href="{link}" target="_blank" title="{title}">{title}</a></h4>
                                <div class="cube-image-container">
                                    {image_html}
                                </div>
                            </div>
                            <div class="rss-cube-meta"> <!-- Bottom part for meta -->
                                <p>📅 {published}</p>
                                <p><img src="{favicon_url}" alt="favicon"> {domain}</p>
                                <hr>
                            </div>
                        </div>
                        """
                        st.markdown(article_html, unsafe_allow_html=True)
    # ——— Bouton “Charger plus” avec on_click ———
    def _load_more():
        st.session_state["rss_offset"] += PAGE_SIZE

    load_more_key = f"load_more_rss_{st.session_state.get('view_mode','')}"
    if len(articles_to_display) < len(articles):
        st.button(
            "🔽 Charger plus d’articles",
            key=load_more_key,
            on_click=_load_more
        )



def display_nodes_page():
    """Displays the Nodes page (D3 visualization)."""
    st.title("🔗 Nodes - Occurrences de Mots-clés dans les Flux")
    st.markdown("""
        Visualisez où un mot-clé spécifique apparaît dans les titres ou résumés de vos flux RSS.
        Le graphe montre les flux comme des nœuds centraux (avec favicon) et les articles correspondants comme des nœuds liés.
    """)

    feeds = load_feeds()
    if not feeds:
        st.warning("Aucun flux disponible. Ajoutez-en dans le 'Gestionnaire de flux'.")
        return

    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        keyword = st.text_input("Rechercher un mot-clé", key="nodes_keyword", placeholder="Ex: intelligence artificielle").lower()
    with col_cfg2:
        link_distance = st.slider("Distance Lien", min_value=50, max_value=500, value=150, step=10, key="nodes_link_dist")
    with col_cfg3:
        feed_repulsion = st.slider("Répulsion Flux", min_value=-2000, max_value=-100, value=-800, step=50, key="nodes_feed_repulsion")

    if not keyword:
        st.info("Veuillez entrer un mot-clé pour lancer la recherche.")
        st.stop()

    nodes_data = []
    links_data = []
    occurrence_counter = 0

    with st.spinner(f"Recherche de '{keyword}' dans {len(feeds)} flux..."):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        feed_results = loop.run_until_complete(fetch_all_feeds_async([feed["url"] for feed in feeds]))

        for i, result in enumerate(feed_results):
            feed_url = result.get("feed_url")
            if not feed_url or result.get("error"):
                logging.warning(f"Skipping feed {feed_url or 'Unknown'} in nodes view due to fetch error: {result.get('error')}")
                continue

            parsed = urlparse(feed_url)
            domain = parsed.netloc.replace("www.", "") if parsed.netloc else f"Source_{i}"
            favicon_url = f"https://www.google.com/s2/favicons?sz=64&domain_url={parsed.netloc}" if parsed.netloc else ""
            feed_node_id = f"feed_{i}"
            feed_occurrences = []

            for article in result.get("articles", []):
                title_l = str(article.get("title", "")).lower()
                summary_l = str(article.get("summary", "")).lower()

                if keyword in title_l or keyword in summary_l:
                    occ_node_id = f"occurrence_{occurrence_counter}"
                    occurrence_counter += 1
                    full_title = article.get("title", "Sans titre")
                    display_label = (full_title[:50] + "...") if len(full_title) > 50 else full_title

                    nodes_data.append({
                        "id": occ_node_id, "type": "occurrence", "label": display_label,
                        "full_label": full_title, "url": article.get("link", "#"), "feed_id": feed_node_id
                    })
                    links_data.append({"source": occ_node_id, "target": feed_node_id}) # Let D3 handle distance or set explicitly
                    feed_occurrences.append(occ_node_id)

            if feed_occurrences:
                 nodes_data.append({
                     "id": feed_node_id, "type": "feed", "label": domain,
                     "favicon": favicon_url, "url": feed_url, "occurrence_count": len(feed_occurrences)
                 })

    if not occurrence_counter:
        st.info(f"Aucune occurrence trouvée pour '{keyword}'.")
        return # Stop if nothing found

    st.success(f"Trouvé {occurrence_counter} occurrence(s) pour '{keyword}'. Affichage du graphe...")

    # Prepare data for JS (ensure it's valid JSON)
    try:
        network_data = {"nodes": nodes_data, "links": links_data}
        # Use Python variables directly in the f-string for simple values
        js_link_distance = link_distance
        js_feed_repulsion = feed_repulsion
        # Dump the complex data structure as JSON
        network_json_str = json.dumps(network_data)
        # Check if data is empty before rendering
        if not nodes_data:
             st.warning("Aucun nœud à afficher dans le graphe après traitement.")
             return

    except Exception as json_err:
         logging.error(f"Error preparing JSON for Nodes graph: {json_err}", exc_info=True)
         st.error("Erreur interne lors de la préparation des données du graphe.")
         return


    # --- D3.js HTML (Simplified & Corrected f-string escaping) ---
    d3_nodes_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>Keyword Occurrences Graph</title>
      <style>
        body {{
margin: 0;
background-color: #1E1E1E;
overflow: hidden;
border: 1px solid #888;         /* ← bordure grise fine */
box-sizing: border-box;         /* ← pour inclure la bordure dans le calcul de taille */
}}
        svg {{ display: block; width: 100%; height: 700px; background-color: #1E1E1E; cursor: grab; }}
        svg:active {{ cursor: grabbing; }}
        .links line {{ stroke: #555; stroke-opacity: 0.7; stroke-width: 1px; }}
        .node {{ cursor: pointer; transition: opacity 0.2s ease; }}
        .node-feed circle {{ fill: transparent; stroke: #777; stroke-width: 1px; stroke-dasharray: 2,2; }}
        .node-feed image {{ pointer-events: none; }}
        .node-occurrence circle {{ fill: #FF8C00; stroke: #FFB86C; stroke-width: 1px; transition: fill 0.2s ease; }}
        .node:hover .node-occurrence circle {{ fill: #FFA500; }}
        .labels text {{ font-family: "Segoe UI", Roboto, sans-serif; fill: #E0E0E0; pointer-events: none; text-anchor: middle; font-size: 10px; paint-order: stroke; stroke: #1E1E1E; stroke-width: 3px; }}
        .label-feed {{ font-size: 12px; font-weight: bold; }}
        .label-occurrence {{ font-size: 9px; }}
        .tooltip {{ position: absolute; background-color: rgba(40, 40, 40, 0.9); color: #E0E0E0; padding: 8px 12px; border-radius: 4px; font-size: 12px; pointer-events: none; opacity: 0; transition: opacity 0.2s ease; max-width: 300px; white-space: normal; z-index: 10; }}
      </style>
    </head>
    <body>
      <div id="tooltip" class="tooltip"></div>
      <svg id="graph-nodes"></svg>
      <script src="https://d3js.org/d3.v7.min.js"></script>
      <script>
        const graphData = {network_json_str}; // Inject JSON string directly
        const nodes = graphData.nodes || [];
        const links = graphData.links || [];
        const linkDistance = {js_link_distance};
        const feedRepulsion = {js_feed_repulsion};

        console.log("Nodes data received:", nodes); // Browser console log
        console.log("Links data received:", links); // Browser console log

        const container = document.getElementById('graph-nodes');
        const tooltip = d3.select("#tooltip");
        const width = container.clientWidth;
        const height = container.clientHeight;

        if (width <= 0 || height <= 0 || nodes.length === 0) {{
            console.error("Graph container has no dimensions or no nodes to render.");
            container.innerHTML = "<p style='color:#E0E0E0; text-align:center; padding-top: 50px;'>Impossible d'afficher le graphe (pas de données ou problème de dimensions).</p>";
        }} else {{
            const svg = d3.select("#graph-nodes")
                .attr("viewBox", [-width / 2, -height / 2, width, height]);

            const simulation = d3.forceSimulation(nodes)
                .force("link", d3.forceLink(links).id(d => d.id).distance(linkDistance).strength(0.1))
                .force("charge", d3.forceManyBody().strength(d => d.type === 'feed' ? feedRepulsion : -50))
                .force("collide", d3.forceCollide().radius(d => (d.type === 'feed' ? 20 + (d.occurrence_count || 0) * 1.5 : 8) + 5 )) // Radius + padding
                .force("center", d3.forceCenter(0, 0))
                .on("tick", ticked);

            const zoom = d3.zoom().scaleExtent([0.1, 5]).on("zoom", (event) => g.attr("transform", event.transform));
            svg.call(zoom);

            const g = svg.append("g");

            const link = g.append("g").attr("class", "links")
              .selectAll("line").data(links).join("line");

            const node = g.append("g").attr("class", "nodes")
              .selectAll("g").data(nodes).join("g")
              .attr("class", d => `node node-${{d.type}}`) // Use JS template literal here
              .call(drag(simulation))
              .on("click", nodeClicked)
              .on("mouseover", nodeMouseover)
              .on("mouseout", nodeMouseout);

            const feedNodes = node.filter(d => d.type === 'feed');
            feedNodes.append("circle").attr("r", d => 20 + (d.occurrence_count || 0) * 1.5);
            feedNodes.append("image")
               .attr("xlink:href", d => d.favicon)
               .attr("width", d => 32 + (d.occurrence_count || 0) * 1)
               .attr("height", d => 32 + (d.occurrence_count || 0) * 1)
               .attr("x", d => -(16 + (d.occurrence_count || 0) * 0.5))
               .attr("y", d => -(16 + (d.occurrence_count || 0) * 0.5));

            node.filter(d => d.type === 'occurrence').append("circle").attr("r", 6);

            const label = g.append("g").attr("class", "labels")
               .selectAll("text").data(nodes).join("text")
               .attr("dy", d => d.type === 'feed' ? (30 + (d.occurrence_count || 0) * 1.5) : 15)
               .attr("class", d => `label-${{d.type}}`) // Use JS template literal
               .text(d => d.label);

            function ticked() {{
              link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
                  .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
              node.attr("transform", d => `translate(${{d.x}}, ${{d.y}})`); // Use JS template literal
              label.attr("x", d => d.x).attr("y", d => d.y);
            }}

            function nodeClicked(event, d) {{
              event.stopPropagation();
              if (d.url && d.url !== '#') window.open(d.url, '_blank', 'noopener,noreferrer');
            }}

            function nodeMouseover(event, d) {{
                tooltip.transition().duration(200).style("opacity", 0.9);
                let ttText = d.type === 'feed' ? `Flux: ${{d.label}}<br>Occurrences: ${{d.occurrence_count}}` : `Article: ${{d.full_label}}`; // JS template literals
                tooltip.html(ttText).style("left", (event.pageX + 10) + "px").style("top", (event.pageY - 15) + "px");
                // Optional highlighting can go here
            }}

            function nodeMouseout() {{
                tooltip.transition().duration(500).style("opacity", 0);
                 // Optional restore highlighting
            }}

            function drag(simulation) {{
              function dragstarted(event, d) {{ if (!event.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }}
              function dragged(event, d) {{ d.fx = event.x; d.fy = event.y; }}
              function dragended(event, d) {{ if (!event.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }}
              return d3.drag().on("start", dragstarted).on("drag", dragged).on("end", dragended);
            }}
        }} // End of else block for rendering
      </script>
    </body>
    </html>
    """
    # Render the HTML component
    components.html(d3_nodes_html, height=710, scrolling=False)


def display_bar_chart_page():
    """Displays the Dashboard page with various charts."""
    st.title("📊 Dashboard - Analyse des Flux RSS")
    st.markdown("Explorez les données de vos flux RSS via différents graphiques.")

    feeds = load_feeds()
    if not feeds:
        st.warning("Aucun flux disponible. Ajoutez-en dans le 'Gestionnaire de flux'.")
        return

    rss_options = {urlparse(feed["url"]).netloc.replace("www.", ""): feed["url"]
                   for feed in feeds if isinstance(feed, dict) and "url" in feed and urlparse(feed["url"]).netloc}

    if not rss_options:
        st.error("Impossible d'extraire des noms de domaine valides des flux enregistrés.")
        return

    st.subheader("Configuration du Dashboard")
    selected_flux_names = st.multiselect(
        "Choisissez les flux à analyser",
        options=list(rss_options.keys()),
        default=list(rss_options.keys()),
        key="dashboard_flux_selector"
    )

    if not selected_flux_names:
        st.info("Veuillez sélectionner au moins un flux.")
        return

    chart_types = ["Activité par Flux (Barres)", "Activité Temporelle (Ligne)", "Timeline des Posts", "Répartition (Treemap)", "Activité Temporelle (Aire)", "Nuage de Mots (Titres)"]
    selected_chart = st.selectbox("Choisissez un type de graphique", chart_types, key="dashboard_chart_selector")

    # --- Data Fetching for Dashboard ---
    all_df_list = []
    fetch_status_placeholder = st.empty()
    fetch_status_placeholder.info(f"Chargement des données pour {len(selected_flux_names)} flux...")
    fetch_errors_dashboard = []

    for flux_name in selected_flux_names:
        rss_url = rss_options[flux_name]
        df_flux = fetch_dashboard_data(rss_url, flux_name) # Returns DataFrame or None
        if df_flux is not None: # Check for None specifically (indicates fetch error)
            if not df_flux.empty:
                all_df_list.append(df_flux)
        else:
            # Error message already shown by fetch_dashboard_data
            fetch_errors_dashboard.append(flux_name)

    fetch_status_placeholder.empty()
    if fetch_errors_dashboard:
         st.warning(f"Échec du chargement des données pour les flux: {', '.join(fetch_errors_dashboard)}")

    if not all_df_list:
        st.error("Aucune donnée d'article n'a pu être chargée pour les flux sélectionnés.")
        return

    df_merged = pd.concat(all_df_list, ignore_index=True)
    st.caption(f"Analyse basée sur {len(df_merged)} articles chargés depuis les flux sélectionnés.")
    st.markdown("---")
    st.subheader(f"Graphique : {selected_chart}")

    # --- Chart Rendering ---
    try:
        if selected_chart == "Activité par Flux (Barres)":
            if 'Flux' not in df_merged.columns: raise ValueError("Colonne 'Flux' manquante.")
            df_grouped = df_merged.groupby('Flux').size().reset_index(name='Nombre de Posts')
            if df_grouped.empty: raise ValueError("Aucune donnée à afficher après groupement.")
            fig = px.bar(df_grouped, x='Flux', y='Nombre de Posts', color='Flux', title="Nombre Total de Posts par Flux")
            fig.update_layout(xaxis_title=None, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        elif selected_chart == "Activité Temporelle (Ligne)":
            if 'Timestamp' not in df_merged.columns: raise ValueError("Colonne 'Timestamp' manquante.")
            df_line = df_merged.dropna(subset=['Timestamp', 'Flux']).copy() 
            df_line['Timestamp'] = (
            pd.to_datetime(df_line['Timestamp'], errors='coerce', utc=True)
            .dt.tz_convert(None)
            )
            df_line['Date'] = pd.to_datetime(df_line['Timestamp']).dt.date
            df_grouped = df_line.groupby(['Date', 'Flux']).size().reset_index(name='Nombre de Posts')
            if df_grouped.empty: raise ValueError("Aucune donnée après groupement par date/flux.")
            fig = px.line(df_grouped, x='Date', y='Nombre de Posts', color='Flux', title="Évolution du Nombre de Posts par Jour", markers=True)
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

        elif selected_chart == "Timeline des Posts":
            required_cols = ['Timestamp', 'Lien', 'Titre', 'Flux']
            if not all(col in df_merged.columns for col in required_cols):
                raise ValueError(f"Colonnes manquantes pour la timeline: {', '.join(required_cols)}")

            df_timeline = df_merged.dropna(subset=['Timestamp', 'Flux']).copy()
            if df_timeline.empty: raise ValueError("Aucune donnée valide pour la timeline.")

            items = []
            flux_colors = {flux: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                           for i, flux in enumerate(df_timeline['Flux'].unique())}

            for idx, row in df_timeline.iterrows():
                bg_color = flux_colors.get(row['Flux'], '#555')
                r, g, b = tuple(int(bg_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
                text_color = "#000" if luminance > 0.5 else "#FFF"
                link = row.get('Lien', '#')
                title = str(row.get('Titre', 'Sans Titre'))
                title_short = title[:30] + '...' if len(title) > 30 else title

                post_content = f"<a href='{link}' target='_blank' style='color: {text_color}; text-decoration: none; font-weight: bold;'>{title_short}</a>"

                items.append({
                    "id": idx, "content": post_content,
                    "start": pd.to_datetime(row['Timestamp']).isoformat(),
                    "style": f"background-color: {bg_color}; color: {text_color}; border-radius: 3px; font-size: 11px; padding: 2px 4px;",
                    "title": f"{title}\nFlux: {row['Flux']}\nDate: {pd.to_datetime(row['Timestamp']).strftime('%Y-%m-%d %H:%M')}",
                    "group": row['Flux']
                })

            if not items: raise ValueError("Aucun élément à afficher dans la timeline.")

            groups = [{"id": flux, "content": flux} for flux in df_timeline['Flux'].unique()]
            timeline_options = { "stack": False, "verticalScroll": True, "zoomKey": "ctrlKey", "maxHeight": "600px", "groupOrder": "content", "tooltip": {"followMouse": True}, "orientation": {"axis": "top"}, "timeAxis": {"scale": "day", "step": 1}, "zoomMin": 86400000, "zoomMax": 63072000000 } # 1 day to 2 years in ms

            st.info("Utilisez la molette (ou Ctrl+Molette) pour zoomer, et glissez pour naviguer.")
            st_timeline(items, groups=groups, options=timeline_options, height="600px")

        elif selected_chart == "Répartition (Treemap)":
            required_cols = ['Flux', 'Catégorie', 'Titre']
            if not all(col in df_merged.columns for col in required_cols):
                raise ValueError(f"Colonnes manquantes pour le treemap: {', '.join(required_cols)}")

            df_treemap = df_merged.copy()
            df_treemap['Flux'] = df_treemap['Flux'].fillna("Flux Inconnu").astype(str)
            df_treemap['Catégorie'] = df_treemap['Catégorie'].fillna("Sans Catégorie").astype(str)
            df_treemap['Titre'] = df_treemap['Titre'].fillna("Sans Titre").astype(str)
            df_treemap['value'] = 1

            if df_treemap.empty: raise ValueError("Aucune donnée valide pour le treemap.")

            fig = px.treemap(
                df_treemap, path=[px.Constant("Tous"), 'Flux', 'Catégorie', 'Titre'],
                values='value', title="Répartition des Posts par Flux et Catégorie",
                color='Flux',
                hover_data={'Lien': True},   # <-- on enlève 'Titre' ici
                custom_data=['Lien']
            )

            fig.update_traces(textinfo="label+percent root")
            fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
            st.plotly_chart(fig, use_container_width=True)

        elif selected_chart == "Activité Temporelle (Aire)":
            if 'Timestamp' not in df_merged.columns: raise ValueError("Colonne 'Timestamp' manquante.")
            df_area_src = df_merged.dropna(subset=['Timestamp', 'Flux']).copy()
            df_area_src['Timestamp'] = (
                pd.to_datetime(df_area_src['Timestamp'], errors='coerce', utc=True)
                .dt.tz_convert(None)
            )
            df_area_src['Date'] = pd.to_datetime(df_area_src['Timestamp']).dt.date
            df_area = df_area_src.groupby(['Date', 'Flux']).size().unstack(fill_value=0).sort_index()
            if df_area.empty: raise ValueError("Aucune donnée après groupement pour le graphique Aire.")
            df_area.index = pd.to_datetime(df_area.index)

            echarts_options = {
                "backgroundColor": "#252526",
                "title": {"text": "Activité Cumulée des Posts par Flux", "left": "center", "textStyle": {"color": '#E0E0E0'}},
                "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross", "label": {"backgroundColor": '#6a7985'}}},
                "legend": {"data": list(df_area.columns), "top": "bottom", "textStyle": {"color": '#E0E0E0'}},
                "grid": {"left": '3%', "right": '4%', "bottom": '10%', "containLabel": True},
                "xAxis": [{"type": "category", "boundaryGap": False, "data": df_area.index.strftime('%Y-%m-%d').tolist(), "axisLabel": {"color": '#E0E0E0'}}],
                "yAxis": [{"type": "value", "axisLabel": {"color": '#E0E0E0'}}],
                "series": [ { "name": flux, "type": "line", "stack": "Total", "areaStyle": {}, "emphasis": {"focus": "series"}, "data": df_area[flux].tolist(), "smooth": True } for flux in df_area.columns ]
            }
            st_echarts(options=echarts_options, height="500px")

        elif selected_chart == "Nuage de Mots (Titres)":
            if 'Titre' not in df_merged.columns: raise ValueError("Colonne 'Titre' manquante.")
            text = ' '.join(df_merged['Titre'].dropna().astype(str))
            if not text: raise ValueError("Aucun texte de titre disponible pour le nuage de mots.")

            text = text.lower()
            text = ''.join(char for char in text if char.isalnum() or char.isspace())
            # Extend stop words if needed
            stop_words = set(["le", "la", "les", "un", "une", "des", "du", "de", "à", "au", "aux", "et", "ou", "mais", "donc", "or", "ni", "car", "pour", "par", "sur", "en", "dans", "avec", "sans", "sous", "vers", "chez", "ce", "cet", "cette", "ces", "mon", "ton", "son", "ma", "ta", "sa", "mes", "tes", "ses", "notre", "votre", "leur", "nos", "vos", "leurs", "je", "tu", "il", "elle", "nous", "vous", "ils", "elles", "se", "qui", "que", "quoi", "dont", "où", "est", "sont", "ai", "as", "a", "avons", "avez", "ont", "suis", "es", "été", "était", "plus", "moins", "comme", "faire", "fait", "dit", "va", "peut", "très", "tout", "tous", "pas", "ne", "non", "oui", "si", "ça", "cela", "celui", "celle", "ceux", "comment", "quand", "depuis", "aussi", "the", "a", "an", "is", "are", "and", "or", "but", "if", "of", "to", "in", "it", "that", "this", "for", "on", "with", "as", "by", "at", "from", "its", "it's", "be", "was", "were", "will", "can", "not", "no", "yes", "so", "also", "more", "has", "had", "have", "about", "after", "all", "new", "how", "when", "where", "which", "who", "why"])

            wordcloud = WordCloud(width=800, height=400, background_color='#1E1E1E', stopwords=stop_words, colormap='viridis', max_words=150, min_font_size=10, prefer_horizontal=0.9).generate(text)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wordcloud, interpolation='bilinear'); ax.axis('off'); plt.tight_layout(pad=0)
            st.pyplot(fig)

    except ValueError as ve: # Catch data validation errors specifically
         logging.warning(f"Data validation error for chart '{selected_chart}': {ve}")
         st.warning(f"Impossible de générer le graphique '{selected_chart}'. Raison: {ve}")
    except Exception as chart_error:
         logging.error(f"Error generating dashboard chart '{selected_chart}': {chart_error}", exc_info=True)
         st.error(f"Une erreur inattendue est survenue lors de la création du graphique '{selected_chart}'.")


# === NER Page (Entities & Relations) ===

def display_entity_relation_graph_page():
    st.title("💡 Entities & Relations (Extraction et Graphe)")
    st.markdown("Analyse les articles **enregistrés** pour extraire ...")
    
    API_KEY = st.session_state.get("google_api_key", "")
    # --- Helper : enregistrer l’état NER ---
    def save_ner_data() -> None:
        """Sauvegarde l'état NER sur disque."""
        user = st.session_state.get("user")
        if not user:                       # sécurité
            return
        user_dir = os.path.join(USER_DATA_BASE_DIR, user)   # <‑‑ calcule ici
        os.makedirs(user_dir, exist_ok=True)
        ner_file = os.path.join(user_dir, "ner_results.json")

        data = {
            "entities": st.session_state.get("all_entities", {}),
            "relations": st.session_state.get("all_relations", []),
            "processed_post_ids": list(st.session_state.get("processed_post_ids", set())),
        }
        try:
            with open(ner_file, "w", encoding="utf‑8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logging.info("NER data saved.")
        except Exception as e:
            logging.error(f"Save NER failed: {e}", exc_info=True)
            st.error(f"Erreur lors de la sauvegarde NER : {e}")

    # ——— Boutons NER en ligne ———
    c1, c2, c3 = st.columns([1, 1, 1], gap="medium")
    status_placeholder = st.empty()
    progress_bar       = st.progress(0.0)
    progress_text      = st.empty()


    reset_pressed = c1.button("🔄 Réinitialiser NER",
                            key="main_reset_ner",
                            use_container_width=True)

    start_pressed = c2.button("▶️ Démarrer / Reprendre",
                            key="main_start_processing",
                            disabled=not API_KEY or st.session_state.get("processing_active", False),
                            use_container_width=True)

    stop_pressed  = c3.button("⏹️ Arrêter Traitement",
                            key="main_stop_processing",
                            disabled=not st.session_state.get("processing_active", False),
                            use_container_width=True)

    # ——— Gestion des événements ———
    if reset_pressed:
        # ré‑initialise proprement l’état
        st.session_state['all_entities']       = {}
        st.session_state['all_relations']      = []
        st.session_state['processed_post_ids'] = set()
        st.session_state['results_by_post']    = {}
        st.session_state['processing_active']  = False
        save_ner_data()
        st.success("🔄 Données NER réinitialisées.")
        st.rerun()

    if start_pressed:
        st.session_state['processing_active'] = True
        st.success("▶️ Traitement NER lancé.")
        st.rerun()

    if stop_pressed:
        st.session_state['processing_active'] = False
        st.info("⏹️ Traitement NER interrompu.")


    API_KEY = st.text_input(
        "Entrez votre clé API Google Gemini",
        type="password",
        placeholder="sk-XXXX…",
        key="api_input"
    )
    validate = st.button("Valider la clé", key="api_validate")


# le reste de ta logique…



    # Au clic sur Valider, on teste la clé auprès de l'API Gemini
    if validate:
        if not API_KEY:
            st.error("Veuillez saisir une clé API avant de valider.")
        else:
            try:
                test_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash?key={API_KEY}"
                resp = requests.get(test_url, timeout=10)
                if resp.status_code == 200:
                    st.success("✅ Clé API valide !")
                    # On conserve la clé validée en session
                    st.session_state["google_api_key"] = API_KEY
                    st.session_state["api_key_valid"] = True
                else:
                    err = resp.json().get("error", {}).get("message", resp.text)
                    st.error(f"❌ Clé invalide : {err}")
            except Exception as e:
                st.error(f"❌ Erreur lors de la validation : {e}")

    # Tant que la clé n'est pas validée, on bloque la suite
    if not st.session_state.get("api_key_valid", False):
        st.warning("🛑 Merci de valider la clé API pour démarrer l'analyse.")
        return

    # Plus bas dans votre code, utilisez désormais :
    API_KEY = st.session_state["google_api_key"]


    # --- Setup and Initialization ---
    if 'user' not in st.session_state:
        st.error("Utilisateur non connecté."); return
    user = st.session_state.user
    USER_DATA_DIR = os.path.join(USER_DATA_BASE_DIR, user) # Use base dir
    NER_FILE = os.path.join(USER_DATA_DIR, "ner_results.json")

    try:
        os.makedirs(USER_DATA_DIR, exist_ok=True)
    except OSError as e:
        st.error(f"Impossible de créer le répertoire utilisateur {USER_DATA_DIR}: {e}"); return


    # --- State Initialization ---
    def load_ner_data():
        # (Function remains the same as previous version - loads from NER_FILE)
        if os.path.exists(NER_FILE):
            try:
                with open(NER_FILE, "r", encoding="utf-8") as f: data = json.load(f)
                entities = data.get("entities", {})
                relations = data.get("relations", [])
                processed_ids = set(data.get("processed_post_ids", []))
                if not isinstance(entities, dict): entities = {}
                if not isinstance(relations, list): relations = []
                # Ensure processed IDs are strings
                processed_ids = {str(pid) for pid in processed_ids}

                logging.info(f"Loaded NER data from {NER_FILE} for user {user}")
                return entities, relations, processed_ids
            except (json.JSONDecodeError, TypeError) as e:
                st.error(f"Erreur chargement/décodage NER ({NER_FILE}). Réinitialisation. Erreur: {e}")
                return {}, [], set()
            except Exception as e:
                st.error(f"Erreur inconnue chargement NER: {e}")
                return {}, [], set()
        return {}, [], set()

    if 'ner_data_loaded' not in st.session_state:
        # (State loading logic remains same, reconstructing results_by_post)
        st.session_state['all_entities'], st.session_state['all_relations'], st.session_state['processed_post_ids'] = load_ner_data()
        st.session_state['ner_data_loaded'] = True
        st.session_state['results_by_post'] = {}
        st.session_state['next_article_index'] = 0 # Not really used with current batch logic
        st.session_state['processing_active'] = False
        # Reconstruct results_by_post from loaded relations for display
        for rel in st.session_state['all_relations']:
             post_id = str(rel.get("post_id"))
             if post_id:
                 if post_id not in st.session_state['results_by_post']:
                     st.session_state['results_by_post'][post_id] = {"entities": set(), "relations": []}
                 e1 = rel.get("entity1", {})
                 e2 = rel.get("entity2", {})
                 if e1.get("entity") and e1.get("type"):
                     st.session_state['results_by_post'][post_id]["entities"].add((e1["entity"], e1["type"]))
                 if e2.get("entity") and e2.get("type"):
                      st.session_state['results_by_post'][post_id]["entities"].add((e2["entity"], e2["type"]))
                 st.session_state['results_by_post'][post_id]["relations"].append(rel)


    # --- Load Articles from DB ---
    all_articles_tuples = load_articles_from_db() # List of (id, summary)
    st.markdown(f"**Articles chargés pour l'analyse : {len(all_articles_tuples)}**")


    if not all_articles_tuples:
        st.warning("Aucun article trouvé dans votre base de données locale. Enregistrez des articles via le 'Lecteur RSS' avant de lancer l'analyse.")
        if not st.session_state.get('all_entities') and not st.session_state.get('all_relations'):
             st.info("Le graphe des entités et relations apparaîtra ici après traitement.")
             return
        # Fall through to render existing graph data

    # --- Helper Functions (NER Specific - save_ner_data, analyze_text_gemini, extract_structured_result, generate_color) ---
    # (These functions remain the same as the previous version - omitted for brevity)
    def save_ner_data():
        """Saves current NER state (entities, relations, processed IDs) to JSON."""
        if 'user' not in st.session_state: return # Safety check
        ner_data_to_save = {
            "entities": st.session_state.get('all_entities', {}),
            "relations": st.session_state.get('all_relations', []),
            "processed_post_ids": list(st.session_state.get('processed_post_ids', set()))
        }
        try:
            # Ensure directory exists one last time
            os.makedirs(os.path.dirname(NER_FILE), exist_ok=True)
            with open(NER_FILE, "w", encoding="utf-8") as f:
                json.dump(ner_data_to_save, f, ensure_ascii=False, indent=2)
            logging.info(f"NER data saved to {NER_FILE} for user {st.session_state.user}.")
        except Exception as e:
            logging.error(f"Error saving NER data to {NER_FILE}: {e}", exc_info=True)
            st.error(f"Erreur lors de la sauvegarde des données NER : {e}")

    # Replace the entire analyze_text_gemini function with this:

    def analyze_text_gemini(text_batch: str, prompt_template_str: str) -> dict | None:
        """Analyzes text using the Gemini API."""
        if not API_KEY:
            st.error("Clé API Google non configurée. Analyse impossible.")
            return None
        if not text_batch or text_batch.isspace():
            logging.warning("Skipping API call for empty text batch.")
            # Return None or an empty dict structure consistent with expected success format
            # Returning None might be better to signal no result vs. empty result.
            return None

        # *** FIX: Construct the final prompt string FIRST ***
        try:
            final_prompt = prompt_template_str.format(text=text_batch)
        except KeyError as e:
            logging.error(f"KeyError during prompt formatting. Likely stray braces in template. Error: {e}", exc_info=True)
            st.error(f"Erreur interne lors de la préparation de la requête API (KeyError: {e}). Vérifiez le 'prompt_template'.")
            return None
        except Exception as e:
            logging.error(f"Unexpected error during prompt formatting: {e}", exc_info=True)
            st.error(f"Erreur inattendue lors de la préparation de la requête API: {e}")
            return None

        # Log the beginning of the prompt being sent
        logging.info(f"Sending {len(final_prompt)} chars to Gemini API. Prompt start: '{final_prompt[:200]}...'")

        # *** FIX: Use the already formatted 'final_prompt' string ***
        data = {
            "contents": [{"parts": [{"text": final_prompt}]}], # Use the formatted string here
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 8192,
                # Explicitly request JSON output
                "response_mime_type": "application/json"
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
            ]
        }



        gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"

        try:
            response = requests.post(gemini_api_url, headers=HEADERS, json=data, timeout=180)
            logging.info(f"Gemini API Response Status: {response.status_code}")
            raw_response_text = response.text
            logging.info(f"Gemini API Raw Response Text (start): {raw_response_text[:500]}...") # Log raw response
            response.raise_for_status()
            # Attempt to parse JSON directly from the response
            return response.json()
        except requests.exceptions.Timeout:
            logging.error("Gemini API request timed out.")
            st.error("La requête vers l'API Gemini a expiré (Timeout).")
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Gemini API request error: {e}", exc_info=True)
            error_detail = str(e)
            try: # Try to get more detail from response if available
                if e.response is not None:
                    # Log the full error response if possible
                    logging.error(f"Gemini Error Response Content: {e.response.text}")
                    error_detail = e.response.json().get("error", {}).get("message", str(e))
            except: pass # Ignore errors trying to parse the error response
            st.error(f"Erreur communication API Gemini : {error_detail}")
            return None
        except json.JSONDecodeError as json_err: # Catch JSON parsing errors here
            logging.error(f"Failed to decode JSON from Gemini response: {json_err}. Response text: {raw_response_text[:500]}...", exc_info=True)
            st.error("Réponse API reçue mais non décodable en JSON.")
            return None
        except Exception as e:
            logging.error("Unexpected error during Gemini API call/parsing.", exc_info=True)
            st.error(f"Erreur inattendue lors de l'appel API / parsing : {e}")
            return None

    def extract_structured_result(api_response: dict | None) -> dict | None:
        """Extracts the structured JSON result from the Gemini API response."""
        if not api_response: return None
        try:
            candidates = api_response.get("candidates")
            if not candidates:
                prompt_feedback = api_response.get("promptFeedback")
                if prompt_feedback and prompt_feedback.get("blockReason"):
                     reason = prompt_feedback.get("blockReason")
                     logging.warning(f"Gemini API response blocked: {reason}")
                     st.warning(f"API a bloqué la réponse: {reason}")
                else:
                     finish_reason = candidates[0].get("finishReason") if candidates else "NO_CANDIDATES"
                     logging.warning(f"Gemini API response has no candidates or valid content. Finish Reason: {finish_reason}. Response: {str(api_response)[:500]}")
                     st.warning(f"Réponse invalide/vide de l'API (Finish Reason: {finish_reason}).")
                return None

            # Handle potential finish reasons like SAFETY or RECITATION
            finish_reason = candidates[0].get("finishReason")
            if finish_reason not in [None, "STOP", "MAX_TOKENS"]: # Allow None, Stop, MaxTokens
                 logging.warning(f"Gemini response finished unexpectedly: {finish_reason}. Content might be missing.")
                 st.warning(f"Réponse API terminée de manière inattendue ({finish_reason}).")
                 # Continue processing potential partial content if available

            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if not parts or "text" not in parts[0]:
                 logging.warning("Gemini API response part has no text.")
                 return None # Nothing to parse

            text_content = parts[0]["text"]
            cleaned_text = re.sub(r"```(?:json)?\s*|\s*```", "", text_content).strip()

            if not cleaned_text.startswith("{") or not cleaned_text.endswith("}"):
                # Try to find JSON within the text if it's not the whole string
                match = re.search(r"\{.*\}", cleaned_text, re.DOTALL)
                if match:
                    cleaned_text = match.group(0)
                    logging.warning("Extracted JSON object embedded within other text in API response.")
                else:
                    logging.error(f"API response text does not look like JSON: {cleaned_text[:200]}...")
                    st.error("Réponse API non JSON.")
                    return None

            return json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON from API response: {e}. Response text: {cleaned_text[:500]}...", exc_info=True)
            st.error(f"Échec du décodage JSON de la réponse API.")
            return None
        except Exception as e:
             logging.error("Unexpected error parsing API response.", exc_info=True)
             st.error(f"Erreur inattendue (analyse réponse API): {e}")
             return None

    @st.cache_data(show_spinner=False)
    def generate_color(entity_type: str) -> str:
        # (Function remains the same)
        seed = int(hashlib.sha1(str(entity_type).encode('utf-8')).hexdigest(), 16) % (10**6)
        random.seed(seed)
        hue = random.randint(0, 360); saturation = random.randint(60, 90); lightness = random.randint(45, 65)
        return f"hsl({hue}, {saturation}%, {lightness}%)"




    # --- NER Processing Loop ---
    if st.session_state.get('processing_active') and API_KEY:
        batch_size = 15 # Adjust batch size based on typical article length and API limits
        total_articles_in_db = len(all_articles_tuples)

        # Find articles not yet processed
        articles_to_process = [
            (str(id_), summary) # Ensure ID is string
            for id_, summary in all_articles_tuples
            if str(id_) not in st.session_state.get('processed_post_ids', set()) and summary and not summary.isspace()
        ]

        if not articles_to_process:
             status_placeholder.success("✅ Traitement NER terminé.")
             st.session_state['processing_active'] = False
             progress = 1.0
             processed_count = total_articles_in_db
        else:
            current_batch = articles_to_process[:batch_size]
            processed_count = len(st.session_state.get('processed_post_ids', set()))
            progress = min(1.0, processed_count / total_articles_in_db) if total_articles_in_db > 0 else 0.0

            status_placeholder.info(f"Traitement lot ({len(current_batch)} articles)...")
            progress_text.text(f"{processed_count}/{total_articles_in_db} articles traités ({progress:.1%})")
            progress_bar.progress(progress)

            combined_text = "\n\n".join(
                f"POST ID: {post_id}\n{summary.strip()}" for post_id, summary in current_batch
            )

            # Define prompt template (same as before)
# Inside the display_entity_relation_graph_page function, replace the prompt_template definition:

# *** REFINED PROMPT with DOUBLED braces for JSON example ***
# --- nouveau prompt_template ---
            prompt_template = r"""
            Analyze the following text snippets, each marked with 'POST ID: <ID>'.  
            Extract named entities (PERSON, ORGANIZATION, LOCATION, TECHNOLOGY, CONCEPT…)  
            and the relationships that link *entities occurring in the same POST ID*.  
            Normalize entity names to Proper Case.

            Here are the snippets to analyse (keep the order) :

            {text}

            Return ONLY one JSON object whose single key is "relations".  
            Its value MUST be a list where each item has **exactly** this shape :

            ```json
            {{
            "post_id": "<ID>",
            "entity1": {{"entity": "Normalized Entity 1", "type": "EntityType1"}},
            "entity2": {{"entity": "Normalized Entity 2", "type": "EntityType2"}},
            "relationship": "concise relation description"
            }}
            ```"""

            api_response = analyze_text_gemini(combined_text, prompt_template)
            structured_result = extract_structured_result(api_response)

            batch_relations = []
            if structured_result and isinstance(structured_result, dict) and "relations" in structured_result:
                raw_relations = structured_result.get("relations", [])
                if isinstance(raw_relations, list): batch_relations = raw_relations
                else: logging.warning(f"API response 'relations' key is not a list: {type(raw_relations)}")
            elif structured_result: logging.warning(f"API JSON response missing/invalid 'relations' key.")

            new_entities_in_batch = False
            processed_ids_in_this_batch = set()

            for rel in batch_relations:
                # (Validation and state update logic remains the same)
                if not isinstance(rel, dict): continue
                r_post_id_str = str(rel.get("post_id"))
                e1_data = rel.get("entity1", {}); e2_data = rel.get("entity2", {})
                relationship = rel.get("relationship")
                if not isinstance(e1_data, dict) or not isinstance(e2_data, dict): continue
                e1_name = e1_data.get("entity"); e1_type = e1_data.get("type")
                e2_name = e2_data.get("entity"); e2_type = e2_data.get("type")
                if not all([r_post_id_str, e1_name, e1_type, e2_name, e2_type, relationship]): continue

                e1_name = e1_name.strip(); e2_name = e2_name.strip()
                # Add entities if new
                if e1_name and e1_name not in st.session_state['all_entities']:
                    st.session_state['all_entities'][e1_name] = e1_type; new_entities_in_batch = True
                if e2_name and e2_name not in st.session_state['all_entities']:
                    st.session_state['all_entities'][e2_name] = e2_type; new_entities_in_batch = True
                # Add relation
                st.session_state['all_relations'].append(rel)
                # Update results_by_post
                if r_post_id_str not in st.session_state['results_by_post']: st.session_state['results_by_post'][r_post_id_str] = {"entities": set(), "relations": []}
                st.session_state['results_by_post'][r_post_id_str]["entities"].add((e1_name, e1_type))
                st.session_state['results_by_post'][r_post_id_str]["entities"].add((e2_name, e2_type))
                st.session_state['results_by_post'][r_post_id_str]["relations"].append(rel)
                processed_ids_in_this_batch.add(r_post_id_str) # Mark as processed if relation found for it

            # Mark all articles attempted in batch as processed, even if no relations found
            for post_id, _ in current_batch:
                 processed_ids_in_this_batch.add(post_id)
            st.session_state['processed_post_ids'].update(processed_ids_in_this_batch)

            save_ner_data() # Save after each batch

            # Schedule next run
            time.sleep(1.5) # Short delay
            st.rerun()

    # --- Render Graph and Results ---
    legend_placeholder = st.sidebar.empty()
    graph_placeholder = st.empty()

    def render_entity_graph_d3():
        """Renders the D3.js graph of entities and relations with progress bar and right-side legend."""
        # --- Préparation des données ---
        nodes_d3 = []
        links_d3 = []
        entity_types = set()
        color_map = {}

        valid_entities = st.session_state.get('all_entities', {})
        all_relations = st.session_state.get('all_relations', [])
        post_id_to_url_map = st.session_state.get('post_id_to_url', {})

        # Si pas de données, on affiche un message et on quitte
        if not valid_entities and not all_relations:
            st.info("Aucune entité ou relation extraite. Lancez le traitement NER ou vérifiez les résultats.")
            return

        # Construire les nœuds
        entity_post_map = {}
        for rel in all_relations:
            pid = str(rel.get("post_id"))
            if not pid:
                continue
            e1 = rel.get("entity1", {}).get("entity")
            e2 = rel.get("entity2", {}).get("entity")
            if e1:
                entity_post_map.setdefault(e1, set()).add(pid)
            if e2:
                entity_post_map.setdefault(e2, set()).add(pid)

        # Ajouter un nœud par entité
        for name, etype in valid_entities.items():
            nodes_d3.append({
                "id": name,
                "type": etype,
                "posts": sorted(entity_post_map.get(name, []), key=str)
            })
            entity_types.add(etype)
            if etype not in color_map:
                color_map[etype] = generate_color(etype)

        # Construire les liens
        valid_ids = {n["id"] for n in nodes_d3}
        for rel in all_relations:
            src = rel.get("entity1", {}).get("entity")
            tgt = rel.get("entity2", {}).get("entity")
            lbl = rel.get("relationship", "")
            if src in valid_ids and tgt in valid_ids:
                links_d3.append({"source": src, "target": tgt, "relationship": lbl})

        # --- Placeholders de progression (full width) ---
        status_pl = st.empty()
        prog_bar   = st.progress(0.0)
        prog_text  = st.empty()

        # Exemple de mise à jour (à adapter selon ton code)
        n = len(st.session_state.get('processed_post_ids', []))
        total = len(st.session_state.get('post_id_to_url', {}))
        if total > 0:
            fraction = n / total
            status_pl.info(f"{n}/{total} articles traités")
            prog_bar.progress(fraction)
            prog_text.text(f"{fraction:.1%}")

        # --- Colonnes pour graphe et légende ---


        # --- Prepare Graph Data & Render ---
        if not nodes_d3:
             graph_placeholder.info("Aucun nœud à afficher dans le graphe.")
             return

        try:
            postIdToTitle = {
                str(a_id): title
                for a_id, title in get_db_connection().execute(
                    "SELECT id, title FROM articles WHERE username = ?",
                    (user,),
                )
            }

            graph_data = {
                "nodes": nodes_d3,
                "links": links_d3,
                "colorMap": color_map,
                "postIdToUrl": post_id_to_url_map,
                "postIdToTitle": postIdToTitle
            }
            graph_data_json_str = json.dumps(graph_data) # Use standard dumps
            logging.info(f"Rendering NER graph with {len(nodes_d3)} nodes and {len(links_d3)} links.")
        except Exception as json_err:
             logging.error(f"Error preparing JSON for NER graph: {json_err}", exc_info=True)
             st.error("Erreur interne lors de la préparation des données du graphe NER.")
             return


        # --- D3 HTML for the Graph (Simplified & Corrected) ---
        node_radius_py = 8
        node_padding_py = 15
        link_distance_py = 120
        charge_strength_py = -300 # Slightly stronger repulsion

        html_d3 = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Entity-Relation Graph</title>
            <style>
                /* (Styles remain the same as previous version - omitted for brevity) */
                 body {{
                    margin: 0;
                    background-color: #1E1E1E;
                    overflow: hidden;
                    border: 1px solid #888;         /* ← bordure grise fine */
                    box-sizing: border-box;         /* ← pour inclure la bordure dans le calcul de taille */
                    }}
                 svg {{ display: block; width: 100%; height: 750px; background-color: #1E1E1E; cursor: grab; }} svg:active {{ cursor: grabbing; }}
                 .links line {{ stroke: #777; stroke-opacity: 0.5; }} .links line:hover {{ stroke-opacity: 1; stroke: #aaa; }}
                 .nodes circle {{ stroke: #fff; stroke-width: 1px; cursor: pointer; }} .nodes circle:hover {{ stroke-width: 2.5px; stroke: #fff; }}
                 .node-labels text {{ font-family: "Segoe UI", Roboto, sans-serif; font-size: 9px; fill: #E0E0E0; pointer-events: none; text-anchor: middle; paint-order: stroke; stroke: #1E1E1E; stroke-width: 2px; }}
                 .link-labels text {{ font-family: sans-serif; font-size: 8px; fill: #bbb; pointer-events: none; text-anchor: middle; paint-order: stroke; stroke: #1E1E1E; stroke-width: 2px; }}
                 .tooltip {{ position: absolute; background-color: rgba(40, 40, 40, 0.9); color: #E0E0E0; padding: 8px 12px; border-radius: 4px; font-size: 12px; pointer-events: none; opacity: 0; transition: opacity 0.2s ease; max-width: 350px; white-space: normal; z-index: 10; }}
            </style>
        </head>
        <body>
            <div id="tooltip-ner" class="tooltip"></div>
            <svg id="d3-entity-graph"></svg>
            <script src="https://d3js.org/d3.v7.min.js"></script>
            <script>
                // Use a try-catch block for safety during parsing and rendering
                try {{
                    const graphData = JSON.parse(`{graph_data_json_str}`); // Parse the JSON string here
                    const nodes = graphData.nodes || [];
                    const links = graphData.links || [];
                    const colorMap = graphData.colorMap || {{}};
                    const postIdToUrl = graphData.postIdToUrl || {{}};
                    const postIdToTitle = graphData.postIdToTitle || {{}};


                    console.log("NER data received:", {{ nodes: nodes.length, links: links.length }}); // Browser console log

                    const container = document.getElementById('d3-entity-graph');
                    const tooltip = d3.select("#tooltip-ner"); // Use unique ID if needed
                    const width = container.clientWidth;
                    const height = container.clientHeight;
                    const nodeBaseRadius = {node_radius_py};

                    if (width <= 0 || height <= 0 || nodes.length === 0) {{
                        throw new Error("Container has no dimensions or no nodes to render.");
                    }}

                    const svg = d3.select("#d3-entity-graph")
                        .attr("viewBox", [-width / 2, -height / 2, width, height]);

                    const simulation = d3.forceSimulation(nodes)
                        .force("link", d3.forceLink(links).id(d => d.id).distance({link_distance_py}).strength(0.1))
                        .force("charge", d3.forceManyBody().strength({charge_strength_py}))
                        .force("collide", d3.forceCollide().radius(nodeBaseRadius + {node_padding_py}))
                        .force("center", d3.forceCenter(0, 0))
                        .on("tick", ticked);

                    const zoom = d3.zoom().scaleExtent([0.1, 8]).on("zoom", (event) => g.attr("transform", event.transform));
                    const g = svg.append("g");
                    svg.call(zoom).on("dblclick.zoom", null);

                    const link = g.append("g").attr("class", "links")
                        .selectAll("line").data(links).join("line");

                    const linkLabel = g.append("g").attr("class", "link-labels")
                        .selectAll("text").data(links).join("text")
                        .text(d => d.relationship);

                    const nodeGroup = g.append("g").attr("class", "nodes")
                        .selectAll("g").data(nodes).join("g")
                        .call(drag(simulation))
                        .on("click", nodeClicked)
                        .on("mouseover", nodeMouseover)
                        .on("mouseout", nodeMouseout)
                        .on("dblclick", nodeDblClicked);

                    nodeGroup.append("circle")
                        .attr("r", nodeBaseRadius)
                        .attr("fill", d => colorMap[d.type] || "#cccccc");

                    nodeGroup.append("text").attr("class", "node-labels")
                        .attr("dy", - (nodeBaseRadius + 3))
                        .attr("fill", "#FFFFFF")   
                        .text(d => d.id);

                    // Tooltips
                    nodeGroup.select("circle") // Simple browser tooltip
                        .text(d => `${{d.id}} (${{d.type}})\nPosts: ${{d.posts && d.posts.length > 0 ? d.posts.join(', ') : 'N/A'}}`);
                    link.append("title")
                        .text(d => `${{d.source.id}} - ${{d.relationship}} - ${{d.target.id}}`);

                    function ticked() {{
                        link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
                            .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
                        nodeGroup.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
                        linkLabel.attr("x", d => (d.source.x + d.target.x) / 2)
                                 .attr("y", d => (d.source.y + d.target.y) / 2);
                    }}

                    function nodeClicked(event, d) {{ /* ... (same as before) ... */ event.stopPropagation(); if (event.shiftKey) {{ if (d.fx === null) {{ d.fx = d.x; d.fy = d.y; }} else {{ d.fx = null; d.fy = null; simulation.alphaTarget(0.1).restart(); }} }} }}
                    function nodeDblClicked(event, d) {{ /* ... (same as before) ... */ event.stopPropagation(); if (d.posts && d.posts.length > 0) {{ const url = postIdToUrl[d.posts[0]]; if (url && url !=='#') window.open(url, '_blank'); else console.warn(`No valid URL for post ID: ${{d.posts[0]}}`); }} }}

                    function nodeMouseover(event, d) {{
                        tooltip.transition().duration(200).style("opacity", 0.9);

                        let postLinksHtml = "No associated posts found.";
                        if (d.posts && d.posts.length > 0) {{
                            postLinksHtml = d.posts.map(pid => {{
                                const title = postIdToTitle[pid] || `Post ${{pid}}`;
                                const url   = postIdToUrl[pid]   || "#";

                                // Extrait le domaine comme source
                                let source = "";
                                try {{ source = new URL(url).hostname.replace("www.", ""); }} catch {{ }}

                                return url !== "#" && url
                                    ? `
                                    <a href="${{url}}" target="_blank" style="color:#80DFFF; text-decoration:none;">
                                        <strong>${{title}}</strong><br>
                                        <span style="font-size:8px; color:#AAA;">Source : ${{source}}</span>
                                    </a>
                                    `
                                    : `
                                    <div>
                                        <strong>${{title}}</strong><br>
                                        <span style="font-size:8px; color:#AAA;">Source : ${{source}}</span>
                                    </div>
                                    `;
                            }}).join("");
                        }}

                        tooltip
                        .html(`<div style="max-width:200px;">${{postLinksHtml}}</div>`)
                        .style("left", (event.pageX + 15) + "px")
                        .style("top",  (event.pageY - 10) + "px");
                    }}

                    function nodeMouseout() {{
                        tooltip.transition().duration(500).style("opacity", 0);
                    }}
                    

                    function drag(simulation) {{ /* ... (same as before) ... */ function dragstarted(event, d) {{ if (!event.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }} function dragged(event, d) {{ d.fx = event.x; d.fy = event.y; }} function dragended(event, d) {{ if (!event.active) simulation.alphaTarget(0); /* Keep fixed unless shift-clicked */ }} return d3.drag().on("start", dragstarted).on("drag", dragged).on("end", dragended); }}

                }} catch (error) {{
                    console.error("Error rendering D3 graph:", error);
                    container.innerHTML = `<p style='color:red; text-align:center; padding-top: 50px;'>Erreur JavaScript lors du rendu du graphe: ${{error.message}}</p>`;
                }}
            </script>
        </body>
        </html>
        """
        #col_graph, col_legend = st.columns([4, 1], gap="medium")
        col_graph, col_legend = st.columns([4, 1], gap="medium")

        # suppose que tu as une fct dédiée

        with col_graph:
            components.html(html_d3, height=760, scrolling=False)

        with col_legend:
            st.markdown("### Légende (Entités)")
            if not entity_types:
                st.caption("Aucun type.")
            else:
                legend_html = ""
                for etype in sorted(entity_types):
                    color = color_map[etype]
                    legend_html += (
                        f"<div style='display:flex; align-items:center; margin-bottom:4px;'>"
                        f"  <div style='width:12px; height:12px; background:{color}; margin-right:6px; border-radius:3px;'></div>"
                        f"  <span>{etype}</span>"
                        f"</div>"
                    )
                st.markdown(legend_html, unsafe_allow_html=True)

#        with col_bar:
            # on remet à jour la même barre et le même texte
#            progress_bar
#            progress_text


    # Initial render / Render after processing stops
    render_entity_graph_d3()

    # --- Display Results per Article ---
    st.markdown("---")
    st.subheader("Détails par Article Analysé")
    # (Display logic remains the same as previous version - omitted for brevity)
    if not st.session_state.get('results_by_post'):
        st.caption("Aucun détail d'article disponible.")
    else:
        sorted_post_ids = sorted(st.session_state['results_by_post'].keys(), key=lambda x: int(x) if x.isdigit() else 0, reverse=True)
        for post_id in sorted_post_ids:
            data = st.session_state['results_by_post'][post_id]
            post_url = st.session_state.get('post_id_to_url', {}).get(post_id, "#")
            with st.expander(f"Article ID: {post_id} " + (f"([Voir]({post_url}))" if post_url != "#" else "")):
                st.markdown("**Entités:**")
                if data.get("entities"):
                    entities_html = ""
                    sorted_entities = sorted(list(data["entities"]), key=lambda x: x[0])
                    for entity_name, entity_type in sorted_entities:
                        color = generate_color(entity_type); text_color = "#FFF" # Simple white text for now
                        entities_html += f"<span style='background-color:{color}; color:{text_color}; padding: 2px 5px; margin: 2px; border-radius: 3px; display: inline-block; font-size: 0.9em;'>{entity_name} ({entity_type})</span> "
                    st.markdown(entities_html, unsafe_allow_html=True)
                else: st.caption("Aucune.")
                st.markdown("**Relations:**")
                if data.get("relations"):
                    for rel in data["relations"]: st.markdown(f"- `{rel['entity1']['entity']}` **→ {rel['relationship']} →** `{rel['entity2']['entity']}`")
                else: st.caption("Aucune.")


# === Main Application Logic ===

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(
        page_title="KYRRIA App Demo", page_icon="favicon.png", layout="wide",
        initial_sidebar_state="expanded"
    )
    apply_custom_css()

    if 'user' not in st.session_state:
        display_login_signup_forms()
    else:
        pages = {
            "🏠 Home": display_home_page,
            "📡 Gestionnaire de flux": display_feed_manager_page,
            "📰 Lecteur RSS": display_rss_reader_page,
            "🔗 Nodes": display_nodes_page,
            "📊 Dashboard": display_bar_chart_page,
            "💡 Entities & Relations": display_entity_relation_graph_page,
        }
        setup_sidebar(pages)
        page_function = pages.get(st.session_state.selected_page)
        if page_function:
            try:
                page_function()
            except Exception as page_err:
                 logging.error(f"Error rendering page '{st.session_state.selected_page}': {page_err}", exc_info=True)
                 st.error(f"Une erreur inattendue est survenue sur la page '{st.session_state.selected_page}'. Détails dans les logs.")
        else:
            st.session_state.selected_page = "🏠 Home"
            pages["🏠 Home"]() # Fallback

if __name__ == "__main__":
    main()