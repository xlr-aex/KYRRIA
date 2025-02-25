import streamlit as st
import streamlit.components.v1 as components
import feedparser
import requests
import time
import os
import json
import re
import math
from urllib.parse import urlparse
from PIL import Image
from io import BytesIO

# --------------------------------------------------------------------
#                       FONCTIONS UTILES
# --------------------------------------------------------------------
FEEDS_FILE = "feeds.json"

def load_feeds():
    """Charge le fichier JSON contenant la liste des flux."""
    if not os.path.exists(FEEDS_FILE):
        default_feeds = [
            {"url": "https://www.lemonde.fr/rss/une.xml"},
            {"url": "https://www.reddit.com/r/blackhat/.rss"}
        ]
        with open(FEEDS_FILE, "w") as f:
            json.dump(default_feeds, f)
        return default_feeds
    with open(FEEDS_FILE, "r") as f:
        return json.load(f)

def save_feeds(feeds):
    """Sauvegarde la liste des flux dans un fichier JSON."""
    with open(FEEDS_FILE, "w") as f:
        json.dump(feeds, f)

def extraire_image_from_summary(summary_html):
    """
    Cherche la première balise <img src="..."> dans le HTML du 'summary'.
    Retourne l'URL si trouvée, sinon None.
    """
    pattern = r'<img[^>]+src=["\']([^"\']+)["\']'
    match = re.search(pattern, summary_html)
    if match:
        return match.group(1)
    return None

def extraire_image(entry):
    """
    Tente de récupérer l'URL d'une image dans :
      1) 'media_content'
      2) 'media_thumbnail'
      3) En scannant <img> dans le 'summary'
    """
    mc = entry.get("media_content")
    if mc and isinstance(mc, list) and len(mc) > 0:
        url_cand = mc[0].get("url")
        if url_cand:
            return url_cand

    mt = entry.get("media_thumbnail")
    if mt and isinstance(mt, list) and len(mt) > 0:
        url_cand = mt[0].get("url")
        if url_cand:
            return url_cand

    summary_html = entry.get("summary", "")
    return extraire_image_from_summary(summary_html)

def charger_feed_et_articles(url):
    """
    Récupère le flux RSS/Atom via requests + feedparser.
    Retourne un dict simple avec :
      - "bozo": bool
      - "articles": liste d'articles (dictionnaires)
    Chaque article contient uniquement des types simples :
      "title", "link", "published", "summary", "published_parsed",
      "media_content", "media_thumbnail".
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/98.0.4758.102 Safari/537.36"
        )
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        return {"bozo": True, "articles": [], "error": str(e)}

    raw_feed = feedparser.parse(response.content)
    feed_data = {
        "bozo": raw_feed.bozo,
        "articles": []
    }
    for entry in raw_feed.entries:
        pp = entry.get("published_parsed")
        if pp:
            pp = tuple(pp)

        media_content = None
        if hasattr(entry, "media_content"):
            try:
                media_content = list(entry.media_content)
            except Exception:
                media_content = None

        media_thumbnail = None
        if hasattr(entry, "media_thumbnail"):
            try:
                media_thumbnail = list(entry.media_thumbnail)
            except Exception:
                media_thumbnail = None

        feed_data["articles"].append({
            "title": entry.get("title", "Titre non disponible"),
            "link": entry.get("link", "#"),
            "published": entry.get("published", "Date non disponible"),
            "summary": entry.get("summary", ""),
            "published_parsed": pp,
            "media_content": media_content,
            "media_thumbnail": media_thumbnail
        })
    return feed_data

def get_timestamp(entry):
    pp = entry.get("published_parsed")
    if pp:
        return time.mktime(pp)
    return 0

def get_origin(entry):
    link = entry.get("link", "")
    parsed = urlparse(link)
    domain = parsed.netloc.replace("www.", "")
    favicon_url = f"https://www.google.com/s2/favicons?domain={parsed.netloc}"
    return domain, favicon_url

def get_dominant_color(favicon_url):
    """Extrait une couleur dominante approximative (RGB) d'un favicon en ligne, sans colorthief."""
    try:
        response = requests.get(favicon_url, timeout=5)
        response.raise_for_status()
        # Charger l'image dans un buffer
        img = Image.open(BytesIO(response.content))
        # Redimensionner l'image pour simplifier l'analyse
        img = img.resize((16, 16), Image.Resampling.LANCZOS)
        # Convertir en mode RGB si nécessaire
        img = img.convert('RGB')
        # Compter les pixels et trouver la couleur la plus fréquente
        pixel_counts = {}
        for x in range(img.width):
            for y in range(img.height):
                r, g, b = img.getpixel((x, y))
                color = (r, g, b)
                pixel_counts[color] = pixel_counts.get(color, 0) + 1
        dominant_color = max(pixel_counts.items(), key=lambda x: x[1])[0]
        # Format CSS RGB
        return f"rgb({dominant_color[0]}, {dominant_color[1]}, {dominant_color[2]})"
    except Exception:
        return "#444"  # Couleur par défaut

# --------------------------------------------------------------------
#                       CONFIG STREAMLIT
# --------------------------------------------------------------------
st.set_page_config(page_title="KYRRIA App Demo", layout="wide")

pages = [
    "🏠Home",
    "📡Gestionnaire de flux",
    "📰Lecteur RSS",
    "🔗Nodes",
    "🗺️Map monde",
    "📊Camembert",
    "🔵Bubble Chart",
    "🕒Timeline"
]

if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Home"

st.markdown(
    """
    <style>
      html, body { margin: 0; padding: 0; font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; background-color: #f7f7f7; color: #333; }
      .reportview-container { padding: 1rem; }
      .sidebar-header { text-align: center; margin-bottom: 2rem; }
      .sidebar-header h2 { font-size: 1.5rem; margin: 0; color: #007bff; }
      [data-testid="stSidebar"] .stButton button { background-color: #007bff; color: #fff; border: none; padding: 0.8rem 1rem; margin-bottom: 0.5rem; border-radius: 4px; width: 100%; text-align: left; }
      [data-testid="stSidebar"] .stButton button:hover { background-color: #0056b3; }
      [data-testid="stSidebar"] .stButton button:disabled { background-color: #007bff; color: #fff; opacity: 1; cursor: default; }
      .block-container { padding: 2rem; }
      div.stButton > button { width: 100%; height: 50px; font-size: 18px; margin-bottom: 5px; }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h2>KYRRIA</h2>
    </div>
    """, unsafe_allow_html=True)
    for p in pages:
        if p == st.session_state.selected_page:
            if st.button(p, key=p, disabled=True):
                pass
        else:
            if st.button(p, key=p):
                st.session_state.selected_page = p

page = st.session_state.selected_page

# --------------------------------------------------------------------
#                          PAGE HOME
# --------------------------------------------------------------------
if page == "🏠Home":
    st.title("Bienvenue sur KYRRIA App Demo")
    st.markdown("""
    **KYRRIA App Demo** est une application de démonstration présentant diverses visualisations interactives réalisées avec D3.js.  
    Utilisez la barre latérale pour naviguer entre les sections :
    
    - **Gestionnaire de flux** : Ajoutez/supprimez vos flux RSS/Atom.
    - **Lecteur RSS** : Agrégation complète des flux avec options d’affichage.
    - **Nodes** : Visualisation interactive d'un réseau d'entités.
    - **Map monde** : Carte interactive du monde.
    - **Camembert**, **Bubble Chart**, **Timeline** : Différentes visualisations.
    """)

# --------------------------------------------------------------------
#                     PAGE GESTIONNAIRE DE FLUX
# --------------------------------------------------------------------
elif page == "📡Gestionnaire de flux":
    st.title("Gestionnaire de flux RSS/Atom")
    st.markdown("Gérez vos liens RSS/Atom. Les flux sont enregistrés dans le fichier feeds.json.")
    feeds = load_feeds()
    with st.form("flux_form"):
        new_flux = st.text_input("URL du flux", placeholder="https://exemple.com/rss")
        submit_flux = st.form_submit_button("Ajouter")
        if submit_flux:
            if new_flux and new_flux not in [feed["url"] for feed in feeds]:
                feeds.append({"url": new_flux})
                save_feeds(feeds)
                st.success(f"Flux ajouté : {new_flux}")
                st.rerun()
            elif new_flux:
                st.warning("Ce flux est déjà présent.")
            else:
                st.error("Veuillez entrer une URL valide.")
    st.markdown("### Flux enregistrés")
    if feeds:
        for index, feed in enumerate(feeds):
            domain = urlparse(feed["url"]).netloc.replace("www.", "")
            favicon_url = f"https://www.google.com/s2/favicons?domain={domain}"
            col1, col2, col3 = st.columns([1, 6, 1])
            col1.image(favicon_url, width=30)
            col2.write(feed["url"])
            if col3.button("❌", key=f"remove_{index}"):
                del feeds[index]
                save_feeds(feeds)
                st.rerun()
    else:
        st.info("Aucun flux enregistré.")

# --------------------------------------------------------------------
#                        PAGE LECTEUR RSS
# --------------------------------------------------------------------
elif page == "📰Lecteur RSS":
    st.markdown("<div id='lecteur'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <script>
        setTimeout(function() {
            var element = document.getElementById('lecteur');
            if(element) { element.scrollIntoView({ behavior: 'smooth' }); }
        }, 100);
        </script>
        """,
        unsafe_allow_html=True
    )
    st.title("Lecteur de flux RSS/Atom")
    st.markdown("Ce module agrège les flux enregistrés et propose plusieurs options d'affichage.")
    feeds = load_feeds()
    if not feeds:
        st.error("Aucun flux n'est disponible. Veuillez ajouter des flux dans 'Gestionnaire de flux'.")
    else:
        with st.expander("Configuration d'affichage", expanded=False):
            nb_articles = st.number_input(
                "Nombre d'articles à afficher :", min_value=1, max_value=1000, value=50, step=10
            )
            st.markdown("**Mode de visualisation :**")
            if "view_mode" not in st.session_state:
                st.session_state.view_mode = "Liste détaillée"
            col1, col2, col3 = st.columns(3)
            if col1.button("Liste détaillée", key="vis_detailed"):
                st.session_state.view_mode = "Liste détaillée"
            if col2.button("Liste raccourcie", key="vis_compact"):
                st.session_state.view_mode = "Liste raccourcie"
            if col3.button("Vue en cubes", key="vis_cubes"):
                st.session_state.view_mode = "Vue en cubes"
        view_mode = st.session_state.view_mode

        search_keyword = st.text_input("Filtrer par mot-clé dans le titre", "")
        selected_feeds = []

        st.markdown("### Sélectionnez les flux à afficher")
        for feed in feeds:
            parsed = urlparse(feed["url"])
            short_info = feedparser.parse(feed["url"]).feed
            full_title = short_info.get("title", parsed.netloc.replace("www.", "")).strip()
            parts = full_title.split(" - ", 1)
            main_title = parts[0].strip()
            rest_title = parts[1].strip() if len(parts) > 1 else ""
            favicon_url = f"https://www.google.com/s2/favicons?domain={parsed.netloc}"
            cols = st.columns([0.9, 0.1])
            with cols[0]:
                title_html = f"""
                <p style="margin:0; white-space:nowrap;">
                <a href='{feed["url"]}' target='_blank' style='text-decoration:none; color:inherit;'>
                    <img src='{favicon_url}' width='20' style='vertical-align:middle; margin-right:5px;'>
                    {main_title}
                </a>
                {f" - {rest_title}" if rest_title else ""}
                </p>
                """
                st.markdown(title_html, unsafe_allow_html=True)
            with cols[1]:
                if st.checkbox("Sélectionner", key=feed["url"], value=True, label_visibility="collapsed"):
                    selected_feeds.append(feed["url"])

        articles = []
        for feed_url in selected_feeds:
            flux_data = charger_feed_et_articles(feed_url)
            if flux_data.get("bozo") or flux_data.get("error"):
                err_msg = flux_data.get("error", "")
                if err_msg:
                    st.error(f"Flux invalide ou inaccessible: {feed_url}\nErreur: {err_msg}")
                else:
                    st.error(f"Flux invalide: {feed_url}")
            else:
                for item in flux_data["articles"]:
                    item["feed_url"] = feed_url
                    articles.append(item)
                if len(articles) >= 1000:
                    break

        st.markdown(f"**Nombre d'articles disponibles :** {len(articles)} / {nb_articles} affichés")
        if search_keyword:
            articles = [
                entry for entry in articles
                if search_keyword.lower() in entry.get("title", "").lower()
            ]
        articles = sorted(articles, key=get_timestamp, reverse=True)[:nb_articles]

        st.header("Flux Agrégés")
        if not articles:
            st.info("Aucun article ne correspond aux critères.")
        else:
            if view_mode == "Liste détaillée":
                for entry in articles:
                    title = entry.get("title", "Titre non disponible")
                    link = entry.get("link", "#")
                    st.markdown(f"### [{title}]({link})")
                    published = entry.get("published", "Date non disponible")
                    st.markdown(f"_{published}_")
                    image_url = extraire_image(entry)
                    if image_url:
                        st.image(image_url, width=300)
                    st.write(entry.get("summary", "Aucun résumé disponible"))
                    domain, favicon_url = get_origin(entry)
                    st.markdown(f"![]({favicon_url}) {domain}")
                    st.markdown("---")

            elif view_mode == "Liste raccourcie":
                for entry in articles:
                    title = entry.get("title", "Titre non disponible")
                    link = entry.get("link", "#")
                    published = entry.get("published", "Date non disponible")
                    domain, favicon_url = get_origin(entry)
                    st.markdown(
                        f"<p style='margin: 0; line-height: 1;'>"
                        f"<strong><a href='{link}' style='text-decoration:none;color:inherit;'>{title}</a></strong> - <em>{published}</em></p>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"<p style='margin: 0; line-height: 1;'>"
                        f"<a href='{link}'>Lire l'article</a> | <strong>Origine :</strong> "
                        f"<img src='{favicon_url}' width='16' style='vertical-align:middle;'> {domain}</p>",
                        unsafe_allow_html=True
                    )
                    st.markdown("<hr style='margin: 1px 0;'>", unsafe_allow_html=True)

            elif view_mode == "Vue en cubes":
                st.markdown("<h3 style='margin:0; padding:0;'>🟦 Vue en cubes</h3>", unsafe_allow_html=True)
                num_cols = min(4, max(1, len(articles)))
                rows = (len(articles) + num_cols - 1) // num_cols
                for i in range(rows):
                    cols = st.columns(num_cols, gap="small")
                    for j, col in enumerate(cols):
                        index = i * num_cols + j
                        if index < len(articles):
                            entry = articles[index]
                            with col:
                                title = entry.get("title", "Titre non disponible")
                                link = entry.get("link", "#")
                                published = entry.get("published", "Date non disponible")
                                image_url = extraire_image(entry)
                                domain, favicon_url = get_origin(entry)
                                dominant_color = get_dominant_color(favicon_url)
                                # Prépare le "contenu" conditionnel en amont
                                if image_url:
                                    contenu_img_ou_lien = f"<img src='{image_url}' style='width:100%; max-height:200px; object-fit:cover; overflow:hidden; border-radius:5px; margin:0 0 3px 0;'/>"
                                else:
                                    contenu_img_ou_lien = f"<a href='{link}' target='_blank' title=\"Cliquez pour lire l'article\"><div class='cube-link' style='width:100%; height:200px; background-color:{dominant_color}; display:flex; align-items:center; justify-content:center; border-radius:5px; cursor:pointer;'><div style='width:50px; height:50px; background-color:{dominant_color}; display:flex; align-items:center; justify-content:center; border-radius:3px;'><img src='{favicon_url}' width='30' height='30' style='position:absolute;'></div></div></a>"

                                # Puis on construit l'article_html
                                article_html = f"""
                                <div style="margin:0; padding:0; background-color:transparent;">
                                    <h4 style="margin:0; padding:0; font-weight:bold; font-size:1rem;">
                                        <a href="{link}" target="_blank" style="text-decoration:none; color:#007bff;">
                                            {title}
                                        </a>
                                    </h4>
                                    <p style="margin:3px 0; padding:0; font-size:0.9rem; color:#bbb;">
                                        📅 {published}
                                    </p>
                                    {contenu_img_ou_lien}
                                    <p style="margin:0; padding:0; font-size:0.85rem;">
                                        <img src="{favicon_url}" width="14" style="vertical-align:middle;"> {domain}
                                    </p>
                                    <hr style="margin:6px 0 0 0; padding:0; border:none; border-top:1px solid #444;">
                                </div>
                                """
                                st.markdown(article_html, unsafe_allow_html=True)


# --------------------------------------------------------------------
#                        PAGE NODES
# --------------------------------------------------------------------
elif page == "🔗Nodes":
    st.title("Nodes - Réseau de flux et occurrences de mots-clés")
    st.markdown("""
    Entrez un mot-clé pour voir où il apparaît dans vos flux RSS.  
    Ajustez la distance des liens avec le slider et utilisez la molette pour zoomer/dézoomer.  
    Les titres des articles s’affichent en entier avec un fond orange.
    """)

    keyword = st.text_input("Rechercher un mot-clé dans les flux", "").lower()
    
    # Slider pour ajuster la distance des liens
    link_distance = st.slider("Distance des liens", min_value=50, max_value=2000, value=100, step=10)
    
    feeds = load_feeds()
    nodes = []
    links = []
    occurrence_counter = 0
    feed_occurrence_counts = {}

    if keyword:
        feeds_with_occurrences = []
        for feed in feeds:
            flux_data = charger_feed_et_articles(feed["url"])
            if not flux_data.get("bozo") and "articles" in flux_data:
                for article in flux_data["articles"]:
                    if keyword in article.get("title", "").lower() or keyword in article.get("summary", "").lower():
                        feeds_with_occurrences.append(feed)
                        break
        num_feeds = len(feeds_with_occurrences)
        angle_step = 2 * 3.1416 / num_feeds if num_feeds > 0 else 0
        radius = min(800, 600) / 3
    else:
        num_feeds = len(feeds)
        angle_step = 2 * 3.1416 / num_feeds if num_feeds > 0 else 0
        radius = min(800, 600) / 3

    for feed_index, feed in enumerate(feeds):
        feed_url = feed["url"]
        feed_node_id = f"feed_{feed_index}"
        parsed = urlparse(feed_url)
        domain = parsed.netloc.replace("www.", "")
        favicon_url = f"https://www.google.com/s2/favicons?domain={parsed.netloc}"
        feed_title = domain
        occurrence_nodes = []

        if keyword:
            flux_data = charger_feed_et_articles(feed_url)
            if not flux_data.get("bozo") and "articles" in flux_data:
                for article in flux_data["articles"]:
                    title = article.get("title", "").lower()
                    summary = article.get("summary", "").lower()
                    if keyword in title or keyword in summary:
                        occurrence_node_id = f"occurrence_{occurrence_counter}"
                        occurrence_counter += 1
                        occurrence_title = article.get("title", "Sans titre")
                        occurrence_nodes.append({
                            "id": occurrence_node_id,
                            "type": "occurrence",
                            "label": (occurrence_title[:40] + "...") if len(occurrence_title) > 40 else occurrence_title,
                            "full_label": occurrence_title,
                            "url": article.get("link", "#"),
                            "feed_id": feed_node_id
                        })

        occurrence_count = len(occurrence_nodes)
        feed_occurrence_counts[feed_node_id] = occurrence_count

        if not keyword or occurrence_count > 0:
            angle = feed_index * angle_step
            feed_x = 400 + radius * math.cos(angle)
            feed_y = 300 + radius * math.sin(angle)
            nodes.append({
                "id": feed_node_id,
                "type": "feed",
                "label": feed_title,
                "favicon": favicon_url,
                "url": feed_url,
                "occurrences": occurrence_count,
                "x": feed_x,
                "y": feed_y
            })
            article_angle_step = 2 * 3.1416 / occurrence_count if occurrence_count > 0 else 0
            article_radius = link_distance  # Utilisation de la distance choisie via le slider
            for i, occ_node in enumerate(occurrence_nodes):
                angle = i * article_angle_step
                occ_x = feed_x + article_radius * math.cos(angle)
                occ_y = feed_y + article_radius * math.sin(angle)
                occ_node["x"] = occ_x
                occ_node["y"] = occ_y
                nodes.append(occ_node)
                links.append({
                    "source": occ_node["id"],
                    "target": feed_node_id,
                    "distance": article_radius
                })

    if not keyword:
        st.info("Veuillez entrer un mot-clé pour voir les occurrences.")
        network_data = {"nodes": nodes, "links": []}
    else:
        network_data = {"nodes": nodes, "links": links}
        st.markdown(f"**Occurrences trouvées pour '{keyword}' :** {len(links)}")

    network_json = json.dumps(network_data)

    d3_nodes = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body {{
      margin: 0; 
      background: #000; 
    }}
    svg {{
      width: 100%; 
      height: 600px; 
      border: 1px solid #444; 
      background: #000;
    }}
    .node {{
      cursor: pointer;
    }}
    .feed-node {{
      fill: none;
    }}
    .occurrence-node {{
      fill: #ff5733;
      rx: 5;
      ry: 5;
    }}
    .link {{
      stroke: #aaa;
      stroke-opacity: 0.8;
      stroke-width: 1.5;
    }}
    .node:hover .occurrence-node {{
      fill: #ff784e;
    }}
    .occurrence-text {{
      color: #fff;
      padding: 5px;
      max-width: 150px;
      word-wrap: break-word;
      font-size: 14px;
      line-height: 1.2;
      background: none;
      text-align: center;
    }}
  </style>
</head>
<body>
  <svg></svg>
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <script>
    const data = {network_json};

    const svg = d3.select("svg"),
          width = svg.node().clientWidth,
          height = +svg.attr("height") || 600;

    const zoom = d3.zoom()
      .scaleExtent([0.1, 10])
      .on("zoom", (event) => {{
        container.attr("transform", event.transform);
      }});
    svg.call(zoom);

    const container = svg.append("g");

    const simulation = d3.forceSimulation(data.nodes)
        .force("link", d3.forceLink(data.links).id(d => d.id).distance(link => link.distance).strength(1))
        .force("charge", d3.forceManyBody().strength(d => d.type === "feed" ? -200 : 0))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collide", d3.forceCollide().radius(d => d.type === "feed" ? 20 + (d.occurrences || 0) * 2 : 80))
        .on("tick", ticked);

    data.nodes.forEach(d => {{
      d.x = d.x || width / 2;
      d.y = d.y || height / 2;
    }});

    const link = container.append("g")
      .attr("class", "links")
      .selectAll("line")
      .data(data.links)
      .join("line")
      .attr("class", "link");

    const node = container.append("g")
      .attr("class", "nodes")
      .selectAll("g")
      .data(data.nodes)
      .join("g")
      .classed("node", true)
      .on("click", (event, d) => {{
        if (d.url) {{
          window.open(d.url, "_blank");
        }}
      }});

    // Noeuds de type 'feed' => affichage du favicon
    node.filter(d => d.type === "feed")
      .append("image")
      .attr("xlink:href", d => d.favicon)
      .attr("width", d => Math.max(16, 32 + (d.occurrences || 0) * 5))
      .attr("height", d => Math.max(16, 32 + (d.occurrences || 0) * 5))
      .attr("x", d => -Math.max(8, 16 + (d.occurrences || 0) * 2.5))
      .attr("y", d => -Math.max(8, 16 + (d.occurrences || 0) * 2.5))
      .classed("feed-node", true);

    // Noeuds de type 'occurrence' => rectangle orange avec texte complet
    const occurrenceNodes = node.filter(d => d.type === "occurrence");
    
    occurrenceNodes.append("rect")
      .attr("class", "occurrence-node");

    occurrenceNodes.append("foreignObject")
      .attr("width", 150)
      .attr("height", 200)
      .attr("x", -75)
      .attr("y", -10)
      .append("xhtml:div")
      .attr("class", "occurrence-text")
      .text(d => d.full_label);

    occurrenceNodes.each(function(d) {{
      const fo = d3.select(this).select("foreignObject");
      const div = fo.select("div");
      const bbox = div.node().getBoundingClientRect();
      const rect = d3.select(this).select("rect");
      rect.attr("width", bbox.width + 10)
          .attr("height", bbox.height + 10)
          .attr("x", - (bbox.width + 10) / 2)
          .attr("y", - (bbox.height + 10) / 2);
    }});

    // Labels pour les noeuds 'feed'
    node.filter(d => d.type === "feed")
      .append("text")
      .attr("dx", d => Math.max(18, 20 + (d.occurrences || 0) * 2.5))
      .attr("dy", ".35em")
      .attr("text-anchor", "start")
      .text(d => d.label)
      .style("font-size", "20px");

    node.filter(d => d.type === "feed").each(function(d) {{
      const text = d3.select(this).select("text");
      const bbox = text.node().getBBox();
      d3.select(this).insert("rect", "text")
        .attr("x", bbox.x - 2)
        .attr("y", bbox.y - 2)
        .attr("width", bbox.width + 4)
        .attr("height", bbox.height + 4)
        .attr("fill", "rgba(0,0,0,0.7)")
        .attr("rx", 3)
        .attr("ry", 3);
    }});

    // Infobulles
    node.append("title")
      .text(d => d.type === "feed"
                 ? `Flux: ${{d.label}} (Occurrences: ${{d.occurrences || 0}})`
                 : `Article: ${{d.full_label}}`);

    function ticked() {{
      link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);

      node
        .attr("transform", d => `translate(${{d.x}}, ${{d.y}})`);
    }}

    node.call(d3.drag()
      .on("start", dragstarted)
      .on("drag", dragged)
      .on("end", dragended)
    );

    function dragstarted(event, d) {{
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }}
    function dragged(event, d) {{
      d.fx = event.x;
      d.fy = event.y;
    }}
    function dragended(event, d) {{
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }}

    function recenter() {{
      const nodes = simulation.nodes();
      const minX = d3.min(nodes, d => d.x);
      const maxX = d3.max(nodes, d => d.x);
      const minY = d3.min(nodes, d => d.y);
      const maxY = d3.max(nodes, d => d.y);
      const dx = maxX - minX;
      const dy = maxY - minY;
      const x = (minX + maxX) / 2;
      const y = (minY + maxY) / 2;
      const scale = Math.min(width / dx, height / dy) * 0.9;
      const translate = [width / 2 - scale * x, height / 2 - scale * y];

      svg.transition().duration(750).call(
        zoom.transform,
        d3.zoomIdentity.translate(translate[0], translate[1]).scale(scale)
      );
    }}

    svg.on("dblclick", recenter);
  </script>
</body>
</html>
"""
    components.html(d3_nodes, height=600)

# --------------------------------------------------------------------
#                        PAGE MAP MONDE
# --------------------------------------------------------------------
elif page == "🗺️Map monde":
    st.title("Map Monde")
    st.markdown("Carte interactive du monde avec des indicateurs fictifs par pays.")

    d3_map = """
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        text { 
            font: 12px sans-serif; 
            fill: #000; 
            pointer-events: none; 
        }
      </style>
    </head>
    <body>
      <svg></svg>
      <script src="https://d3js.org/d3.v7.min.js"></script>
      <script src="https://unpkg.com/topojson@3"></script>
      <script>
        d3.json("https://raw.githubusercontent.com/holtzy/D3-graph-gallery/master/DATA/world.geojson")
          .then(world => {
            const svg = d3.select("svg");
            const width = +svg.attr("width") || 960;
            const height = +svg.attr("height") || 600;

            const projection = d3.geoNaturalEarth1().fitSize([width, height], world);
            const path = d3.geoPath().projection(projection);

            const color = d3.scaleSequential(d3.interpolateBlues)
                            .domain([0, 100]);

            // On ajoute un champ 'index' aléatoire pour la coloration fictive
            world.features.forEach(d => { d.properties.index = Math.floor(Math.random() * 101); });

            svg.append("g")
              .selectAll("path")
              .data(world.features)
              .join("path")
                .attr("fill", d => color(d.properties.index))
                .attr("stroke", "#fff")
                .attr("stroke-width", 0.5)
                .attr("d", path);

            const legend = svg.append("g")
                              .attr("transform", "translate(20,20)");

            const legendScale = d3.scaleLinear().domain([0, 100]).range([0, 100]);
            const legendAxis = d3.axisRight(legendScale).ticks(5);

            legend.selectAll("rect")
                  .data(d3.range(0, 101, 10))
                  .enter()
                  .append("rect")
                  .attr("x", 0)
                  .attr("y", d => d)
                  .attr("width", 20)
                  .attr("height", 10)
                  .attr("fill", d => color(d));

            legend.append("g")
                  .attr("transform", "translate(20,0)")
                  .call(legendAxis);
          });
      </script>
    </body>
    </html>
    """
    components.html(d3_map, height=620)

# --------------------------------------------------------------------
#                         PAGE CAMEMBERT
# --------------------------------------------------------------------
elif page == "📊Camembert":
    st.title("Camembert")
    st.markdown("Répartition fictive des sources d'information dans le secteur de la santé.")

    d3_pie = """
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body { margin: 0; background: #f7f7f7; font: 12px sans-serif; }
        svg { width: 500px; height: 500px; }
      </style>
    </head>
    <body>
      <svg></svg>
      <script src="https://d3js.org/d3.v7.min.js"></script>
      <script>
        const data = { "Articles Médicaux": 45, "Rapports de Recherche": 35, "Études Cliniques": 20 };
        const width = 500, height = 500, radius = Math.min(width, height) / 2;

        const svg = d3.select("svg")
            .attr("viewBox", `0 0 ${width} ${height}`)
          .append("g")
            .attr("transform", `translate(${width / 2},${height / 2})`);

        const color = d3.scaleOrdinal()
            .domain(Object.keys(data))
            .range(["#007bff", "#28a745", "#ffc107"]);

        const pie = d3.pie().value(d => d[1]);
        const data_ready = pie(Object.entries(data));

        const arc = d3.arc().innerRadius(0).outerRadius(radius);

        svg.selectAll("path")
          .data(data_ready)
          .join("path")
            .attr("d", arc)
            .attr("fill", d => color(d.data[0]))
            .attr("stroke", "white")
            .style("stroke-width", "2px");

        svg.selectAll("text")
          .data(data_ready)
          .join("text")
          .attr("transform", d => `translate(${arc.centroid(d)})`)
          .attr("text-anchor", "middle")
          .attr("dy", "0.35em")
          .text(d => d.data[0]);
      </script>
    </body>
    </html>
    """
    components.html(d3_pie, height=520)

# --------------------------------------------------------------------
#                        PAGE BUBBLE CHART
# --------------------------------------------------------------------
elif page == "🔵Bubble Chart":
    st.title("Bubble Chart")
    st.markdown("Diagramme à bulles illustrant la fréquence de termes-clés dans les données.")

    d3_bubble = """
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body { margin: 0; background: #f7f7f7; }
        svg { width: 800px; height: 600px; background: #fff; }
        text { font: 12px sans-serif; fill: #fff; text-anchor: middle; }
      </style>
    </head>
    <body>
      <svg></svg>
      <script src="https://d3js.org/d3.v7.min.js"></script>
      <script>
        const data = [
          {term: "Innovation", freq: 90},
          {term: "AI", freq: 70},
          {term: "Santé", freq: 110},
          {term: "Recherche", freq: 60},
          {term: "Consulting", freq: 40},
          {term: "Pharma", freq: 80},
          {term: "Data", freq: 50}
        ];

        const svg = d3.select("svg"),
              width = +svg.attr("width") || 800,
              height = +svg.attr("height") || 600;

        const pack = d3.pack()
                       .size([width, height])
                       .padding(5);

        const root = d3.hierarchy({children: data}).sum(d => d.freq);
        const nodes = pack(root).leaves();

        svg.selectAll("circle")
            .data(nodes)
            .join("circle")
            .attr("cx", d => d.x)
            .attr("cy", d => d.y)
            .attr("r", d => d.r)
            .attr("fill", "#007bff")
            .attr("stroke", "#fff");

        svg.selectAll("text")
          .data(nodes)
          .join("text")
          .attr("x", d => d.x)
          .attr("y", d => d.y)
          .attr("dy", "0.35em")
          .text(d => d.data.term);
      </script>
    </body>
    </html>
    """
    components.html(d3_bubble, height=620)

# --------------------------------------------------------------------
#                        PAGE TIMELINE
# --------------------------------------------------------------------
elif page == "🕒Timeline":
    st.title("Timeline - Chronologie des publications")
    st.markdown("""
    Chronologie interactive des publications des flux RSS, classées par date de publication.  
    Cliquez sur un événement pour ouvrir l'article correspondant, passez la souris pour plus d'informations.
    """)

    feeds = load_feeds()
    if not feeds:
        st.error("Aucun flux n'est disponible. Veuillez ajouter des flux dans 'Gestionnaire de flux'.")
    else:
        articles = []
        for feed in feeds:
            feed_url = feed["url"]
            flux_data = charger_feed_et_articles(feed_url)
            if not flux_data.get("bozo") and "articles" in flux_data:
                for article in flux_data["articles"]:
                    published = article.get("published", "Date non disponible")
                    published_parsed = article.get("published_parsed")
                    if published_parsed:
                        timestamp = time.mktime(published_parsed)
                        date_str = time.strftime("%Y-%m-%d %H:%M:%S", published_parsed)
                    else:
                        timestamp = time.time()  # fallback
                        date_str = published
                    articles.append({
                        "title": article.get("title", "Titre non disponible"),
                        "link": article.get("link", "#"),
                        "published": date_str,
                        "timestamp": timestamp,
                        "feed_url": feed_url
                    })

        articles = sorted(articles, key=lambda x: x["timestamp"], reverse=True)

        timeline_data = [
            {
                "label": f"{article['title']} - {get_origin({'link': article['feed_url']})[0]}",
                "date": article["published"],
                "url": article["link"]
            }
            for article in articles
        ]

        if not timeline_data:
            st.info("Aucun article avec date de publication n'est disponible.")
        else:
            timeline_json = json.dumps(timeline_data)
            HEIGHT = 300

            # IMPORTANT: doubles accolades {{ }} pour le code JS, simple {timeline_json} pour la variable Python
            d3_timeline = f"""
            <!DOCTYPE html>
            <html>
            <head>
              <meta charset="utf-8">
              <style>
                body {{ margin: 20px; background: #f7f7f7; font: 12px sans-serif; }}
                svg {{ width: 100%; max-width: 900px; height: {HEIGHT}px; background: #fff; border: 1px solid #ddd; }}
                .event circle {{ fill: #007bff; cursor: pointer; }}
                .event text {{ fill: #000; font-size: 12px; text-anchor: middle; }}
                .axis {{ stroke: #000; stroke-width: 1; }}
                .event:hover circle {{ fill: #0056b3; }}
                .event:hover text {{ font-weight: bold; fill: #0056b3; }}
              </style>
            </head>
            <body>
              <svg></svg>
              <script src="https://d3js.org/d3.v7.min.js"></script>
              <script>
                const data = {timeline_json};

                const svg = d3.select("svg"),
                      width = svg.node().clientWidth || 900,
                      height = {HEIGHT};

                const parseTime = d3.timeParse("%Y-%m-%d %H:%M:%S");
                data.forEach(d => {{
                  d.date = parseTime(d.date);
                }});

                const x = d3.scaleTime()
                    .domain(d3.extent(data, d => d.date))
                    .range([50, width - 50]);

                svg.append("line")
                   .attr("class", "axis")
                   .attr("x1", 50)
                   .attr("x2", width - 50)
                   .attr("y1", height / 2)
                   .attr("y2", height / 2)
                   .attr("stroke-width", 2);

                svg.append("g")
                   .attr("class", "axis")
                   .attr("transform", `translate(0, ${height / 2 + 10})`)
                   .call(d3.axisBottom(x).ticks(5));

                const events = svg.selectAll(".event")
                  .data(data)
                  .join("g")
                  .attr("class", "event")
                  .attr("transform", d => `translate(${{x(d.date)}}, ${height / 2})`)
                  .on("click", (event, d) => {{
                    if (d.url) {{
                      window.open(d.url, "_blank");
                    }}
                  }})
                  .on("mouseover", function(event, d) {{
                    d3.select(this).select("circle").attr("fill", "#0056b3");
                    d3.select(this).select("text").style("font-weight", "bold").style("fill", "#0056b3");
                  }})
                  .on("mouseout", function(event, d) {{
                    d3.select(this).select("circle").attr("fill", "#007bff");
                    d3.select(this).select("text").style("font-weight", "normal").style("fill", "#000");
                  }});

                events.append("circle")
                  .attr("r", 8);

                events.append("text")
                  .attr("y", -15)
                  .text(d => d.label.length > 30 ? d.label.substring(0, 30) + "..." : d.label)
                  .style("font-size", "12px");

                events.append("title")
                  .text(d => `Article: ${{d.label}}\\nDate: ${{d3.timeFormat("%Y-%m-%d %H:%M:%S")(d.date)}}`);
              </script>
            </body>
            </html>
            """

            components.html(d3_timeline, height=HEIGHT)
