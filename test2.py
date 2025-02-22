import streamlit as st
import streamlit.components.v1 as components
import feedparser
import requests
import time
import os
import json
import re
from urllib.parse import urlparse

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
        # On ne garde que des types simples (str, bool, None, listes de dictionnaires simples)
        # Pour published_parsed, on le transforme en tuple (il est normalement picklable, mais on peut aussi le convertir)
        pp = entry.get("published_parsed")
        if pp:
            pp = tuple(pp)  # convertir en tuple

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
    st.markdown("""
        <script>
        setTimeout(function() {
            var element = document.getElementById('lecteur');
            if(element) { element.scrollIntoView({ behavior: 'smooth' }); }
        }, 100);
        </script>
        """, unsafe_allow_html=True)
    st.title("Lecteur de flux RSS/Atom")
    st.markdown("Ce module agrège les flux enregistrés et propose plusieurs options d'affichage.")
    feeds = load_feeds()
    if not feeds:
        st.error("Aucun flux n'est disponible. Veuillez ajouter des flux dans 'Gestionnaire de flux'.")
    else:
        with st.expander("Configuration d'affichage", expanded=False):
            nb_articles = st.number_input("Nombre d'articles à afficher :", min_value=1, max_value=1000, value=50, step=10)
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
            articles = [entry for entry in articles if search_keyword.lower() in entry.get("title", "").lower()]
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
                                { f"<img src='{image_url}' style='width:100%; max-height:200px; object-fit:cover; overflow:hidden; border-radius:5px; margin:0 0 3px 0;'/>" if image_url else "<div style='width:100%; height:200px; background-color:#444; display:flex; align-items:center; justify-content:center; color:white; font-size:14px; border-radius:5px;'>Pas d'image</div>" }
                                <p style="margin:0; padding:0; font-size:0.85rem;">
                                    📌 <img src="{favicon_url}" width="14" style="vertical-align:middle;"> {domain}
                                </p>
                                <hr style="margin:6px 0 0 0; padding:0; border:none; border-top:1px solid #444;">
                                </div>
                                """
                                st.markdown(article_html, unsafe_allow_html=True)

# --------------------------------------------------------------------
#                        PAGE NODES, MAP, CAMEMBERT, etc.
# --------------------------------------------------------------------
# (Les autres pages restent inchangées)
# ... [code pour Nodes, Map monde, Camembert, Bubble Chart, Timeline] ...


# --------------------------------------------------------------------
#                        PAGE NODES
# --------------------------------------------------------------------
elif page == "🔗Nodes":
    st.title("Nodes")
    st.markdown("Visualisation interactive d'un réseau d'entités avec des données significatives.")
    d3_nodes = """
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body { margin: 0; background: #f7f7f7; }
        svg { width: 800px; height: 600px; border: 1px solid #e6e6e6; background: #fff; }
        text { font: 12px sans-serif; pointer-events: none; fill: #333; }
      </style>
    </head>
    <body>
      <svg></svg>
      <script src="https://d3js.org/d3.v7.min.js"></script>
      <script>
        const nodes = [
          {id: "Hôpital"},
          {id: "Clinique"},
          {id: "Laboratoire"},
          {id: "Pharmacie"},
          {id: "Fournisseur"},
          {id: "Médecin"}
        ];
        const links = [
          {source: "Hôpital", target: "Clinique"},
          {source: "Hôpital", target: "Laboratoire"},
          {source: "Clinique", target: "Pharmacie"},
          {source: "Laboratoire", target: "Fournisseur"},
          {source: "Pharmacie", target: "Médecin"},
          {source: "Hôpital", target: "Médecin"}
        ];
        const svg = d3.select("svg");
        const width = +svg.attr("width") || 800;
        const height = +svg.attr("height") || 600;
        const simulation = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(links).id(d => d.id).distance(150))
            .force("charge", d3.forceManyBody().strength(-400))
            .force("center", d3.forceCenter(width / 2, height / 2));

        const link = svg.append("g")
            .attr("stroke", "#aaa")
            .attr("stroke-opacity", 0.8)
          .selectAll("line")
          .data(links)
          .join("line")
            .attr("stroke-width", 2);

        const node = svg.append("g")
            .attr("stroke", "#fff")
            .attr("stroke-width", 1.5)
          .selectAll("circle")
          .data(nodes)
          .join("circle")
            .attr("r", 15)
            .attr("fill", "#007bff")
            .call(drag(simulation));

        const label = svg.append("g")
            .selectAll("text")
            .data(nodes)
            .join("text")
            .text(d => d.id)
            .attr("x", 18)
            .attr("y", 5);

        simulation.on("tick", () => {
          link
              .attr("x1", d => d.source.x)
              .attr("y1", d => d.source.y)
              .attr("x2", d => d.target.x)
              .attr("y2", d => d.target.y);
          node
              .attr("cx", d => d.x)
              .attr("cy", d => d.y);
          label
              .attr("x", d => d.x)
              .attr("y", d => d.y);
        });

        function drag(simulation) {
          function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          }
          function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
          }
          function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          }
          return d3.drag()
              .on("start", dragstarted)
              .on("drag", dragged)
              .on("end", dragended);
        }
      </script>
    </body>
    </html>
    """
    components.html(d3_nodes, height=620)

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
        body { margin: 0; background: #f7f7f7; }
        svg { width: 960px; height: 600px; background: #fff; }
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
    st.title("Timeline")
    st.markdown("Chronologie interactive d'événements marquants dans le secteur.")
    d3_timeline = """
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        body { margin: 20px; background: #f7f7f7; font: 12px sans-serif; }
        svg { width: 900px; height: 250px; background: #fff; }
        .event circle { fill: #007bff; }
        .event text { fill: #000; font-size: 10px; text-anchor: middle; }
        .axis { stroke: #000; }
      </style>
    </head>
    <body>
      <svg></svg>
      <script src="https://d3js.org/d3.v7.min.js"></script>
      <script>
        const events = [
          {label: "Lancement Produit", date: new Date("2024-06-01")},
          {label: "Mise à jour Majeure", date: new Date("2024-09-15")},
          {label: "Partenariat Stratégique", date: new Date("2025-01-10")},
          {label: "Acquisition", date: new Date("2025-04-20")}
        ];

        const svg = d3.select("svg"),
              width = +svg.attr("width") || 900,
              height = +svg.attr("height") || 250,
              margin = {left: 50, right: 50, top: 20, bottom: 20};

        const x = d3.scaleTime()
            .domain(d3.extent(events, d => d.date))
            .range([margin.left, width - margin.right]);

        // ligne "timeline"
        svg.append("line")
           .attr("class", "axis")
           .attr("x1", margin.left)
           .attr("x2", width - margin.right)
           .attr("y1", height/2)
           .attr("y2", height/2)
           .attr("stroke-width", 2);

        // points + labels
        svg.selectAll("g.event")
          .data(events)
          .join("g")
          .attr("class", "event")
          .attr("transform", d => `translate(${x(d.date)}, ${height/2})`)
          .each(function(d) {
            const g = d3.select(this);
            g.append("circle").attr("r", 12);
            g.append("text").attr("y", -20).text(d.label);
          });
      </script>
    </body>
    </html>
    """
    components.html(d3_timeline, height=260)
