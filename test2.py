import streamlit as st
import streamlit.components.v1 as components
import feedparser
import time
import os
import json
from urllib.parse import urlparse

# --- Fonctions de gestion du fichier JSON des flux ---
FEEDS_FILE = "feeds.json"

def load_feeds():
    if not os.path.exists(FEEDS_FILE):
        # Initialisation avec un flux par défaut
        default_feeds = [{"url": "https://www.lemonde.fr/rss/une.xml"}]
        with open(FEEDS_FILE, "w") as f:
            json.dump(default_feeds, f)
        return default_feeds
    with open(FEEDS_FILE, "r") as f:
        return json.load(f)

def save_feeds(feeds):
    with open(FEEDS_FILE, "w") as f:
        json.dump(feeds, f)

# --- Configuration de la page ---
st.set_page_config(page_title="KYRRIA App Demo", layout="wide")

# --- Définition des pages disponibles ---
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

# Initialisation de la page sélectionnée dans session_state
if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Home"

# --- CSS personnalisé pour la sidebar ---
st.markdown(
    """
    <style>
      html, body {
          margin: 0;
          padding: 0;
          font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
          background-color: #f7f7f7;
          color: #333;
      }
      .reportview-container { padding: 1rem; }
      .sidebar-header {
          text-align: center;
          margin-bottom: 2rem;
      }
      .sidebar-header h2 {
          font-size: 1.5rem;
          margin: 0;
          color: #007bff;
      }
      [data-testid="stSidebar"] .stButton button {
          background-color: #007bff;
          color: #fff;
          border: none;
          padding: 0.8rem 1rem;
          margin-bottom: 0.5rem;
          border-radius: 4px;
          width: 100%;
          text-align: left;
      }
      [data-testid="stSidebar"] .stButton button:hover {
          background-color: #0056b3;
      }
      [data-testid="stSidebar"] .stButton button:disabled {
          background-color: #007bff;
          color: #fff;
          opacity: 1;
          cursor: default;
      }
      .block-container { padding: 2rem; }
     div.stButton > button {
        width: 100%;
        height: 50px;
        font-size: 18px;
        margin-bottom: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Sidebar de navigation ---
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

# --- Détermination du contenu en fonction de la page sélectionnée ---
page = st.session_state.selected_page

if page == "🏠Home":
    st.title("Bienvenue sur KYRRIA App Demo")
    st.markdown("""
    **KYRRIA App Demo** est une application de démonstration présentant diverses visualisations interactives réalisées avec D3.js.  
    Utilisez la barre latérale pour naviguer entre les sections :
    
    - **Gestionnaire de flux** : Ajoutez et supprimez vos flux RSS (les modifications sont enregistrées dans un fichier JSON).
    - **Lecteur RSS** : Agrégation complète des flux RSS avec options d’affichage.
    - **Nodes** : Visualisation interactive d'un réseau d'entités.
    - **Map monde** : Carte interactive du monde avec indicateurs.
    - **Camembert** : Répartition d'une donnée catégorielle.
    - **Bubble Chart** : Diagramme à bulles illustrant des métriques.
    - **Timeline** : Chronologie d'événements marquants.
    """)

elif page == "📡Gestionnaire de flux":
    st.title("Gestionnaire de flux RSS")
    st.markdown("Gérez vos liens RSS. Les flux sont enregistrés dans le fichier `feeds.json`.")

    feeds = load_feeds()

    with st.form("flux_form"):
        new_flux = st.text_input("URL du flux RSS", placeholder="https://exemple.com/rss")
        submit_flux = st.form_submit_button("Ajouter")
        if submit_flux:
            if new_flux and new_flux not in [feed["url"] for feed in feeds]:
                feeds.append({"url": new_flux})
                save_feeds(feeds)
                st.success(f"Flux ajouté : {new_flux}")
                st.experimental_rerun()
            elif new_flux:
                st.warning("Ce flux est déjà présent.")
            else:
                st.error("Veuillez entrer une URL valide.")

    st.markdown("### Flux RSS enregistrés")
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
                st.experimental_rerun()
    else:
        st.info("Aucun flux RSS enregistré.")

elif page == "📰Lecteur RSS":
    # Insertion d'une ancre pour le scroll
    st.markdown("<div id='lecteur'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <script>
        setTimeout(function() {
            var element = document.getElementById('lecteur');
            if(element) {
                element.scrollIntoView({ behavior: 'smooth' });
            }
        }, 100);
        </script>
        """,
        unsafe_allow_html=True
    )
    st.title("Lecteur de flux RSS")
    st.markdown("Ce module agrège les flux RSS enregistrés (via le Gestionnaire de flux) et propose différentes options d'affichage.")

    feeds = load_feeds()
    if not feeds:
        st.error("Aucun flux n'est disponible. Veuillez ajouter des flux dans la page 'Gestionnaire de flux'.")
    else:
        # --- Configuration d'affichage dans un expander ---
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
 # --- Sélection des flux à afficher ---
        selected_feeds = []
        st.markdown("### Sélectionner les flux à afficher")
        for feed in feeds:
            parsed = urlparse(feed["url"])
            feed_meta = feedparser.parse(feed["url"]).feed
            full_title = feed_meta.get("title", parsed.netloc.replace("www.", "")).strip()
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
                if st.checkbox("", key=feed["url"], value=True):
                    selected_feeds.append(feed["url"])

        # --- CHARGE LES ARTICLES AVANT D'AFFICHER LE COMPTEUR ---
        @st.cache_data(show_spinner=False)
        def charger_flux(url):
            return feedparser.parse(url)

        # Initialisation de la liste des articles
        articles = []
        for feed_url in selected_feeds:
            flux = charger_flux(feed_url)
            if flux.bozo:
                st.error(f"Erreur lors du chargement du flux : {feed_url}")
            else:
                for entry in flux.entries:
                    entry["feed_url"] = feed_url
                    articles.append(entry)

        # --- Affichage du nombre de posts disponibles (AJOUTÉ APRÈS LE CHARGEMENT) ---
        st.markdown(f"**Nombre de posts disponibles :** {len(articles)} / {nb_articles} affichés")

        # --- Chargement et agrégation des flux ---
        @st.cache_data(show_spinner=False)
        def charger_flux(url):
            return feedparser.parse(url)

        articles = []
        for feed_url in selected_feeds:
            flux = charger_flux(feed_url)
            if flux.bozo:
                st.error(f"Erreur lors du chargement du flux : {feed_url}")
            else:
                for entry in flux.entries:
                    entry["feed_url"] = feed_url
                    articles.append(entry)
                    if len(articles) >= 1000:  # Stop collecting after 1000 articles
                        break

        # Filtrage par mot-clé (insensible à la casse)
        if search_keyword:
            articles = [entry for entry in articles if search_keyword.lower() in entry.get("title", "").lower()]

        # Tri des articles par date (les plus récents en premier)
        def get_timestamp(entry):
            if "published_parsed" in entry and entry.published_parsed:
                return time.mktime(entry.published_parsed)
            return 0
        articles = sorted(articles, key=get_timestamp, reverse=True)[:nb_articles]

        st.header("Flux RSS Agrégés")
        def extraire_image(entry):
            image_url = None
            if "media_thumbnail" in entry:
                image_url = entry.media_thumbnail[0]["url"]
            elif "media_content" in entry:
                image_url = entry.media_content[0]["url"]
            return image_url

        def get_origin(entry):
            link = entry.get("link", "")
            parsed = urlparse(link)
            domain = parsed.netloc.replace("www.", "")
            favicon_url = f"https://www.google.com/s2/favicons?domain={parsed.netloc}"
            return domain, favicon_url

        if not articles:
            st.info("Aucun article ne correspond aux critères.")
        else:
            if view_mode == "Liste détaillée":
                for entry in articles:
                    title = entry.get("title", "Titre non disponible")
                    link = entry.get("link", "#")
                    st.markdown(f"### [{title}]({link})")  # Titre cliquable
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

                                # Build the HTML for the article
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

                                {
                                    f"<img src='{image_url}' style='width:100%; max-height:200px; object-fit:cover; overflow:hidden; border-radius:5px; margin:0 0 3px 0;'/>"
                                    if image_url else
                                    "<div style='width:100%; height:200px; background-color:#444; display:flex; align-items:center; justify-content:center; color:white; font-size:14px; border-radius:5px;'>Pas d'image</div>"
                                }


                                <p style="margin:0; padding:0; font-size:0.85rem;">
                                    📌 <img src="{favicon_url}" width="14" style="vertical-align:middle;"> {domain}
                                </p>
                                <hr style="margin:6px 0 0 0; padding:0; border:none; border-top:1px solid #444;">
                                </div>
                                """
                                
                                st.markdown(article_html, unsafe_allow_html=True)

# --- Visualisations D3.js ---
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
        const pack = d3.pack().size([width, height]).padding(5);
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
        svg.append("line")
           .attr("class", "axis")
           .attr("x1", margin.left)
           .attr("x2", width - margin.right)
           .attr("y1", height/2)
           .attr("y2", height/2)
           .attr("stroke-width", 2);
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
