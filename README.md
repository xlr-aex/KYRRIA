# 🚀 KYRRIA App Demo

> **Un lecteur RSS avancé & explorateur de graphes interactifs**

---

## 📖 Préambule

KYRRIA est une application **open-source** développée en Python/Streamlit pour les utilisateurs exigeants :

- **Agrégation & gestion** de flux RSS/Atom via une interface intuitive  
- **Lecture** enrichie : filtrage, pagination, mise en cache  
- **Stockage local** des articles (« Enregistrer en DB ») pour analyse ultérieure  
- **Visualisations variées** : timeline, barres, treemap, nuage de mots…  
- **Graphes D3.js** pour explorer entités & relations extraites (via Google Gemini)  

Disponible en local (Python 3.12) et immédiatement en cloud sur :  
➡️ [https://kyrria.streamlit.app](https://kyrria.streamlit.app)  
Code source : [github.com/xlr-aex/KYRRIA](https://github.com/xlr-aex/KYRRIA)

---

## 🛠️ Prérequis

- **Python ≥ 3.12**  
- **Git**  
- Windows 10+ ou Linux/macOS

---

## ⚙️ Installation & exécution locale

1. **Cloner le repo** dans `D:\KYRRIA_final` (ou chemin de votre choix)  
   ```bash
   git clone https://github.com/xlr-aex/KYRRIA.git D:\KYRRIA_final
   cd D:\KYRRIA_final
````

2. **Ne pas versionner le venv**

   ```bash
   cat <<EOF >> .gitignore
   # Virtual environments
   .venv/
   venv/
   env/
   __pycache__/
   *.py[cod]
   EOF
   git add .gitignore
   git commit -m "🧹 Ajoute .gitignore pour ignorer le venv"
   ```

3. **Créer et activer l’environnement virtuel**

   ```bash
   python3 -m venv .venv
   # Sous Windows
   .\.venv\Scripts\activate
   # Sous Linux/macOS
   source .venv/bin/activate
   ```

4. **Mettre à jour pip & installer les dépendances**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Lancer l’application**

   ```bash
   streamlit run Kyrria.py
   ```

---

## 🌐 Déploiement Streamlit Cloud

Ce repo est connecté à Streamlit Cloud :

* URL : [kyrria.streamlit.app](https://kyrria.streamlit.app)
* Branche de production : `main`

---

## 🔄 Pour écraser le dépôt distant avec votre dossier local

> **Attention :** cette opération force la mise à jour du `main` sur GitHub.

```bash
cd D:\KYRRIA_final

# Initialiser Git si besoin
git init

# Définir l’origin et écraser
git remote remove origin 2>/dev/null || true
git remote add origin https://github.com/xlr-aex/KYRRIA.git

# Ajouter, committer et forcer le push
git add .
git commit -m "🔥 Mise à jour complète depuis local"
git push -u origin main --force
```

---

## 🎯 Usage rapide

* **🏠 Home** : présentation
* **📡 Gestionnaire de flux** : ajouter/supprimer vos RSS
* **📰 Lecteur RSS** : lire & enregistrer vos articles
* **🔗 Nodes** : graphe d’occurrences par mot-clé
* **📊 Dashboard** : graphiques barres, treemap, timeline…
* **💡 Entities & Relations** : extraction NER + graphe D3.js

---

1. Forkez ce dépôt
2. Créez une branche `feature/…`
3. Ouvrez un **Pull Request**
4. Nous validerons et fusionnerons

---
