# 🚀 KYRRIA App Demo

> **Plateforme de Veille Informationnelle et d'Analyse Relationnelle**

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
➡️ [github.com/xlr-aex/KYRRIA](https://github.com/xlr-aex/KYRRIA)

## 🎯 Usage rapide

* **🏠 Home** : présentation
* **📡 Gestionnaire de flux** : ajouter/supprimer vos RSS
* **📰 Lecteur RSS** : lire & enregistrer vos articles
* **🔗 Nodes** : graphe d’occurrences par mot-clé
* **📊 Dashboard** : graphiques barres, treemap, timeline…
* **💡 Entities & Relations** : extraction NER + graphe D3.js

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
   ```

2. **Créer et activer l’environnement virtuel**

   ```bash
   # Créer l'environnement virtuel
   python3 -m venv .venv
   ```
   ```bash
   # Sous Windows
   .\.venv\Scripts\activate
   # Sous Linux/macOS
   source .venv/bin/activate
   ```

3. **Mettre à jour pip & installer les dépendances**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Lancer l’application**

   ```bash
   streamlit run Kyrria.py
   ```

---

## 🌐 Déploiement Streamlit Cloud

Ce repo est connecté à Streamlit Cloud :

* URL : [kyrria.streamlit.app](https://kyrria.streamlit.app)
---
Licence



© 2025 xlr-eax

Permission is hereby granted, free of charge, to any person obtaining a copyof this software and associated documentation files (the "Software"), to dealin the Software without restriction, including without limitation the rightsto use, copy, modify, merge, publish, distribute, sublicense, and/or sellcopies of the Software, and to permit persons to whom the Software isfurnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included inall copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS ORIMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THEAUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHERLIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS INTHE SOFTWARE.


