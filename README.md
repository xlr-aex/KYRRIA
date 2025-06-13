````markdown
# 🚀 KYRRIA App Demo

> **Un lecteur RSS avancé & explorateur de graphes interactif**

---

## 📝 Introduction

KYRRIA est une application **open-source** pensée pour les utilisateurs exigeants :  
- **Agrégation** et **gestion** de flux RSS/Atom via une interface claire.  
- **Lecture** enrichie avec filtrage, pagination et mise en cache.  
- **Stockage** local des articles (« Enregistrer en DB ») pour analyse ultérieure.  
- **Visualisations** variées : timeline, barres, treemap, nuage de mots…  
- **Graphes D3.js** pour explorer relations et entités extraites (via Google Gemini).  

Disponible en local avec Python 3.12, ou immédiatement en cloud :  
➡️ **https://kyrria.streamlit.app**

---

## 🌐 Version Cloud

Testez sans installation :  
➡️ **https://kyrria.streamlit.app**

---

## 📋 Prérequis

- **Python 3.12.x** (strictement)  
- **pip ≥ 22.0**  
- **Git** (optionnel)

Vérifiez vos versions :

<details>
<summary>Windows (PowerShell)</summary>

```powershell
python --version      # → Python 3.12.x
pip --version         # → pip ≥ 22.0
````

</details>

<details>
<summary>Linux / macOS</summary>

```bash
python3 --version     # → Python 3.12.x
pip3 --version        # → pip ≥ 22.0
```

</details>

---

## 🛠️ Installation & Lancement

### 1. Récupérer le code

```bash
git clone https://…/KYRRIA_final.git
cd KYRRIA_final
```

> Si vous n’avez pas Git, téléchargez l’archive ZIP et décompressez-la.

---

### 2. Créer et activer l’environnement virtuel

| Plate-forme              | Commandes                                                            |
| ------------------------ | -------------------------------------------------------------------- |
| **Windows – PowerShell** | `powershell<br>python -m venv .venv<br>. .venv\Scripts\Activate.ps1` |
| **Windows – CMD**        | `cmd<br>python -m venv .venv<br>.venv\Scripts\activate.bat`          |
| **Linux & macOS**        | `bash<br>python3 -m venv .venv<br>source .venv/bin/activate`         |

> Un préfixe `(.venv)` doit apparaître dans votre invite.

---

### 3. Mettre à jour pip & installer les dépendances

```bash
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
```

> **Astuce** :
>
> * En cas de conflit (numpy, matplotlib…), assurez-vous d’être sous Python 3.12.
> * Pour repartir de zéro, supprimez et recréez le venv :
>
>   ```bash
>   deactivate  
>   rm -rf .venv  
>   python3.12 -m venv .venv  
>   source .venv/bin/activate  
>   pip install -r requirements.txt
>   ```

---

### 4. Lancer l’application

```bash
streamlit run Kyrria.py
```

Ouvrez votre navigateur à l’URL indiquée (par défaut → `http://localhost:8501`).

---

## 📖 Usage rapide

1. **Inscription / Connexion**
2. **Gestionnaire de flux** → ajoutez ou supprimez vos RSS/Atom.
3. **Lecteur RSS** → lisez, filtrez et enregistrez vos articles.
4. **Dashboard** → visualisez statistiques & graphiques.
5. **Nodes & Entities & Relations** → explorez relations et entités extraites via D3.js.

---

## 🆘 Support & Debug

* **Logs** : activez le log dans `Kyrria.py`

  ```python
  import logging
  logging.basicConfig(level=logging.DEBUG, filename='kyrria.log')
  ```
* **Problèmes d’installation** :

  1. Vérifiez la version de Python.
  2. Supprimez et recréez le venv.
  3. Ouvrez une issue sur GitHub.

---

> **BON TEST !** 🚀
> KYRRIA – votre compagnon RSS & data-viz.
> **Cloud** : [https://kyrria.streamlit.app](https://kyrria.streamlit.app)

```
```
