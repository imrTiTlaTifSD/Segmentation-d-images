# 🧩 Segmentation d’images — Clustering non supervisé (Streamlit)

Projet Machine Learning (Ynov) : segmentation d’images par **clustering** (non supervisé).  
Chaque pixel est représenté par des features (RGB et optionnellement position x,y), puis assigné à un cluster.

## 🎯 Objectifs
- Segmentation d’image **non supervisée**
- Comparer **au moins 3 modèles**
- Fournir une application **Streamlit** interactive

## 🧠 Modèles
- **KMeans** : minimise la distance aux centroïdes
- **Gaussian Mixture (GMM)** : clustering probabiliste (assignation par probabilité)
- **Agglomerative Clustering** : clustering hiérarchique (linkage)

## 🧩 Features
- **RGB** (normalisé)
- Option **(x,y)** : position des pixels (améliore souvent la cohérence spatiale)

## 🚀 Lancer l’application

### 1) Créer et activer l’environnement
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
### 2) Installer les dépendances
pip install -r requirements.txt

### 3) Lancer Streamlit
streamlit run app.py

📁 Structure
projet_segmentation/
├─ app.py
├─ requirements.txt
├─ README.md
├─ .gitignore
└─ src/
   ├─ kmeans_segmentation.py
   ├─ gmm_segmentation.py
   └─ agglomerative_segmentation.py
