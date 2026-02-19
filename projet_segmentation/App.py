import streamlit as st
from PIL import Image
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from kmeans_segmentation import segment_kmeans
from gmm_segmentation import segment_gmm
from agglomerative_segmentation import segment_agglomerative

import io

def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --- Page config ---
st.set_page_config(page_title="Projet Segmentation - Ynov", page_icon="🧩", layout="wide")

# --- Petite touche visuelle ---
st.markdown(
    """
    <style>
      .hero {
        padding: 18px 20px;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(16,185,129,0.12));
        border: 1px solid rgba(255,255,255,0.15);
        margin-bottom: 14px;
      }
      .chip {
        display:inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.18);
        background: rgba(255,255,255,0.06);
        margin-right: 8px;
        font-size: 0.9rem;
      }
      .panel {
        padding: 14px 16px;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.03);
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Header ---
st.markdown(
    """
    <div class="hero">
      <h1 style="margin:0;">🧩 Segmentation d’image — Projet Machine Learning (Ynov)</h1>
      <p style="margin:6px 0 0 0;">
        Objectif : regrouper les pixels en <b>clusters</b> (non supervisé) pour produire une image segmentée.
      </p>
      <div style="margin-top:10px;">
        <span class="chip">Non supervisé</span>
        <span class="chip">Clustering</span>
        <span class="chip">Pixels → features</span>
        <span class="chip">Streamlit</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar (plus “app”) ---
st.sidebar.title("⚙️ Configuration")
model_name = st.sidebar.selectbox("Modèle", ["KMeans", "Gaussian Mixture (GMM)", "Agglomerative (Hierarchical)"])
k = st.sidebar.slider("Nombre de clusters (k)", min_value=2, max_value=20, value=6, step=1)
use_xy = st.sidebar.checkbox("Ajouter la position (x, y)", value=True)
if model_name == "Agglomerative (Hierarchical)":
    linkage = st.sidebar.selectbox("Linkage (Agglo)", ["ward", "complete", "average", "single"])
else:
    linkage = "ward"

if model_name == "Agglomerative (Hierarchical)":
    max_pixels = st.sidebar.slider("Max pixels (Agglo)", 5000, 120000, 40000, step=5000)
else:
    max_pixels = 40000

st.sidebar.caption("On ajoutera ici les modèles (KMeans, GMM, Agglo) et leurs paramètres.")

with st.sidebar.expander("📚 Référence au cours (perso)", expanded=True):
    st.markdown(
        """
        Dans le cours **Machine Learning (Ynov)**, on a vu que :
        - en **clustering**, on cherche des groupes sans labels (non supervisé) ;
        - une image peut être vue comme une matrice de **pixels**, donc on peut faire du ML sur des features (RGB et éventuellement position x,y) ;
        - le paramètre clé est souvent le **nombre de clusters (k)**.
        """
    )

# --- Main layout ---
left, right = st.columns([1.05, 1])

with left:
    st.subheader("1) Importer une image")
    uploaded = st.file_uploader("PNG / JPG", type=["png", "jpg", "jpeg"])

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.write("✅ Astuce : commence avec une image pas trop grande (ex: 400–800 px de large) pour garder un calcul rapide.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("2) Lancer la segmentation")
    run = st.button("🚀 Segmenter", type="primary", use_container_width=True)


with right:
    st.subheader("3) Résultats")

    if uploaded:
        img = Image.open(uploaded).convert("RGB")

        colA, colB = st.columns(2)
        with colA:
            st.image(img, caption="Image d’entrée", use_container_width=True)


        if run:
            with st.spinner("Segmentation en cours…"):
                if model_name == "KMeans":
                    seg = segment_kmeans(img, n_clusters=k, use_xy=use_xy)
                elif model_name == "Gaussian Mixture (GMM)":
                    seg = segment_gmm(img, n_clusters=k, use_xy=use_xy)
                else:
                    seg = segment_agglomerative(img,n_clusters=k,use_xy=use_xy,linkage=linkage,max_pixels=max_pixels)



            with colB:
                st.image(seg, caption=f"Segmentée ({model_name}, k={k})", use_container_width=True)


            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.write(f"**Modèle utilisé :** {model_name}")
            st.write(f"**Nombre de clusters (k) :** {k}")
            st.write(f"**Features :** {'RGB + (x,y)' if use_xy else 'RGB uniquement'}")

            if model_name == "KMeans":
                st.write("**Lien avec le cours :** KMeans minimise la distance aux centroïdes (clusters plutôt ‘ronds’).")
            elif model_name == "Gaussian Mixture (GMM)":
                st.write("**Lien avec le cours :** GMM assigne par probabilité (clustering probabiliste, clusters elliptiques).")
            else:
                st.write(f"**Downscale :** max {max_pixels} pixels (pour éviter l’explosion mémoire)")

            st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.info("Régle les paramètres puis clique sur **Segmenter**.")

    else:
        st.info("Charge une image pour afficher l’aperçu.")

