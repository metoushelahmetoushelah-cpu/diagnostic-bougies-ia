import streamlit as st
from teachable_machine import TeachableMachine
from PIL import Image
import time
import cv2
import numpy as np

st.set_page_config(page_title="Station de Diagnostic - Metoushelah", page_icon="⚙️", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Roboto', sans-serif; }
    .stButton>button { border-radius: 5px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

connaissances_mecaniques = {
    "normal": {
        "titre": "✅ MOTEUR SAIN (Combustion optimale)",
        "causes": "Aucune anomalie. Le mélange air/essence est idéal et l'étincelle est parfaite.",
        "solutions": "Aucune intervention requise. Poursuivre l'entretien régulier.",
        "couleur": "success"
    },
    "riche": {
        "titre": "⚠️ MÉLANGE RICHE (Calaminée / Suie sèche)",
        "causes": "Excès de carburant ou manque d'air. Dépôts noirs secs (suie).",
        "solutions": "1. Remplacer le filtre à air.\n2. Nettoyer/Tester les injecteurs.\n3. Vérifier la sonde lambda.\n4. Contrôler le capteur de pression d'admission.",
        "couleur": "warning"
    },
    "huile": {
        "titre": "🚨 MOTEUR HUILEUX (Dépôts gras et humides)",
        "causes": "L'huile moteur pénètre dans la chambre de combustion et brûle avec l'essence.",
        "solutions": "1. Effectuer un test de compression (segments usés).\n2. Vérifier l'usure des guides de soupapes.\n3. Contrôler le système de ventilation du carter (PCV).\n4. Prévoir une révision mécanique lourde.",
        "couleur": "error"
    },
    "pauvre": {
        "titre": "⚠️ MÉLANGE PAUVRE (Surchauffe / Isolant blanc)",
        "causes": "Manque de carburant ou excès d'air. Combustion trop chaude.",
        "solutions": "1. Chercher une prise d'air (filtre, collecteur).\n2. Vérifier la pression de la pompe à carburant.\n3. Nettoyer les injecteurs bouchés.\n4. Vérifier le calage de l'allumage.",
        "couleur": "warning"
    },
    "endommag": {
        "titre": "❌ BOUGIE ENDOMMAGÉE / FONDUE",
        "causes": "Surchauffe extrême ou détonation. Indice thermique inadapté.",
        "solutions": "1. Remplacer impérativement la bougie par un modèle adapté.\n2. Vérifier l'avance à l'allumage.\n3. Contrôler le système de refroidissement.\n4. Vérifier la qualité du carburant.",
        "couleur": "error"
    }
}

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2043/2043331.png", width=70)
    st.divider()
    st.markdown("📧 **Contact Professionnel**")
    st.markdown("[metoushelahmetoushelah@gmail.com](mailto:metoushelahmetoushelah@gmail.com)")
    st.divider()
    st.markdown("🖥️ **Système Expert & IA**")
    st.success("🟢 Base de connaissances : Active")
    st.caption("Version du logiciel : v2.4 (Calibrée)")
    st.divider()

st.title("⚙️ Station de Diagnostic Électromécanique")
st.markdown("Analyse intelligente des bougies couplée aux manuels techniques d'atelier.")
st.divider()

@st.cache_resource
def load_model():
    return TeachableMachine(model_path="keras_model.h5", model_type='h5', labels_file_path="labels.txt")

model = load_model()

col1, col2 = st.columns([1, 1.3], gap="large")

with col1:
    st.subheader("📷 Module d'Acquisition")
    uploaded_file = st.file_uploader("Insérez l'image pour analyse...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Échantillon brut", use_container_width=True)

with col2:
    st.subheader("📊 Rapport de Diagnostic")
    if uploaded_file is not None:
        if st.button("⚡ Lancer l'analyse technique", type="primary", use_container_width=True):
            with st.spinner("Analyse visuelle et croisement avec les manuels..."):
                time.sleep(1)
                try:
                    img_array = np.array(image)
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    
                    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                    luminosite_moyenne = np.mean(gris)
                    
                    if luminosite_moyenne > 100:
                        final_img = img_bgr.copy()
                    else:
                        result_wb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
                        avg_a = np.average(result_wb[:, :, 1])
                        avg_b = np.average(result_wb[:, :, 2])
                        result_wb[:, :, 1] = result_wb[:, :, 1] - ((avg_a - 128) * (result_wb[:, :, 1] / 255.0) * 0.8)
                        result_wb[:, :, 2] = result_wb[:, :, 2] - ((avg_b - 128) * (result_wb[:, :, 2] / 255.0) * 0.8)
                        img_corrected = cv2.cvtColor(result_wb, cv2.COLOR_LAB2BGR)
                        
                        lab = cv2.cvtColor(img_corrected, cv2.COLOR_BGR2LAB)
                        l, a, b = cv2.split(lab)
                        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
                        cl = clahe.apply(l)
                        limg = cv2.merge((cl,a,b))
                        final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

                    final_img_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
                    final_pil = Image.fromarray(final_img_rgb)
                    
                    temp_path = "temp_image.jpg"
                    final_pil.save(temp_path)
                    
                    result = model.classify_image(image_path=temp_path)
                    class_name = result["highest_class_name"].lower()
                    confidence = float(result["highest_class_confidence"])
                    
                    gris_final = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
                    pixels_noirs = cv2.countNonZero(cv2.inRange(gris_final, 0, 70))
                    densite_noire = pixels_noirs / gris_final.size
                    
                    hsv_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2HSV)
                    saturation_moyenne = np.mean(hsv_img[:, :, 1])
                    
                    diagnostic_final = class_name
                    
                    # LE SEUIL EST PASSÉ À 0.03 (3%) ICI
                    if ("pauvre" in class_name or "normal" in class_name) and densite_noire > 0.03:
                        if saturation_moyenne > 60:
                            diagnostic_final = "huile"
                        else:
                            diagnostic_final = "riche"
                    elif "riche" in class_name and saturation_moyenne > 70:
                        diagnostic_final = "huile"
                    elif "huile" in class_name and saturation_moyenne < 30 and densite_noire > 0.05:
                        diagnostic_final = "riche"
                    
                    st.metric(label="Indice de confiance", value=f"{confidence * 100:.1f} %")
                    
                    with st.expander("👁️ Détails de l'analyse visuelle"):
                        st.image(final_pil, caption="Image optimisée par OpenCV", use_container_width=True)
                        st.write(f"**Densité de calamine (Noir) :** {densite_noire * 100:.1f} %")
                        st.write(f"**Indice de brillance/humidité :** {saturation_moyenne:.1f} (Élevé = Huileux)")
                    
                    if confidence < 0.60:
                        st.warning("⚠️ **Image non concluante.** L'angle ou la lumière ne permet pas un diagnostic sûr.")
                        st.info("Veuillez reprendre la photo en isolant bien la bougie sur fond neutre.")
                        st.stop()
                    
                    fiche = connaissances_mecaniques.get("normal", connaissances_mecaniques["normal"])
                    for key in connaissances_mecaniques.keys():
                        if key in diagnostic_final:
                            fiche = connaissances_mecaniques[key]
                            break
                            
                    tab1, tab2 = st.tabs(["🔍 Diagnostic", "🛠️ Manuel de Maintenance"])
                    
                    with tab1:
                        if fiche["couleur"] == "success":
                            st.success(f"**{fiche['titre']}**")
                        elif fiche["couleur"] == "warning":
                            st.warning(f"**{fiche['titre']}**")
                        else:
                            st.error(f"**{fiche['titre']}**")
                            
                        st.markdown("### 🔬 Causes identifiées :")
                        st.info(fiche["causes"])
                        
                    with tab2:
                        st.markdown("### 🔧 Protocole de réparation (Manuel) :")
                        st.write(fiche["solutions"])
                    
                    st.divider()
                    
                    rapport = f"RAPPORT DE DIAGNOSTIC\n\nÉtat: {fiche['titre']}\nConfiance: {confidence*100:.1f}%\n\nCauses: {fiche['causes']}\n\nSolutions:\n{fiche['solutions']}"
                    st.download_button("📥 Télécharger le rapport technique", data=rapport, file_name="rapport_diagnostic.txt")
                    
                except Exception as e:
                    st.error(f"Erreur système : {e}")
    else:
        st.info("En attente d'une image pour lancer l'algorithme de diagnostic...")
