import streamlit as st
from teachable_machine import TeachableMachine
from PIL import Image
import time

st.set_page_config(page_title="Station de Diagnostic - Metoushelah", page_icon="⚙️", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }
    .stButton>button {
        border-radius: 5px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2043/2043331.png", width=70)
    
    st.divider()
    
    st.markdown("📧 **Contact Professionnel**")
    st.markdown("[metoushelahmetoushelah@gmail.com](mailto:metoushelahmetoushelah@gmail.com)")
    
    st.divider()
    
    st.markdown("🖥️ **Statut du Système**")
    st.success("🟢 Moteur d'inférence IA : En ligne")
    st.caption("Version du logiciel : v1.0.0")
    
    st.divider()
    
    with st.expander("📋 Spécifications Techniques"):
        st.write("- **Modèle :** CNN (Keras/TensorFlow)")
        st.write("- **Entrée :** Image 224x224 px")
        st.write("- **Classes :** 5 signatures de combustion")
        st.write("- **Année :** 2026")

st.title("⚙️ Station de Diagnostic Électromécanique")
st.markdown("Analyse des signatures visuelles des bougies d'allumage par intelligence artificielle.")
st.divider()

@st.cache_resource
def load_model():
    return TeachableMachine(model_path="keras_model.h5", model_type='h5', labels_file_path="labels.txt")

model = load_model()

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.subheader("📷 Module d'Acquisition")
    uploaded_file = st.file_uploader("Insérez l'image de la bougie pour analyse...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Échantillon en cours de traitement", use_container_width=True)

with col2:
    st.subheader("📊 Rapport de Diagnostic")
    
    if uploaded_file is not None:
        if st.button("⚡ Lancer l'analyse technique", type="primary", use_container_width=True):
            
            temp_path = "temp_image.jpg"
            image.save(temp_path)
            
            with st.spinner("Traitement des données de combustion..."):
                time.sleep(1.5)
                try:
                    result = model.classify_image(image_path=temp_path)
                    class_name = result["highest_class_name"].lower()
                    confidence = result["highest_class_confidence"]
                    
                    st.metric(label="Indice de confiance du modèle", value=f"{confidence * 100:.1f} %")
                    
                    tab1, tab2 = st.tabs(["🔍 Diagnostic", "🛠️ Plan de Maintenance"])
                    
                    rapport_texte = ""
                    
                    if "normal" in class_name:
                        st.toast("✅ Moteur Sain - Combustion optimale", icon="⚙️")
                        with tab1:
                            st.success("**État de la bougie : NORMALE**")
                            st.write("La couleur de l'isolant indique une combustion complète et équilibrée.")
                        with tab2:
                            st.info("**Action :** Aucune intervention requise. Poursuivre la maintenance standard.")
                        rapport_texte = "Diagnostic: Normal\nAction: Aucune"
                            
                    elif "riche" in class_name or "carbon" in class_name or "fouled" in class_name:
                        st.toast("⚠️ Alerte : Anomalie détectée", icon="🚨")
                        with tab1:
                            st.warning("**État de la bougie : MÉLANGE RICHE (Encrassée)**")
                            st.error("**Causes probables :**\n- Filtre à air colmaté\n- Injecteur qui fuit\n- Sonde lambda défaillante")
                        with tab2:
                            st.write("**Actions correctives :**\n1. Contrôler le filtre à air.\n2. Tester les injecteurs.\n3. Diagnostiquer la sonde lambda.")
                        rapport_texte = "Diagnostic: Mélange Riche\nAction: Contrôle admission/injection"
                            
                    elif "huile" in class_name or "oil" in class_name:
                        st.toast("🚨 Danger : Usure interne", icon="⚠️")
                        with tab1:
                            st.error("**État de la bougie : HUILEUSE**")
                            st.error("**Causes probables :**\n- Usure des segments de piston\n- Guides de soupapes usés")
                        with tab2:
                            st.write("**Actions correctives :**\n1. Test de compression.\n2. Inspection endoscopique.\n3. Révision mécanique.")
                        rapport_texte = "Diagnostic: Huileuse\nAction: Test compression"
                            
                    elif "pauvre" in class_name or "lean" in class_name:
                        st.toast("⚠️ Alerte : Surchauffe", icon="🌡️")
                        with tab1:
                            st.warning("**État de la bougie : MÉLANGE PAUVRE**")
                            st.error("**Causes probables :**\n- Prise d'air\n- Injecteur bouché\n- Pompe faible")
                        with tab2:
                            st.write("**Actions correctives :**\n1. Test d'étanchéité.\n2. Contrôle pression carburant.\n3. Vérifier l'allumage.")
                        rapport_texte = "Diagnostic: Mélange Pauvre\nAction: Contrôle prise d'air"
                            
                    elif "endommag" in class_name or "damage" in class_name:
                        st.toast("🚨 Critique : Endommagée", icon="❌")
                        with tab1:
                            st.error("**État de la bougie : ENDOMMAGÉE / FONDUE**")
                            st.error("**Causes probables :**\n- Surchauffe sévère\n- Indice thermique inadapté")
                        with tab2:
                            st.write("**Actions correctives :**\n1. Remplacer la bougie.\n2. Vérifier le calage.\n3. Contrôler le carburant.")
                        rapport_texte = "Diagnostic: Endommagée\nAction: Remplacement"
                    
                    else:
                        st.warning(f"**État détecté : {class_name}**")
                        st.write("Anomalie non répertoriée.")
                        rapport_texte = f"Diagnostic: {class_name}\nAction: A verifier"
                    
                    st.divider()
                    
                    st.download_button(
                        label="📥 Télécharger le rapport de diagnostic",
                        data=rapport_texte,
                        file_name="rapport_diagnostic.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                        
                    with st.expander("📈 Données brutes du réseau de neurones"):
                        st.write(result["all_predictions"])
                        
                except Exception as e:
                    st.error(f"Erreur système : {e}")
    else:
        st.info("En attente d'une image pour lancer l'algorithme de diagnostic...")