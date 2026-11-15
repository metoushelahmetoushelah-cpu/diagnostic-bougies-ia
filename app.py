import streamlit as st
from PIL import Image
import base64
import io
import json
import os
from fpdf import FPDF
from datetime import datetime
from openai import OpenAI

st.set_page_config(page_title="Station de Diagnostic - Metoushelah", page_icon="⚙️", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Roboto', sans-serif; }
    .stButton>button { border-radius: 5px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

if "OPENROUTER_API_KEY" in st.secrets:
    api_key = st.secrets["OPENROUTER_API_KEY"]
else:
    st.error("⚠️ Clé API manquante. Veuillez configurer le fichier secrets.toml ou les Secrets Streamlit.")
    st.stop()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

if "diagnostic_texte" not in st.session_state:
    st.session_state.diagnostic_texte = None

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2043/2043331.png", width=70)
    st.divider()
    st.markdown("📧 **Contact Professionnel**")
    st.markdown("[metoushelahmetoushelah@gmail.com](mailto:metoushelahmetoushelah@gmail.com)")
    st.divider()
    st.markdown("🖥️ **Architecture du Système**")
    st.success("🟢 Module d'Analyse Avancée : Actif")
    st.info("⚙️ Moteur d'inférence : En ligne")
    st.caption("Version du logiciel : v12.1 (Finale)")
    st.divider()
    if st.button("🔄 Réinitialiser l'application", use_container_width=True):
        st.session_state.diagnostic_texte = None
        st.rerun()

st.title("⚙️ Station de Diagnostic Multicritères")
st.markdown("Spécialisé pour les moteurs à allumage commandé (Essence).")
st.divider()

tab1, tab2 = st.tabs(["🔍 Nouveau Diagnostic", "📁 Historique des Interventions"])

def creer_pdf(texte):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, txt="Rapport de Diagnostic Mecanique", ln=True, align='C')
    pdf.cell(0, 10, txt=f"Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True, align='C')
    pdf.ln(10)
    
    for ligne in texte.split('\n'):
        ligne_encodee = ligne.encode('latin-1', 'replace').decode('latin-1')
        if ligne_encodee.strip() == "":
            pdf.ln(5)
        else:
            pdf.multi_cell(0, 10, txt=ligne_encodee)
            
    return bytes(pdf.output())

with tab1:
    col1, col2 = st.columns([1, 1.3], gap="large")

    with col1:
        st.subheader("📷 Module d'Acquisition")
        option_photo = st.radio("Méthode :", ["Uploader une image", "Utiliser la caméra"])
        
        image = None
        if option_photo == "Uploader une image":
            uploaded_file = st.file_uploader("Insérez l'image...", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
        else:
            camera_file = st.camera_input("Prenez une photo de la bougie")
            if camera_file is not None:
                image = Image.open(camera_file).convert("RGB")
                
        if image is not None:
            st.image(image, caption="Échantillon brut", use_container_width=True)

    with col2:
        st.subheader("🚗 Contexte Technique et Sensoriel")
        
        with st.form("formulaire_vehicule"):
            col_a, col_b = st.columns(2)
            with col_a:
                marque = st.text_input("Marque et Modèle", "Ex: Toyota Corolla")
                kilometrage = st.number_input("Kilométrage (km)", min_value=0, value=150000, step=10000)
            with col_b:
                code_obd = st.text_input("Code Défaut OBD-II (si disponible)", "Ex: P0300")
            
            st.markdown("**🖐️ Analyse Sensorielle (Crucial)**")
            
            etat_toucher = st.selectbox(
                "Au toucher, la bougie est :", 
                ["Sèche (mate ou avec suie)", "Humide d'essence", "Grasse (huile)"]
            )
            
            submit_button = st.form_submit_button("⚡ Lancer le Diagnostic Croisé", type="primary", use_container_width=True)

        if submit_button and image is not None:
            status_placeholder = st.empty()
            status_placeholder.info("⏳ Croisement des données en cours... Veuillez patienter.")
            
            try:
                image.thumbnail((500, 500))
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                prompt = f"""
                Tu es un ingénieur expert en électromécanique automobile. Analyse cette image de bougie d'allumage (moteur essence) en la croisant avec les données du véhicule et le retour sensoriel.
                
                DONNÉES DU VÉHICULE :
                - Véhicule : {marque}
                - Kilométrage : {kilometrage} km
                - Code OBD-II : {code_obd if code_obd else "Aucun"}
                
                RETOUR SENSORIEL :
                - État au toucher : {etat_toucher}
                
                RÈGLES STRICTES DE DIAGNOSTIC ET DE VOCABULAIRE :
                0. DÉCODAGE OBD : Si un code OBD est fourni, décode sa signification exacte et croise-le avec l'état de la bougie.
                1. Si "Grasse (huile)" : Le diagnostic DOIT être "Présence d'huile dans la chambre de combustion". Ne mentionne JAMAIS le mot "carburant" dans ce cas.
                2. Si "Sèche (mate ou avec suie)" et image noire : Le diagnostic DOIT être "Mélange trop riche (excès de carburant)".
                3. Si "Humide d'essence" : Le diagnostic DOIT être "Défaut d'injection ou d'allumage".
                
                Réponds EXACTEMENT dans ce format :
                DIAGNOSTIC: [Le nom de l'état et l'analyse croisée]
                CAUSES: [Explication des causes]
                SOLUTIONS: [Les 3 étapes de maintenance]
                """
                
                modeles = [
                    "google/gemma-4-26b-a4b-it:free",
                    "google/gemma-4-31b-it:free",
                    "nvidia/nemotron-nano-12b-v2-vl:free",
                    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
                ]
                
                succes = False
                texte_ia = ""
                
                for modele in modeles:
                    try:
                        response = client.chat.completions.create(
                            model=modele,
                            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}]}],
                            temperature=0.2,
                        )
                        texte_ia = response.choices[0].message.content
                        succes = True
                        break
                    except Exception:
                        continue
                
                status_placeholder.empty()
                
                if succes:
                    st.session_state.diagnostic_texte = texte_ia
                    
                    try:
                        historique = []
                        if os.path.exists("historique.json"):
                            with open("historique.json", "r", encoding="utf-8") as f:
                                historique = json.load(f)
                        
                        historique.append({"date": datetime.now().strftime('%d/%m/%Y %H:%M'), "vehicule": marque, "diagnostic": texte_ia})
                        
                        with open("historique.json", "w", encoding="utf-8") as f:
                            json.dump(historique, f, ensure_ascii=False, indent=4)
                    except Exception:
                        pass
                else:
                    st.error("Le module d'analyse est actuellement indisponible. Veuillez réessayer.")
            except Exception as e:
                status_placeholder.empty()
                st.error(f"Erreur technique : {e}")
        
        elif submit_button and image is None:
            st.warning("Veuillez fournir une image de la bougie pour lancer l'analyse.")

        if st.session_state.diagnostic_texte:
            st.success("✅ Diagnostic Croisé terminé.")
            
            texte_ia = st.session_state.diagnostic_texte
            diag_text = texte_ia
            causes_text = ""
            solutions_text = ""
            
            try:
                parts = texte_ia.split("CAUSES:")
                part1 = parts[0]
                rest = parts[1]
                parts2 = rest.split("SOLUTIONS:")
                causes_text = parts2[0].strip()
                solutions_text = parts2[1].strip()
                diag_text = part1.split("DIAGNOSTIC:")[1].strip() if "DIAGNOSTIC:" in part1 else part1.strip()
            except Exception:
                pass
                
            res_tab1, res_tab2, res_tab3 = st.tabs(["🔍 Diagnostic", "🔬 Causes", "🛠️ Solutions"])
            
            with res_tab1:
                st.info(diag_text)
            with res_tab2:
                st.warning(causes_text)
            with res_tab3:
                st.success(solutions_text)
            
            st.divider()
            st.markdown("### 📄 Exportation du Rapport")
            
            try:
                pdf_data = creer_pdf(texte_ia)
                st.download_button(
                    label="📥 Télécharger le rapport en PDF",
                    data=pdf_data,
                    file_name=f"Rapport_{datetime.now().strftime('%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.warning(f"Le PDF n'a pas pu être généré ({e}).")
                st.download_button(
                    label="📥 Télécharger le rapport (Format Texte)",
                    data=texte_ia,
                    file_name=f"Rapport_{datetime.now().strftime('%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

with tab2:
    st.subheader("📁 Historique des Interventions")
    if os.path.exists("historique.json"):
        try:
            with open("historique.json", "r", encoding="utf-8") as f:
                historique = json.load(f)
                for item in reversed(historique):
                    with st.expander(f"🚗 {item['vehicule']} - {item['date']}"):
                        st.write(item['diagnostic'])
        except Exception:
            st.error("Le fichier d'historique est corrompu. Veuillez le supprimer.")
    else:
        st.info("Aucune intervention enregistrée pour le moment.")
