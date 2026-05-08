import streamlit as st
import os
import sys
import tempfile
import time
import warnings
import logging

# Suppress annoying library warnings & Fix Windows symlink issues
os.environ["SPEECHBRAIN_LOG_LEVEL"] = "ERROR"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# SpeechBrain/torchaudio specific suppression for terminal cleanliness
logging.getLogger("speechbrain").setLevel(logging.ERROR)

# Ensure project root and its subfolders are in sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
scripts_path = os.path.join(project_root, 'scripts')
parent_dir = os.path.dirname(project_root)

for path in [project_root, scripts_path, parent_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Robust imports from scripts
try:
    from utils_ui import predict_text, run_voice_prediction, run_video_prediction, predict_text_batch
except ImportError:
    from scripts.utils_ui import predict_text, run_voice_prediction, run_video_prediction, predict_text_batch

# --- Model Caching ---
@st.cache_resource
def get_voice_model():
    """Cache the SpeechBrain model to avoid redundant loading."""
    try:
        from load_model import load_deepfake_model
    except ImportError:
        from scripts.load_model import load_deepfake_model
    print("[*] Accessing SpeechBrain model from memory...")
    return load_deepfake_model(silent=True)

# --- 0. Authentication Logic ---
from modules.auth_logic import init_db, create_user, verify_user

# Initialize Database
init_db()

# Initialize Session State
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""

def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()

# --- Page Configuration ---
st.set_page_config(
    page_title="AFAD Deepfake Detector",
    page_icon="🛡️",
    layout="wide",
)

# --- 1. LOGIN / REGISTRATION UI ---
if not st.session_state.logged_in:
    st.title("🛡️ AFAD - Access Portal")
    
    auth_mode = st.radio("Choose Action:", ["Login", "Register"], horizontal=True)
    
    with st.form("auth_form"):
        st.subheader(auth_mode)
        user_input = st.text_input("Username")
        pass_input = st.text_input("Password", type="password")
        submit_btn = st.form_submit_button(auth_mode)
        
        if submit_btn:
            if auth_mode == "Register":
                if user_input and pass_input:
                    success = create_user(user_input, pass_input)
                    if success:
                        st.success("Registration successful! Please login.")
                    else:
                        st.error("Username already exists.")
                else:
                    st.warning("Please fill all fields.")
            
            elif auth_mode == "Login":
                if verify_user(user_input, pass_input):
                    st.session_state.logged_in = True
                    st.session_state.username = user_input
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
    
    st.stop() # Prevent dashboard from loading

# --- 2. DASHBOARD (LOGGED IN) ---

# --- Header & Logout ---
col_head, col_logout = st.columns([0.85, 0.15])
with col_head:
    st.title("🛡️ AFAD Deepfake Detection Suite")
    st.markdown(f"Welcome back, **{st.session_state.username}**! Multi-modal protection active.")

with col_logout:
    st.write("") # Padding
    if st.button("Logout"):
        logout()

# --- Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #ddd;
        background-color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["💬 Text Analysis", "🔊 Voice Analysis", "🎬 Video Analysis"])

# --- 1. Text Analysis ---
with tab1:
    st.header("Social Engineering & Text Deepfake Detection")
    st.info("Analyze messages for manipulation flags and regional scam patterns.")
    
    text_input = st.text_area("Paste the message here:", height=200, placeholder="e.g., hey bro urgent send money...")
    
    if st.button("Analyze Text"):
        if text_input.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Running AFAD Hybrid Analysis..."):
                label, risk = predict_text(text_input)
                
                # Display Results
                st.subheader("Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    color = "red" if label == "Attack" else "green"
                    if label == "Suspicious": color = "orange"
                    st.markdown(f"### Classification: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
                
                with col2:
                    st.metric("Risk Score", f"{risk:.2f}%")
                
                if label == "Attack":
                    st.error("⚠️ HIGH RISK: This message matches known scam patterns or rule-based triggers.")
                elif label == "Suspicious":
                    st.warning("⚠️ CAUTION: This message shows suspicious manipulation cues.")
                else:
                    st.success("✅ SAFE: No significant attack indicators found.")

# --- 2. Voice Analysis ---
with tab2:
    st.header("AI Voice Deepfake Detection")
    st.info("Upload a .wav file to verify if the voice is real or AI-generated.")
    
    audio_file = st.file_uploader("Upload Audio (.wav)", type=["wav"])
    
    if 'audio_result' not in st.session_state:
        st.session_state.audio_result = None
    if 'last_audio_name' not in st.session_state:
        st.session_state.last_audio_name = None

    if audio_file is not None:
        if audio_file.name != st.session_state.last_audio_name:
            st.session_state.audio_result = None
            st.session_state.last_audio_name = audio_file.name
            
        st.audio(audio_file)
        
        if st.button("Verify Audio"):
            with st.spinner("Analyzing spectral features..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(audio_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    voice_model = get_voice_model()
                    result = run_voice_prediction(tmp_path, original_filename=audio_file.name, model=voice_model)
                    st.session_state.audio_result = result
                finally:
                    if os.path.exists(tmp_path): os.remove(tmp_path)
        
        if st.session_state.audio_result:
            result = st.session_state.audio_result
            st.subheader("Results")
            if result == "fake": st.error("🚨 DETECTED: This audio is likely AI-GENERATED (FAKE).")
            elif result == "real": st.success("✅ VERIFIED: This audio is likely REAL.")
            else: st.warning(f"Result: {result}")

# --- 3. Video Analysis ---
with tab3:
    st.header("Visual Deepfake & Face Swap Detection")
    st.info("Upload an .mp4 video to check for facial manipulations.")
    
    video_file = st.file_uploader("Upload Video (.mp4)", type=["mp4"])
    
    if 'video_result' not in st.session_state:
        st.session_state.video_result = None
    if 'last_video_name' not in st.session_state:
        st.session_state.last_video_name = None

    if video_file is not None:
        if video_file.name != st.session_state.last_video_name:
            st.session_state.video_result = None
            st.session_state.last_video_name = video_file.name
            
        st.video(video_file)
        
        if st.button("Scan Video"):
            with st.spinner("Analyzing frames..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                    tmp_file.write(video_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    prediction, prob, count = run_video_prediction(tmp_path, original_filename=video_file.name, max_frames=50)
                    st.session_state.video_result = (prediction, prob, count)
                except Exception as e:
                    st.session_state.video_result = ("Error", 0.0, 0)
                    st.error(f"Execution Error: {e}")
                finally:
                    if os.path.exists(tmp_path): os.remove(tmp_path)

        if st.session_state.video_result:
            prediction, prob, count = st.session_state.video_result
            
            # --- Check for Module Failure ---
            if isinstance(prediction, str) and prediction.startswith("Error"):
                st.warning("⚠️ Video detection unavailable: Please check dependencies (OpenCV/PyTorch).")
            else:
                st.subheader("Results")
                col1, col2 = st.columns(2)
                with col1:
                    color = "red" if prediction == "DEEPFAKE" else "green"
                    st.markdown(f"### Result: <span style='color:{color}'>{prediction}</span>", unsafe_allow_html=True)
                with col2:
                    st.metric("Confidence Score", f"{prob:.2f}")
                st.write(f"Analyzed {count} faces.")

# --- Footer ---
st.divider()
st.caption("AFAD Prototype - Advanced Fraud & Attack Detection System | Session Active")
