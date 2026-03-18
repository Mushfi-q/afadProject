import streamlit as st
import os
import sys
import tempfile
import time
import warnings
import logging

# Suppress annoying library warnings
os.environ["SPEECHBRAIN_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# SpeechBrain/torchaudio specific suppression for terminal cleanliness
logging.getLogger("speechbrain").setLevel(logging.ERROR)

# Ensure project root and its parent are in sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(project_root)

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Try to import from the scripts subfolder
try:
    from scripts.utils_ui import predict_text, run_voice_prediction, run_video_prediction
except ImportError:
    # Fallback if scripts isn't treated as a package
    sys.path.append(os.path.join(project_root, 'scripts'))
    from utils_ui import predict_text, run_voice_prediction, run_video_prediction

# --- Model Caching ---
@st.cache_resource
def get_voice_model():
    """Cache the SpeechBrain model to avoid redundant loading."""
    from scripts.load_model import load_deepfake_model
    print("[*] Accessing SpeechBrain model from memory...")
    return load_deepfake_model(silent=True)

# --- Page Configuration ---
st.set_page_config(
    page_title="AFAD Deepfake Detector",
    page_icon="🛡️",
    layout="wide",
)

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

# --- Header ---
st.title("🛡️ AFAD Deepfake Detection Suite")
st.markdown("Multi-modal protection against AI-generated threats.")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["💬 Text Analysis", "🔊 Voice Analysis", "🎬 Video Analysis"])

# --- 1. Text Analysis ---
with tab1:
    st.header("Social Engineering & Text Deepfake Detection")
    st.info("Analyze messages for psychological cues like urgency, emotional manipulation, and authority.")
    
    text_input = st.text_area("Paste the message here:", height=200, placeholder="e.g., hey bro urgent send money...")
    
    if st.button("Analyze Text"):
        if text_input.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing psychological cues..."):
                print(f"[*] Processing Text Analysis...")
                label, risk = predict_text(text_input)
                
                # Display Results
                st.subheader("Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    color = "red" if label == "Attack" else "green"
                    st.markdown(f"### Classification: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
                
                with col2:
                    st.metric("Risk Score", f"{risk:.2f}%")
                
                if label == "Attack":
                    st.error("⚠️ HIGH RISK: This message contains strong indicators of social engineering.")
                else:
                    st.success("✅ LOW RISK: This message appears to be safe.")

# --- 2. Voice Analysis ---
with tab2:
    st.header("AI Voice Deepfake Detection")
    st.info("Upload a .wav file to verify if the voice is real or AI-generated.")
    
    # --- Audio Upload & Persistence ---
    audio_file = st.file_uploader("Upload Audio (.wav)", type=["wav"])
    
    # Initialize session state for audio
    if 'audio_result' not in st.session_state:
        st.session_state.audio_result = None
    if 'last_audio_name' not in st.session_state:
        st.session_state.last_audio_name = None

    if audio_file is not None:
        # Reset result if a DIFFERENT file is uploaded
        if audio_file.name != st.session_state.last_audio_name:
            st.session_state.audio_result = None
            st.session_state.last_audio_name = audio_file.name
            
        st.audio(audio_file)
        
        if st.button("Verify Audio"):
            print(f"[*] Processing Audio Analysis: {audio_file.name}")
            with st.spinner("Analyzing spectral features..."):
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(audio_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    voice_model = get_voice_model()
                    result = run_voice_prediction(tmp_path, original_filename=audio_file.name, model=voice_model)
                    st.session_state.audio_result = result # Persist!
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
        
        # Display persistent result
        if st.session_state.audio_result:
            result = st.session_state.audio_result
            st.subheader("Results")
            if result == "fake":
                st.error("🚨 DETECTED: This audio is likely AI-GENERATED (FAKE).")
            elif result == "real":
                st.success("✅ VERIFIED: This audio is likely REAL.")
            else:
                st.warning(f"Result: {result}")
    else:
        st.session_state.audio_result = None
        st.session_state.last_audio_name = None

# --- 3. Video Analysis ---
with tab3:
    st.header("Visual Deepfake & Face Swap Detection")
    st.info("Upload an .mp4 video to check for facial manipulations using XceptionNet.")
    
    # --- Video Upload & Persistence ---
    video_file = st.file_uploader("Upload Video (.mp4)", type=["mp4"])
    
    # Initialize session state for video
    if 'video_result' not in st.session_state:
        st.session_state.video_result = None
    if 'last_video_name' not in st.session_state:
        st.session_state.last_video_name = None

    if video_file is not None:
        # Reset result if a DIFFERENT file is uploaded
        if video_file.name != st.session_state.last_video_name:
            st.session_state.video_result = None
            st.session_state.last_video_name = video_file.name
            
        st.video(video_file)
        
        if st.button("Scan Video"):
            print(f"[*] Processing Video Analysis: {video_file.name}")
            with st.spinner("Extracting frames and analyzing faces (this may take a moment)..."):
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                    tmp_file.write(video_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    prediction, prob, count = run_video_prediction(tmp_path, original_filename=video_file.name, max_frames=50)
                    st.session_state.video_result = (prediction, prob, count) # Persist!
                except Exception as e:
                    st.error(f"Error during video processing: {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

        # Display persistent result
        if st.session_state.video_result:
            prediction, prob, count = st.session_state.video_result
            st.subheader("Results")
            col1, col2 = st.columns(2)
            
            with col1:
                color = "red" if prediction == "DEEPFAKE" else "green"
                st.markdown(f"### Result: <span style='color:{color}'>{prediction}</span>", unsafe_allow_html=True)
            
            with col2:
                st.metric("Confidence Score", f"{prob:.2f}")
            
            st.write(f"Analyzed {count} faces across extracted frames.")
            
            if prediction == "DEEPFAKE":
                st.error("🚨 WARNING: High probability of facial manipulation detected.")
            else:
                st.success("✅ CLEAR: No significant signs of deepfake manipulation found.")
    else:
        st.session_state.video_result = None
        st.session_state.last_video_name = None

# --- Footer ---
st.divider()
st.caption("AFAD Prototype - Advanced Fraud & Attack Detection System (Internal Training Phase)")
