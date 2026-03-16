import os
import sys
import pandas as pd
import joblib
import torch
import time
from scipy.sparse import hstack

# Ensure project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
# Add both AFAD_Project and its parent to path to handle different run contexts
current_dir = os.path.dirname(os.path.abspath(__file__)) # c:\...\AFAD_Project\scripts
project_root = os.path.dirname(current_dir)             # c:\...\AFAD_Project

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Try flexible imports
try:
    from features.custom_feature_extractor import extract_custom_features
except ImportError:
    # If the parent is not in path yet, add it
    project_root = os.path.dirname(current_dir)
    sys.path.append(project_root)
    from features.custom_feature_extractor import extract_custom_features
# Try flexible imports for other scripts in the same folder
try:
    from predict_audio import predict_audio
    from predict_video import predict_video
except ImportError:
    from scripts.predict_audio import predict_audio
    from scripts.predict_video import predict_video

# --- TEXT DETECTION LOGIC ---

def predict_text(message):
    """
    Predicts if a text message is an Attack or Safe.
    Returns: label, risk_score (%)
    """
    try:
        model_path = os.path.join(project_root, "models/saved/afad_model.pkl")
        vectorizer_path = os.path.join(project_root, "models/saved/afad_vectorizer.pkl")
        
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            return "Error: Model files not found", 0.0
            
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        # a. TF-IDF Vectorization
        msg_tfidf = vectorizer.transform([message])
        
        # b. Psychological Cues
        msg_df = pd.DataFrame([message], columns=['message'])
        msg_custom = extract_custom_features(msg_df)
        
        # c. Combine Features
        msg_final = hstack([msg_tfidf, msg_custom.values])
        
        # d. Predict
        prediction_num = model.predict(msg_final)[0]
        prediction_label = "Attack" if prediction_num == 1 else "Safe"
        
        # e. Get Probability (Risk Score)
        proba = model.predict_proba(msg_final)[0][1] * 100
        
        # --- NEW: Behavioral Safety Filter ---
        familiarity_score = msg_custom['familiarity_score'].values[0]
        urgency_score = msg_custom['urgency_score'].values[0]
        
        # Rule: If no familiarity cues are detected, we downgrade the risk score
        # to prevent false positives from generic "urgent" or "please" messages.
        if familiarity_score == 0:
            if proba > 60:
                proba = proba * 0.4 # Dramatically reduce (e.g., 80% becomes 32%)
            elif proba > 30:
                proba = proba * 0.7 # Moderately reduce
        
        # Recalculate label based on filtered probability
        prediction_label = "Attack" if proba >= 50 else "Safe"
        
        # --- NEW: Logging ---
        log_path = os.path.join(project_root, "results/textLog.txt")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a") as f:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp}] Message: \"{message[:50]}...\"\n")
            f.write(f"Cues -> Familiarity: {familiarity_score}, Urgency: {urgency_score}\n")
            f.write(f"ML Raw Prob: {model.predict_proba(msg_final)[0][1]*100:.2f}%\n")
            f.write(f"Filtered Risk: {proba:.2f}% | Final Decision: {prediction_label}\n")
            f.write("-" * 50 + "\n")

        return prediction_label, proba
        
    except Exception as e:
        return f"Error: {e}", 0.0

# --- WRAPPER FOR STREAMLIT ---

def run_voice_prediction(audio_path, original_filename=None, model=None):
    """
    Wrapper for voice prediction to be used in UI.
    """
    return predict_audio(audio_path, original_filename=original_filename, model=model)

def run_video_prediction(video_path, original_filename=None, max_frames=50):
    """
    Wrapper for video prediction to be used in UI.
    """
    return predict_video(video_path, original_filename=original_filename, max_frames=max_frames)
