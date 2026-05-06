import os
import sys
import pandas as pd
import joblib
import torch
import time
import re
import string
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
    from features.custom_feature_extractor import extract_custom_features, extract_keyword_flags
except ImportError:
    # If the parent is not in path yet, add it
    project_root = os.path.dirname(current_dir)
    sys.path.append(project_root)
    from features.custom_feature_extractor import extract_custom_features, extract_keyword_flags
# Try flexible imports for other scripts in the same folder
try:
    try:
        from predict_audio import predict_audio
        from predict_video import predict_video
    except ImportError:
        from scripts.predict_audio import predict_audio
        from scripts.predict_video import predict_video
except Exception as e:
    print(f"Warning: Audio/Video prediction dependencies not fully loaded: {e}")
    def predict_audio(*args, **kwargs): return "Error: Audio deps missing", 0.0
    def predict_video(*args, **kwargs): return "Error: Video deps missing", 0.0, 0

# --- TEXT PREPROCESSING ---

def clean_text(text):
    """
    Cleans text by removing email headers, converting to lowercase, 
    removing URLs, punctuation, and extra whitespace.
    Matches the logic in prepare_afad_dataset.py.
    """
    if not isinstance(text, str):
        return ""
    
    # Remove email headers
    text = re.sub(r'^[A-Za-z-]+:.*$', '', text, flags=re.MULTILINE)
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

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
        
        # a. Clean and Vectorize
        cleaned_message = clean_text(message)
        msg_tfidf = vectorizer.transform([cleaned_message])
        
        # b. Keyword Flags (Phase 2)
        msg_df = pd.DataFrame([cleaned_message], columns=['message'])
        msg_flags = extract_keyword_flags(msg_df)
        
        # c. Combine Features (Must match training shape: TF-IDF + 3 Flags = 5003)
        msg_final = hstack([msg_tfidf, msg_flags.values])
        
        # e. Get Probability (Risk Score)
        proba = model.predict_proba(msg_final)[0][1] * 100
        
        # --- NEW: False Positive Reduction Layer ---
        # If no manipulation keywords are present, we reduce the probability 
        # to prevent common phrases from being flagged as suspicious.
        has_money = msg_flags['has_money_term'].values[0]
        has_urgency = msg_flags['has_urgency'].values[0]
        has_payment = msg_flags['has_payment_term'].values[0]
        
        if has_money == 0 and has_urgency == 0 and has_payment == 0:
            if proba > 50:
                proba = proba * 0.5  # Reduce significantly (e.g. 73% -> 36.5%)
            else:
                proba = proba * 0.8  # Reduce slightly
                
        # --- PHASE 5: Threshold Tuning ---
        # Adjusted to reduce false positives:
        # Attack: > 85% | Suspicious: 50% - 85% | Safe: < 50%
        if proba > 85:
            prediction_label = "Attack"
        elif proba >= 50:
            prediction_label = "Suspicious"
        else:
            prediction_label = "Safe"
            
        # --- PHASE 4: Rule-Based Safety Layer (OVERRIDE) ---
        # Phase 4 override remains unchanged and takes priority
        is_overridden = False
        if has_money == 1 and has_urgency == 1:
            prediction_label = "Attack"
            is_overridden = True
        
        # --- NEW: Logging ---
        log_path = os.path.join(project_root, "results/textLog.txt")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a") as f:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp}] Message: \"{message[:50]}...\"\n")
            f.write(f"Flags -> Money: {has_money}, Urgency: {has_urgency}\n")
            f.write(f"ML Raw Prob: {model.predict_proba(msg_final)[0][1]*100:.2f}%\n")
            f.write(f"Override: {is_overridden} | Final Decision: {prediction_label} | Risk: {proba:.2f}%\n")
            f.write("-" * 50 + "\n")
        
        return prediction_label, proba
        
    except Exception as e:
        return f"Error: {e}", 0.0

def predict_text_batch(messages):
    """
    Predicts labels for a list of messages.
    Ensures the Phase 4 rule is applied to each message.
    Returns: list of (label, risk_score)
    """
    results = []
    for msg in messages:
        results.append(predict_text(msg))
    return results

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
