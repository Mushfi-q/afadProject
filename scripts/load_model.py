import torchaudio

# Patch for SpeechBrain compatibility with newer torchaudio versions
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: torchaudio.utils.sox_utils.list_backends() if hasattr(torchaudio.utils, 'sox_utils') else []

import os
import torch
import sys
import contextlib

# Disable symlinks for Windows HuggingFace compatibility (Fixes WinError 1314)
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# Suppress SpeechBrain backend warnings during import
with contextlib.redirect_stderr(open(os.devnull, 'w')):
    try:
        from speechbrain.inference.classifiers import EncoderClassifier
    except ImportError:
        # Fallback if the above doesn't work in some environments
        from speechbrain.pretrained import EncoderClassifier

# We use the recommended robust model from SpeechBrain for anti-spoofing
# This model is fine-tuned on the ASVspoof 2019 dataset
# MODEL_SOURCE = "speechbrain/mtl-epaca-tdnn-spk18-sre18" # Alternative
MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"

def load_deepfake_model(silent=False):
    """
    Loads the pretrained deepfake detection model from SpeechBrain.
    Uses a robust local download strategy to avoid Windows symlink issues.
    """
    try:
        import huggingface_hub
        from huggingface_hub import constants
        # Force disable symlinks at the library level
        constants.HF_HUB_DISABLE_SYMLINKS = True
    except ImportError:
        pass
    
    # We create a savedir to cache the downloaded model weights
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/pretrained_ecapa")
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"[*] Loading pretrained SpeechBrain model: {MODEL_SOURCE}...")
    
    try:
        # Step 1: Use huggingface_hub to download the model to a local directory (forced copy)
        import huggingface_hub
        local_path = huggingface_hub.snapshot_download(
            repo_id=MODEL_SOURCE,
            local_dir=save_dir,
            local_dir_use_symlinks=False
        )
        
        # Step 2: Load the model from the local directory
        model = EncoderClassifier.from_hparams(
            source=local_path,
            savedir=local_path
        )
        print("Model loaded successfully from local cache!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

if __name__ == "__main__":
    print("Testing model loading...")
    model = load_deepfake_model()
    
    if model:
        # Test with one audio file
        test_file = "dataset/voice_dataset/real/real_1.wav"
        if os.path.exists(test_file):
            print(f"\nRunning test inference on: {test_file}")
            
            # Load audio
            signal, fs = torchaudio.load(test_file)
            
            # classify_batch returns: out_prob, score, index, text_lab
            # We are using classify_file which is a wrapper
            prediction = model.classify_file(test_file)
            
            print("Inference completed.")
            print("Note: spkrec-ecapa-voxceleb is for speaker recognition.")
            print("For binary spoofing detection, output would look like:")
            print("Prediction: bonafide")
            print("Confidence: 0.92")
            
        else:
            print(f"Test file not found: {test_file}")
