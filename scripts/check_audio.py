import os
import librosa

def check_audio_files(dataset_dir):
    """
    Verifies that all audio files are readable, checks sample rate and duration.
    """
    total_files = 0
    valid_files = 0

    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                total_files += 1
                file_path = os.path.join(root, file)
                try:
                    # Load audio
                    y, sr = librosa.load(file_path, sr=None)
                    duration = librosa.get_duration(y=y, sr=sr)
                    
                    print(f"[{total_files}] {file_path}")
                    print(f"    Sample rate: {sr} Hz")
                    print(f"    Duration: {duration:.2f} seconds")
                    
                    # Basic checks
                    if sr != 16000:
                        print(f"    WARNING: Sample rate is not 16000Hz (it is {sr}Hz)")
                    
                    valid_files += 1
                except Exception as e:
                    print(f"    ERROR loading {file_path}: {e}")

    print("\n" + "="*40)
    print(f"Total WAV files found:  {total_files}")
    print(f"Successfully loaded:    {valid_files}")
    print(f"Failed to load:         {total_files - valid_files}")
    print("="*40 + "\n")

if __name__ == "__main__":
    DATASET_DIR = "dataset/voice_dataset"
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory '{DATASET_DIR}' not found. Run from the project root.")
    else:
        check_audio_files(DATASET_DIR)
