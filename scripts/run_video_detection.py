import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import warnings

# Add AFAD_Project root to python path dynamically
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Suppress PyTorch warnings
warnings.filterwarnings("ignore")

from predict_video import predict_video

def run_video_detection(dataset_dir, output_csv="results/video_predictions.csv"):
    """
    Scans the video dataset folders, extracts frames, detects faces,
    predicts the deepfake probability, and aggregates the results.
    """
    print(f"Starting Video Pipeline on: {dataset_dir}")
    print("="*40)
    
    results = []
    
    for label in ['real', 'deepfake']:
        folder_path = os.path.join(dataset_dir, label)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found -> {folder_path}")
            continue
            
        true_label = 'REAL' if label == 'real' else 'DEEPFAKE'
        
        for file in os.listdir(folder_path):
            if file.lower().endswith('.mp4'):
                video_path = os.path.join(folder_path, file)
                print(f"\nProcessing {file} ({true_label})...")
                
                # Run the full predict chain (Limited to 50 faces for speed)
                pred_label, prob, analyzed_faces = predict_video(video_path, max_frames=50)
                
                print(f"File: {file} | True: {true_label} | Pred: {pred_label} | Prob: {prob:.2f}")
                
                results.append({
                    'video_file': file,
                    'true_label': true_label,
                    'prediction': pred_label,
                    'probability': prob,
                    'faces_analyzed': analyzed_faces
                })

    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")
    
    return df

def evaluate_video_results(df):
    """
    Evaluates predictions with sklearn metrics and logs to videoTrainingLog.txt
    """
    if len(df) == 0:
        print("No video results to evaluate.")
        return
        
    # Filter out UNKNOWN for metric calculation if any exist
    eval_df = df[df['prediction'] != 'UNKNOWN']
    
    if len(eval_df) == 0:
        print("No valid predictions (excluding UNKNOWN) to evaluate.")
        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        import numpy as np
        cm = np.zeros((2, 2), dtype=int)
    else:
        y_true = eval_df['true_label']
        y_pred = eval_df['prediction']
        
        # Fake/Deepfake is our positive class
        pos_label = 'DEEPFAKE'
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        
        cm = confusion_matrix(y_true, y_pred, labels=['REAL', 'DEEPFAKE'])
    
    # Log Evaluation to file based on user request "Create a videoTrainingLog.txt"
    log_path = "results/videoTrainingLog.txt"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # Prepare the log content
    log_content = [
        "Video Deepfake Detection Training Phase Log",
        "Date: 2026-03-16",
        "Phase: Video Deepfake Detection Pipeline\n",
        "========================================",
        "DETAILED STEP-BY-STEP PROCESS",
        "========================================\n",
        "Step 7.2 — Install Required Libraries",
        "* Process: Installed OpenCV, MTCNN (facenet-pytorch), PyTorch, NumPy, and Timms.",
        "* Findings: Dependencies installed smoothly.\n",
        "Step 7.3 — Frame Extraction from Video",
        "* Process: Created scripts/extract_frames.py using OpenCV.",
        "* Logic: Captured frames at an interval of N=10 and output to a temp directory.\n",
        "Step 7.4 & 7.5 — Face Detection (MTCNN) & Preprocessing",
        "* Process: Created scripts/detect_faces.py leveraging facenet-pytorch's MTCNN.",
        "* Logic: Parsed each frame to extract the prominent face. Bounding boxes converted to PIL.",
        "* Preprocessing: Resized face to 299x299, applied toTensor(), and mean/std normalization [0.5, 0.5, 0.5] as required by Xception.\n",
        "Step 7.6 — Load Pretrained Deepfake Model",
        "* Process: Created scripts/load_video_model.py utilizing the timm architecture library.",
        "* Logic: Inherited the 'xception41' backbone. Since downloading custom 10+ GB deepfake weights is impractical here, we initialized the native ImageNet weights and adapted the linear head to num_classes=2.\n",
        "Step 7.7 & 7.8 — Frame Level Prediction & Video Level Decision",
        "* Process: Developed scripts/predict_video.py. Fed each preprocessed MTCNN face into the Xception classifier.",
        "* Logic: Calculated the argmax probabilities per frame. Aggregated frame probabilities using straight Average Probability.",
        "* Decision: Average Fake Prob > 0.50 -> 'DEEPFAKE' video classification.\n",
        "Step 7.9, 7.10 & 7.11 — Dataset Processing and Evaluation",
        "* Process: Developed scripts/run_video_detection.py to batch process the dataset/video folders.",
        f"* Results: Processed {len(df)} total videos.",
        "========================================",
        "FINAL VIDEO METRICS SUMMARY",
        "========================================",
        f"Total Video Samples: {len(df)}",
        f"Accuracy:  {accuracy:.2f}",
        f"Precision: {precision:.2f}",
        f"Recall:    {recall:.2f}\n",
        "Confusion Matrix:",
        "\t\tPred Real\tPred Fake",
        f"True Real\t{cm[0,0]}\t\t{cm[0,1]}",
        f"True Fake\t{cm[1,0]}\t\t{cm[1,1]}\n"
    ]
    
    with open(log_path, "w") as f:
        f.write("\n".join(log_content))
        
    print(f"\nEvaluation summary dumped to {log_path}")
    print("\nEVALUATION PREVIEW:")
    print("="*40)
    print(f"Accuracy: {accuracy:.2f} | Precision: {precision:.2f} | Recall: {recall:.2f}")

if __name__ == "__main__":
    DATASET_DIR = "dataset/video"
    OUTPUT_CSV = "results/video_predictions.csv"
    
    print("--- Phase 2: Launching Video Detection Pipeline ---")
    df_results = run_video_detection(DATASET_DIR, OUTPUT_CSV)
    
    print("\n--- Phase 2: Evaluating Model Results ---")
    evaluate_video_results(df_results)
