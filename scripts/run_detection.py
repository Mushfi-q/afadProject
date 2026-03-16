import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import warnings

# Suppress warnings from SpeechBrain/PyTorch
warnings.filterwarnings("ignore")

from scripts.predict_audio import predict_audio

def run_detection(dataset_dir, output_csv="results/predictions.csv"):
    """
    Scans the dataset folders, runs predictions on all WAV files,
    and returns a DataFrame of the results.
    """
    print(f"Starting detection on dataset: {dataset_dir}")
    
    results = []
    
    for label in ['real', 'deepfake']:
        folder_path = os.path.join(dataset_dir, label)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found -> {folder_path}")
            continue
            
        # The true label string we want
        true_label = 'real' if label == 'real' else 'fake'
        
        for file in os.listdir(folder_path):
            if file.lower().endswith('.wav'):
                file_path = os.path.join(folder_path, file)
                
                # Get prediction
                prediction = predict_audio(file_path)
                
                results.append({
                    'file': file,
                    'true_label': true_label,
                    'prediction': prediction
                })
                
                print(f"Processed: {file} | True: {true_label} | Pred: {prediction}")

    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")
    
    return df

def evaluate_results(df):
    """
    Evaluates the predictions using scikit-learn metrics.
    """
    if len(df) == 0:
        print("No results to evaluate.")
        return
        
    y_true = df['true_label']
    y_pred = df['prediction']
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Precision and recall need a designated 'positive' class
    # Let's say detecting a 'fake' is our positive class (what we want to detect)
    pos_label = 'fake'
    
    precision = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=['real', 'fake'])
    
    # Print results
    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print(f"Total samples: {len(df)}")
    print(f"Accuracy:  {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print("\nConfusion Matrix:")
    print("\t\tPred Real\tPred Fake")
    print(f"True Real\t{cm[0,0]}\t\t{cm[0,1]}")
    print(f"True Fake\t{cm[1,0]}\t\t{cm[1,1]}")
    print("="*40)
    
    # Log to file
    with open("results/voiceTrainingLog.txt", "a") as f:
        f.write("\n" + "="*40 + "\n")
        f.write("Evaluation Results\n")
        f.write("="*40 + "\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Accuracy: {accuracy:.2f}\n")
        f.write(f"Precision: {precision:.2f}\n")
        f.write(f"Recall: {recall:.2f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write("\t\tPred Real\tPred Fake\n")
        f.write(f"True Real\t{cm[0,0]}\t\t{cm[0,1]}\n")
        f.write(f"True Fake\t{cm[1,0]}\t\t{cm[1,1]}\n")

if __name__ == "__main__":
    DATASET_DIR = "dataset/voice_dataset"
    OUTPUT_CSV = "results/predictions.csv"
    
    print("--- STEP 5: Run Detection ---")
    df_results = run_detection(DATASET_DIR, OUTPUT_CSV)
    
    print("\n--- STEP 6 & 7: Evaluate Model & Print Results ---")
    evaluate_results(df_results)
