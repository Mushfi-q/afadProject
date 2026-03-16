import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from AFAD_Project.models.train_text_model import load_split_data, vectorize_text, combine_features

def train_afad_model(X_train_final, y_train):
    """
    Step 5.7: Train Logistic Regression Model
    Trains a Logistic Regression model on the combined feature set.
    """
    print("\n--- Starting Model Training ---")
    # Step 5.7: Initialize Logistic Regression with class balancing
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    
    # Fit the model
    print("Training Logistic Regression Classifier...")
    model.fit(X_train_final, y_train)
    print("Training complete.")
    
    return model

def evaluate_model(model, X_test_vec, y_test):
    """
    Step 5.8: Test the Model
    Evaluates the model on the unseen test set.
    """
    print("\n--- Step 5.8: Test the Model ---")
    y_pred = model.predict(X_test_vec)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return y_pred

if __name__ == "__main__":
    try:
        # Step 5.3: Load the High-Quality Re-Cleaned Dataset (No Headers)
        dataset_path = "data/final_afad_dataset_cleaned.csv"
        df = pd.read_csv(dataset_path)
        
        print("Dataframe Head:")
        print(df[['message', 'label']].head())
        print("\nClass Distribution:")
        print(df['label'].value_counts())
        
        # Step 5.4: Split Features and Labels
        X = df["message"]
        y = df["label"]
        
        # Step 5.5: Train/Test Split
        # 80% training, 20% testing with stratification
        from sklearn.model_selection import train_test_split
        print("\n--- Step 5.5: Stratified Train/Test Split (80/20) ---")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Testing set size: {len(X_test)}")
        
        # 2. Vectorize (TF-IDF)
        X_train_tfidf, X_test_tfidf, vectorizer = vectorize_text(X_train, X_test)
        
        # 3. Extract Psychological Features
        from AFAD_Project.features.custom_feature_extractor import extract_custom_features
        train_custom = extract_custom_features(pd.DataFrame(X_train, columns=['message']))
        test_custom = extract_custom_features(pd.DataFrame(X_test, columns=['message']))
        
        # 4. Combine Features
        X_train_final = combine_features(X_train_tfidf, train_custom)
        X_test_final = combine_features(X_test_tfidf, test_custom)
        
        # 5. Train
        model = train_afad_model(X_train_final, y_train)
        
        # 6. Evaluate
        evaluate_model(model, X_test_final, y_test)
        
        # 7. Save Model and Vectorizer
        os.makedirs("AFAD_Project/models/saved", exist_ok=True)
        joblib.dump(model, "AFAD_Project/models/saved/afad_model.pkl")
        joblib.dump(vectorizer, "AFAD_Project/models/saved/afad_vectorizer.pkl")
        print("\nModel and vectorizer saved to 'AFAD_Project/models/saved/'")
        
    except Exception as e:
        print(f"Error during training: {e}")
