import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from AFAD_Project.models.train_text_model import (
    KEYWORD_FLAG_COLUMNS,
    combine_features,
    extract_keyword_flags,
    load_split_data,
    vectorize_text,
)

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
        # Step 5.3: Load the pre-split datasets with keyword flags.
        train_df, test_df = load_split_data()
        
        print("Training Dataframe Head:")
        print(train_df[['message', 'label'] + KEYWORD_FLAG_COLUMNS].head())
        print("\nClass Distribution:")
        print(train_df['label'].value_counts())
        
        # Step 5.4: Split Features and Labels
        X_train = train_df["message"]
        X_test = test_df["message"]
        y_train = train_df["label"]
        y_test = test_df["label"]
        
        print(f"Training set size: {len(X_train)}")
        print(f"Testing set size: {len(X_test)}")
        
        # 2. Vectorize (TF-IDF)
        X_train_tfidf, X_test_tfidf, vectorizer = vectorize_text(X_train, X_test)
        
        # 3. Extract Keyword Flag Features
        train_keyword_flags = extract_keyword_flags(train_df)
        test_keyword_flags = extract_keyword_flags(test_df)
        
        # 4. Combine Features
        X_train_final = combine_features(X_train_tfidf, train_keyword_flags)
        X_test_final = combine_features(X_test_tfidf, test_keyword_flags)
        print(f"TF-IDF features: {X_train_tfidf.shape[1]}")
        print(f"Keyword flag features: {len(KEYWORD_FLAG_COLUMNS)}")
        print(f"Expected final features: {X_train_tfidf.shape[1] + len(KEYWORD_FLAG_COLUMNS)}")
        
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
