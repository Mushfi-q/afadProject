import pandas as pd
import joblib
from AFAD_Project.features.custom_feature_extractor import extract_custom_features

def manual_check():
    # 1. Load Model and Vectorizer
    model_path = "AFAD_Project/models/saved/afad_model.pkl"
    vectorizer_path = "AFAD_Project/models/saved/afad_vectorizer.pkl"
    
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # 2. Define Test Messages
    test_messages = [
        "hey bro urgent send money",
        "hello see you tomorrow"
    ]
    
    print("\n--- Running Sanity Check ---")
    
    for msg in test_messages:
        # a. TF-IDF Vectorization
        msg_tfidf = vectorizer.transform([msg])
        
        # b. Psychological Cues
        msg_df = pd.DataFrame([msg], columns=['message'])
        msg_custom = extract_custom_features(msg_df)
        
        # c. Combine Features
        from scipy.sparse import hstack
        msg_final = hstack([msg_tfidf, msg_custom.values])
        
        # d. Predict
        prediction_num = model.predict(msg_final)[0]
        prediction_label = "Attack" if prediction_num == 1 else "Safe"
        
        # e. Get Probability (Risk Score)
        proba = model.predict_proba(msg_final)[0][1] * 100
        
        print(f"Input: \"{msg}\"")
        print(f"Prediction: {prediction_label}")
        print(f"Risk Score: {proba:.2f}%")
        print("-" * 30)

if __name__ == "__main__":
    try:
        manual_check()
    except Exception as e:
        print(f"Error during sanity check: {e}")
