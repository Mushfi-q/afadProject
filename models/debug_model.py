import joblib
import pandas as pd
import numpy as np

def debug_model():
    model_path = "AFAD_Project/models/saved/afad_model.pkl"
    vectorizer_path = "AFAD_Project/models/saved/afad_vectorizer.pkl"
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    feature_names = list(vectorizer.get_feature_names_out())
    # Add custom cue names
    feature_names.extend(['familiarity_cue', 'urgency_cue', 'emotional_cue', 'authority_cue'])
    
    coefs = model.coef_[0]
    
    # Create a DataFrame for visualization
    feat_importances = pd.DataFrame({'feature': feature_names, 'weight': coefs})
    feat_importances['abs_weight'] = feat_importances['weight'].abs()
    feat_importances = feat_importances.sort_values(by='abs_weight', ascending=False)
    
    print("\nTop 20 Malicious (Positive Weight) Features:")
    print(feat_importances[feat_importances['weight'] > 0].head(20))
    
    print("\nTop 20 Safe (Negative Weight) Features:")
    print(feat_importances[feat_importances['weight'] < 0].head(20))

    # Check "hello see you tomorrow" specifically
    msg = "hello see you tomorrow"
    msg_tfidf = vectorizer.transform([msg])
    
    # Check which features are active in this message
    indices = msg_tfidf.indices
    print(f"\nActive TF-IDF features for '{msg}':")
    for idx in indices:
        name = feature_names[idx]
        weight = coefs[idx]
        value = msg_tfidf[0, idx]
        print(f"Feature: {name}, Value: {value:.4f}, Weight: {weight:.4f}, Contribution: {value*weight:.4f}")

if __name__ == "__main__":
    debug_model()
