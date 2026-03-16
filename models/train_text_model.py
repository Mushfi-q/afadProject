import pandas as pd
from sklearn.model_selection import train_test_split
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from AFAD_Project.features.custom_feature_extractor import extract_custom_features

def vectorize_text(X_train, X_test):
    """
    Step 4.2: Convert Text to TF-IDF Features
    """
    print("\nVectorizing text data using TF-IDF...")
    
    # Define custom stop words to filter technical headers and common noise
    technical_stop_words = [
        '7bit', 'binary', 'charsetusascii', 'contenttransferencoding', 
        'contenttype', 'mimeversion', 'messageid', 'javamailevansth',
        'evansth', 'thalia', 'usascii', 'textplain', 'pst', 'pdt',
        'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun',
        'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
        '0700', '0800', '2000', '2001', '2002', 'date'
    ]
    
    # Combine with standard English stop words
    from sklearn.feature_extraction import text
    stop_words = list(text.ENGLISH_STOP_WORDS.union(technical_stop_words))
    
    vectorizer = TfidfVectorizer(
        max_features=5000, 
        stop_words=stop_words, # Using combined stop words
        ngram_range=(1, 2)
    )
    
    # Fit and transform the training data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # Transform the test data (using the same vocabulary from training)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"Vectorization complete.")
    print(f"X_train_tfidf shape: {X_train_tfidf.shape}")
    print(f"X_test_tfidf shape: {X_test_tfidf.shape}")
    
    return X_train_tfidf, X_test_tfidf, vectorizer

def combine_features(tfidf_matrix, custom_features_df):
    """
    Step 4.5: Combine TF-IDF + Psychological Features
    Merges Sparse TF-IDF matrix with dense psychological counters.
    """
    print("Combining TF-IDF and psychological features...")
    # Convert custom features to sparse format and stack with TF-IDF
    combined_matrix = hstack([tfidf_matrix, custom_features_df.values])
    print(f"Feature combination complete. Final shape: {combined_matrix.shape}")
    return combined_matrix

def load_split_data():
    """
    Loads pre-split AFAD training and testing datasets.
    """
    train_path = r"AFAD_Project\dataset\text\train\afad_train.csv"
    test_path = r"AFAD_Project\dataset\text\test\afad_test.csv"
    
    print(f"Loading training data from: {train_path}")
    print(f"Loading testing data from: {test_path}")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    return train_df['message'], test_df['message'], train_df['label'], test_df['label']

if __name__ == "__main__":
    try:
        # Step 1 & 2: Load Pre-split Data
        X_train, X_test, y_train, y_test = load_split_data()
        
        # Step 4.2: Convert Text to TF-IDF Features
        X_train_tfidf, X_test_tfidf, vectorizer = vectorize_text(X_train, X_test)
        
        # Step 4.4 & 4.5: Extract and Combine Psychological Features
        print("\n--- Extracting Psychological Features ---")
        train_custom = extract_custom_features(pd.DataFrame(X_train, columns=['message']))
        test_custom = extract_custom_features(pd.DataFrame(X_test, columns=['message']))
        
        # Combine Step
        X_train_final = combine_features(X_train_tfidf, train_custom)
        X_test_final = combine_features(X_test_tfidf, test_custom)
        
        # Step 9: Inspect Important Features
        print("\nInspecting sampled feature names (learned vocabulary):")
        feature_names = vectorizer.get_feature_names_out()
        # Print a mix of early, middle, and late features to see diversity
        print(f"Total features: {len(feature_names)}")
        print("Sample features:", feature_names[100:120])
        
        # Look specifically for markers the user mentioned
        targeted_terms = ['send money', 'urgent', 'immediately', 'verify', 'transfer', 'login', 'account', 'approve']
        found_terms = [t for t in feature_names if any(kw == t or kw + ' ' in t or ' ' + kw in t for kw in targeted_terms)]
        print(f"\nTargeted familiarity indicators found in vocabulary: {found_terms[:20]}")
        
        # Verify sample transformation
        print("\nFeature Engineering steps completed successfully.")
        
    except Exception as e:
        print(f"Error: {e}")
