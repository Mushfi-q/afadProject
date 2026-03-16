import pandas as pd
import re
import string
import os

def clean_text(text):
    """
    Cleans text by removing email headers, converting to lowercase, 
    removing URLs, punctuation, and extra whitespace.
    """
    if not isinstance(text, str):
        return ""
    
    # Remove email headers (lines starting with 'Word:')
    # This handles Message-ID, Subject, From, To, etc. in Enron/Synthetic data
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

def prepare_dataset():
    # 1. Load the three CSV files
    # Note: Enron dataset is very large, so we only load necessary columns
    print("Loading datasets...")
    
    # SMS Spam Collection
    # Usually has encoding issues, 'latin-1' is common for this dataset
    sms_df = pd.read_csv('Text/spam.csv', encoding='latin-1')
    
    # Enron Email Dataset
    # Loading only the 'message' column to save memory
    enron_df = pd.read_csv('Text/emails.csv', usecols=['message'])
    
    # Synthetic Familiarity Attack Dataset
    synthetic_df = pd.read_csv('Text/synthetic_familiarity_clean.csv')

    # 3. Standardize column names
    print("Standardizing columns...")
    
    # SMS dataset: labels are in 'v1', messages in 'v2'
    sms_df = sms_df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})
    
    # Enron dataset: add label column
    enron_df['label'] = 0
    enron_df = enron_df[['message', 'label']]
    
    # Synthetic dataset: ensure exactly message and label
    synthetic_df = synthetic_df[['message', 'label']]

    # 4. & 5. Convert labels
    print("Converting labels...")
    
    # SMS: ham -> 0, spam -> 1
    sms_df['label'] = sms_df['label'].map({'ham': 0, 'spam': 1})
    
    # 7. Concatenate the three datasets
    print("Merging datasets...")
    final_df = pd.concat([sms_df, enron_df, synthetic_df], ignore_index=True)

    # 8. Clean the text messages
    print("Cleaning text (lowercase, URLs, punctuation, whitespace)...")
    final_df['message'] = final_df['message'].apply(clean_text)

    # 9. Remove duplicate and empty messages
    print("Removing duplicates and empty messages...")
    initial_count = len(final_df)
    final_df.drop_duplicates(subset=['message'], inplace=True)
    final_df = final_df[final_df['message'].notna()]
    final_df = final_df[final_df['message'].str.strip() != ""]
    
    # NEW: Filter by message length (minimum 3 words)
    print("Filtering messages with less than 3 words...")
    final_df = final_df[final_df['message'].apply(lambda x: len(str(x).split()) >= 3)]
    cleaned_count = len(final_df)

    # 10. Downsampling Safe Class (Label 0)
    print("Downsampling Safe class to 5000 records...")
    safe_df = final_df[final_df['label'] == 0]
    attack_df = final_df[final_df['label'] == 1]
    
    # Randomly sample 5000 safe messages if available
    n_safe = min(len(safe_df), 5000)
    safe_df_downsampled = safe_df.sample(n=n_safe, random_state=42)
    
    # Combine back
    final_df = pd.concat([safe_df_downsampled, attack_df], ignore_index=True)

    # 11. Shuffle the dataset randomly
    print("Shuffling balanced dataset...")
    # frac=1 shuffles the whole thing
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 12. Train/Test Split (80% Train, 20% Test)
    print("Performing 80/20 train/test split...")
    train_size = int(0.8 * len(final_df))
    train_df = final_df.iloc[:train_size]
    test_df = final_df.iloc[train_size:]

    # 13. Print dataset statistics
    total_rows = len(final_df)
    safe_messages = len(final_df[final_df['label'] == 0])
    attack_messages = len(final_df[final_df['label'] == 1])
    
    print("\n--- Final Dataset Statistics ---")
    print(f"Total rows: {total_rows}")
    print(f"Safe messages (label 0): {safe_messages}")
    print(f"Attack messages (label 1): {attack_messages}")
    print(f"Train set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    print(f"Null messages: {final_df['message'].isnull().sum()}")
    print(f"Safe : Attack Ratio: {safe_messages/attack_messages:.2f} : 1")

    # 14. Save the final datasets
    final_df.to_csv('final_afad_text_dataset.csv', index=False)
    train_df.to_csv('afad_train.csv', index=False)
    test_df.to_csv('afad_test.csv', index=False)
    
    print("\nFiles saved:")
    print("- final_afad_text_dataset.csv")
    print("- afad_train.csv")
    print("- afad_test.csv")

if __name__ == "__main__":
    prepare_dataset()

if __name__ == "__main__":
    prepare_dataset()
