import pandas as pd
try:
    from features.cue_dictionary import FAMILIARITY_CUES, URGENCY_CUES, EMOTIONAL_CUES, AUTHORITY_CUES
except ImportError:
    from cue_dictionary import FAMILIARITY_CUES, URGENCY_CUES, EMOTIONAL_CUES, AUTHORITY_CUES

def count_cues(text, cue_list):
    """
    Counts how many cues from the list appear in the text.
    """
    if not isinstance(text, str):
        return 0
    
    text_lower = text.lower()
    count = 0
    for cue in cue_list:
        if cue.lower() in text_lower:
            count += 1
    return count

def extract_custom_features(df):
    """
    Computes psychology-based feature scores for each message in the dataframe.
    """
    print("Computing psychological feature counters...")
    
    df_features = pd.DataFrame()
    
    df_features['familiarity_score'] = df['message'].apply(lambda x: count_cues(x, FAMILIARITY_CUES))
    df_features['urgency_score'] = df['message'].apply(lambda x: count_cues(x, URGENCY_CUES))
    df_features['emotional_score'] = df['message'].apply(lambda x: count_cues(x, EMOTIONAL_CUES))
    df_features['authority_score'] = df['message'].apply(lambda x: count_cues(x, AUTHORITY_CUES))
    
    return df_features

if __name__ == "__main__":
    # Test with a sample message
    sample_text = "Hey bro urgent please help send money"
    
    features = {
        "familiarity": count_cues(sample_text, FAMILIARITY_CUES),
        "urgency": count_cues(sample_text, URGENCY_CUES),
        "emotional": count_cues(sample_text, EMOTIONAL_CUES),
        "authority": count_cues(sample_text, AUTHORITY_CUES)
    }
    
    print(f"Sample Text: '{sample_text}'")
    print(f"Computed Features: {list(features.values())}")
