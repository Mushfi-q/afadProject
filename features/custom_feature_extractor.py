import pandas as pd
try:
    from features.cue_dictionary import FAMILIARITY_CUES, URGENCY_CUES, EMOTIONAL_CUES, AUTHORITY_CUES, MONEY_TERMS, URGENCY_TERMS, PAYMENT_TERMS
except ImportError:
    from cue_dictionary import FAMILIARITY_CUES, URGENCY_CUES, EMOTIONAL_CUES, AUTHORITY_CUES, MONEY_TERMS, URGENCY_TERMS, PAYMENT_TERMS

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

def check_any_cue(text, cue_list):
    """
    Returns 1 if any cue from the list appears in the text, else 0.
    """
    if not isinstance(text, str):
        return 0
    
    text_lower = text.lower()
    for cue in cue_list:
        if cue.lower() in text_lower:
            return 1
    return 0

def extract_keyword_flags(df):
    """
    Computes binary keyword flags for each message.
    Used for Phase 2.1 & 2.2.
    """
    df_flags = pd.DataFrame()
    df_flags['has_money_term'] = df['message'].apply(lambda x: check_any_cue(x, MONEY_TERMS))
    df_flags['has_urgency'] = df['message'].apply(lambda x: check_any_cue(x, URGENCY_TERMS))
    df_flags['has_payment_term'] = df['message'].apply(lambda x: check_any_cue(x, PAYMENT_TERMS))
    return df_flags

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
