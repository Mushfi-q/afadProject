import pandas as pd
import re
import string

def aggressive_clean(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Remove common email header prefixes and technical metadata
    # The current dataset has lowercased words like 'messageid', 'mimeversion', 'contenttype'
    patterns = [
        r'messageid \d+', 
        r'mimeversion \d+', 
        r'contenttype textplain.*', 
        r'contenttransferencoding \w+',
        r'xfrom .*', r'xto .*', r'xcc .*', r'xbcc .*', 
        r'xfolder .*', r'xorigin .*', r'xfilename .*',
        r'javamailevansthyme', r'charsetusascii',
        r'date \w{3} \d+ \w{3} \d+ \d+ \d+ \w+', # Space-separated date header
        r'from .* to .* subject .*', # From/To/Subject chain
        r'pdt', r'pst', r'0700', r'0800' # Common timezone/offset artifacts
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text)
    
    # 2. General cleaning (URLs, punctuation, extra whitespace)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def reclean_dataset():
    input_file = 'data/final_afad_dataset.csv'
    output_file = 'data/final_afad_dataset_cleaned.csv'
    
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    
    print("Applying aggressive cleaning to strip headers and metadata...")
    df['message'] = df['message'].apply(aggressive_clean)
    
    # Remove messages that became too short after stripping headers
    df = df[df['message'].apply(lambda x: len(str(x).split()) >= 3)]
    
    print(f"Saving re-cleaned dataset to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Optimization complete.")

if __name__ == "__main__":
    reclean_dataset()
