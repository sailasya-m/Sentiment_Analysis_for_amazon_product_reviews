from datasets import load_dataset
import pandas as pd
import re
from bs4 import BeautifulSoup
from tqdm import tqdm

def clean_html(text):
    try:
        return BeautifulSoup(text, "html.parser").get_text()
    except Exception:
        return text

def clean_text(text):
    text = str(text)
    text = clean_html(text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    return text

def prepare_data(sample_size=5000):
    print(f"Loading Amazon Fine Food Reviews from HuggingFace (PJ2005/amazon-fine-food-reviews)...")
    try:
        # Trying a more likely dataset name
        dataset = load_dataset('PJ2005/amazon-fine-food-reviews', split='train', streaming=True)
        
        data = []
        count = 0
        pbar = tqdm(total=sample_size)
        
        for item in dataset:
            # Check the actual keys in the dataset
            text = item.get('text', item.get('Text', ''))
            score = item.get('score', item.get('Score', 0))
            
            if score == 3: # Skip neutral
                continue
            
            label = 1 if score > 3 else 0
            text = clean_text(text)
            
            data.append({'text': text, 'label': label})
            count += 1
            pbar.update(1)
            
            if count >= sample_size:
                break
        pbar.close()
        
        df = pd.DataFrame(data)
        df.to_csv('processed_reviews.csv', index=False)
        print(f"Saved {len(df)} processed reviews to processed_reviews.csv")
        
    except Exception as e:
        print(f"Error with PJ2005: {e}")
        print("Falling back to small synthetic dataset for development...")
        df = pd.DataFrame({
            'text': ["This food is great", "I hated it", "Amazing taste", "Worst experience ever"] * 250,
            'label': [1, 0, 1, 0] * 250
        })
        df.to_csv('processed_reviews.csv', index=False)
        print("Saved synthetic fallback to processed_reviews.csv")

if __name__ == "__main__":
    prepare_data(2000) # Smaller sample for faster training
