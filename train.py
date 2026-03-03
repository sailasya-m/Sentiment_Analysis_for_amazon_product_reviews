import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, get_linear_schedule_with_warmup
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from model import HybridSentimentModel
import os

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }

def train():
    # Parameters
    MAX_LEN = 128
    BATCH_SIZE = 16
    EPOCHS = 2
    LEARNING_RATE = 2e-5
    
    # Check for processed data
    if not os.path.exists('processed_reviews.csv'):
        print("Error: processed_reviews.csv not found. Run preprocess.py first.")
        return
        
    df = pd.read_csv('processed_reviews.csv')
    df = df.dropna()
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    train_dataset = SentimentDataset(train_df.text.to_list(), train_df.label.to_list(), tokenizer, MAX_LEN)
    val_dataset = SentimentDataset(val_df.text.to_list(), val_df.label.to_list(), tokenizer, MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = HybridSentimentModel()
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    history = {'train_loss': []}
    
    for epoch in range(EPOCHS):
        model.train()
        print(f"Epoch {epoch+1}/{EPOCHS}")
        epoch_loss = 0
        
        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss/len(train_loader)
        history['train_loss'].append(avg_loss)
        print(f"Train Loss: {avg_loss:.4f}")
        
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), history['train_loss'], marker='o', label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_history.png')
    print("Training history plot saved to training_history.png")
        
    # Evaluation
    model.eval()
    val_preds = []
    val_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            preds = torch.sigmoid(outputs).round().flatten()
            
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
            
    # Evaluation report
    print("\nEvaluation Report:")
    report = classification_report(val_labels, val_preds)
    print(report)
    print(f"Accuracy: {accuracy_score(val_labels, val_preds):.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(val_labels, val_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Hybrid BERT-BiGRU-Attention')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to confusion_matrix.png")
    
    # Save model
    torch.save(model.state_dict(), 'hybrid_bert_model.pth')
    print("Model saved to hybrid_bert_model.pth")

if __name__ == "__main__":
    train()
