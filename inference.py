import torch
from transformers import BertTokenizer
from model import HybridSentimentModel
import torch.nn.functional as F

def predict_sentiment(text, model, tokenizer, device, max_len=128):
    model.eval()
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        # Use sigmoid because the model output is raw logits for binary classification
        probability = torch.sigmoid(outputs).item()
        prediction = 1 if probability >= 0.5 else 0
        
    return prediction, probability

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Initialize and load model
    model = HybridSentimentModel()
    if torch.cuda.is_available():
        model.load_state_dict(torch.load('hybrid_bert_model.pth'))
    else:
        model.load_state_dict(torch.load('hybrid_bert_model.pth', map_location=device))
    model.to(device)

    test_reviews = [
        "This product is absolutely amazing! I love the quality and the taste.",
        "Terrible experience. The box was broken and the food tasted stale.",
        "It's okay, but I've had better. Not really worth the price.",
        "Highly recommended for everyone who loves gourmet snacks!"
    ]

    print("\n--- Model Inference Results ---")
    for review in test_reviews:
        pred, prob = predict_sentiment(review, model, tokenizer, device)
        sentiment = "POSITIVE" if pred == 1 else "NEGATIVE"
        print(f"\nReview: {review}")
        print(f"Analysis: {sentiment} (Confidence: {prob:.2%})")
