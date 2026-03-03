import torch
from flask import Flask, request, jsonify, send_from_directory
from transformers import BertTokenizer
from model import HybridSentimentModel
import os

app = Flask(__name__)

# Load model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = HybridSentimentModel()
model_path = os.path.join(os.path.dirname(__file__), 'hybrid_bert_model.pth')

if torch.cuda.is_available():
    model.load_state_dict(torch.load(model_path))
else:
    model.load_state_dict(torch.load(model_path, map_location=device))

model.to(device)
model.eval()

def predict_sentiment(text, max_len=128):
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
        probability = torch.sigmoid(outputs).item()
        prediction = 1 if probability >= 0.5 else 0
        
    return prediction, probability

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    prediction, probability = predict_sentiment(text)
    sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
    
    return jsonify({
        'sentiment': sentiment,
        'confidence': probability
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)
