import torch
import torch.nn as nn
from transformers import BertModel

class HybridSentimentModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', gru_hidden_dim=256, num_heads=8, dropout=0.1):
        super(HybridSentimentModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # Freezing some BERT layers if needed (optional)
        # for param in self.bert.parameters():
        #     param.requires_grad = False
            
        self.gru = nn.GRU(
            input_size=768, 
            hidden_size=gru_hidden_dim, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        
        # BiGRU output is 2 * gru_hidden_dim
        self.attention = nn.MultiheadAttention(
            embed_dim=2 * gru_hidden_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * gru_hidden_dim, 1) # Binary classification
        
    def forward(self, input_ids, attention_mask):
        # 1. BERT Backbone
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state # [batch_size, seq_len, 768]
        
        # 2. Bidirectional GRU
        gru_output, _ = self.gru(sequence_output) # [batch_size, seq_len, 2 * gru_hidden_dim]
        
        # 3. Multi-Head Attention
        # query, key, value = gru_output
        attn_output, _ = self.attention(gru_output, gru_output, gru_output)
        
        # 4. Global Average Pooling (or Max Pooling) over the sequence
        pooled_output = torch.mean(attn_output, dim=1)
        
        # 5. Classification Head
        logits = self.fc(self.dropout(pooled_output))
        return logits
