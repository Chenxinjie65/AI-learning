import torch
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch.nn as nn
import random
import math

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Transformer模型
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, nhead, num_layers, dim_feedforward=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(embedding_dim, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, text):
        padding_mask = text == 0
        embedded = self.embedding(text)
        embedded = self.pos_encoder(embedded)
        embedded = self.dropout(embedded)
        output = self.transformer_encoder(embedded, src_key_padding_mask=padding_mask)
        output = output.mean(dim=1)
        return self.fc(self.dropout(output))

def process_text(text, tokenizer, vocab):
    tokens = tokenizer(text)
    if len(tokens) > 500:  # 限制序列长度为500
        tokens = tokens[:500]
    return torch.tensor([vocab[token] for token in tokens], dtype=torch.long).unsqueeze(0)

# 数据处理
tokenizer = get_tokenizer("basic_english")
train_iter, test_iter = IMDB()
vocab = build_vocab_from_iterator(
    (tokenizer(text) for _, text in train_iter 
    if len(tokenizer(text)) > 0),
    min_freq=10
)
vocab.insert_token("<unk>", len(vocab))
vocab.set_default_index(len(vocab) - 1)

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDING_DIM = 64
N_HEAD = 4
N_LAYERS = 2

model = TransformerClassifier(
    len(vocab), EMBEDDING_DIM, N_HEAD, N_LAYERS
).to(device)
checkpoint = torch.load("./Transformer_imdb_model.pth")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 测试
print("测试示例：")
print("-" * 50)

_, test_iter = IMDB()
test_data = list(test_iter)
random.shuffle(test_data)
count = 0

with torch.no_grad():
    for label, text in test_data[:20]:
        input_tensor = process_text(text, tokenizer, vocab).to(device)
        
        # 预测
        output = model(input_tensor)
        predicted_label = output.argmax(1).item()
        true_label = int(label) - 1
        
        # 显示结果
        print(f"样本 {count + 1}:")
        print(f"文本: {text[:500]}...")  
        print(f"真实标签: {'正面' if true_label == 1 else '负面'}")
        print(f"预测标签: {'正面' if predicted_label == 1 else '负面'}")
        print(f"预测{'正确' if predicted_label == true_label else '错误'}")
        print("-" * 50)
        
        count += 1