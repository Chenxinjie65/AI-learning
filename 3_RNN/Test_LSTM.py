import torch
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch.nn as nn
import random

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return self.fc(self.dropout(hidden))

def process_text(text, tokenizer, vocab):
    tokens = tokenizer(text)
    if len(tokens) > 500:
        tokens = tokens[:500]
    return torch.tensor([vocab[token] for token in tokens], dtype=torch.long).unsqueeze(0)

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
EMBEDDING_DIM = 50
HIDDEN_DIM = 128

model = LSTM(len(vocab), EMBEDDING_DIM, HIDDEN_DIM).to(device)
checkpoint = torch.load("./LSTM_imdb_model.pth")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 测试
print("测试示例：")
print("-" * 50)

_, test_iter = IMDB()
test_data = list(test_iter)
random.shuffle(test_data) #打乱顺序
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