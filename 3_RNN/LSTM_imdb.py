import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import time
import json
from datetime import datetime

# 1. 导入数据
tokenizer = get_tokenizer("basic_english")

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

# 加载数据集
train_iter, test_iter = IMDB()

# 构建词汇表
vocab = build_vocab_from_iterator(yield_tokens(train_iter), min_freq=10)
unk_index = vocab["<unk>"] if "<unk>" in vocab else len(vocab)
vocab.insert_token("<unk>", unk_index)
vocab.set_default_index(unk_index)
train_iter, test_iter = IMDB()


# 2.数据处理
def process_data(data_iter):
    processed_data = []
    for label, text in data_iter:
        tokens = tokenizer(text)
        if len(tokens) > 500:  # 限制序列长度为500
            tokens = tokens[:500]
        tensor = torch.tensor([vocab[token] for token in tokens], dtype=torch.long)
        label = int(label) - 1
        processed_data.append((tensor, label))
    return processed_data

def collate_batch(batch):
    text_list, labels = zip(*batch)
    text_list = pad_sequence(text_list, batch_first=True)
    labels = torch.tensor(labels, dtype=torch.long)
    return text_list, labels


# 3.定义lstm模型
class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, 2)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        # 使用最后一个时间步的前向和后向隐藏状态
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return self.fc(self.dropout(hidden))


# 4.训练模型函数
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for text, labels in train_loader:
        text, labels = text.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(text)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


# 5.评估函数
def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for text, labels in test_loader:
            text, labels = text.to(device), labels.to(device)
            output = model(text)
            total_loss += criterion(output, labels).item()
            correct += (output.argmax(1) == labels).sum().item()
    return total_loss / len(test_loader), correct / len(test_loader.dataset)


# 6.训练
EMBEDDING_DIM = 50
HIDDEN_DIM = 128
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data = process_data(train_iter)
test_data = process_data(test_iter)
train_loader = DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=collate_batch)
model = LSTM(len(vocab), EMBEDDING_DIM, HIDDEN_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

start_time = time.time()
best_accuracy = 0
history = []
for epoch in range(EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion, device)
    test_loss, accuracy = evaluate(model, test_loader, criterion, device)
    history.append(
        {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "accuracy": accuracy,
        }
    )
    print(f"Epoch: {epoch+1}/{EPOCHS}")
    print(f"\tTrain Loss: {train_loss:.3f}")
    print(f"\tTest Loss: {test_loss:.3f}")
    print(f"\tAccuracy: {accuracy:.3f}")
    # 保存最佳模型
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "accuracy": accuracy,
            },
            "./LSTM_imdb_model.pth",
        )
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print("Early stopping!")
            break

end_time = time.time()
training_time = end_time - start_time
print(f"训练总耗时: {end_time - start_time:.2f} 秒")
print(f"最佳准确率: {best_accuracy:.3f}")

# 保存训练历史到日志文件
log_data = {
    "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_parameters": {
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
    },
    "best_accuracy": best_accuracy,
    "training_time": training_time,
    "history": history,
}
# 保存为JSON
with open("./LSTM_imdb_training_log.json", "w", encoding="utf-8") as f:
    json.dump(log_data, f, ensure_ascii=False, indent=4)

# 保存为txt
with open("./LSTM_imdb_training_log.txt", "w", encoding="utf-8") as f:
    f.write(f"训练日期: {log_data['training_date']}\n")
    f.write(f"模型参数:\n")
    f.write(f"  Embedding Dimension: {EMBEDDING_DIM}\n")
    f.write(f"  Hidden Dimension: {HIDDEN_DIM}\n")
    f.write(f"  Batch Size: {BATCH_SIZE}\n")
    f.write(f"  Epochs: {EPOCHS}\n")
    f.write(f"  Learning Rate: {LEARNING_RATE}\n\n")
    f.write(f"训练结果:\n")
    f.write(f"  最佳准确率: {best_accuracy:.3f}\n")
    f.write(f"  训练时间: {training_time:.2f} 秒\n\n")
    f.write("训练历史:\n")
    for record in history:
        f.write(f"Epoch {record['epoch']}:\n")
        f.write(f"  Train Loss: {record['train_loss']:.3f}\n")
        f.write(f"  Test Loss: {record['test_loss']:.3f}\n")
        f.write(f"  Accuracy: {record['accuracy']:.3f}\n")