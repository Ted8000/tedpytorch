import torch
from transformers import *
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import dataloader, Dataset, dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import classification_report


# 数据读取
df_train = pd.read_csv('intent_train.txt', sep='\t', header=None)
df_test = pd.read_csv('intent_test.txt', sep='\t', header=None)
df_train = shuffle(df_train)
df_test = shuffle(df_test)
train_x, train_y = df_train[1].tolist(), df_train[0].tolist()
test_x, test_y = df_test[1].tolist(), df_test[0].tolist()
le = LabelEncoder()
le.fit(train_y)
train_y = le.transform(train_y)
test_y = le.transform(test_y)

num_class = len(set(train_y))
lr = 1e-5
batch_size = 32
epochs = 5

hidden = 768
pretrain_model = 'bert-base-chinese'
tokenizer_base = BertTokenizer.from_pretrained(pretrain_model, cache_dir='./cache')
model_base = torch.load('bert_base.ckpt')

# Teacher model不参与训练
for p in model_base.parameters():
    p.requires_grad = False

class BERTClass(torch.nn.Module):
    def __init__(self, pretrain_name, hidden, num_class):
        super(BERTClass, self).__init__()
        self.l1 = AutoModel.from_pretrained(pretrain_name, cache_dir='./cache')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(hidden, num_class)
    
    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output
    
class CustomDataset(Dataset):
    def __init__(self, tokenizer_t, tokenizer_s, train_x, train_y):
        self.tokenizer_t = tokenizer_t
        self.tokenizer_s = tokenizer_s
        self.train_x = train_x
        self.train_y = train_y

    def __len__(self):
        return len(self.train_x)
  
    def __getitem__(self, index):
        batch_t = self.tokenizer_t(self.train_x[index], padding='max_length', truncation=True, max_length=32, return_tensors="pt")
        batch_t['label'] = torch.tensor(self.train_y[index])
        batch_s = self.tokenizer_s(self.train_x[index], padding='max_length', truncation=True, max_length=32, return_tensors="pt")
        batch_s['label'] = torch.tensor(self.train_y[index])
        return batch_t, batch_s

hidden = 312
pretrain_model = 'voidful/albert_chinese_tiny'
tokenizer_albert_tiny = BertTokenizer.from_pretrained(pretrain_model, cache_dir='./cache')
model_albert_tiny = BERTClass(pretrain_model, hidden, num_class)


training_set = CustomDataset(tokenizer_base, tokenizer_albert_tiny, train_x, train_y)
test_set = CustomDataset(tokenizer_base, tokenizer_albert_tiny, test_x, test_y)

# 优化器
optimizer = AdamW(model_albert_tiny.parameters(), lr=lr)

# 损失函数
criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.KLDivLoss()


training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
testing_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_base.eval()
model_albert_tiny.train()
model_albert_tiny.to(device)

loss_ = []
total_step = len(training_loader)

for epoch in range(epochs):
    for i, (batch_t, batch_s) in enumerate(training_loader):  
        # Move tensors to the configured device
        # teacher forward
        input_ids = batch_t['input_ids'].squeeze(1).to(device)
        token_type_ids = batch_t['token_type_ids'].squeeze(1).to(device)
        attention_mask = batch_t['attention_mask'].squeeze(1).to(device)
#         labels = batch_t['label'].to(device)

        # Forward pass
        outputs_t = model_base(input_ids, token_type_ids, attention_mask)[0]
        
        # student forward
        input_ids = batch_s['input_ids'].squeeze(1).to(device)
        token_type_ids = batch_s['token_type_ids'].squeeze(1).to(device)
        attention_mask = batch_s['attention_mask'].squeeze(1).to(device)
        labels = batch_s['label'].to(device)

        # Forward pass
        outputs_s = model_albert_tiny(input_ids, token_type_ids, attention_mask)
        
        # hard label loss
        loss1 = criterion1(outputs_s, labels)
        
        # soft label loss
#         T = 2
        alpha = 0.5
#         outputs_S = F.log_softmax(outputs_s/T, dim=1)
#         outputs_T = F.softmax(outputs_t/T, dim=1)
        loss2 = criterion2(outputs_s, outputs_t)
        
        # 综合loss
        loss = loss1*(1-alpha) + loss2*alpha

        loss_.append(loss.item())

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 1000 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, epochs, i+1, total_step, loss.item()))
    
    with torch.no_grad():
        correct = 0
        total = 0
        for (batch_t, batch_s) in valid_loader:
            input_ids = batch_s['input_ids'].squeeze(1).to(device)
            token_type_ids = batch_s['token_type_ids'].squeeze(1).to(device)
            attention_mask = batch_s['attention_mask'].squeeze(1).to(device)
            labels = batch_s['label'].to(device)
            outputs = model_albert_tiny(input_ids, token_type_ids, attention_mask)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the valid data: {} %'.format(100 * correct / total))


    def torch_predict(model, test_x, test_y):
        y_pred = []
        y_real = []
        with torch.no_grad():
            correct = 0
            total = 0
            for (batch_t, batch_s) in valid_loader:
                input_ids = batch_s['input_ids'].squeeze(1).to(device)
                token_type_ids = batch_s['token_type_ids'].squeeze(1).to(device)
                attention_mask = batch_s['attention_mask'].squeeze(1).to(device)
                labels = batch_s['label'].to(device)
                outputs = model_albert_tiny(input_ids, token_type_ids, attention_mask)
                _, predicted = torch.max(outputs.data, 1)
                y_pred.extend(predicted.cpu())
                y_real.extend(labels.cpu())
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        a = []
        b = []
        for i in range(len(y_pred)):
            a.append(y_pred[i].tolist())
            b.append(y_real[i].tolist())
        a = le.inverse_transform(a).tolist()
        b = le.inverse_transform(b).tolist()
        print(classification_report(b, a))
        print('Accuracy of the network on the valid data: {} %'.format(100 * correct / total))
        return b, a
    
y_real, y_pred = torch_predict(model_albert_tiny, test_x, test_y)

torch.save(model_albert_tiny.state_dict(), 'model_albert_tiny.ckpt')
