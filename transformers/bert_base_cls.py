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
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df_train = pd.read_csv('./train.txt', sep='\t', header=None)
df_test = pd.read_csv('./test.txt', sep='\t', header=None)
df_train = shuffle(df_train)
df_test = shuffle(df_test)
train_x, train_y = df_train[1].tolist(), df_train[0].tolist()
test_x, test_y = df_test[1].tolist(), df_test[0].tolist()
le = LabelEncoder()
le.fit(train_y)
train_y = le.transform(train_y)
test_y = le.transform(test_y)

    
class CustomDataset(Dataset):
    def __init__(self, tokenizer, train_x, train_y):
        self.tokenizer = tokenizer
        self.train_x = train_x
        self.train_y = train_y

    def __len__(self):
        return len(self.train_x)
  
    def __getitem__(self, index):
        batch = self.tokenizer(self.train_x[index], padding='max_length', truncation=True, max_length=32, return_tensors="pt")
        batch['label'] = torch.tensor(self.train_y[index])
        return batch
    

num_class = len(set(train_y))
lr = 5e-5
batch_size = 32
epochs = 5

pretrain_model = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(pretrain_model, cache_dir='../../cache')
model = BertForSequenceClassification.from_pretrained(pretrain_model, num_labels=num_class, cache_dir='../../cache')

training_set = CustomDataset(tokenizer, train_x, train_y)
test_set = CustomDataset(tokenizer, test_x, test_y)

optimizer = AdamW(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
testing_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)



model.to(device)

loss_ = []
total_step = len(training_loader)

for epoch in range(epochs):
    model.train()
    for i, batch in enumerate(training_loader):  
    # Move tensors to the configured device
        input_ids = batch['input_ids'].squeeze(1).to(device)
        token_type_ids = batch['token_type_ids'].squeeze(1).to(device)
        attention_mask = batch['attention_mask'].squeeze(1).to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        outputs = model(input_ids, token_type_ids, attention_mask)[0]
        
        loss = criterion(outputs, labels)
        
        loss_.append(loss.item())
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 1000 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                .format(epoch+1, epochs, i+1, total_step, loss.item()))

    model.eval()
    best_acc = 0
    
    with torch.no_grad():
        correct = 0
        total = 0
        for i, batch in enumerate(valid_loader):
            input_ids = batch['input_ids'].squeeze(1).to(device)
            token_type_ids = batch['token_type_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, token_type_ids, attention_mask)[0]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc_now = correct / total   
        
        if acc_now > best_acc:
            best_acc = acc_now
            model_best = copy.deepcopy(model)
            
        print('Accuracy of the network on the valid data: {} %'.format(100 * correct / total))
    


torch.save(model_best, 'bert_base_broker_813.ckpt')
