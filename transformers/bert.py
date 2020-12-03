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
import os
import sys
from tqdm import tqdm
import argparse

# run
# python bert.py -d1 ./data/train_1111 -d2 ./data/test.csv -s ./models/roberta_2.pt

parser = argparse.ArgumentParser()
parser.add_argument("-d1", "--path1", nargs='?', const=1, type=str, default="/home/lupinsu001/data/classification/train",
                        help="train path load")
parser.add_argument("-d2", "--path2", nargs='?', const=1, type=str, default="/home/lupinsu001/data/classification/test",
                        help="test path load")
parser.add_argument("-token", "--token_path", nargs='?', const=1, type=str, default="/home/lupinsu001/cache/roberta/",
                        help="model path load")
parser.add_argument("-m", "--path3", nargs='?', const=1, type=str, default="/home/lupinsu001/cache/roberta/",
                        help="model path load")
parser.add_argument("-s", "--path4", nargs='?', const=1, type=str, default="/home/lupinsu001/data/models/model_tmp.pt",
                        help="model path save")
parser.add_argument("-label", "--label_path", nargs='?', const=1, type=str,  default="/home/lupinsu001/data/classification/label_save",
                        help="model path save")
parser.add_argument("-g", "--gpu", nargs='?', const=1, type=int, default=0,
                        help="model path save")

args = parser.parse_args()


device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

# sentence process
df_train = pd.read_csv(args.path1, sep='\t', header=None)
df_test = pd.read_csv(args.path2, sep='\t', header=None)
df_train = shuffle(df_train)
train_x, train_y = df_train.iloc[:,0].tolist(), df_train.iloc[:,1].tolist()
test_x, test_y = df_test.iloc[:,0].tolist(), df_test.iloc[:,1].tolist()

# label load
if os.path.exists(args.label_path):
    with open(args.label_path, 'r') as f:
        labels = f.read().split()
else:
    labels = set(train_y)|set(test_y)

label2id = {}
id2label = {}
for k,v in enumerate(labels):
    id2label[k] = v
    label2id[v] = k
    
train_y = list(map(lambda x:label2id[x], train_y))
test_y = list(map(lambda x:label2id[x], test_y))

# model load
class BERTClass(torch.nn.Module):
    def __init__(self, pretrain_name, num_class, cache_dir=None):
        super(BERTClass, self).__init__()
        if cache_dir == None:
            self.l1 = BertModel.from_pretrained(pretrain_name)
        else:
            self.l1 = BertModel.from_pretrained(pretrain_name, cache_dir=cache_dir)
        hidden = self.l1.pooler.dense.out_features
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(hidden, num_class)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        _, output_1= self.l1(input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output
    
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

num_class = len(set(labels))
lr = 5e-5
batch_size = 256
epochs = 10

tokenizer = BertTokenizer.from_pretrained(args.token_path)
model = BERTClass(args.path3, num_class)

training_set = CustomDataset(tokenizer, train_x, train_y)
test_set = CustomDataset(tokenizer, test_x, test_y)

optimizer = AdamW(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
stepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.75, last_epoch=-1)

training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=4)
valid_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)
testing_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)

model.train()
model.to(device)

loss_ = []
best_acc = 0
total_step = len(training_loader)
step_ = len(training_loader)//3

# model train
for epoch in range(epochs):
    model.train()
    for i, batch in enumerate(tqdm(training_loader)):  
    # Move tensors to the configured device
        input_ids = batch['input_ids'].squeeze(1).to(device)
        token_type_ids = batch['token_type_ids'].squeeze(1).to(device)
        attention_mask = batch['attention_mask'].squeeze(1).to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        outputs = model(input_ids, attention_mask, token_type_ids)
        
        loss = criterion(outputs, labels)
        
        loss_.append(loss.item())
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % step_ == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                .format(epoch+1, epochs, i+1, total_step, loss.item()))
    
    stepLR.step()
    
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for i, batch in enumerate(valid_loader):
            input_ids = batch['input_ids'].squeeze(1).to(device)
            token_type_ids = batch['token_type_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask, token_type_ids)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        now_acc = 100 * correct / total
        if now_acc > best_acc:
            best_model = copy.deepcopy(model)
            best_acc = now_acc
        print('Accuracy of the network on the valid data: {} %'.format(100 * correct / total))
        
torch.save(best_model, args.path4)