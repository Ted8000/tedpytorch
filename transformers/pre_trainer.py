# 预训练
from transformers import *
from typing import Dict, List, Optional
from torch.utils.data import Dataset
import torch
from torch.nn import CrossEntropyLoss
import os
import argparse

# run
# python pre_trainer.py -d data/raw_data -s ./models/roberta_2.pt

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--path1", nargs='?', const=1, type=str, default="/home/lupinsu001/data/classification/train",
                        help="train path load")
parser.add_argument("-model", "--model_path", nargs='?', const=1, type=str, default="bert-base-chinese",
                        help="test path load")
parser.add_argument("-token", "--token_path", nargs='?', const=1, type=str, default="bert-base-chinese",
                        help="model path load")
parser.add_argument("-m", "--path3", nargs='?', const=1, type=str, default="/home/lupinsu001/EsperBERTo/checkpoint-150000/",
                        help="model path load")
parser.add_argument("-s", "--path4", nargs='?', const=1, type=str, default="",
                        help="model path save")
parser.add_argument("-g", "--gpu", nargs='?', const=1, type=int, default=0,
                        help="model path save")
args = parser.parse_args()


class MyDataset(Data.Dataset):
    def __init__(self,filepath, tokenizer):
        number = 0
        with open(filepath,"r") as f:
            # 获得训练数据的总行数
            for _ in tqdm(f,desc="load training dataset"):
                number+=1
        self.number = number
        self.fopen = open(filepath,'r')
        self.tokenizer = tokenizer
    def __len__(self):
        return self.number
    def __getitem__(self,index):
        line = self.fopen.__next__()
        data = self.tokenizer(line, truncation=True, padding='max_length', max_length=32, return_tensors='pt')
        return {'input_ids':data['input_ids'].squeeze()}

class Raw_data_load(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        
        files = glob.glob(file_path)

        self.dataset = load_dataset('text', data_files=files, split='train')
        
        def encode(examples):
            return tokenizer(examples['text'], add_special_tokens=True, truncation=True, max_length=block_size)
        self.examples = self.dataset.map(encode, batched=True)
        self.examples.set_format(type='torch', columns=['input_ids'])
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]
    
class BERT_ML_Class(torch.nn.Module):
    def __init__(self, pretrain_name, num_class, cache_dir=None):
        super(BERT_ML_Class, self).__init__()
        self.num_class = num_class
        
        if cache_dir == None:
            self.l1 = BertModel.from_pretrained(pretrain_name)
        else:
            self.l1 = BertModel.from_pretrained(pretrain_name, cache_dir=cache_dir)
        hidden = self.l1.pooler.dense.out_features
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(hidden, self.num_class)
    
    def forward(self, input_ids, attention_mask, labels):
        sequence_output, output_1= self.l1(input_ids, attention_mask=attention_mask)
        
        output_2 = self.l2(sequence_output)
        prediction_scores = self.l3(output_2)
        
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.num_class), labels.view(-1))
        
        return ((masked_lm_loss,) + (output_1,)) if masked_lm_loss is not None else output_1

tokenizer = BertTokenizerFast.from_pretrained(args.token_path)
model = BERT_ML_Class(args.model_path, tokenizer.vocab_size)
dataset  = Raw_data_load(tokenizer, args.data_path, block_size=64)
print('length of dataset:', len(dataset))


# model load
# config = RobertaConfig(
#     vocab_size=21128,
#     max_position_embeddings=32,
#     num_attention_heads=6,
#     num_hidden_layers=4,
#     type_vocab_size=1,
# )
# model = RobertaForMaskedLM(config=config)

print('model_parameters_nums:', model.num_parameters())

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)


training_args = TrainingArguments(
    output_dir="./EsperBERTo",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=128,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    prediction_loss_only=True,
)

trainer.train()

# save model
trainer.save_model('./models/pretrained_model.ckpt')
torch.save(model, './models/pre_train_model.pt')