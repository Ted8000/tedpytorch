from datasets import load_dataset
from nlp import load_dataset
import glob
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

files = glob.glob('data/raw_data/shards/shard_*')

dataset = load_dataset('text', data_files=files, split='train')

def encode(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=32)
dataset = dataset.map(encode, batched=True)
dataset.set_format(type='torch', columns=['input_ids'])