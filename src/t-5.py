# using t-5 for training neurips mia model 

from transformers import AutoTokenizer, AutoModelWithLMHead, T5ForConditionalGeneration
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch



def read_seq_split(split_dir): # read the sequence and split it into chunks
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["", ""]: # for each label
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text())

    return texts, labels

# replace with the sequence data we want to use 
train_texts, train_labels = read_seq_split('figures/train') 
test_texts, test_labels = read_seq_split('figures/test')

# wraps input validation and application to input data into a single call
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

class MIADataset(torch.utils.data.Dataset): # create a custom dataset for neurips mia model
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# assuming we want to use trainer in leiu of custom pytorch trainer

# need to change training args based on raz input on the model
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

model = AutoModelWithLMHead.from_pretrained("t5-small")

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

trainer.train()



'''
# example on how to use the model

tokenizer = AutoTokenizer.from_pretrained("t5-small")

model = AutoModelWithLMHead.from_pretrained("t5-small")

input_ids = tokenizer("Translate English to French: Have you eaten yet?", return_tensors="pt").input_ids 
# input_ids = tokenizer('translate English to German: The house is wonderful.', return_tensors='pt').input_ids
outputs = model.generate(input_ids)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
'''