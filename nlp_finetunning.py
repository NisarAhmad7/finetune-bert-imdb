
# pip install transformers datasets torch

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import load_dataset


model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 2 classes: positive/negative


dataset = load_dataset("imdb")


def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize, batched=True)

training_args = TrainingArguments(
    output_dir="./finetuned-bert",       # where to save the model
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"].shuffle(seed=42).select(range(1000)),  # small sample for demo
    eval_dataset=tokenized_dataset["test"].shuffle(seed=42).select(range(500))
)


trainer.train()

model.save_pretrained("./finetuned-bert")
tokenizer.save_pretrained("./finetuned-bert")

print("✅ Finetuning complete! Model saved in ./finetuned-bert")