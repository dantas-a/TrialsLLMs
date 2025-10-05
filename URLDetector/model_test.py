from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from utils import compute_metrics

# load the dataset
print("Loading Dataset...")
dataset_dict = load_dataset("shawhin/phishing-site-classification")

model_path = "bert-phishing-classifier_teacher/final_model"

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Loading the model for classification with 2 labels
print("Loading Model...")
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def preprocess_function(examples):
    # return tokenized text with truncation
    return tokenizer(examples['text'], truncation=True)

# Here we have the tokenized dataset
print("Processing dataset...")
tokenized_data = dataset_dict.map(preprocess_function,batched=True)

trainer = Trainer(model=model, tokenizer=tokenizer)

print("Making Predictions...")
predictions = trainer.predict(tokenized_data["test"])

logits = predictions.predictions
labels = predictions.label_ids

metrics = compute_metrics((logits, labels))
print(metrics)