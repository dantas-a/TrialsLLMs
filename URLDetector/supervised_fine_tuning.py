from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from utils import compute_metrics
from transformers import DataCollatorWithPadding

# load the dataset
print("Loading Dataset...")
dataset_dict = load_dataset("shawhin/phishing-site-classification")

# Name of the model we want to load
model_path = "google-bert/bert-base-uncased"

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# convert id to labels 
id2label = {0: "Safe",1: "Not Safe"}
label2id = {"Safe": 0, "Not Safe": 1}

# Loading the model for classification with 2 labels
print("Loading Model...")
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2, id2label=id2label,label2id=label2id)

# Freeze every parameter in the model (because my computer can't handle tuning so much parameters)
print("Freezing everything...")
for name, param in model.base_model.named_parameters():
    param.requires_grad = False

# Unfreeze only some parameters in the model
print("Unfreezing Pooling layers...")
for name, param in model.base_model.named_parameters():
    if "pooler" in name:
        param.requires_grad = True
         
def preprocess_function(examples):
    # return tokenized text with truncation
    return tokenizer(examples['text'], truncation=True)

# Here we have the tokenized dataset
print("Processing dataset...")
tokenized_data = dataset_dict.map(preprocess_function,batched=True)
     
# Important : in each batch, the tokenized data must have the same length so it will allow padding.   
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# hyperparameters
lr = 2e-4
batch_size = 8
num_epoch = 10

training_args = TrainingArguments(
    output_dir = "bert-phishing-classifier_teacher",
    learning_rate = lr,
    per_device_train_batch_size = batch_size,
    per_device_eval_batch_size = batch_size,
    num_train_epochs = num_epoch,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)
   
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
) 

print("Training...")
trainer.train()

trainer.save_model("bert-phishing-classifier_teacher/final_model")
tokenizer.save_pretrained("bert-phishing-classifier_teacher/final_model")

