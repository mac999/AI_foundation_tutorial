import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM
from torch.optim import AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Load Pre-trained Tokenizer and Add Custom Token
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
custom_token = "[CoT]"
tokenizer.add_tokens([custom_token])  # Add custom token
print(f"Added token: {custom_token} -> ID: {tokenizer.convert_tokens_to_ids(custom_token)}")

# Step 2: Load Pre-trained Model and Resize Token Embeddings
model = BertForMaskedLM.from_pretrained("bert-base-uncased")
model.resize_token_embeddings(len(tokenizer))
model.to(device)

inference_text = "Question: What is 10 - 3?" # before training, we can test the model with a simple inference text
encoded_input = tokenizer(inference_text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")

input_ids = encoded_input["input_ids"].to(device)
attention_mask = encoded_input["attention_mask"].to(device)

with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    predictions = outputs.logits # [1, 128, 30523], [batch_size, sequence_length, vocab_size]
    predicted_ids = torch.argmax(predictions, dim=-1)
    decoded_output = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    print(f"Decoded Output: {decoded_output}")

# Step 3: Define CoT Dataset
class CoTDataset(Dataset):
    def __init__(self, tokenizer, texts, max_length=128):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # Split the text into input_text and labels_text using the custom token [CoT]
        split_text = text.split("[CoT]")
        input_text = split_text[0].strip() + " [CoT]"
        labels_text = split_text[1].strip() if len(split_text) > 1 else ""

        encoded_input = self.tokenizer(input_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt",) # Tokenize input text, padding and truncating to max_length, returning PyTorch tensors (example: input_ids=[101, 2023, 2003, 102], attention_mask=[1, 1, 1, 1, 0, 0])
        encoded_labels = self.tokenizer(labels_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt",)

        return {
            "input_ids": encoded_input["input_ids"].squeeze(0),
            "attention_mask": encoded_input["attention_mask"].squeeze(0),
            "labels": encoded_labels["input_ids"].squeeze(0),  # Use labels for loss calculation. squeeze(0) removes the batch dimension, making it a 1D tensor
        }

# Step 4: Sample CoT Dataset
texts = [
    "Question: What is 2 + 2? [CoT] Reasoning: First, we know that adding 2 to 2 gives 4. Final Answer: 4.",
    "Question: What is 5 * 3? [CoT] Reasoning: Multiply 5 by 3 to get 15. Final Answer: 15.",
    "Question: What is 10 - 3? [CoT] Reasoning: Subtract 3 from 10 to get 7. Final Answer: 7.",
    "Question: What is 7 + 5? [CoT] Reasoning: Adding 7 and 5 gives 12. Final Answer: 12.",
]

dataset = CoTDataset(tokenizer, texts)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Step 5: Define Optimizer and Device
optimizer = AdamW(model.parameters(), lr=5e-5)

# Step 6: Training Loop
model.train()
epochs = 100
for epoch in range(epochs):
    # print(f"Epoch {epoch + 1}/{epochs}")
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward Pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward Pass and Optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch}. Loss: {loss.item()}")

# Step 7: Save Fine-Tuned Model and Tokenizer
model_path = "./cot_bert_model"
tokenizer_path = "./cot_bert_tokenizer"

model.save_pretrained(model_path)
tokenizer.save_pretrained(tokenizer_path)

print(f"Model saved to {model_path}")
print(f"Tokenizer saved to {tokenizer_path}")

# Step 8: Load Fine-Tuned Model and Tokenizer for Inference
model = BertForMaskedLM.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
model.to(device)

# Step 9: Inference with Fine-Tuned Model
model.eval()

# Example inference text
inference_text = "Question: What is 10 - 3? [CoT]"

# Tokenize input
encoded_input = tokenizer(inference_text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")

input_ids = encoded_input["input_ids"].to(device)
attention_mask = encoded_input["attention_mask"].to(device)

# Predict masked tokens
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    predictions = outputs.logits # [1, 128, 30523], [batch_size, sequence_length, vocab_size]

# Decode predicted tokens
predicted_ids = torch.argmax(predictions, dim=-1)
decoded_output = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

print(f"Input: {inference_text}")
print(f"Output: {decoded_output}")