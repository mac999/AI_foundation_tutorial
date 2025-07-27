import torch
from transformers import BertTokenizer, BertForMaskedLM
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm import tqdm

# 1. 데이터셋 정의
class MaskedDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128, mask_probability=0.15):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_probability = mask_probability

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Tokenize input text
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Create masked input and labels
        labels = input_ids.clone() # tokenize labels for loss calculation
        rand = torch.rand(input_ids.shape)
        mask_arr = (rand < self.mask_probability) * (input_ids != self.tokenizer.cls_token_id) * (input_ids != self.tokenizer.sep_token_id) * (input_ids != self.tokenizer.pad_token_id) # Create a boolean mask to determine which tokens can be masked, excluding special tokens ([CLS], [SEP], [PAD])

        input_ids[mask_arr] = self.tokenizer.mask_token_id  # Replace with [MASK]

        return {
            "input_ids": input_ids, # Masked tokenized input text
            "attention_mask": attention_mask,  # Attention mask for the input
            "labels": labels # Original tokenized input text (used as labels for loss calculation)
        }

# 2. 데이터 준비
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world.",
    "BERT stands for Bidirectional Encoder Representations from Transformers."
]

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
dataset = MaskedDataset(texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 3. 모델 및 옵티마이저 초기화
model = BertForMaskedLM.from_pretrained("bert-base-uncased")
optimizer = AdamW(model.parameters(), lr=5e-5)

# 4. 학습 루프
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 500
for epoch in range(epochs):
    model.train()
    # loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}")

    for batch in dataloader:
        # Move data to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        print(f"Epoch {epoch + 1} Loss: {loss.item():.4f}")

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update progress bar
        # loop.set_postfix(loss=loss.item())

# Test the trained model
#
test_texts = [
    "The quick brown [MASK] jumps over the lazy dog.",
    "Artificial intelligence is [MASK] the world.",
    "BERT stands for [MASK] Encoder Representations from Transformers."
]

# Tokenize the test texts
test_encodings = tokenizer(
    test_texts,
    max_length=128,
    padding="max_length",
    truncation=True,
    return_tensors="pt"
)

# Move data to the appropriate device
input_ids = test_encodings["input_ids"].to(device)
attention_mask = test_encodings["attention_mask"].to(device)

# Set the model to evaluation mode
model.eval()

# Perform inference
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    predictions = outputs.logits

# Decode the predictions
for i, test_text in enumerate(test_texts):
    masked_index = (input_ids[i] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0].item()
    predicted_token_id = predictions[i, masked_index].argmax(dim=-1).item()
    predicted_token = tokenizer.decode([predicted_token_id])

    print(f"Original: {test_text}")
    print(f"Prediction: {test_text.replace(tokenizer.mask_token, predicted_token)}")
    print()

