import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM
from torch.optim import AdamW

# Step 1: 대화 데이터셋 정의
conversation_data = [
    {"question": "안녕", "answer": "안녕, 나는 LLM 이야. 너는 누구야?"},
    {"question": "너 뭐 할 수 있어?", "answer": "나는 다양한 질문에 답할 수 있어."},
    {"question": "날씨 어때?", "answer": "날씨는 내가 있는 곳에서는 확인할 수 없어."},
    {"question": "몇 살이야?", "answer": "나는 나이가 없지만, 최신 정보를 제공하려 노력해."},
    {"question": "너 좋아하는 건 뭐야?", "answer": "나는 데이터를 분석하고 정보를 제공하는 걸 좋아해."},
]

# Step 2: 데이터셋 클래스 정의
class ConversationDataset(Dataset):
    def __init__(self, tokenizer, data, max_length=64):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 질문과 답변을 각각 토큰화
        question_tokens = self.tokenizer(
            item["question"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        answer_tokens = self.tokenizer(
            item["answer"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": question_tokens["input_ids"].squeeze(0),
            "attention_mask": question_tokens["attention_mask"].squeeze(0),
            "labels": answer_tokens["input_ids"].squeeze(0),  # 답변을 정답으로 설정
        }

# Step 3: 토크나이저 및 데이터셋 준비
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
dataset = ConversationDataset(tokenizer, conversation_data)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Step 4: 모델 준비
model = BertForMaskedLM.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Step 5: Optimizer 설정
optimizer = AdamW(model.parameters(), lr=5e-5)

# Step 6: 학습 루프
epochs = 100
model.train()
for epoch in range(epochs):
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Step 7: 모델 저장
model_path = "./conversation_bert_mlm_model"
tokenizer_path = "./conversation_bert_tokenizer"
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

# Step 10: Example inference text
inference_text =  "안녕."

def gen_model(question):
    # Tokenize input
    encoded_input = tokenizer(
        question,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoded_input["input_ids"].to(device)
    attention_mask = encoded_input["attention_mask"].to(device)

    # Predict masked tokens
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = outputs.logits

    # Decode predicted tokens
    predicted_ids = torch.argmax(predictions, dim=-1)
    decoded_output = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

    return decoded_output
    
decoded_output = gen_model(inference_text)
print(f"Input: {inference_text}")
print(f"Output: {decoded_output}")

