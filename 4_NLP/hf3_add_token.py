from transformers import BertTokenizerFast, BertModel
import torch
from torch import nn

# BERT 토크나이저 사전학습모델 로딩
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
print(tokenizer.tokenize("[CLS] Hello world, how are you?"))

print(tokenizer.tokenize("[newtoken] Hello world, how are you?"))
tokenizer.add_tokens(['[newtoken]'])

# 토큰을 추가하고 다시 토큰화를 한다.
tokenizer.add_tokens(['[newtoken]'])
tokenizer.tokenize("[newtoken] Hello world, how are you?")

# 제대로 토큰화가 된다. 
# ['[newtoken]', 'hello', 'world', ',', 'how', 'are', 'you', '?’]

# 토큰값을 확인해 본다.
tokenized = tokenizer("[newtoken] Hello world, how are you?", add_special_tokens=False, return_tensors="pt")
print(tokenized['input_ids'])

tkn = tokenized['input_ids'][0, 0]
print("First token:", tkn)
print("Decoded:", tokenizer.decode(tkn))

# 다음과 같이, 토큰값이 잘 할당된 것을 알 수 있다.
# tensor([[30522,  7592,  2088,  1010,  2129,  2024,  2017,  1029]])
# First token: tensor(30522)
# Decoded: [newtoken]
