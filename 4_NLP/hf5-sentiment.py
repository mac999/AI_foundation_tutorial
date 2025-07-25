import transformers, torch
from transformers import pipeline

print(transformers.__version__)
print(torch.cuda.is_available())

print('text-classification')
pipe = pipeline("text-classification", device=0)
msg = pipe("This restaurant is awesome")
print(msg)

print('sentiment-analysis')
classifier = pipeline("sentiment-analysis", device=0) # model="nlptown/bert-base-multilingual-uncased-sentiment", 
msg = classifier("We are very happy to show you the 🤗 Transformers library.")
for m in msg:
    print(m)
