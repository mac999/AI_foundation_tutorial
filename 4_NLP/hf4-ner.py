from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "My name is Wolfgang and I live in Berlin"
print(example)
ner_results = nlp(example)
print(ner_results)

example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
print(example)
ner_results = nlp(example)
print(ner_results)
