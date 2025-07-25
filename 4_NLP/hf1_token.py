from transformers import BertTokenizer, BertModel
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

example_sen = (
    """
    The United States and Russia sought to lower the temperature in a 
    heated standoff over Ukraine,even as they reported no breakthroughs 
    in high-stakes talks on Friday aimed at preventing a feared Russian invasion
    """
)
print(bert_tokenizer.tokenize(example_sen))