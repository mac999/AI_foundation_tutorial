
from dotenv import load_dotenv
import os
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
hf_api_token = os.getenv("HF_API_TOKEN")
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["HF_TOKEN"] = hf_api_token

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

#set the qunatization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

#Load the model and Tokenizer
model_id = "google/gemma-2b-it" # https://huggingface.co/google/gemma-2b-it

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto") # {"":0})
tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)

from datasets import load_dataset
dataset = load_dataset("mamachang/medical-reasoning") # https://huggingface.co/datasets/mamachang/medical-reasoning
print(dataset)
print(f'train shape: {dataset["train"].shape}')
df = dataset["train"].to_pandas()
print(df.head(10))

def generate_prompt(data_point): # https://ai.google.dev/gemma/docs/core/prompt-structure
    # Generate prompt
    prefix_text = 'Below is an instruction that describes a task. Write a response that ' \
               'appropriately completes the request.\n\n'
    # Samples with additional context into.
    if data_point['input']: # https://huggingface.co/google/gemma-2b-it#chat-template
        text = f"""<start_of_turn>user {prefix_text} {data_point["instruction"]} here are the inputs {data_point["input"]} <end_of_turn>\n<start_of_turn>model{data_point["output"]} <end_of_turn>"""
    # Without
    else:
        text = f"""<start_of_turn>user {prefix_text} {data_point["instruction"]} <end_of_turn>\n<start_of_turn>model{data_point["output"]} <end_of_turn>"""
    return text

print(dataset["train"].features)

# add the "prompt" column in the dataset
text_column = [generate_prompt(data_point) for data_point in dataset["train"]]
print(text_column[:5])

dataset = dataset["train"].add_column("prompt", text_column)
dataset = dataset.shuffle(seed=1234)  # Shuffle dataset here

# dataset = dataset.map(lambda samples: tokenizer(samples["prompt"]), batched=True)
def tokenize_samples(samples):
    return tokenizer(samples["prompt"])
dataset = dataset.map(tokenize_samples, batched=True)
print(f'input_ids: {dataset["input_ids"][0][0:10]}, mask: {dataset["attention_mask"][0][0:10]}')

dataset = dataset.train_test_split(test_size=0.1)
train_data = dataset["train"]
test_data = dataset["test"]
print(train_data)
print(test_data)

# for testing, convert the dataset's input_ids to torch tensors
max_len = 100 # max length of the input sequence or padding should be
trimmed_dataset_input_ids = [row[:max_len] for row in train_data["input_ids"]]
tensor_dataset_input_ids = torch.tensor(trimmed_dataset_input_ids, dtype=torch.float32, device="cuda")
print(f'train_data shape: {tensor_dataset_input_ids.shape}')
print(f'train_data input_ids: {tensor_dataset_input_ids[0][0:10]}')

# clean
import torch, gc
try:
  del model
  del merged_model # delete model 1 variable
except:
  pass
gc.collect()
torch.cuda.empty_cache()
