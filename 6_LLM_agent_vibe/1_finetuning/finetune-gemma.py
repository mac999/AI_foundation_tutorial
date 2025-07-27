# https://daddynkidsmakers.blogspot.com/2025/02/gemma2.html
# 
import os, torch, gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.

from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

load_dotenv()
hf_api_key = os.getenv("HF_TOKEN")
login(token=hf_api_key)

#set the qunatization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

#Load the model and Tokenizer
model_id = "google/gemma-2b-it" # before using it, you need grant access to the model from Hugging Face
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto") # {"":0})
tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)

from datasets import load_dataset
dataset = load_dataset("mamachang/medical-reasoning")
print(dataset)

df = dataset["train"].to_pandas()
print(df.head(10))

def generate_prompt(data_point):
    """Gen. input text based on a prompt, task instruction, (context info.), and answer

    :param data_point: dict: Data point
    :return: dict: tokenzed prompt
    """

    # Generate prompt
    prefix_text = 'Below is an instruction that describes a task. Write a response that ' \
               'appropriately completes the request.\n\n'
    # Samples with additional context into.
    if data_point['input']:
        text = f"""<start_of_turn>user {prefix_text} {data_point["instruction"]} here are the inputs {data_point["input"]} <end_of_turn>\n<start_of_turn>model{data_point["output"]} <end_of_turn>"""
    # Without
    else:
        text = f"""<start_of_turn>user {prefix_text} {data_point["instruction"]} <end_of_turn>\n<start_of_turn>model{data_point["output"]} <end_of_turn>"""
    return text

dataset['train']

# add the "prompt" column in the dataset
text_column = [generate_prompt(data_point) for data_point in dataset["train"]]
dataset = dataset["train"].add_column("prompt", text_column)
dataset = dataset.shuffle(seed=1234)  # Shuffle dataset here
dataset = dataset.map(lambda samples: tokenizer(samples["prompt"]), batched=True)
dataset = dataset.train_test_split(test_size=0.1)
train_data = dataset["train"]
test_data = dataset["test"]
print(train_data)
print(test_data)


from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
print(model)

lora_config = LoraConfig(
    r=64,
    lora_alpha=32,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

import bitsandbytes as bnb
def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        if 'lm_head' in lora_module_names: # needed for 16-bit
            lora_module_names.remove('lm_head')
    return list(lora_module_names)

modules = find_all_linear_names(model)
print(modules)

trainable, total = model.get_nb_trainable_parameters()
print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total*100:.4f}%")
print('\n')
train_data
test_data

import transformers
from trl import SFTTrainer

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side='right'
torch.cuda.empty_cache()

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=test_data,
    # dataset_text_field="prompt",
    peft_config=lora_config,
    # max_seq_length=2500,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=3,
        max_steps=100,
        learning_rate=2e-4,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        save_strategy="epoch",
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

new_model = "gemma-medical-qa-finetune"
trainer.model.save_pretrained(new_model)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map={"": 0},
)
merged_model= PeftModel.from_pretrained(base_model, new_model)
merged_model= merged_model.merge_and_unload()

# Save the merged model
#save_adapter=True, save_config=True
merged_model.save_pretrained("merged_model",safe_serialization=True)
tokenizer.save_pretrained("merged_model")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
#
# Push the model and tokenizer to the Hugging Face Model Hub
merged_model.push_to_hub(new_model, use_temp_dir=False)
tokenizer.push_to_hub(new_model, use_temp_dir=False)

def get_completion(query: str, model, tokenizer) -> str:
  device = "cuda:0"

  prompt_template = """
  <start_of_turn>user
  Below is an instruction that describes a task. Write a response that appropriately completes the request.
  {query}
  <end_of_turn>\n<start_of_turn>model


  """
  prompt = prompt_template.format(query=query)

  encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

  model_inputs = encodeds.to(device)


  generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)
  # decoded = tokenizer.batch_decode(generated_ids)
  decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
  return (decoded)
#
query = """\n\n Please answer with one of the option in the bracket. Write reasoning in between <analysis></analysis>. Write answer in between <answer></answer>. here are the inputs Q:An 8-year-old boy is brought to the pediatrician by his mother with nausea, vomiting, and decreased frequency of urination. He has acute lymphoblastic leukemia for which he received the 1st dose of chemotherapy 5 days ago. His leukocyte count was 60,000/mm3 before starting chemotherapy. The vital signs include: pulse 110/min, temperature 37.0°C (98.6°F), and blood pressure 100/70 mm Hg. The physical examination shows bilateral pedal edema. Which of the following serum studies and urinalysis findings will be helpful in confirming the diagnosis of this condition? ? \n{'A': 'Hyperkalemia, hyperphosphatemia, hypocalcemia, and extremely elevated creatine kinase (MM)', 'B': 'Hyperkalemia, hyperphosphatemia, hypocalcemia, hyperuricemia, urine supernatant pink, and positive for heme', 'C': 'Hyperuricemia, hyperkalemia, hyperphosphatemia, lactic acidosis, and urate crystals in the urine', 'D': 'Hyperuricemia, hyperkalemia, hyperphosphatemia, and urinary monoclonal spike', 'E': 'Hyperuricemia, hyperkalemia, hyperphosphatemia, lactic acidosis, and oxalate crystals'}"""

result = get_completion(query=query, model=merged_model, tokenizer=tokenizer)
print(result)

print(f"Model Answer : \n {result.split('model')[-1]}")


query = """Please answer with one of the option in the bracket. Write reasoning in between <analysis></analysis>. Write answer in between <answer></answer>.here are the inputs:Q:A 34-year-old man presents to a clinic with complaints of abdominal discomfort and blood in the urine for 2 days. He has had similar abdominal discomfort during the past 5 years, although he does not remember passing blood in the urine. He has had hypertension for the past 2 years, for which he has been prescribed medication. There is no history of weight loss, skin rashes, joint pain, vomiting, change in bowel habits, and smoking. On physical examination, there are ballotable flank masses bilaterally. The bowel sounds are normal. Renal function tests are as follows:\nUrea 50 mg/dL\nCreatinine 1.4 mg/dL\nProtein Negative\nRBC Numerous\nThe patient underwent ultrasonography of the abdomen, which revealed enlarged kidneys and multiple anechoic cysts with well-defined walls. A CT scan confirmed the presence of multiple cysts in the kidneys. What is the most likely diagnosis?? \n{'A': 'Autosomal dominant polycystic kidney disease (ADPKD)', 'B': 'Autosomal recessive polycystic kidney disease (ARPKD)', 'C': 'Medullary cystic disease', 'D': 'Simple renal cysts', 'E': 'Acquired cystic kidney disease'}"""
result = get_completion(query=query, model=merged_model, tokenizer=tokenizer)
print(f"Model Answer : \n {result.split('model')[-1]}")

try:
  del model
  del merged_model # delete model 1 variable
except:
  pass

gc.collect()
torch.cuda.empty_cache()


