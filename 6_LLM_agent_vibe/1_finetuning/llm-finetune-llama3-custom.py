import os, pandas as pd, json, torch, wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login # https://www.kaggle.com/discussions/product-feedback/114053
from trl import SFTTrainer, setup_chat_format
from datasets import Dataset, load_dataset
from pathlib import Path
from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	BitsAndBytesConfig,
	HfArgumentParser,
	TrainingArguments,
	pipeline,
	logging,
)
from peft import (
	LoraConfig,
	PeftModel,
	prepare_model_for_kbit_training,
	get_peft_model,
)

''' from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("d70e1595ba3bc6490103a46aee0c1b0a")
wb_token = user_secrets.get_secret("wandb")
'''

custom_all_data = []
def load_dataset_from_folder(folder_path):
	global custom_all_data
	for file_name in os.listdir(folder_path):
		if file_name.endswith('.json'):
			file_path = os.path.join(folder_path, file_name)
			with open(file_path, 'r', encoding='utf-8') as file:
				data_list = json.load(file)

				for data in data_list['responses']:
					if 'answer' in data and isinstance(data['answer'], list):
						data['answer'] = ', '.join(data['answer'])
					# custom_all_data.extend(data)
					custom_all_data.append(data)

	df = pd.DataFrame(custom_all_data)
	df = df.astype(str)
	return Dataset.from_pandas(df)

def train():
	base_model = "Undi95/Meta-Llama-3-8B-hf"
	dataset_name = "ruslanmv/ai-medical-chatbot" # 사용할 데이터셋
	new_model = "llama-3-8b-chat-doctor"        # 파인튜닝 모델 이름

	login(token = '')
	wandb.login(key='')
	run = wandb.init(
		project=new_model,
		job_type="training",
		anonymous="allow"
	) 

	attn_implementation = "eager"
	global custom_all_data
	custom_dataset = load_dataset_from_folder('./dataset')
	custom_dataset = custom_dataset.shuffle(seed=65) # .select(range(3000))  # Use only 1000 examples for fine-tuning

	# QLoRA 설정
	bnb_config = BitsAndBytesConfig(
		load_in_4bit=True, # 4bit 양자화
		bnb_4bit_quant_type="nf4",
		bnb_4bit_compute_dtype=torch.float16,
		bnb_4bit_use_double_quant=True,
	)

	model = AutoModelForCausalLM.from_pretrained(
		base_model,
		quantization_config=bnb_config,
		device_map="auto",
		attn_implementation=attn_implementation, 
		local_files_only=False
	)

	# 토크나이저 및 모델 로딩
	tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=False)
	if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
		tokenizer.chat_template = None
	model, tokenizer = setup_chat_format(model, tokenizer)

	# 테스트
	messages = [
		{
			"role": "user",
			"content": "Hello doctor, I have bad acne. How do I get rid of it?"
		}
	]
	print('before fine tuning')
	print(messages)
	prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
	inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")
	outputs = model.generate(**inputs, max_length=150, num_return_sequences=1)
	text = tokenizer.decode(outputs[0], skip_special_tokens=True)
	print(text.split("assistant")[1])


	# Lora 설정
	peft_config = LoraConfig(
		r=16,
		lora_alpha=32,
		lora_dropout=0.05,
		bias="none",
		task_type="CAUSAL_LM",
		target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
	)
	model = get_peft_model(model, peft_config)  # 파라메터 튜닝 설정

	# 데이터셋 로딩
	dataset = load_dataset(dataset_name, split="all")
	dataset = dataset.shuffle(seed=65).select(range(1000)) # 파인튜닝 예시 위해 1000개만 사용

	def format_chat_template(row):
		row_json = [{"role": "user", "content": row["Patient"]},
				{"role": "assistant", "content": row["Doctor"]}]
		row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
		return row

	dataset = dataset.map(
		format_chat_template,
		num_proc=1,
	)

	def bim_format_chat_template(row):
		row_json = [{"role": "user", "content": row["question"]},
					{"role": "assistant", "content": row["answer"]}]
		row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
		return row

	custom_dataset = custom_dataset.map(
		bim_format_chat_template,
		num_proc=1,
	)

	from datasets import concatenate_datasets
	print(custom_dataset['text'][100])
	custom_question = custom_dataset['text'][100]
	dataset = concatenate_datasets([dataset, custom_dataset])
	print(dataset['text'][3])

	dataset = dataset.train_test_split(test_size=0.1) # 데이터 검증 세트 분할

	# 모델 하이퍼파라메터 설정. 참고. Fine-Tuning LLaMA 2: A Step-by-Step Guide to Customizing the Large Language Model | DataCamp
	training_arguments = TrainingArguments(
		output_dir=new_model,
		per_device_train_batch_size=1,
		per_device_eval_batch_size=1,
		gradient_accumulation_steps=2,
		optim="paged_adamw_32bit",
		num_train_epochs=10, #15
		# evaluation_strategy="steps",
		eval_steps=0.2,
		logging_steps=1,
		warmup_steps=10,
		logging_strategy="steps",
		learning_rate=2e-4,
		fp16=False,
		bf16=False,
		group_by_length=True,
		report_to="wandb"
	)

	# 지도미세조정(SFT) 설정
	trainer = SFTTrainer(
		model=model,
		train_dataset=dataset["train"],
		eval_dataset=dataset["test"],
		peft_config=peft_config,
		max_seq_length=512,
		dataset_text_field="text",
		tokenizer=tokenizer,
		args=training_arguments,
		packing= False,
	)

	# 미세조정 모델학습
	trainer.train()

	# W&B 로그 종료
	wandb.finish()
	model.config.use_cache = True

	# 테스트
	print('after fine tuning')
	messages = [{
			"role": "user",
			"content": "Hello doctor, I have bad acne. How do I get rid of it?"
		}]
	print(messages)
	prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
	inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")
	outputs = model.generate(**inputs, max_length=150, num_return_sequences=1)
	text = tokenizer.decode(outputs[0], skip_special_tokens=True)
	print(text.split("assistant")[1])

	messages = [{
			"role": "user",
			"content": 'What are the profile properties of IfcStructuralCurveMemberVarying described by?'
		}]
	print(messages)
	prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
	inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")
	outputs = model.generate(**inputs, max_length=150, num_return_sequences=1)
	text = tokenizer.decode(outputs[0], skip_special_tokens=True)
	print(text.split("assistant")[1])

	trainer.model.save_pretrained(new_model)
	print('done!!!!!')

	trainer.model.push_to_hub(new_model, use_temp_dir=False)
	print('push!!!!!')

def generate_answer(messages, model, tokenizer):
	print('\n')
	print(messages)
	prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
	inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")
	outputs = model.generate(**inputs, max_length=150, num_return_sequences=1)
	text = tokenizer.decode(outputs[0], skip_special_tokens=True)
	print(text.split("assistant")[1])

def test():
	# https://www.philschmid.de/fine-tune-flan-t5-peft
	# https://discuss.huggingface.co/t/loading-and-saving-a-model/74870
	base_model = "Undi95/Meta-Llama-3-8B-hf"
	bnb_config = BitsAndBytesConfig(
		load_in_4bit=True, # 4bit 양자화
		bnb_4bit_quant_type="nf4",
		bnb_4bit_compute_dtype=torch.float16,
		bnb_4bit_use_double_quant=True,
	)
	model = AutoModelForCausalLM.from_pretrained(
		base_model,
		quantization_config=bnb_config,
		device_map="auto",
		attn_implementation="eager", local_files_only=False
	)
	tokenizer = AutoTokenizer.from_pretrained(base_model)
	model, tokenizer = setup_chat_format(model, tokenizer)
	model.eval()

	print('before fine tuning')
	messages = [{
			"role": "user",
			"content": "Hello doctor, I have bad acne. How do I get rid of it?"
		}]
	generate_answer(messages, model, tokenizer)

	messages = [{
			"role": "user",
			"content": 'What are the profile properties of IfcStructuralCurveMemberVarying described by?'
		}]
	generate_answer(messages, model, tokenizer)

	peft_config = LoraConfig(
		r=16,
		lora_alpha=32,
		lora_dropout=0.05,
		bias="none",
		task_type="CAUSAL_LM",
		target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
	)
	model = get_peft_model(model, peft_config)  # 파라메터 튜닝 설정
	new_model = "./llama-3-8b-chat-doctor15_20240706/checkpoint-18000"        # 파인튜닝 모델 이름
	model = PeftModel.from_pretrained(model, new_model, device_map={"":0})
	model = model.merge_and_unload() # https://stackoverflow.com/questions/76459034/how-to-load-a-fine-tuned-peft-lora-model-based-on-llama-with-huggingface-transfo
	# tokenizer.bos_token_id = 1 
	model, tokenizer = setup_chat_format(model, tokenizer)
	model.eval()

	print('\n\n\nafter fine tuning')
	messages = [{
			"role": "user",
			"content": "Hello doctor, I have bad acne. How do I get rid of it?"
		}]
	generate_answer(messages, model, tokenizer)

	messages = [{
			"role": "user",
			"content": 'What are the profile properties of IfcStructuralCurveMemberVarying described by?'
		}]
	generate_answer(messages, model, tokenizer)

if __name__ == "__main__":
	train()
	test()
	 