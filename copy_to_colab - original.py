import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import logging

# Setup logging
def setup_logging():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger

logger = setup_logging()

# Step 1: Setup GPU & Install Dependencies (Run in a Colab cell before running this script)
# !pip install transformers accelerate peft datasets bitsandbytes

# Step 2: Load TinyLLaMA model in 4-bit
model_name = "unsloth/tinyllama"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
logger.info("Tokenizer loaded successfully.")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
logger.info("Model loaded successfully.")

# Step 3: Load and preprocess dataset
dataset = load_dataset("yahma/alpaca-cleaned", split="train").shuffle(seed=42).select(range(int(51760 * 0.001)))  # Reduce dataset size to 1/1000
logger.info(f"Dataset loaded. Total samples: {len(dataset)}")

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n### Input:\n{input}\n### Response:\n{output}"

def format_example(example):
    instruction = example["instruction"]
    inp = example.get("input", "")
    output = example["output"]
    text = PROMPT_TEMPLATE.format(instruction=instruction, input=inp, output=output)
    tokenized = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    return {"input_ids": tokenized["input_ids"].squeeze(), "labels": tokenized["input_ids"].squeeze()}

dataset = dataset.map(format_example, remove_columns=dataset.column_names)
dataset = dataset.train_test_split(test_size=0.005, seed=42)
train_data, eval_data = dataset["train"], dataset["test"]
logger.info(f"Data split completed. Train size: {len(train_data)}, Eval size: {len(eval_data)}")

# Step 4: Apply LoRA fine-tuning
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
logger.info("LoRA fine-tuning applied.")

# Step 5: Define Training Arguments
output_dir = "tiny-llama-finetuned"
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,  # Reduce epochs for faster training
    per_device_train_batch_size=1,  # Reduce batch size
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    eval_strategy="steps",
    eval_steps=10,  # Reduce evaluation frequency
    logging_steps=5,
    save_steps=20,  # Reduce checkpoint frequency
    save_total_limit=1,
    learning_rate=1e-4,  # Adjust learning rate for small dataset
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    optim="adamw_bnb_8bit",
    report_to="none",
    remove_unused_columns=False
)
logger.info("Training arguments defined.")

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    data_collator=data_collator
)

# Step 6: Train the Model
logger.info("Starting model training...")
try:
    trainer.train()
    logger.info("Training completed successfully.")
except Exception as e:
    logger.error(f"Training failed: {e}")

# Step 7: Save LoRA Adapters
peft_model_dir = "tiny-llama-lora-adapter"
model.save_pretrained(peft_model_dir)
tokenizer.save_pretrained(peft_model_dir)
logger.info("LoRA adapters saved.")

# Step 8: Evaluate the Fine-tuned Model
def generate_response(prompt):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("### Response:")[-1].strip()

# Example inference test
instruction = "What is the capital of France?"
input_text = ""
prompt = f"### Instruction:\n{instruction}\n### Input:\n{input_text}\n### Response:\n"
logger.info("Testing inference...")
try:
    response = generate_response(prompt)
    logger.info(f"Generated response: {response}")
    print(response)
except Exception as e:
    logger.error(f"Inference failed: {e}")