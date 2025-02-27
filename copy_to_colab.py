import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import logging

# ------------------------
# Step 1: Setup Logging
# ------------------------
# This function initializes logging to keep track of important events during the execution of the script.
def setup_logging():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger

logger = setup_logging()

# ------------------------
# Step 2: Install Dependencies (Run in a Colab cell before running this script)
# ------------------------
# Install necessary packages if running in Google Colab.
# Uncomment and run the following in a Colab cell:
# !pip install transformers accelerate peft datasets bitsandbytes

# ------------------------
# Step 3: Load TinyLLaMA Model in 4-bit Precision
# ------------------------

# Define the model name (pre-trained TinyLLaMA)
model_name = "unsloth/tinyllama"

# Configure 4-bit quantization for efficient memory usage
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Load model in 4-bit precision
    bnb_4bit_use_double_quant=True,  # Enables double quantization for better compression
    bnb_4bit_quant_type="nf4",  # Uses NF4 quantization type
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16  # Uses bf16 if supported, otherwise fp16
)

# Load tokenizer for the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
logger.info("Tokenizer loaded successfully.")

# Load the pre-trained model with quantization settings
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"  # Automatically assigns model to available GPU/CPU
)
logger.info("Model loaded successfully.")

# ------------------------
# Step 4: Load and Preprocess Dataset
# ------------------------

# Load the Alpaca dataset (fine-tuning dataset)
dataset = load_dataset("yahma/alpaca-cleaned", split="train")

# Shuffle the dataset and reduce size to 1/1000 for faster training (subset of 51 samples)
dataset = dataset.shuffle(seed=42).select(range(int(51760 * 0.001)))
logger.info(f"Dataset loaded. Total samples: {len(dataset)}")

# Define prompt template for instruction-based learning
PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n### Input:\n{input}\n### Response:\n{output}"

# Function to format dataset examples into a structured prompt format
def format_example(example):
    instruction = example["instruction"]
    inp = example.get("input", "")
    output = example["output"]
    
    # Create formatted prompt
    text = PROMPT_TEMPLATE.format(instruction=instruction, input=inp, output=output)
    
    # Tokenize the text and prepare input tensors
    tokenized = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    
    return {"input_ids": tokenized["input_ids"].squeeze(), "labels": tokenized["input_ids"].squeeze()}

# Apply the formatting function to the dataset
dataset = dataset.map(format_example, remove_columns=dataset.column_names)

# Split dataset into training and evaluation sets (0.5% for evaluation)
dataset = dataset.train_test_split(test_size=0.005, seed=42)
train_data, eval_data = dataset["train"], dataset["test"]
logger.info(f"Data split completed. Train size: {len(train_data)}, Eval size: {len(eval_data)}")

# ------------------------
# Step 5: Apply LoRA Fine-Tuning
# ------------------------

# Define LoRA (Low-Rank Adaptation) configuration for fine-tuning
lora_config = LoraConfig(
    r=16,  # LoRA rank (controls adaptation complexity)
    lora_alpha=32,  # Scaling factor for LoRA
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # List of layers to apply LoRA
    lora_dropout=0.05,  # Dropout rate to prevent overfitting
    bias="none",
    task_type=TaskType.CAUSAL_LM  # Specifies task type for causal language modeling
)

# Apply LoRA modifications to the model
model = get_peft_model(model, lora_config)

# Print the number of trainable parameters
model.print_trainable_parameters()
logger.info("LoRA fine-tuning applied.")

# ------------------------
# Step 6: Define Training Arguments
# ------------------------

# Output directory for storing model checkpoints
output_dir = "tiny-llama-finetuned"

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,  # Reduce training epochs for efficiency
    per_device_train_batch_size=1,  # Reduce batch size to fit in memory
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    eval_strategy="steps",  # Evaluate every N steps
    eval_steps=10,  # Evaluate every 10 steps
    logging_steps=5,  # Log every 5 steps
    save_steps=20,  # Save model checkpoint every 20 steps
    save_total_limit=1,  # Keep only the latest checkpoint
    learning_rate=1e-4,  # Learning rate for fine-tuning
    bf16=torch.cuda.is_bf16_supported(),  # Use bfloat16 if supported
    fp16=not torch.cuda.is_bf16_supported(),  # Use fp16 if bf16 isn't supported
    optim="adamw_bnb_8bit",  # Optimizer for efficient training
    report_to="none",  # Disable logging to external platforms
    remove_unused_columns=False
)

logger.info("Training arguments defined.")

# Data collator to handle padding in batches
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    data_collator=data_collator
)

# ------------------------
# Step 7: Train the Model
# ------------------------

logger.info("Starting model training...")
try:
    trainer.train()
    logger.info("Training completed successfully.")
except Exception as e:
    logger.error(f"Training failed: {e}")

# ------------------------
# Step 8: Save LoRA Adapters
# ------------------------

# Define directory for saving the fine-tuned model's LoRA adapters
peft_model_dir = "tiny-llama-lora-adapter"

# Save model and tokenizer
model.save_pretrained(peft_model_dir)
tokenizer.save_pretrained(peft_model_dir)
logger.info("LoRA adapters saved.")

# ------------------------
# Step 9: Evaluate the Fine-Tuned Model
# ------------------------

# Function to generate responses using the fine-tuned model
def generate_response(prompt):
    model.eval()  # Set model to evaluation mode
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")  # Tokenize input and move to GPU
    outputs = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)  # Generate response
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("### Response:")[-1].strip()  # Extract response

# Example test prompt
instruction = "What is the capital of France?"
input_text = ""
prompt = f"### Instruction:\n{instruction}\n### Input:\n{input_text}\n### Response:\n"

logger.info("Testing inference...")
try:
    response = generate_response(prompt)
    logger.info(f"Generated response: {response}")
    print(response)  # Output the response
except Exception as e:
    logger.error(f"Inference failed: {e}")
