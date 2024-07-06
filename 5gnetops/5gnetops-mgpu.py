# This is the Multi-GPU Parallel Training Version of 5gnetops
# Author: Fatioh E. NAR
# Run from a cli with; torchrun --nproc_per_node=2 5gnetops-mgpu.py
#
import os
import lzma
import shutil
import pandas as pd
import gc
import torch
from datasets import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator

# Set TOKENIZERS_PARALLELISM to false to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
gc.collect()

# Initialize the accelerator with GPU use
accelerator = Accelerator()

def preprocess_function(examples, tokenizer):
    inputs = examples['input_text']
    targets = examples['target_text']
    model_inputs = tokenizer(inputs, max_length=512, padding='max_length', truncation=True)
    labels = tokenizer(text_target=targets, max_length=128, padding='max_length', truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

def load_data():
    # Extract the .xz file
    with lzma.open('data/5G_netops_data_100K.csv.xz', 'rb') as f_in:
        with open('data/5G_netops_data_100K.csv', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Load the synthetic telecom data
    data_path = "data/5G_netops_data_100K.csv"
    data = pd.read_csv(data_path)

    # Fill NaN values and prepare input and target texts
    data = data.fillna('')
    
    # Ensure 'Zip' column is treated as a string
    data['Zip'] = data['Zip'].astype(str)
    
    # Create input and target texts
    data['input_text'] = data.apply(lambda row: f"Setup {row['Connection Setup Success Rate (%)']} Availability {row['Cell Availability (%)']} Changes {row['Parameter Changes']} Alarms {row['Alarm Count']}", axis=1)
    data['target_text'] = data.apply(lambda row: f"Success {row['Successful Configuration Changes (%)']} DropRate {row['Call Drop Rate (%)']}", axis=1)
    
    # Prepare the dataset
    dataset = Dataset.from_pandas(data)
    return dataset

def train_ddp(rank, world_size, model_name, model_save_path):
    print("Accelerator initialized with loaded configuration")

    # Load and preprocess data
    dataset = load_data()
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True, num_proc=4)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])

    # Load model
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Define PEFT/LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=2,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=['q', 'v', 'k', 'o']
    )
    model = get_peft_model(model, lora_config)

    # Prepare the model and data for distributed training
    model, train_dataset, eval_dataset = accelerator.prepare(
        model, train_dataset, eval_dataset
    )

    # Set training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",  # Output directory
        overwrite_output_dir=True,  # Overwrite the content of the output directory
        num_train_epochs=100,  # Number of training epochs
        per_device_train_batch_size=34,  # Batch size per device during training
        gradient_accumulation_steps=42,  # Accumulate gradients over multiple steps
        learning_rate=5e-5,  # Learning rate
        save_steps=2000,  # Save checkpoint every 2000 steps
        save_total_limit=2,  # Limit the total amount of checkpoints
        eval_strategy="steps",  # Evaluate during training at each `logging_steps`
        logging_steps=500,  # Log every 500 steps
        eval_steps=2000,  # Evaluate every 2000 steps
        load_best_model_at_end=True,  # Load the best model at the end of training
        metric_for_best_model="loss",  # Use loss to evaluate the best model
        predict_with_generate=True,  # Use generation for evaluation
        fp16=True,  # Load mixed precision training for CUDA only
        remove_unused_columns=False,  # Remove unused columns from the dataset
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )

    trainer.train()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    # Save the model and tokenizer
    model_save_path = "models/5gran_faultprediction_model_multigpu"
    model_name = "t5-small"
    train_ddp(0, world_size, model_name, model_save_path)
