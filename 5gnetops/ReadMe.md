# 5G Network Predictions with Fine-Tuning Distilgpt2 using PEFT & LoRA
Attention: Still Under Progress <br> 

This repository guides you through fine-tuning the Distilgpt2 model using Parameter-Efficient Fine-Tuning (PEFT) with Low-Rank Adaptation (LoRA) to reduce computational and memory requirements.

## Prerequisites

Install the required libraries:

```sh
pip install torch transformers datasets peft
```

Set the environment variable for MAC with AMD GPUs: (PyTorch uses the new Metal Performance Shaders (MPS) backend for GPU training acceleration. This MPS backend extends the PyTorch framework, providing scripts and capabilities to set up and run operations on Mac. )

```sh
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.9
```
![macmpsgpu](https://raw.githubusercontent.com/fenar/etc-ai-wrx/main/5gnetops/data/macmpsgpu.png)<br> 

## Setup

### Load Model and Tokenizer

```python
from transformers import GPTJForCausalLM, AutoTokenizer

model_name = "EleutherAI/gpt-j-6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = GPTJForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))
```

### Prepare Dataset

```python
def tokenize_function(examples):
    inputs = tokenizer(examples['input_text'], padding="max_length", truncation=True, max_length=512)
    targets = tokenizer(examples['target_text'], padding="max_length", truncation=True, max_length=512)
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'], 'labels': targets['input_ids']}

# Load and preprocess dataset
tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
```
![5G Data Set Snapshot](https://raw.githubusercontent.com/fenar/etc-ai-wrx/main/5gnetops/data/5gdatasetsnapshot.png)<br>

### Apply LoRA with PEFT

```python
from peft import get_peft_model, LoraConfig, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=4,
    lora_alpha=32,
    lora_dropout=0.1
)

model = get_peft_model(model, lora_config)
```

### Training

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

trainer.train()
```
