from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    HfArgumentParser
)
from dataset import tokenize_dataset, get_dataset
from dataclasses import dataclass, field
from peft import LoraConfig, IA3Config, get_peft_model, PeftModelForCausalLM
import torch
import math
import sys
import shutil
import os

@dataclass
class ModelConfig:
    model_to_train: str = field(default="models/finsight")
    seq_len: int = field(default=512)
    attention_type: str = field(default="flash_attention_2")
    dataset: str = field(default="")
    use_lora: bool = field(default=False)

if len(sys.argv) == 1:
    raise ValueError("Missing configuration file.")

def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def train(model, trainer, training_args):
    model.train()
    trainer.train()
    trainer.save_model(training_args.output_dir)

def evaluate(trainer, test_dataset):
    results = trainer.evaluate(eval_dataset=test_dataset)
    print(f"Perplexity: {math.exp(results['eval_loss']):.2f}")

def main():
    device = get_device()
    print(f"Using device: {device}")
    config_file = sys.argv[1]

    # Load model and training configurations
    parser = HfArgumentParser((ModelConfig, TrainingArguments))
    model_config, training_args = parser.parse_json_file(json_file=config_file)

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_to_train)
    tokenizer.padding_side = "right"
    tokenizer.add_special_tokens({'pad_token': '<|padding|>'})

    # Load Model
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_to_train,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,                           
        attn_implementation=model_config.attention_type,
        use_cache=False
    )
    # Load LoRA configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    # Load IA3 configuration
    # peft_config = IA3Config(
    #     # target_modules=["k_proj", "v_proj", "down_proj"], feedforward_modules=["down_proj"]
    #     target_modules="all-linear",
    #     task_type="CAUSAL_LM",
    # )

    # Load PEFT model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    # total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Total trainable parameters: {total_trainable_params}")
    model.resize_token_embeddings(len(tokenizer))

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if model_config.use_lora:
            model.enable_input_require_grads()

    # Load Dataset
    train_dataset, test_dataset = get_dataset(model_config.dataset)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    tokenized_train_dataset = train_dataset.map(lambda x: tokenize_dataset(tokenizer, x, model_config.seq_len), batched=True, remove_columns=train_dataset.column_names)
    tokenized_test_dataset = test_dataset.map(lambda x: tokenize_dataset(tokenizer, x, model_config.seq_len), batched=True, remove_columns=test_dataset.column_names)

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Training and Evaluation
    train(model, trainer, training_args)
    evaluate(trainer, tokenized_test_dataset)
    print("Done!")


if __name__ == "__main__":
    main()
