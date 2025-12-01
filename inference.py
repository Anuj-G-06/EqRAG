from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, AutoPeftModelForCausalLM
import torch
import json
from dataclasses import dataclass
from typing import Optional
import re
from dataset import get_dataset
import sys
from pathlib import Path
from tqdm.auto import tqdm

@dataclass
class InferenceConfig:
    base_model_path: str
    model_path: str
    dataset_name: str
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    max_new_tokens: int = 512
    device: str = "auto"
    peft: bool = False
    repetition_penalty: float | None = None
    batch_size: int = 1

def load_json_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file {path} not found")
    with p.open("r") as f:
        return json.load(f)

def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def load_model(config: InferenceConfig):
    device = get_device() if config.device == "auto" else config.device

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    tokenizer.padding_side = "left"
    
    # Load model
    # Method 1: Load model (No PEFT)
    if not config.peft:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            device_map="auto",
            dtype=torch.bfloat16,
            trust_remote_code=True,
            use_cache=True
        )
        model.resize_token_embeddings(len(tokenizer))
    # Method 2: Load and merge PEFT weights with base model
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_path,
            device_map="auto",
            dtype=torch.bfloat16,
            trust_remote_code=True,
            use_cache=True
        )
        base_model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(
            base_model,
            config.model_path,
            device_map="auto"
        )
        model = model.merge_and_unload()

    model = torch.compile(model, mode="reduce-overhead")
    model.eval()
    return model, tokenizer

def generate_batch_responses(model, tokenizer, prompts: str, config: InferenceConfig) -> str:
    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=2048
    ).to(model.device)

    with torch.no_grad():
        # Set up generation config
        gen_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "max_new_tokens": config.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if config.temperature is not None and config.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = config.temperature
            if config.top_k is not None:
                gen_kwargs["top_k"] = config.top_k
            if config.top_p is not None:
                gen_kwargs["top_p"] = config.top_p
            if config.repetition_penalty is not None:
                gen_kwargs["repetition_penalty"] = config.repetition_penalty
        else:
            gen_kwargs["do_sample"] = False

        outputs = model.generate(**gen_kwargs)

    responses = []
    for i, output in enumerate(outputs):
        input_length = inputs["input_ids"][i].shape[0]
        response = tokenizer.decode(output[input_length:], skip_special_tokens=True)
        responses.append(response)
    return responses

def extract_action(response: str) -> Optional[str]:
    # Extract action from response
    match = re.search(r"\[Action\]:\s*(BUY|HOLD|SELL)", response)
    return match.group(1) if match else None

def extract_prediction_percent(response: str) -> Optional[float]:
    # Extract prediction percentage from response
    text = response.lower()
    range_pattern = r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*%"
    m = re.search(range_pattern, text)
    if m:
        low, high = float(m.group(1)), float(m.group(2))
        return (low + high) / 2
    
    # Pattern for single %
    single_pattern = r"(up|down)\s+by\s+(\d+(?:\.\d+)?)\s*%"
    m = re.search(single_pattern, text)
    if m:
        direction, value = m.group(1), float(m.group(2))
        return value if direction == "up" else -value
    
    return None

def evaluate_response(response: str, ground_truth_action: Optional[str] = None) -> dict:
    """Custom evaluation of model response"""
    action = extract_action(response)
    prediction_percent = extract_prediction_percent(response)
    
    evaluation = {
        "response": response,
        "extracted_action": action,
        "extracted_prediction": prediction_percent,
        "has_action": action is not None,
        "has_prediction": prediction_percent is not None,
    }
    
    if ground_truth_action:
        evaluation["action_match"] = action == ground_truth_action
    
    return evaluation

def run_inference(test_dataset, config: InferenceConfig):
    model, tokenizer = load_model(config)
    results = []

    batch_size = config.batch_size
    num_batches = (len(test_dataset) + batch_size - 1) // batch_size
    
    for b in tqdm(range(num_batches), desc="Running Batched Inference", unit="batch"):
        i = b * batch_size
        j = min(i + batch_size, len(test_dataset))
        batch = test_dataset.select(range(i, j))

        prompts = list(batch["prompt"])
        responses = generate_batch_responses(model, tokenizer, prompts, config)

        for k, response in enumerate(responses):
            ground_truth_action = batch["action"][k]
            evaluation = evaluate_response(response, ground_truth_action)
            evaluation["symbol"] = batch["symbol"][k]
            evaluation["period"] = batch["period"][k]
            evaluation["label"] = batch["label"][k]
            results.append(evaluation)

    # Calculate accuracy
    if any(r.get('action_match') is not None for r in results):
        correct = sum(1 for r in results if r.get('action_match') == True)
        total = sum(1 for r in results if r.get('action_match') is not None)
        accuracy = correct / total if total > 0 else 0
        print(f"Overall Action Accuracy: {accuracy:.2%} ({correct}/{total})")

    return results

def main():
    if len(sys.argv) == 1:
        raise ValueError("Missing model configuration file.")
    model_config = load_json_config(sys.argv[1])

    config = InferenceConfig(
        base_model_path=model_config["model_to_train"],
        model_path=model_config["output_dir"],
        dataset_name=model_config["dataset"],
        temperature=0.7,
        top_k=50,
        top_p=None,
        max_new_tokens=600,
        peft=model_config["use_peft"],
        # repetition_penalty=1.2,
        batch_size=4
    )

    # Load test dataset
    test_dataset = get_dataset(config.dataset_name)[1]

    # Run inference
    results = run_inference(test_dataset, config)

    # Save results
    with open(f"outputs/inference_{model_config["run_name"]}.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()