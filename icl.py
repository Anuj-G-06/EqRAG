from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, AutoPeftModelForCausalLM
import torch
import json
from dataclasses import dataclass
from typing import Optional
import re
from dataset import get_dataset
from icl_dataset import get_icl_dataset
import sys
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, classification_report
from sklearn.preprocessing import label_binarize


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
    icl: bool = False

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
    if config.icl and tokenizer.pad_token is None:
      tokenizer.pad_token = tokenizer.eos_token#'<|padding|>'
    
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
        if not config.icl:
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
            "use_cache": True,
            "min_new_tokens": 10
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
        # input_length = inputs["input_ids"][i].shape[0]
        # input_length = inputs["attention_mask"][i].sum().item()
        # response = tokenizer.decode(output[input_length:], skip_special_tokens=True)
        generated_ids = output[-config.max_new_tokens:]  # Alternative: slice from end
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        # response = tokenizer.decode(output[input_length:], skip_special_tokens=True)
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
            evaluation["ground_truth_action"] = ground_truth_action
            results.append(evaluation)

    metrics = {}
    # Calculate accuracy
    if any(r.get('action_match') is not None for r in results):
        correct = sum(1 for r in results if r.get('action_match') == True)
        total = sum(1 for r in results if r.get('action_match') is not None)
        accuracy = correct / total if total > 0 else 0
        metrics['accuracy'] = accuracy
        metrics['correct_predictions'] = correct
        metrics['total_predictions'] = total
        print(f"Overall Action Accuracy: {accuracy:.2%} ({correct}/{total})")

    # Calculate classification metrics where ground truth is available
    y_true = [r["ground_truth_action"] for r in results if r.get("ground_truth_action") is not None and r.get("extracted_action") is not None]
    y_pred = [r["extracted_action"] for r in results if r.get("ground_truth_action") is not None and r.get("extracted_action") is not None]

    if len(y_true) > 0:
        classes = sorted(list(set(y_true) | set(y_pred)))
        precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0, labels=classes)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0, labels=classes)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0, labels=classes)

        # average_precision_score expects binary indicators per class; use label_binarize.
        y_true_bin = label_binarize(y_true, classes=classes)
        y_pred_bin = label_binarize(y_pred, classes=classes)
        try:
            avg_precision = average_precision_score(y_true_bin, y_pred_bin, average="macro")
        except ValueError:
            avg_precision = float("nan")

        metrics['precision_macro'] = float(precision_macro)
        metrics['recall_macro'] = float(recall_macro)
        metrics['f1_macro'] = float(f1_macro)
        metrics['average_precision'] = float(avg_precision)
        
        # Per-class metrics
        metrics['per_class_metrics'] = {}
        for i, cls in enumerate(classes):
            metrics['per_class_metrics'][cls] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1': float(f1_per_class[i])
            }
        
        # Get classification report as dict
        report_dict = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
        metrics['classification_report'] = report_dict

        print("Classification metrics (macro):")
        print(f"  Precision: {precision_macro:.4f}")
        print(f"  Recall:    {recall_macro:.4f}")
        print(f"  F1:        {f1_macro:.4f}")
        print(f"  AvgPrec:   {avg_precision:.4f}")
        print("\nPer-class report:")
        print(classification_report(y_true, y_pred, zero_division=0))
    else:
        print("No predictions with ground truth available to compute metrics.")


    return results, metrics

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
        batch_size=12,
        icl=model_config["icl"]
    )

    # Load test dataset
    # test_dataset = get_dataset(config.dataset_name)[1]
    # Few shot test dataset
    test_dataset = get_icl_dataset(config.dataset_name, icl_mode=True)[1]

    # Run inference
    generations, metrics = run_inference(test_dataset, config)

    # Save results
    inference_results = {
        "config": {
            "model_path": config.model_path,
            "dataset_name": config.dataset_name,
            "temperature": config.temperature,
            "top_k": config.top_k,
            "max_new_tokens": config.max_new_tokens,
            "batch_size": config.batch_size,
            "icl": config.icl,
            "peft": config.peft,
        },
        "metrics": metrics,
        "predictions": generations
    }
    with open(f"outputs/inference_{model_config["run_name"]}.json", "w") as f:
        json.dump(inference_results, f, indent=2)

if __name__ == "__main__":
    main()