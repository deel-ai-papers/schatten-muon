import argparse
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# SMuon ecosystem imports
from smuon.wrap_model import ActivationRecorder
from smuon.optimizers.adaptive import SMuonWithAuxAdam, SingleDeviceSMuonWithAuxAdam
from smuon.optimizers.baseline import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam


def load_config_with_overrides(config_path: str) -> argparse.Namespace:
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}

    parser = argparse.ArgumentParser(description="Classic PyTorch LoRA Training")

    parser.add_argument("--model_name", type=str, default="facebook/opt-125m")
    parser.add_argument("--dataset_id", type=str, default="stanfordnlp/sst2")
    parser.add_argument("--output_dir", type=str, default="./lora-out")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_run_name", type=str, default="classic-lora-run")

    parser.add_argument(
        "--optimizer", type=str, default="smuon", choices=["adamw", "muon", "smuon"]
    )
    parser.add_argument(
        "--moment_type", type=str, default="none", choices=["none", "padafactor"]
    )
    parser.add_argument("--adam_lr", type=float, default=3e-4)
    parser.add_argument("--adam_wd", type=float, default=0.0)
    parser.add_argument("--muon_lr", type=float, default=0.001)
    parser.add_argument("--muon_wd", type=float, default=0.0)
    parser.add_argument("--muon_momentum", type=float, default=0.95)
    parser.add_argument("--sv_momentum", type=float, default=0.95)
    parser.add_argument("--beta1", type=float, default=0.90)
    parser.add_argument("--beta2", type=float, default=0.95)

    parser.add_argument("--p_method", type=str, default="exact_momentum")
    parser.add_argument("--subsampling_ratio", type=float, default=1.0)
    parser.add_argument("--smuon_interval", type=int, default=100)

    existing_args = {action.dest for action in parser._actions}
    for key, value in config.items():
        if key not in existing_args:
            arg_type = type(value) if value is not None else str
            parser.add_argument(f"--{key}", type=arg_type, default=value)

    cli_args, _ = parser.parse_known_args()
    for key, value in vars(cli_args).items():
        if value is not None:
            config[key] = value

    return argparse.Namespace(**config)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_gsm8k_prompt(example):
    return f"Question: {example['question']}\nAnswer: {example['answer']}"


def format_sst2_prompt(example):
    label_str = "POSITIVE" if example["label"] == 1 else "NEGATIVE"
    return f"Sentence: {example['sentence']}\nSentiment: {label_str}"


def extract_gsm8k_answer(text: str):
    """Extracts the final numerical answer from GSM8k format."""
    # First, look for the standard GSM8k delimiter
    match = re.search(r"####\s*(-?[\d\.,]+)", text)
    if match:
        return match.group(1).replace(",", "")

    # Fallback: just find the last number in the generated text
    nums = re.findall(r"-?[\d\.,]+", text)
    if nums:
        return nums[-1].replace(",", "")
    return None


def evaluate_gsm8k(model, tokenizer, dataset, num_samples=100):
    """Evaluates the model on a subset of the GSM8k test set."""
    model.eval()
    correct = 0
    total = min(num_samples, len(dataset))

    # Evaluate on a random subset to save time during training
    indices = random.sample(range(len(dataset)), total)

    for idx in tqdm(indices, desc="Evaluating GSM8k (Sub-sample)", leave=False):
        example = dataset[idx]
        prompt = f"Question: {example['question']}\nAnswer:"
        true_answer = extract_gsm8k_answer(example["answer"])

        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=256
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,  # Enough tokens to generate the math and final answer
                temperature=0.0,  # Greedy decoding for math
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        pred_answer = extract_gsm8k_answer(generated_text)

        if pred_answer == true_answer:
            correct += 1

    model.train()
    return correct / total


def train_lora_model(args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main_process = local_rank == 0
    device_map = {"": local_rank}

    if args.seed is not None:
        set_seed(args.seed)

    # Initialize WandB manually
    if is_main_process:
        wandb.init(project="smuon-lora", name=args.wandb_run_name, config=vars(args))

    # ==========================================
    # Model & Tokenizer Setup
    # ==========================================
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"Loading {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map=device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "gate_proj"],
    )
    model = get_peft_model(model, peft_config)

    # ==========================================
    # Manual Dataset Preparation
    # ==========================================
    print("Tokenizing dataset...")
    raw_dataset = load_dataset(
        args.dataset_id,
        "default" if "sst2" in args.dataset_id else "main",
        split="train",
    )

    # Load eval dataset if using GSM8k
    if "gsm8k" in args.dataset_id:
        eval_dataset = load_dataset(args.dataset_id, "main", split="test")
    else:
        eval_dataset = None

    def tokenize_function(examples):
        if "sst2" in args.dataset_id:
            texts = [
                format_sst2_prompt({"sentence": s, "label": l})
                for s, l in zip(examples["sentence"], examples["label"])
            ]
        elif "gsm8k" in args.dataset_id:
            texts = [
                format_gsm8k_prompt({"question": q, "answer": a})
                for q, a in zip(examples["question"], examples["answer"])
            ]
        else:
            raise ValueError(
                f"Dataset formatting not implemented for {args.dataset_id}"
            )

        return tokenizer(texts, truncation=True, max_length=256)

    tokenized_dataset = raw_dataset.map(
        tokenize_function, batched=True, remove_columns=raw_dataset.column_names
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(
        tokenized_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator
    )

    # ==========================================
    # Optimizer Routing
    # ==========================================
    muon_params, adam_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            muon_params.append(param)
        else:
            adam_params.append(param)

    param_groups = [
        {
            "params": muon_params,
            "use_muon": True,
            "lr": args.muon_lr,
            "momentum": args.muon_momentum,
            "sv_momentum": args.sv_momentum,
            "weight_decay": args.muon_wd,
        },
        {
            "params": adam_params,
            "use_muon": False,
            "lr": args.adam_lr,
            "betas": (args.beta1, args.beta2),
            "weight_decay": args.adam_wd,
        },
    ]

    print(f"Initializing {args.optimizer.upper()} optimizer...")
    if args.optimizer == "smuon":
        optimizer = SingleDeviceSMuonWithAuxAdam(
            param_groups,
            p_method=args.p_method,
            subsampling_ratio=args.subsampling_ratio,
            moment_type=args.moment_type,
        )
    elif args.optimizer == "muon":
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
    else:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.adam_lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.adam_wd,
        )

    # Scheduler
    total_steps = (len(dataloader) // args.grad_accum) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    # ==========================================
    # Classic PyTorch Training Loop
    # ==========================================
    recorder = ActivationRecorder(model, use_gram=False)
    use_smuon = args.optimizer == "smuon"
    global_step = 0

    print(f"Starting Training: {args.wandb_run_name}")

    for epoch in range(args.epochs):
        model.train()
        epoch_iterator = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            disable=not is_main_process,
        )

        for step, batch in enumerate(epoch_iterator):
            batch = {k: v.to(model.device) for k, v in batch.items()}

            # SMuon Logic: Only record on specific accumulation boundaries
            is_accum_boundary = (step + 1) % args.grad_accum == 0
            should_record = (
                use_smuon
                and is_accum_boundary
                and (
                    ((global_step % args.smuon_interval == 0) and (global_step != 0))
                    or (global_step == 10)
                )
            )

            # 1. Forward Pass
            if should_record:
                with recorder.recording():
                    outputs = model(**batch)
            else:
                outputs = model(**batch)

            loss = outputs.loss / args.grad_accum

            # Crash Guard
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n[CRITICAL] NaN Loss detected at step {step}!")
                raise ValueError("Training halted due to NaN loss.")

            # 2. Backward Pass
            loss.backward()

            # 3. Optimizer Step (only on accumulation boundary)
            if is_accum_boundary:
                if should_record:
                    optimizer.update_p_state(
                        activations=recorder.get_activations(), use_gram=False
                    )
                    recorder.clear()

                    if hasattr(optimizer, "get_p_state_for_logging"):
                        p_state = optimizer.get_p_state_for_logging()
                        for param_name, state in p_state.items():
                            safe_name = param_name.replace(".", "/")
                            if is_main_process:
                                wandb.log(
                                    {f"p_star/{safe_name}": state["p_star"]},
                                    step=global_step,
                                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if is_main_process:
                    wandb.log(
                        {
                            "train/loss": loss.item() * args.grad_accum,
                            "train/lr": scheduler.get_last_lr()[0],
                        },
                        step=global_step,
                    )
                epoch_iterator.set_postfix(loss=(loss.item() * args.grad_accum))
                global_step += 1

        # ==========================================
        # End of Epoch Evaluation
        # ==========================================
        if is_main_process and "gsm8k" in args.dataset_id and eval_dataset is not None:
            print(f"\nRunning GSM8k Evaluation for Epoch {epoch + 1}...")
            # Generating text is slow; 100 samples is a good baseline. Adjust if needed.
            gsm8k_acc = evaluate_gsm8k(model, tokenizer, eval_dataset, num_samples=100)
            print(
                f"Epoch {epoch + 1} GSM8k Accuracy (100 samples): {gsm8k_acc * 100:.2f}%"
            )

            wandb.log(
                {"eval/gsm8k_accuracy": gsm8k_acc},
                step=global_step,
            )

    # Save model
    if is_main_process:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        wandb.finish()


if __name__ == "__main__":
    config_path = Path(__file__).parent / "configs" / "lora.yaml"
    args = load_config_with_overrides(config_path)
    train_lora_model(args)
