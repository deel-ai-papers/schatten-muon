import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import timm
import wandb
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from accelerate import Accelerator

from smuon.utils import get_optimizer
from smuon.wrap_model import ActivationRecorder


def load_config_with_overrides(config_path: str) -> argparse.Namespace:
    """Load config from YAML and allow CLI argument overrides.

    Arguments are dynamically created from the config file keys.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser(description="Vision model training")

    # Dynamically add arguments based on config keys and infer types
    for key, value in config.items():
        arg_type = type(value) if value is not None else str
        parser.add_argument(f"--{key}", type=arg_type, default=None)

    cli_args = parser.parse_args()

    # Override config with CLI args (only if provided)
    for key, value in vars(cli_args).items():
        if value is not None:
            config[key] = value

    return argparse.Namespace(**config)


def set_seed(seed: int):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_timm_model(
    args,
    data_dir=os.environ.get("IMAGENETTE"),
):
    if args.seed is not None:
        set_seed(args.seed)

    accelerator = Accelerator()

    # Initialize wandb on main process only
    if accelerator.is_main_process:
        wandb.init(
            project="smuon-ImageNette",
            name=getattr(args, "wandb_run_name", None),
            config=vars(args),
            tags=[args.tag],
        )

    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"), transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "val"), transform=val_transform
    )

    generator = torch.Generator()
    if args.seed is not None:
        generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    model = timm.create_model(args.model_name, pretrained=False, num_classes=10)
    optimizer = get_optimizer(model, args)

    # Cosine annealing LR scheduler
    total_steps = len(train_loader) * args.epochs
    eta_min = getattr(args, "eta_min", 0.0)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min)

    loss_fn = nn.CrossEntropyLoss()

    # Wrap model for activation recording (used by smuon)
    # Use Gram matrices for correct distributed aggregation across GPUs
    use_gram = True
    recorder = ActivationRecorder(model, use_gram=use_gram)
    use_smuon = args.optimizer == "smuon"
    smuon_interval = getattr(args, "smuon_interval", 100)

    # Keep reference to unwrapped optimizer for custom methods
    unwrapped_optimizer = optimizer

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    global_step = 0
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0
        for i, data in enumerate(train_loader):
            x, y = data
            optimizer.zero_grad()

            # Record activations every smuon_interval steps
            should_record = use_smuon and (
                ((global_step % smuon_interval == 0) and (global_step != 0))
                or (global_step == 10)
            )
            if should_record:
                with recorder.recording():
                    outputs = model(x)
            else:
                outputs = model(x)

            loss = loss_fn(outputs, y)
            accelerator.backward(loss)

            # Update p state with recorded activations
            if should_record:
                unwrapped_optimizer.update_p_state(
                    activations=recorder.get_activations(),
                    use_gram=use_gram,
                )
                recorder.clear()

                # Log p values and coefficients
                if accelerator.is_main_process:
                    p_state = unwrapped_optimizer.get_p_state_for_logging()
                    for param_name, state in p_state.items():
                        safe_name = param_name.replace(".", "/")
                        wandb.log(
                            {f"p_star/{safe_name}": state["p_star"]},
                            step=global_step,
                        )

            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

            # Log step-level train loss and learning rate
            if accelerator.is_main_process:
                current_lr = scheduler.get_last_lr()[0]
                wandb.log(
                    {
                        "train/loss_step": loss.item(),
                        "train/lr": current_lr,
                    },
                    step=global_step,
                )

            global_step += 1

        avg_train_loss = train_loss / len(train_loader)

        # Evaluate
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                outputs = model(x)
                loss = loss_fn(outputs, y)
                val_loss += loss.item()

                # gather_for_metrics prevents padded batch duplication across GPUs
                preds = outputs.argmax(dim=-1)
                preds, y = accelerator.gather_for_metrics((preds, y))

                correct += (preds == y).sum().item()
                total += y.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total

        accelerator.print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # Log epoch-level metrics
        if accelerator.is_main_process:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss_epoch": avg_train_loss,
                    "val/loss": avg_val_loss,
                    "val/accuracy": val_acc,
                },
                step=global_step,
            )

    # Finish wandb run
    if accelerator.is_main_process:
        wandb.finish()

    accelerator.end_training()


if __name__ == "__main__":
    config_path = Path(__file__).parent / "configs" / "vision.yaml"
    args = load_config_with_overrides(config_path)
    train_timm_model(args)
