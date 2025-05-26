from datetime import datetime 
from tqdm import tqdm 

import os
import wandb
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from training.dataloader import get_train_dataloader, get_val_dataloader
from training.eval import evaluate_metrics_only, evaluate_yolo_model, run_full_evaluation
from models.factory import create_model
from training.loss import FocalLoss, WeightedLoss, LabelSmoothingLoss
from utils.classes import get_class_weights
from config import CONFIG
from utils.input_size import get_input_size_for_model
from training.early_stopping import EarlyStopping

def _initialize_wandb(config):
    return wandb.init(
        project=CONFIG['project_name'],
        name=f"{config['model_name']}_{config['model_variant']}_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        job_type = "Training",
        config = CONFIG,
        tags = ["training", config['model_name'], config['model_variant']],
    )

def _setup_training_environment(config):
    CONFIG['img_size'] = get_input_size_for_model(config['model_name'], config['model_variant'])
    
    config.setdefault("loss_type", "focal")  # or "labelsmoothing", "weighted"
    config.setdefault("early_stopping_patience", 5)
    config.setdefault("batch_size", 256)
    config.setdefault("learning_rate", 0.0004175)
    config.setdefault("weight_decay", 0.125)
    config.setdefault("epochs", 8)
    print(f"[INFO] Using input size: {CONFIG['img_size']} for model: {config['model_name']}")
    print(f"[INFO] Using num classes: {config['num_classes']} for model: {config['model_name']}")
    print(f"[INFO] Using batch size: {config['batch_size']} for model: {config['model_name']}")
    print(f"[INFO] Using learning rate: {config['learning_rate']} for model: {config['model_name']}")
    print(f"[INFO] Using weight decay: {config['weight_decay']} for model: {config['model_name']}")
    print(f"[INFO] Using epochs: {config['epochs']} for model: {config['model_name']}")
    print(f"[INFO] Using early stopping patience: {config['early_stopping_patience']} for model: {config['model_name']}")
    print(f"[INFO] Using loss type: {config['loss_type']}")


def _select_loss_function(weights, device, config):
    loss_type = config['loss_type']
    if loss_type == 'focal':
        alpha = config.get('focal_alpha', 0.25)
        gamma = config.get('focal_gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma, class_weights=weights, device=device)
    elif loss_type == 'labelsmoothing':
        epsilon = config.get('labelsmoothing_epsilon', 0.1)
        return LabelSmoothingLoss(epsilon=epsilon)
    elif loss_type == 'weighted':
        return WeightedLoss(weights)
    else:
        raise ValueError(f"Unsupported loss function type: {loss_type}")
    
def _select_optimizer(model, config):
    if config['optimizer'] == 'adam':
        return optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'adamw':
        return optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'sgd':
        return optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9, weight_decay=config['weight_decay'])
    else:
        raise ValueError(f"Unsupported optimizer type: {config['optimizer']}")
    
def _select_scheduler(optimizer, config):
    # TODO : Implement a more sophisticated scheduler if needed & make parameters sweepable
    return CosineAnnealingLR(optimizer, T_max=100)

def _train_one_epoch(model, train_loader, criterion, optimizer, device, epoc, total_epochs):
    model.train()
    running_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoc + 1}/{total_epochs}", unit="batch", total=len(train_loader))
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(train_loader)

def train_model(model, config):
    _setup_training_environment(config)

    run= _initialize_wandb(config)
    run_name = run.name

    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")

    if config["model_name"] == "yolov11":
        config['model_variant'] = "Default-cls"
        model = create_model('yolov11')
        model_path = model.train(run, config)
        evaluate_yolo_model(model_path, config['yolo_dataset_path'], run_name)
        return
    
    model.to(device)
    train_loader = get_train_dataloader(config)
    val_loader = get_val_dataloader(config)
    _, class_weights = get_class_weights(strategy="balanced")

    optimizer = _select_optimizer(model, config)
    scheduler = _select_scheduler(optimizer, config)
    criterion = _select_loss_function(class_weights, device, config)

    # Set up early stopping
    checkpoint_path = os.path.join(wandb.run.dir, "checkpoints", "checkpoint.pt")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    early_stopper = EarlyStopping(
        patience=config["early_stopping_patience"],
        delta=0.015,
        path=checkpoint_path,
        monitor_acc=False,
    )

    for epoch in range(config['epochs']):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, config['epochs'])

        val_loss, acc1, acc5, *_ = evaluate_metrics_only(model, val_loader, criterion=criterion)

        early_stopper(val_loss, acc1, model)

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "metrics/accuracy_top1": acc1,
            "metrics/accuracy_top5": acc5,
            "lr": scheduler.get_last_lr()[0],
        }, step=epoch + 1)

        print(f"[Epoch {epoch + 1}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Top-1 Acc: {acc1:.4f} | Top-5 Acc: {acc5:.4f}")

        scheduler.step()

        if early_stopper.early_stop:
            print(f"[INFO] Early stopping at epoch {epoch + 1}. Best Val Loss: {early_stopper.best_loss:.4f}, Top-1 Acc: {early_stopper.best_acc:.4f}")
            break

    print("[INFO] Running final evaluation on best model...")
    model.load_state_dict(torch.load(checkpoint_path))
    run_full_evaluation(model, val_loader, config['yolo_dataset_path'])

    final_model_path = os.path.join("model_weights", config["model_name"], config["model_variant"], run.name, "best.pt")
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save(model.state_dict(), final_model_path)

    print(f"[INFO] Best model saved at: {final_model_path}")




