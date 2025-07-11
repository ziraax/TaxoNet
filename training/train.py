from datetime import datetime 
from sklearn.metrics import f1_score, recall_score
from tqdm import tqdm 

import os
import wandb
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from training.dataloader import get_test_dataloader, get_train_dataloader, get_val_dataloader
from training.eval import evaluate_metrics_only, evaluate_yolo_model, run_test_full_evaluation
from models.factory import create_model
from training.loss import FocalLoss, WeightedLoss, LabelSmoothingLoss
from utils.classes import get_class_weights
from config import DEFAULT_CONFIG
from utils.input_size import get_input_size_for_model
from training.early_stopping import EarlyStopping

def _initialize_wandb(config):

    model_name = config["model_name"]

    # Get the appropriate variant depending on the model
    if model_name == "resnet":
        variant = config.get("resnet_variant", "unknown")
    elif model_name == "densenet":
        variant = config.get("densenet_variant", "unknown")
    elif model_name == "efficientnet":
        variant = config.get("efficientnet_variant", "unknown")
    else:
        variant = "n/a"

    config['model_variant'] = variant

    return wandb.init(
        project=DEFAULT_CONFIG['project_name'],
        name=f"{config['model_name']}_{config['model_variant']}_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        job_type = "Training",
        config = DEFAULT_CONFIG,
        tags = ["training", config['model_name'], config['model_variant']],
    )

def _setup_training_environment(config):
    if 'img_size' not in config:
        print(f"[WARNING] 'img_size' not found in config. Using default input size for model: {config['model_name']}")
        config['img_size'] = get_input_size_for_model(config, config['model_name'], config.get('model_variant', 'Default-cls'))
    
    if 'num_classes' not in config:
        print(f"[WARNING] 'num_classes' not found in config. Using class weights to determine number of classes.")
        config['num_classes'] = len(get_class_weights(config, strategy="balanced")[0])

    config.setdefault("loss_type", "focal")  # or "labelsmoothing", "weighted"
    config.setdefault("focal_gamma", 2.0)
    config.setdefault("use_per_class_alpha", True)  # scalar fallback for focal
    config.setdefault("early_stopping_patience", 20)
    config.setdefault("batch_size", 64)
    config.setdefault("optimizer", "adamw")  # or "adam", "sgd"
    config.setdefault("learning_rate", 0.0004175)
    config.setdefault("weight_decay", 0.125)
    config.setdefault("epochs", 8)
    config.setdefault("labelsmoothing_epsilon", 0.1)


    print(f"[INFO] Using input size: {config['img_size']} for model: {config['model_name']}")
    print(f"[INFO] Using num classes: {config['num_classes']} for model: {config['model_name']}")
    print(f"[INFO] Using batch size: {config['batch_size']} for model: {config['model_name']}")
    print(f"[INFO] Using learning rate: {config['learning_rate']} for model: {config['model_name']}")
    print(f"[INFO] Using weight decay: {config['weight_decay']} for model: {config['model_name']}")
    print(f"[INFO] Using epochs: {config['epochs']} for model: {config['model_name']}")
    print(f"[INFO] Using early stopping patience: {config['early_stopping_patience']} for model: {config['model_name']}")
    print(f"[INFO] Using loss type: {config['loss_type']}")
    print(f"[INFO] Using focal gamma: {config['focal_gamma']})")
    print(f"[INFO] Using focal alpha per class?: {config.get('use_per_class_alpha')}")



def _select_loss_function(weights, device, config):
    loss_type = config['loss_type']
    if loss_type == 'focal':
        gamma = config.get('focal_gamma', 2.0)
        # Use per-class alpha if specified
        if config.get('use_per_class_alpha', False):
            _, class_frequencies = get_class_weights(config, strategy="balanced")  # You may already have this
            alpha = torch.tensor(class_frequencies, dtype=torch.float32).to(device)  # or adjust per your needs
        else:
            alpha = config.get('focal_alpha', 0.25)  # scalar fallback
        return FocalLoss(alpha=alpha, gamma=gamma, device=device)
    elif loss_type == 'labelsmoothing':
        epsilon = config.get('labelsmoothing_epsilon', 0.1)
        return LabelSmoothingLoss(epsilon=epsilon)
    elif loss_type == 'weighted':
        return WeightedLoss(class_weights=weights, device=device)
    else:
        raise ValueError(f"Unsupported loss function type: {loss_type}")
    
def _select_optimizer(model, config):
    lr = float(config['learning_rate'])
    wd = float(config['weight_decay'])

    if config['optimizer'] == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif config['optimizer'] == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif config['optimizer'] == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
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
        config['batch_size'] = config.get('batch_size', 256)
        config['epochs'] = config.get('epochs', 65)
        config['initial_lr'] = config.get('learning_rate', 0.0004175)
        config['optimizer'] = config.get('optimizer', 'adamw')
        config['weight_decay'] = config.get('weight_decay', 0.125)
        config['momentum'] = config.get('momentum', 1.632)

        model = create_model('yolov11')
        model_path = model.train(run, config)
        evaluate_yolo_model(model_path, config['final_dataset_path'], run_name, config)
        return
    
    model.to(device)
    train_loader = get_train_dataloader(config)
    val_loader = get_val_dataloader(config)
    test_loader = get_test_dataloader(config)
    _, class_weights = get_class_weights(config, strategy="balanced")

    optimizer = _select_optimizer(model, config)
    scheduler = _select_scheduler(optimizer, config)
    criterion = _select_loss_function(class_weights, device, config)

    # Set up early stopping
    checkpoint_path = os.path.join(wandb.run.dir, "checkpoints", "checkpoint.pt")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    early_stopper = EarlyStopping(
        patience=config["early_stopping_patience"],
        delta=0.001,
        path=checkpoint_path,
        monitor_acc=False,
    )

    for epoch in range(config['epochs']):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, config['epochs'])

        val_loss, acc1, acc5, y_true, y_pred, *_ = evaluate_metrics_only(model, val_loader, criterion=criterion)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)

        early_stopper(val_loss, acc1, model)

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/metrics/accuracy_top1": acc1,
            "val/metrics/accuracy_top5": acc5,
            "val/metrics/f1_macro": f1_macro,
            "val/metrics/recall_macro": recall_macro,
            "lr": scheduler.get_last_lr()[0],
        }, step=epoch + 1)

        print(f"[Epoch {epoch + 1}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Top-1 Acc: {acc1:.4f} | Top-5 Acc: {acc5:.4f}")

        scheduler.step()

        if early_stopper.early_stop:
            print(f"[INFO] Early stopping at epoch {epoch + 1}. Best Val Loss: {early_stopper.best_loss:.4f}, Top-1 Acc: {early_stopper.best_acc:.4f}")
            break

    print("[INFO] Running final evaluation on best model...")

    model.load_state_dict(torch.load(checkpoint_path))
    run_test_full_evaluation(model, test_loader, config['final_dataset_path'])

    final_model_path = os.path.join("model_weights", config["model_name"], config["model_variant"], run.name, "best.pt")
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save(model.state_dict(), final_model_path)

    print(f"[INFO] Best model saved at: {final_model_path}")




