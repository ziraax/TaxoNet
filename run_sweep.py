import wandb
import yaml
from config import CONFIG
from models.factory import create_model
from training.train import train_model
from utils.classes import get_num_classes

def sweep_train():
    #initialize wandb
    wandb.init()
    config = wandb.config
    CONFIG.update(dict(config))
    CONFIG["num_classes"] = get_num_classes()


    model = create_model(
        model_name=config.model_name,
        num_classes=CONFIG["num_classes"],
        pretrained=config.get("pretrained", True),  # if you sweep over this too
        freeze_backbone=config.freeze_backbone,
        efficientnet_variant=config.get("efficientnet_variant", "b0"),
        resnet_variant=config.get("resnet_variant", "50"),
        densenet_variant=config.get("densenet_variant", "121"),
        mc_dropout=config.mc_dropout,
        mc_p=config.mc_p,
    )

    
    train_model(model, CONFIG)

if __name__ == "__main__":
    # Load the YAML file properly
    with open("sweep.yaml") as f:
        sweep_config = yaml.safe_load(f)

    # This returns the sweep ID
    sweep_id = wandb.sweep(sweep=sweep_config, project=CONFIG['project_name'])

    # This launches the agent to run sweeps
    wandb.agent(sweep_id, function=sweep_train)
