import argparse
import wandb

from config import CONFIG

from inference.runner import run_inference
from inference.utils import save_results
from inference.visualize import visualize_predictions
from training.train import train_model
from models.factory import create_model

from utils.classes import get_num_classes
from utils.input_size import get_input_size_for_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train a custom model.")
    
    #------------ Training Arguments ------------
    parser.add_argument('--model_name', type=str, default='densenet',
        choices=['yolov11', 'resnet', 'densenet', 'efficientnet'], help='Name of the model to train (default: densenet)'
    )
    # Add pretrained argument like --pretrained true or --pretrained false
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='Use pretrained weights')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false', help='Do not use pretrained weights')
    parser.set_defaults(pretrained=True)  # default = True

    # Add freeze_backbone argument like --freeze_backbone true or --freeze_backbone false
    parser.add_argument('--freeze_backbone', dest='freeze_backbone', action='store_true', help='Freeze the backbone of the model')
    parser.add_argument('--no-freeze_backbone', dest='freeze_backbone', action='store_false', help='Do not freeze the backbone of the model')
    parser.set_defaults(freeze_backbone=False)  # default = False

    # Add efficientnet_variant argument like --efficientnet_variant b0 or --efficientnet_variant b1
    parser.add_argument('--efficientnet_variant', type=str, default='b0',
        choices=['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'], help="Variant of EfficientNet to use (default: b0)"
    )

    # Add densenet_variant argument like --densenet_variant 121 or --densenet_variant 169
    parser.add_argument('--densenet_variant', type=str, default='densenet121',
        choices=['121', '169', '201', '161'], help="Variant of DenseNet to use (default: densenet121)"
    )

    # add resnet_variant argument like --resnet_variant 18 or --resnet_variant 34
    parser.add_argument("--resnet_variant", type=str, default='50',
        choices=['18', '34', '50', '101', '152'], help="Variant of ResNet to use (default: 50)"
    )

    parser.add_argument("--mc_dropout",   action="store_true", help="activate MC-Dropout during inference")
    parser.add_argument("--mc_p", type=float, default=0.3, help="dropout probability when mc_dropout is on")

    return parser.parse_args()

def configure_model(args):
    variant_map = {
        'densenet': args.densenet_variant,
        'efficientnet': args.efficientnet_variant,
        'resnet': args.resnet_variant,
        'yolov11': 'Default-cls'
    }

    model_variant = variant_map.get(args.model_name, 'None')
    num_classes = get_num_classes()
    img_size = get_input_size_for_model(args.model_name, model_variant)

    CONFIG.update({
        'model_name': args.model_name,
        'model_variant': model_variant,
        'num_classes': num_classes,
        'img_size': img_size,
    })


def main():
    try:
        args = parse_args()
        configure_model(args)

        print(f"[INFO] Creating model: {CONFIG['model_name']} | Variant: {CONFIG['model_variant']} | "
              f"Pretrained: {args.pretrained} | Freeze Backbone: {args.freeze_backbone}")      
          
        model = create_model(
                args.model_name, 
                num_classes=CONFIG['num_classes'], 
                pretrained=args.pretrained,
                freeze_backbone=args.freeze_backbone,
                efficientnet_variant=args.efficientnet_variant,
                resnet_variant=args.resnet_variant,
                densenet_variant=args.densenet_variant,
                mc_dropout=args.mc_dropout,
                mc_p=args.mc_p
        )
            
        # Run preprocessing
        # full_preprocessing()
        train_model(model, CONFIG)

        
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        raise
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()