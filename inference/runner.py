import sys
import os    
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from torch.utils.data import DataLoader
from inference.dataset import InferenceDataset
from models.factory import create_model
from inference.utils import topk_predictions
from utils.classes import get_classes_names
from utils.input_size import get_input_size_for_model


def run_inference(args, CONFIG):
    """
    Run inference on a set of images using a specified model.
    Args:
        args: Command line arguments containing model and data parameters.
        CONFIG: Configuration dictionary containing model settings.
    Returns:
        all_results: List of dictionaries containing image paths and predictions.

    Supports:
        - Batch inference for multiple images.
    """


    image_paths = [os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    class_names = get_classes_names()
    CONFIG['num_classes'] = len(class_names)
    if args.model_name == 'densenet':
        CONFIG['model_variant'] = args.densenet_variant
    elif args.model_name == 'resnet':
        CONFIG['model_variant'] = args.resnet_variant
    else:
        CONFIG['model_variant'] = args.efficientnet_variant
    CONFIG['img_size'] = get_input_size_for_model(args.model_name, CONFIG['model_variant'])

    all_results = []

    if args.model_name == "yolov11":
        # Load YOLOv11 classification model
        model = YOLO(args.weights_path, "classify")

        # Perform batch inference
        results = model(image_paths, imgsz=CONFIG['img_size'], device=args.device)

        for path, result in zip(image_paths, results):
            top5_indices = result.probs.top5
            top5_confs = result.probs.top5conf
            preds = [(class_names[i], round(top5_confs[j].item(), 4)) for j, i in enumerate(top5_indices)]
            all_results.append({
                'image_path': path,
                'predictions': preds
            })

    else:
        # Torchvision models
        dataset = InferenceDataset(image_paths, CONFIG['img_size'])
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        model = create_model(
            args.model_name,
            num_classes=CONFIG['num_classes'],
            pretrained=False,
            efficientnet_variant=args.efficientnet_variant,
            densenet_variant=args.densenet_variant,
            resnet_variant=args.resnet_variant,
            mc_dropout=args.mc_dropout,
            mc_p=args.mc_dropout_p if hasattr(args, "mc_dropout_p") else 0.5  # optional dropout prob        
        )

        model.load_state_dict(torch.load(args.weights_path, map_location=args.device))
        model = model.to(args.device)

        # By default, we enable eval mode
        model.eval()

        def enable_mc_dropout(m):
            if isinstance(m, torch.nn.Dropout):
                m.train()

        #--------- Monte Carlo Dropout ---------
        if args.mc_dropout:
            model.apply(enable_mc_dropout)
            mc_iters = args.mc_iterations
        else:
            mc_iters = 1
        #---------------------------------------
        # Perform batch inference
        with torch.no_grad():
            for batch_images, batch_paths in dataloader:
                batch_images = batch_images.to(args.device)

                #--- Collect logits over N passes
                mc_logits = []	
                for _ in range(mc_iters):
                    logits = model(batch_images)            # (B, C)
                    mc_logits.append(logits.unsqueeze(0))   # (1, B, C)
                
                logits_stack = torch.cat(mc_logits, dim=0)  # (N, B, C)

                # Compute probabilities (softmax) for each MC pass
                probs_stack = F.softmax(logits_stack, dim=2)
                # (N, B, C) -> (B, C)
                mean_probs = probs_stack.mean(dim=0)
                std_probs = probs_stack.std(dim=0)
                

                for i, path in enumerate(batch_paths):
                    # Top-k indices and scores from mean probs
                    topk_scores, topk_indices = mean_probs[i].topk(args.top_k)

                    preds_with_uncertainty = []
                    for score, idx in zip(topk_scores, topk_indices):
                        class_name = class_names[idx]
                        uncertainty = std_probs[i, idx].item()
                        preds_with_uncertainty.append((class_name, round(score.item(), 4), round(uncertainty, 4)))

                    all_results.append({
                        'image_path': path,
                        'predictions': preds_with_uncertainty
                    })
    if args.save_csv:
        from inference.utils import save_results
        save_results(all_results, args.save_csv)

    return all_results
