import pandas as pd 
import json 
import torch

def save_results(results, output_path):
    """
    Save the inference results to a CSV file.

    Args:
        results (list): List of tuples containing image path, predicted class index, and confidence score.
        output_path (str): Path to save the CSV file.
    """

    flat_rows = []
    for item in results: 
        row = {
            'image_path': item['image_path'],
        }
        if 'uncertainty' in item:
            row['uncertainty'] = item['uncertainty']
        # Dynamically create column headers for top-k
        for i, (label, score) in enumerate(item['predictions']):
            row[f'top{i+1}_label'] = label
            row[f'top{i+1}_score'] = score
        flat_rows.append(row)

    if output_path.endswith('.json'):
        with open(output_path, 'w') as f:
            json.dump(flat_rows, f, indent=2)
    else:
        df = pd.DataFrame(flat_rows)
        df.to_csv(output_path, index=False)


def topk_predictions(logits, class_names, top_k=3):
    """
    Get the top K predictions from the model output.

    Args:
        logits (torch.Tensor): Model output logits.
        class_names (list): List of class names.
        top_k (int): Number of top predictions to return.

    Returns:
        list: List of tuples containing class name and confidence score.
    """
    probs = torch.softmax(logits, dim=1)
    topk_conf, topk_idx = torch.topk(probs, k=top_k, dim=1)

    results = []
    for confs, indices in zip(topk_conf, topk_idx):
        preds = [(class_names[i], round(c.item(), 4)) for i, c in zip(indices, confs)]
        results.append(preds)
    return results






    