import matplotlib.pyplot as plt
from PIL import Image

def visualize_predictions(results, max_images=16):
    """
    Visualize the predictions with images and their corresponding labels.
    
    Args:
        results (list): List of tuples containing image path, predicted class index, and confidence score.
        max_images (int): Maximum number of images to display.
    """
    num_images = min(len(results), max_images)
    cols = 4
    rows = (num_images + cols - 1) // cols
    plt.figure(figsize=(16, 4 * rows))
    for i in range(num_images):
        item = results[i]
        img = Image.open(item['image_path']).convert("RGB")
        preds = item['predictions']
        title = "\n".join([f"{cls} ({conf:.2f})" for cls, conf in preds])
        plt.subplot(rows, cols, i+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(title)
    plt.tight_layout()
    plt.show()