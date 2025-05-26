import torch 

def batch_mc_predictions(model, images, T=20):
    """
    Run T stochastic forward passes and return mean & std (epistemic).
    """
    mean, std = model.predict_mc(images, T=T) # [B, C]
    return mean.cpu(), std.cpu()

