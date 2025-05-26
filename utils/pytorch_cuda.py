import torch

def check_pytorch_cuda():
    """Prints PyTorch and CUDA-related information."""
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device name:", torch.cuda.get_device_name(0))
        print("CUDA device count:", torch.cuda.device_count())

if __name__ == "__main__":
    check_pytorch_cuda()
