import torch

def check_gpu():
    if torch.cuda.is_available():
        print(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("CUDA is not available. Running on CPU.")
        return False

if __name__ == "__main__":
    check_gpu()