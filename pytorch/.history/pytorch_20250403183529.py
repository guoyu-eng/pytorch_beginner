import torch

def check_pytorch_availability():
    if torch.cuda.is_available():
        print("CUDA is available. PyTorch can use GPU.")
    else:
        print("CUDA is not available. PyTorch will use CPU.")

    print(f"PyTorch version: {torch.__version__}")
    print(dir(torch))

if __name__ == "__main__":
    check_pytorch_availability()
