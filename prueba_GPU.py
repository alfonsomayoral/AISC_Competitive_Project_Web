import torch

def main():
    print("CUDA available? ", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("Device #0 name:   ", torch.cuda.get_device_name(0))
        print("Torch CUDA version:", torch.version.cuda)

if __name__ == "__main__":
    main()