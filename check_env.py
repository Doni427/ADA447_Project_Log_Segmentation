import torch

def verify_environment():
    print(f"PyTorch Version: {torch.__version__}")
    
    # Check for Apple Silicon (MPS)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Apple Silicon Acceleration (MPS) is available.")
    # Check for NVIDIA CUDA
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ CUDA Acceleration is available. Device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️ No hardware acceleration found. Defaulting to CPU.")
        
    # Run a quick tensor verification
    x = torch.rand(3, 3, device=device)
    print(f"Successfully allocated dummy tensor on: {x.device}")

if __name__ == "__main__":
    verify_environment()