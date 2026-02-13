"""Verify CUDA + PyTorch setup on GB10."""
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"VRAM: {props.total_memory / 1e9:.1f} GB")
    print(f"Compute capability: {props.major}.{props.minor}")

    # Quick tensor test
    x = torch.randn(1024, 1024, device="cuda")
    y = torch.randn(1024, 1024, device="cuda")
    torch.cuda.synchronize()

    import time
    start = time.perf_counter()
    for _ in range(100):
        z = torch.mm(x, y)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tflops = (2 * 1024**3 * 100) / elapsed / 1e12
    print(f"MatMul throughput: {tflops:.2f} TFLOPS (FP32, 1024x1024)")
else:
    print("ERROR: CUDA not available!")
    exit(1)
