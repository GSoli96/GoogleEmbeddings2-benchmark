"""
check_gpu.py
============
Checks available GPU acceleration for this project's stack:
PyTorch (CUDA / MPS), FAISS, and SentenceTransformers.
"""

import sys


def section(title: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print('─' * 50)


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------
section("Python")
print(f"  Version : {sys.version.split()[0]}")
print(f"  Path    : {sys.executable}")


# ---------------------------------------------------------------------------
# PyTorch
# ---------------------------------------------------------------------------
section("PyTorch")
try:
    import torch
    print(f"  Version : {torch.__version__}")

    cuda_ok = torch.cuda.is_available()
    print(f"  CUDA    : {'✓ available' if cuda_ok else '✗ not available'}")
    if cuda_ok:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            vram = p.total_memory / 1024 ** 3
            print(f"            [{i}] {p.name}  {vram:.1f} GB VRAM")

    mps_ok = torch.backends.mps.is_available()
    print(f"  MPS     : {'✓ available (Apple Silicon)' if mps_ok else '✗ not available'}")
    if mps_ok:
        # Quick smoke-test: move a small tensor to MPS
        try:
            t = torch.ones(4, device="mps")
            _ = (t * 2).cpu()
            print("            smoke-test: OK")
        except Exception as e:
            print(f"            smoke-test FAILED: {e}")

    active = "cuda" if cuda_ok else ("mps" if mps_ok else "cpu")
    print(f"\n  → LocalEmbedder will use: {active.upper()}")

except ImportError:
    print("  ✗ PyTorch not installed")


# ---------------------------------------------------------------------------
# FAISS
# ---------------------------------------------------------------------------
section("FAISS")
try:
    import faiss
    print(f"  Version : {faiss.__version__}")
    print(f"  Threads : {faiss.omp_get_max_threads()} (OMP max)")

    gpu_ok = hasattr(faiss, "StandardGpuResources")
    print(f"  GPU     : {'✓ faiss-gpu build' if gpu_ok else '✗ CPU-only build (faiss-cpu)'}")
    if gpu_ok:
        try:
            res = faiss.StandardGpuResources()
            idx = faiss.GpuIndexFlatIP(res, 16)
            print("            GPU index init: OK")
        except Exception as e:
            print(f"            GPU index init FAILED: {e}")
    else:
        print("  → FAISS will run on CPU (GPU requires CUDA + faiss-gpu)")

except ImportError:
    print("  ✗ FAISS not installed")


# ---------------------------------------------------------------------------
# SentenceTransformers
# ---------------------------------------------------------------------------
section("SentenceTransformers")
try:
    import sentence_transformers as st
    print(f"  Version : {st.__version__}")
    print("  (device selection delegated to PyTorch — see above)")
except ImportError:
    print("  ✗ sentence-transformers not installed")


# ---------------------------------------------------------------------------
# Google GenAI SDK
# ---------------------------------------------------------------------------
section("Google GenAI SDK (for ge2 model)")
try:
    from google import genai
    print(f"  Version : {genai.__version__}")
    print("  → Embeddings run on Google Cloud — no local GPU needed")
except ImportError:
    print("  ✗ google-genai not installed")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
section("Summary")
try:
    cuda_ok
except NameError:
    cuda_ok = False
try:
    mps_ok
except NameError:
    mps_ok = False

if cuda_ok:
    print("  GPU acceleration: FULL (CUDA — PyTorch + potentially FAISS)")
elif mps_ok:
    print("  GPU acceleration: PARTIAL (Apple MPS — PyTorch/SentenceTransformers only)")
    print("                    FAISS runs on CPU (requires CUDA for GPU mode)")
else:
    print("  GPU acceleration: NONE — everything runs on CPU")
print()
