from pathlib import Path

# Find the repository whether Jupyter starts in the root or this part's folder.
ROOT = Path(__file__).resolve().parents[1]
if ROOT is None:
    raise FileNotFoundError("Open the notebook inside the tutorial repository.")
MODELS = ROOT / "models"
JOBLIB_PATH = MODELS / "linear_regression.joblib"
ONNX_PATH = MODELS / "linear_regression.onnx"

import numpy as np
import onnxruntime as ort
from time import perf_counter

if not ONNX_PATH.is_file() or not JOBLIB_PATH.is_file():
    raise FileNotFoundError("Create both artifacts in Part 1 before running this tutorial.")
samples = np.array([[0.0], [25.0], [100.0]], dtype=np.float32)
print("Available providers:", ort.get_available_providers())

import joblib

# Load only the trusted artifact created in Part 1; no fit() call is needed.
model = joblib.load(JOBLIB_PATH)
joblib_predictions = model.predict(samples.astype(np.float64))
print("Joblib CPU:", joblib_predictions)

cpu_session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])

def predict(session, batch):
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    return session.run([output_meta.name], {input_meta.name: batch})[0].reshape(-1)

for item in cpu_session.get_inputs():
    print("Input contract:", item.name, item.shape, item.type)
cpu_predictions = predict(cpu_session, samples)
print("ONNX CPU:", cpu_predictions)
np.testing.assert_allclose(cpu_predictions, joblib_predictions, rtol=1e-5, atol=1e-4)

gpu_session = None
if "CUDAExecutionProvider" not in ort.get_available_providers():
    print("CUDA provider unavailable; GPU experiment skipped.")
else:
    try:
        candidate = ort.InferenceSession(
            str(ONNX_PATH), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        if "CUDAExecutionProvider" in candidate.get_providers():
            # Disable Python-level retry after provider failure; graph partitioning
            # may still assign unsupported operators to CPU.
            candidate.disable_fallback()
            gpu_predictions = predict(candidate, samples)
            np.testing.assert_allclose(gpu_predictions, cpu_predictions, rtol=1e-5, atol=1e-4)
            gpu_session = candidate
            print("CUDA-preferred providers:", candidate.get_providers())
            print("CUDA-preferred predictions:", gpu_predictions)
        else:
            print("CUDA initialization fell back to CPU; GPU experiment skipped.")
    except (RuntimeError, OSError, ValueError) as error:
        print("CUDA session failed:", error)

def benchmark(run, warmup=30, iterations=300):
    for _ in range(warmup):
        run()
    durations = []
    for _ in range(iterations):
        start = perf_counter()
        run()
        durations.append((perf_counter() - start) * 1000)
    return np.mean(durations), np.percentile(durations, 95)

for batch_size in (1, 1024):
    batch = np.linspace(0, 99, batch_size, dtype=np.float32).reshape(-1, 1)
    batch64 = batch.astype(np.float64)
    runners = {"Joblib CPU": lambda: model.predict(batch64),
               "ONNX CPU": lambda: predict(cpu_session, batch)}
    if gpu_session is not None:
        runners["ONNX CUDA-preferred"] = lambda: predict(gpu_session, batch)
    for name, run in runners.items():
        mean, p95 = benchmark(run)
        print(f"{batch_size:4d} | {name:22s} | mean={mean:.4f} ms | p95={p95:.4f} ms")
