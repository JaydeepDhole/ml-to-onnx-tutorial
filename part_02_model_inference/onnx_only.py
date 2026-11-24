"""Deploy the ONNX artifact without scikit-learn or Joblib."""
import argparse
from pathlib import Path
import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    args = parser.parse_args()
    model_path = Path(__file__).resolve().parents[1] / "models" / "linear_regression.onnx"
    if not model_path.is_file():
        raise FileNotFoundError("Run Part 1 to generate the ONNX artifact.")
    provider = "CUDAExecutionProvider" if args.device == "gpu" else "CPUExecutionProvider"
    if provider not in ort.get_available_providers():
        raise SystemExit(f"Requested provider unavailable: {provider}")
    providers = [provider, "CPUExecutionProvider"] if args.device == "gpu" else [provider]
    session = ort.InferenceSession(str(model_path), providers=providers)
    if provider not in session.get_providers():
        raise SystemExit(f"Requested provider failed to initialize: {provider}")
    session.disable_fallback()
    samples = np.array([[0.0], [25.0], [100.0]], dtype=np.float32)
    result = session.run(None, {session.get_inputs()[0].name: samples})[0]
    print("Session providers:", session.get_providers())
    for celsius, fahrenheit in zip(samples.ravel(), result.ravel()):
        print(f"{celsius:g} °C -> {fahrenheit:.3f} °F")
    if args.device == "gpu":
        print("CUDA preferred; individual graph nodes may execute on CPU.")


if __name__ == "__main__":
    main()
