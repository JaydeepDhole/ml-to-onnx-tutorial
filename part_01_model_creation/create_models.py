from pathlib import Path

# Find the repository whether Jupyter starts in the root or this part's folder.
ROOT = Path(__file__).resolve().parents[1]
if ROOT is None:
    raise FileNotFoundError("Open the notebook inside the tutorial repository.")
MODELS = ROOT / "models"
JOBLIB_PATH = MODELS / "linear_regression.joblib"
ONNX_PATH = MODELS / "linear_regression.onnx"

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.read_csv(ROOT / "data" / "temperature_conversion.csv")
# One feature per row; use a two-dimensional array for the model input.
X = data[["input"]].to_numpy(dtype=np.float64)
y = data["output"].to_numpy(dtype=np.float64)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression().fit(X_train, y_train)
predictions = model.predict(X_test)
print(f"Coefficient: {model.coef_[0]:.6f}; intercept: {model.intercept_:.6f}")
print(f"Test MAE: {mean_absolute_error(y_test, predictions):.8f}")
print(f"Test R²: {r2_score(y_test, predictions):.6f}")
np.testing.assert_allclose(predictions, y_test, atol=1e-9, rtol=0)

MODELS.mkdir(exist_ok=True)
joblib.dump(model, JOBLIB_PATH)
# Reload our own trusted artifact to verify that serialization preserves predictions.
loaded_model = joblib.load(JOBLIB_PATH)
np.testing.assert_allclose(loaded_model.predict(X_test), predictions, rtol=0, atol=0)
print(f"Saved {JOBLIB_PATH.name}: {JOBLIB_PATH.stat().st_size} bytes")

import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# None allows a variable batch size; 1 is the number of input features.
graph_model = convert_sklearn(
    loaded_model, initial_types=[("celsius", FloatTensorType([None, 1]))],
    target_opset=15,
)
# Validate the graph before publishing the artifact.
onnx.checker.check_model(graph_model)
onnx.save(graph_model, ONNX_PATH)
print(f"Saved {ONNX_PATH.name}: {ONNX_PATH.stat().st_size} bytes")

graph_model = onnx.load(ONNX_PATH)
for opset in graph_model.opset_import:
    print("Opset:", opset.domain or "ai.onnx", opset.version)
for node in graph_model.graph.node:
    print("Operator:", node.domain or "ai.onnx", node.op_type)
for tensor in [*graph_model.graph.input, *graph_model.graph.output]:
    print(tensor.name, onnx.helper.printable_type(tensor.type))
