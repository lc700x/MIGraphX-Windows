"""ONNX parser smoke test — export ResNet18 from PyTorch, run on GPU via MIGraphX.

Run with the venv python:
    F:\\MIGraphxWin\\venv\\Scripts\\python.exe test_onnx.py

Requires: pip install onnx
The migraphx wheel's .pth sets up ROCm DLL paths automatically.
"""
import numpy as np
import torch
import torchvision
import migraphx

ONNX_PATH = "resnet18.onnx"

# Export ResNet18 to ONNX (legacy exporter — no onnxscript needed)
model = torchvision.models.resnet18(weights=None).eval()
torch.onnx.export(
    model, torch.randn(1, 3, 224, 224), ONNX_PATH,
    input_names=["input"], output_names=["output"],
    opset_version=17, dynamo=False,
)
print(f"Exported {ONNX_PATH}")

# Parse + compile for GPU + run
prog = migraphx.parse_onnx(ONNX_PATH)
prog.compile(migraphx.get_target("gpu"))

x = np.random.randn(1, 3, 224, 224).astype(np.float32)
name = list(prog.get_parameter_shapes().keys())[0]
out = np.array(prog.run({name: migraphx.argument(x)})[0])

print(f"Output shape: {out.shape}")
assert out.shape == (1, 1000), f"unexpected shape {out.shape}"
print("=== parse_onnx + GPU inference OK ===")
