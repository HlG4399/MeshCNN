import onnx
onnx_model = onnx.load("mesh_classifier.onnx")
onnx.checker.check_model(onnx_model)

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mesh_index = 0
dummy_input = (torch.randn(16, 5, 750, requires_grad=True).to(device), torch.tensor([mesh_index]).to(device))

import onnxruntime as ort
sess = ort.InferenceSession("mesh_classifier.onnx")
onnx_output = sess.run(None, {"input_name": dummy_input})[0]
