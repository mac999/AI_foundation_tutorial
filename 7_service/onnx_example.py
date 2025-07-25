import os, torch, onnxruntime, numpy as np

# Defining PyTorch model
class MyModel(torch.nn.Module):
	def __init__(self):
		super(MyModel, self).__init__()
		self.fc1 = torch.nn.Linear(10, 512)   # 10 -> 512
		self.fc2 = torch.nn.Linear(512, 1024) # 512 -> 1024
		self.fc3 = torch.nn.Linear(1024, 512) # 1024 -> 512
		self.fc4 = torch.nn.Linear(512, 256)  # 512 -> 256
		self.fc5 = torch.nn.Linear(256, 10)   # 256 -> 10

	def forward(self, x):
		x = torch.relu(self.fc1(x))
		x = torch.relu(self.fc2(x))
		x = torch.relu(self.fc3(x))
		x = torch.relu(self.fc4(x))
		x = self.fc5(x)
		return x

# Creating an instance
model = MyModel()

# Training data
X_train = torch.randn(1000, 10)
y_train = torch.randn(1000, 10)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

for epoch in range(200):
	optimizer.zero_grad()
	outputs = model(X_train)
	loss = criterion(outputs, y_train)
	loss.backward()
	optimizer.step()
	
	if epoch % 20 == 0:
		print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# Defining input example
example_input = torch.randn(1, 10)

# Exporting to ONNX format
module_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(module_path, "model.onnx")
torch.onnx.export(model, example_input, model_path)

# Using ONNX model
session = onnxruntime.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
onnx_output = session.run(None, {input_name: example_input.numpy()})

# Compare outputs
pytorch_output = model(example_input)
print("PyTorch Output:", pytorch_output.detach().numpy())
print("ONNX Output:", onnx_output[0])

# Quantizing the ONNX model
from onnxruntime.quantization import quantize_dynamic, QuantType

quantized_model_path = os.path.join(module_path, "model_quantized.onnx")
quantize_dynamic(model_path, quantized_model_path, weight_type=QuantType.QUInt8)

# Load the quantized model
quantized_session = onnxruntime.InferenceSession(quantized_model_path)
quantized_output = quantized_session.run(None, {input_name: example_input.numpy()})
print("Quantized ONNX Output:", quantized_output[0])

# Compare file sizes
original_size = os.path.getsize(model_path)
quantized_size = os.path.getsize(quantized_model_path)
print(f"Original model size: {original_size / 1024:.2f} KB")
print(f"Quantized model size: {quantized_size / 1024:.2f} KB")
print(f"Size reduction: {(1 - quantized_size/original_size)*100:.1f}%")
