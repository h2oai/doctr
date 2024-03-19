# Import required libraries
import time
import torch
import numpy as np
import onnxruntime
from doctr.models import ocr_predictor
from openvino.runtime import Core


start_load_time = time.time()
device = torch.device('cpu')
model = ocr_predictor(det_arch='db_resnet50', pretrained=True).det_predictor.model
model.to(device).eval()
model_load_time = time.time() - start_load_time
print(f"PyTorch Model Load Time: {model_load_time} seconds")


# Define a function for PyTorch inference and benchmarking
def pytorch_inference(model, input_tensor):
    with torch.no_grad():
        return model(input_tensor).detach().cpu().numpy()


# Define a function to benchmark ONNX inference and verify accuracy
def benchmark_onnx_inference_and_verify(model_path, input_tensor, pytorch_output):
    session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    start_time = time.time()
    onnx_output = session.run(None, {"input": input_tensor.numpy()})
    inference_time = time.time() - start_time

    # Verify accuracy
    np.testing.assert_allclose(pytorch_output, onnx_output[0], rtol=1e-3, atol=1e-5)
    print("ONNX Runtime verification passed")
    return inference_time


# Define a function to benchmark OpenVINO inference and verify accuracy
def benchmark_openvino_inference_and_verify(model_path, input_tensor, pytorch_output):
    ie = Core()
    model_onnx = ie.read_model(model=model_path)
    compiled_model = ie.compile_model(model=model_onnx, device_name="CPU")
    output_layer = compiled_model.output(0)

    start_time = time.time()
    openvino_output = compiled_model([input_tensor.numpy()])[output_layer]
    inference_time = time.time() - start_time

    # Verify accuracy
    np.testing.assert_allclose(pytorch_output, openvino_output, rtol=1e-3, atol=1e-5)
    print("OpenVINO Runtime verification passed")
    return inference_time


torch.onnx.export(model,
                  torch.randn(1, 3, 1024, 1024),
                  'db_resnet50.onnx',
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names=["input"],
                  output_names=["output"],
                  dynamic_axes={"input": {0: "batch_size", 2: "height", 3: "width"},
                                "output": {0: "batch_size", 2: "height", 3: "width"}})

# Example input tensor
input_tensor_1024 = torch.randn(1, 3, 1024, 1024)
input_tensor_1536 = torch.randn(1, 3, 1536, 1536)

# Perform PyTorch inference and capture output
start = time.time()
pytorch_output_1024 = pytorch_inference(model, input_tensor_1024)
print(f"PyTorch Inference Time (1024x1024): {time.time() - start}")

start = time.time()
pytorch_output_1536 = pytorch_inference(model, input_tensor_1536)
print(f"PyTorch Inference Time (1536x1536): {time.time() - start}")


# Benchmark and verify ONNX Runtime
time_onnx_1024 = benchmark_onnx_inference_and_verify('db_resnet50.onnx', input_tensor_1024,
                                                     pytorch_output_1024)
print(f"ONNX Runtime Inference Time (1024x1024): {time_onnx_1024}")

time_onnx_1536 = benchmark_onnx_inference_and_verify('db_resnet50.onnx', input_tensor_1536,
                                                        pytorch_output_1536)
print(f"ONNX Runtime Inference Time (1536x1536): {time_onnx_1536}")

# Benchmark and verify OpenVINO
time_openvino_1024 = benchmark_openvino_inference_and_verify('db_resnet50.onnx', input_tensor_1024,
                                                             pytorch_output_1024)
print(f"OpenVINO Inference Time (1024x1024): {time_openvino_1024}")

time_openvino_1536 = benchmark_openvino_inference_and_verify('db_resnet50.onnx', input_tensor_1536,
                                                                pytorch_output_1536)
print(f"OpenVINO Inference Time (1536x1536): {time_openvino_1536}")
