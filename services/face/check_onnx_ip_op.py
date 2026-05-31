import onnx

model = onnx.load(r"triton_face_detection/model_repository/retinaface/1/model.onnx")

print("Inputs:")
for input in model.graph.input:
    print(input.name)

print("\nOutputs:")
for output in model.graph.output:
    print(output.name)