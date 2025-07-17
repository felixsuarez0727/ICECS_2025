import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

import numpy as np
import tensorflow as tf

def convert_and_test_model(keras_model_path: str, tflite_model_path: str) -> int:
    # Load Keras model
    model_keras = tf.keras.models.load_model(keras_model_path)

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model_keras)
    tflite_model = converter.convert()

    # Save TFLite model
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)

    # Load TFLite model interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Prepare dummy input
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    input_data = np.random.random(input_shape).astype(input_dtype)

    # Run Keras inference
    output_keras = model_keras.predict(input_data)

    # Run TFLite inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_tflite = interpreter.get_tensor(output_details[0]['index'])

    # Print comparison
    print("Keras model output:\n", output_keras)
    print("TFLite model output:\n", output_tflite)
    print("Absolute difference:\n", np.abs(output_keras - output_tflite))
    print("Are outputs close (tolerance=1e-5)?", np.allclose(output_keras, output_tflite, atol=1e-5))

    # Count unique operations
    ops = set(d['op_name'] for d in interpreter._get_ops_details())
    return len(ops)

def bin_to_c_header(bin_path: str, header_path: str, array_name: str = "model_tflite", num_ops: int = None):
    with open(bin_path, "rb") as f:
        data = f.read()

    guard_name = os.path.basename(header_path).upper().replace(".", "_")

    with open(header_path, "w") as f:
        f.write(f"#ifndef {guard_name}\n")
        f.write(f"#define {guard_name}\n\n")

        f.write(f"unsigned char {array_name}[] = {{\n")
        for i in range(0, len(data), 12):
            line = data[i:i+12]
            hex_bytes = ", ".join(f"0x{b:02x}" for b in line)
            f.write(f"  {hex_bytes},\n")
        f.write("};\n")
        f.write(f"unsigned int {array_name}_len = {len(data)};\n\n")

        if num_ops is not None:
            f.write(f"#define TF_NUM_OPS {num_ops}\n\n")

        f.write(f"#endif // {guard_name}\n")

if __name__ == "__main__":
    keras_model_path = "../results2/nas/best_model_recovered.h5"
    tflite_model_path = "model.tflite"
    c_header_path = "model.h"

    num_ops = convert_and_test_model(keras_model_path, tflite_model_path)
    bin_to_c_header(tflite_model_path, c_header_path, num_ops=num_ops)
