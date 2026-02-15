"""
Export the trained Keras model to TensorFlow.js format manually.
This avoids the buggy tensorflowjs pip package by writing the model.json
and weight .bin files directly.
"""

import os
import sys
import io
import json
import struct
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'saved_model', 'mnist_cnn.keras')
tfjs_dir = os.path.join(BASE_DIR, 'tfjs_model')
os.makedirs(tfjs_dir, exist_ok=True)

print(f"Loading model from: {model_path}")
model = tf.keras.models.load_model(model_path)

# Verify accuracy
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test.astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)
y_test = tf.keras.utils.to_categorical(y_test, 10)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# ──── Build TF.js compatible model topology ────
# Map Keras layer configs to TF.js format

def get_dtype_str(dtype):
    dtype_str = str(dtype)
    if 'float32' in dtype_str:
        return 'float32'
    elif 'int32' in dtype_str:
        return 'int32'
    return 'float32'

# Collect all weights
weight_specs = []
weight_data = bytearray()

for layer in model.layers:
    weights = layer.get_weights()
    for i, w in enumerate(weights):
        w = w.astype(np.float32)
        weight_name = layer.weights[i].name
        
        weight_specs.append({
            'name': weight_name,
            'shape': list(w.shape),
            'dtype': 'float32'
        })
        weight_data.extend(w.tobytes())

# Save weights binary
weights_path = os.path.join(tfjs_dir, 'group1-shard1of1.bin')
with open(weights_path, 'wb') as f:
    f.write(weight_data)

print(f"Weights file: {len(weight_data)} bytes")

# ──── Build model.json ────
# Get the Keras config
keras_config = model.get_config()

model_topology = {
    'class_name': keras_config['class_name'] if 'class_name' in keras_config else 'Sequential',
    'config': keras_config,
    'keras_version': tf.keras.__version__,
    'backend': 'tensorflow'
}

model_json = {
    'format': 'layers-model',
    'generatedBy': 'manual-converter',
    'convertedBy': None,
    'modelTopology': model_topology,
    'weightsManifest': [{
        'paths': ['group1-shard1of1.bin'],
        'weights': weight_specs
    }]
}

json_path = os.path.join(tfjs_dir, 'model.json')
with open(json_path, 'w') as f:
    json.dump(model_json, f)

print(f"model.json saved to: {json_path}")
print("TF.js export complete!")
