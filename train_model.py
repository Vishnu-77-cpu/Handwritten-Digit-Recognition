"""
Save the trained model and convert to TensorFlow.js format.
Separate script to avoid re-training.
"""

import os
import sys
import io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import tensorflow as tf
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("Re-training model (quick) and saving...")

# ──── Load MNIST ────
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# ──── Build Model (same architecture, NO data augmentation for export) ────
# Data augmentation layers cause issues in TF.js conversion,
# so we build a clean inference model

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu',
                           kernel_initializer='he_normal', input_shape=(28, 28, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu',
                           kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu',
                           kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu',
                           kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu',
                           kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ──── Train with callbacks ────
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy', factor=0.5, patience=3, min_lr=1e-6, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1
    )
]

print("Training...")
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=20,
    validation_data=(x_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# ──── Evaluate ────
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# ──── Save Keras model ────
model_dir = os.path.join(BASE_DIR, 'saved_model')
os.makedirs(model_dir, exist_ok=True)
save_path = os.path.join(model_dir, 'mnist_cnn.keras')
model.save(save_path)
print(f"Keras model saved to: {save_path}")

# ──── Convert to TensorFlow.js ────
tfjs_dir = os.path.join(BASE_DIR, 'tfjs_model')
os.makedirs(tfjs_dir, exist_ok=True)

import subprocess
result = subprocess.run([
    sys.executable, '-m', 'tensorflowjs.converters.converter',
    '--input_format=keras',
    save_path,
    tfjs_dir
], capture_output=True, text=True)

if result.returncode == 0:
    print(f"TensorFlow.js model saved to: {tfjs_dir}")
    print("Done! Model is ready for the web app.")
else:
    print(f"TF.js conversion error: {result.stderr}")
    # Try alternative conversion method
    print("Trying alternative conversion via Python API...")
    import tensorflowjs as tfjs
    tfjs.converters.save_keras_model(model, tfjs_dir)
    print(f"TensorFlow.js model saved to: {tfjs_dir}")
    print("Done! Model is ready for the web app.")
