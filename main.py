import tkinter as tk
from tkinter import Canvas
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# Suppress warnings (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MODEL_PATH = "mnist_model_v2_10.h5"

# Load or train the model
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5)
    model.save(MODEL_PATH)

class DigitRecognizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")

        self.canvas = Canvas(self.root, width=280, height=280, bg='white')
        self.canvas.pack()

        self.button_predict = tk.Button(self.root, text="Predict", command=self.predict_digit)
        self.button_predict.pack()

        self.label_result = tk.Label(self.root, text="", font=("Helvetica", 18))
        self.label_result.pack()

        self.button_clear = tk.Button(self.root, text="Clear", command=self.clear_canvas)
        self.button_clear.pack()

        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def draw_lines(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='black')
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill='black')

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.label_result.config(text="")

    def predict_digit(self):
        img_resized = self.image.resize((28, 28))
        img_array = np.array(img_resized)
        img_array = 255 - img_array
        img_array = img_array / 255.0
        img_array = img_array.reshape(1, 28, 28)

        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)
        self.label_result.config(text=f"Predicted Digit: {predicted_digit}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizer(root)
    root.mainloop()
