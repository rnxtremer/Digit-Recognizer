# Handwritten Digit Recognizer - Full Project Explanation

This document explains **every single line and functional block** of the Handwritten Digit Recognizer project.

This project is broken down into two main files:
1. `app.py` - The user interface (UI) and image processing frontend built with Streamlit.
2. `src/neural_network.py` - The custom Artificial Neural Network backend built from scratch using NumPy.

---

## Part 1: `src/neural_network.py` 
This is the core "brain" of our project. It builds a Multi-Layer Perceptron (MLP) entirely from scratch using only matrix math operations, teaching the computer how to learn patterns from pixels without using frameworks like TensorFlow or PyTorch.

### Imports
```python
import numpy as np
import pickle
```
* **Line 1:** `import numpy as np` - Imports NumPy, a wildly fast math library for Python. We use it to perform matrix multiplications and calculus operations across thousands of numbers instantly.
* **Line 2:** `import pickle` - Imports Python's built-in object serialization library. This allows us to "save" our trained neural network (its learned weights and biases) to a file on your hard drive, so we don't have to retrain it from scratch every time we launch the app.

### The NeuralNetwork Class Initialization
```python
class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01, seed=42):
```
* **Line 5-7:** Defines our master `NeuralNetwork` blueprint. The constructor `__init__` takes 3 parameters: `layer_sizes` (a list telling us how many neurons are in each layer, e.g., `[784, 128, 64, 10]`), `learning_rate` (how large of a step the network takes when correcting its mistakes), and `seed` (locks randomness so results are consistent).

```python
        np.random.seed(seed)
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes)
```
* **Line 17-21:** We lock the random seed, and then save our layer architecture sizes, learning rate, and calculate exactly how many layers exist (in our case, 4 layers: 1 input, 2 hidden, 1 output).

### Initializing Weights and Biases (The Brain's Connections)
```python
        self.weights = []
        self.biases = []
        for i in range(self.num_layers - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            
            self.weights.append(w)
            self.biases.append(b)
```
* **Line 24-33:** Weights are the "strength" of connections between neurons. Biases act as thresholds. 
* We loop through each layer and create connection matrices between them. 
* We use **He Initialization** (`np.sqrt(2.0 / layer_sizes[i])`) when setting up `w` (weights) because it prevents the numbers from getting too astronomically massive or vanishingly small when they pass through ReLU activation functions.
* `b` (biases) always start at exactly zero.

### Tracking State
```python
        self.cache = {}
        self.history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}
```
* **Line 36-44:** `cache` remembers the mathematical results of every single neuron during a forward pass so we can use them later to calculate calculus derivatives (gradients) during backward propagation. `history` just tracks how well the model is learning over time so we could plot it.

### Activation Functions (Making it Non-Linear)
```python
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def relu_derivative(self, Z):
        return (Z > 0).astype(float)
```
* **Line 47-52:** The **ReLU** function (Rectified Linear Unit) is applied to hidden layers. It looks at a number; if it's less than 0, it turns it to 0. If it's greater than 0, it leaves it alone. This introduces "non-linearity", which allows the network to learn complex curves instead of just straight lines.
* The derivative of ReLU is just `1` if the number is positive, and `0` if negative. We need this for backpropagation calculus.

```python
    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
```
* **Line 55-59:** The **Softmax** function is placed exclusively on the very last output layer. If the network outputs raw scores for 10 digits (like: "I am 80 confident it's a zero, and 120 confident it's a one"), Softmax mathematically crushes those raw scores into **percentages that perfectly add up to 100%** (e.g., "0% chance it's zero, 99.9% chance it's a one"). 
* Subtracting `np.max` is a genius math trick to prevent your computer from overflowing and crashing when calculating large exponents.

### Forward Propagation (Guessing the Answer)
```python
    def forward_propagation(self, X):
        self.cache['A0'] = X
        
        for i in range(self.num_layers - 2):
            Z = np.dot(self.cache[f'A{i}'], self.weights[i]) + self.biases[i]
            A = self.relu(Z)
            self.cache[f'Z{i+1}'] = Z
            self.cache[f'A{i+1}'] = A
```
* **Line 62-71:** "Forward Prop" is the act of taking an image (`X`) and throwing it through the network.
* `np.dot` multiplies the current layer's pixel signals by their connection weights, and adds the bias. This creates raw output signals (`Z`).
* We then run `Z` through the `relu` activation function to get the activated outputs (`A`).
* We stash `Z` and `A` in the cache memory so we don't have to recalculate them later during learning.

```python
        Z_out = np.dot(self.cache[f'A{self.num_layers-2}'], self.weights[-1]) + self.biases[-1]
        A_out = self.softmax(Z_out)
        self.cache[f'Z{self.num_layers-1}'] = Z_out
        self.cache[f'A{self.num_layers-1}'] = A_out
        return A_out
```
* **Line 74-81:** After passing through the hidden layers, the final layer connects to our 10 output neurons (digits 0-9). We use `.softmax()` instead of ReLU here so our final output `A_out` is cleanly formatted as probabilities.

### Computing Loss (How Wrong Are We?)
```python
    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
        return loss
```
* **Line 84-88:** We use **Categorical Cross-Entropy Loss**. It takes the true answer (`y_true`), compares it to our guess (`y_pred`), and calculates a massive "penalty score" if we are highly confident in a wildly wrong answer. The `+1e-8` is injected purely so we never accidentally ask python to calculate `log(0)`, which throws a fatal crash.

### Backward Propagation (Learning from Mistakes via Calculus)
```python
    def backward_propagation(self, y_true):
        m = y_true.shape[0]
        dW = [None] * (self.num_layers - 1)
        db = [None] * (self.num_layers - 1)
        
        dZ = self.cache[f'A{self.num_layers-1}'] - y_true
```
* **Line 91-99:** "Backprop" is where the actual learning happens through the Chain Rule of calculus. Using the results of our forward guess, we calculate how far off we were at the very end (`A_out - y_true`), which gives us the error delta `dZ`.

```python
        for i in range(self.num_layers - 2, -1, -1):
            dW[i] = np.dot(self.cache[f'A{i}'].T, dZ) / m
            db[i] = np.sum(dZ, axis=0, keepdims=True) / m
            
            if i > 0:
                dA = np.dot(dZ, self.weights[i].T)
                dZ = dA * self.relu_derivative(self.cache[f'Z{i}'])
```
* **Line 102-110:** We step through the network **backwards**. At every layer, we use calculus derivatives to figure out exactly how much each specific weight matrix (`dW`) and bias matrix (`db`) is responsible for the final error in our guess. We pass what's left of the error backwards (`dA`) to the previous layer infinitely until we reach the start. 

```python
        for i in range(self.num_layers - 1):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]
```
* **Line 113-115:** **Gradient Descent.** Now that we know which direction makes the error worse or better, we slightly adjust every single weight and bias in the entire brain in the direction of "less error", multiplying the push by our `learning_rate`.

### Training and Utility Functions
```python
    def train_on_batch(self, X_batch, y_batch):
        y_pred = self.forward_propagation(X_batch)
        loss = self.compute_loss(y_batch, y_pred)
        self.backward_propagation(y_batch)
        return loss
```
* **Line 118-138:** Combines guessing (Forward Prop), grading the guess (Compute Loss), and learning from the mistake (Backward Prop) together into a single action for training loops.

```python
    def predict(self, X):
        y_pred_proba = self.forward_propagation(X)
        return np.argmax(y_pred_proba, axis=1)
```
* **Line 141-143:** When interacting with the frontend, we don't care about training anymore. We just push an image into the network (`forward_propagation`), get a bunch of percentages, and use `np.argmax` to pick the digit with the highest percentage confidence.

```python
    def save_model(self, filepath):
        model_data = {'weights': self.weights, 'biases': self.biases, 'layer_sizes': self.layer_sizes, 'learning_rate': self.learning_rate}
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
    def load_model(self, filepath): ...
```
* **Line 157-180:** Simply saves our matrices directly to the hard drive into a `.pkl` file (or loads them from the file), so our AI doesn't get amnesia when we reboot the server.

---

## Part 2: `app.py`
This is your frontend user interface. It draws the application, handles CSS styling trickery to fit the screen, processes the canvas sketch you draw, heavily modifies the image to look like the MNIST training data format, and passes it to the `NeuralNetwork` to get the answer.

### Imports & Layout Initialization
```python
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import pickle
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.neural_network import NeuralNetwork

st.set_page_config(page_title="Digit Recognizer Dashboard", page_icon="🔢", layout="wide")
```
* **Lines 1-7:** Core imports. `streamlit` draws the frontend. `st_canvas` is the sketchpad component. `cv2` (OpenCV) handles powerful image processing algorithms.
* **Line 9-10:** Adds the project folder to python's system path so it won't crash when trying to import our custom `NeuralNetwork` module from the `src` folder.
* **Line 12:** Configures the initial page settings like the browser tab name and icon, and sets the base structural layout logic to `wide`.

### Loading the AI Brain into Memory
```python
@st.cache_resource
def load_model():
    try:
        with open("models/digit_recognizer.pkl", 'rb') as f:
            model_data = pickle.load(f)
        layer_sizes = model_data.get('layer_sizes', [784, 128, 64, 10])
        lr = model_data.get('learning_rate', 0.01)
        nn = NeuralNetwork(layer_sizes, learning_rate=lr)
        nn.weights = model_data['weights']
        nn.biases = model_data['biases']
        return nn
    except Exception as e:
        return None

nn = load_model()
```
* **Line 14-30:** The `@st.cache_resource` decorator tells Streamlit to only run this massive file-loading task ONE time when the app starts, instead of re-loading it every single time you click a button.
* We open the saved `.pkl` matrix file containing our pre-trained brain, rebuild the brain's 784-128-64-10 architecture, inject the learned weights into it, and keep it active in a variable called `nn` continuously.

### Real-Time Computer Vision Pipeline (The `predict_image` function)
The brain expects to be fed extremely crisp, perfectly centered, 28x28 pixel images exactly like MNIST. However, your canvas sketch is 800x400 and full of noise. This incredibly complex pipeline isolates, slices, crops, pads, and mathematically aligns your sketch to trick the AI into processing it flawlessly.

```python
def predict_image(img_array, is_light_theme):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
    if is_light_theme:
        gray = cv2.bitwise_not(gray)
```
* **Line 31-38:** Takes your raw drawing. Replaces RGBA color data with Grayscale to eliminate color complexity. The AI is trained on white numbers existing on a black background (Dark Mode inherently), so if you drew on a bright light background, we use `cv2.bitwise_not` to mathematically invert the pixels from black-on-white to white-on-black.

```python
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, -5)
    kernel_dilate = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel_dilate, iterations=1)
```
* **Line 43-47:** Smooths jagged pixel edges (`GaussianBlur`), applies dramatic 100% black/white contrast snapping to eliminate light gray anti-aliasing artifacts (`adaptiveThreshold`), and aggressively thickens the white lines so they are easier for the algorithm to trace (`cv2.dilate`).

```python
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```
* **Line 49:** Uses OpenCV's phenomenal "Blob Detection" math algorithm to find distinct, separate islands of white pixels (like drawing a '1' next to a '2'). This allows us to read multiple numbers at the exact same time without the AI getting confused!

```python
    valid_contours = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        
        # Omissions (lines 58-65 exclude tiny dots, extreme stretched anomalies, edge clips)
        valid_contours.append((x, y, w, h, c))
```
* **Line 51-68:** We draw invisible bounding boxes around every shape it found. Then we filter out the garbage. If a box is microscopic (like a speck of dust), or touches the exact edge of the canvas, or is stretched ridiculously wide, we delete it from memory. The rest are valid drawings.

```python
    valid_contours = sorted(valid_contours, key=lambda b: b[0])
    output_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
```
* **Line 69:** We forcefully sort our box coordinates by their X-axis location. This simple hack ensures that if you draw "1 4 2", it actually reads them left-to-right structurally instead of reading whoever happens to be vertically higher first.

### Center of Mass Physics and Bounding Boxes
For every valid digit drawn on the canvas left-to-right:

```python
    for (x, y, w, h, c) in valid_contours:
        pad = max(5, int(0.1 * h))
        roi = gray[y1:y2, x1:x2]
        
        # Cropping and padding logic lines 82-89 ensures the crop is a perfect square box so resizing doesn't warp/squish the number unrecognizably.
        squared_roi = np.pad(roi, ... ) 
        resized_20 = cv2.resize(squared_roi, (20, 20), interpolation=cv2.INTER_AREA)
        roi_resized = np.pad(resized_20, ((4, 4), (4, 4)), mode='constant', constant_values=0)
```
* **Lines 73-89:** We slice open only the chunk of picture containing the digit. We math out how to pad the surrounding edges with black pixels to create a perfect square, scale it down cleanly to a TINY 20x20 box, and forcefully pad another 4 pixel black wall around it, making it identically `28x28`.

```python
        M = cv2.moments(roi_resized)
        if M["m00"] != 0:
            cX = M["m10"] / M["m00"]
            cY = M["m01"] / M["m00"]
            shift_x = 13.5 - cX
            shift_y = 13.5 - cY
            translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            roi_resized = cv2.warpAffine(roi_resized, translation_matrix, (28, 28))
```
* **Lines 91-98:** This is "Center of Mass" logic. MNIST wasn't trained on numbers floating arbitrarily in a 28x28 grid. MNIST physically clustered the mass of the white pixels and locked their centroid to exactly pixel location (13.5, 13.5) dead center. We calculate the geometric mass of your drawing, calculate the shift required, and warp the image translation matrix mathematically to force your drawing strictly into the absolute dead center of the square. Without this, your accuracy would crumble to 15%.

```python
        roi_norm = roi_resized.astype(np.float32) / 255.0
        roi_flat = roi_norm.reshape(1, 784)
        pred_proba = nn.forward_propagation(roi_flat)
        digit = np.argmax(pred_proba, axis=1)[0]
        predictions.append(str(digit))
```
* **Lines 100-105:** We divide pixel values by 255 to squash them between `0.0` and `1.0`. We flatten the 28x28 grid into a flat list of 784 numbers (`.reshape`). We throw it to our Neural Network we built back in part 1. The network guesses the number and we extract its highest guess, saving it to our text sequence string.

```python
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(output_img, str(digit), (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
```
* **Lines 107-108:** For fun, we draw those shiny green bounding boxes and floating red text predictions directly onto the visual output image using openCV shapes, so the user can literally "see" how the AI chopped up their canvas. 

### Frontend UI Layout
```python
st.markdown("""
<style>
/* ... Highly Aggressive CSS Constraints removed for brevity ... */
</style>
""", unsafe_allow_html=True)
```
* **Line 120:** An injection block where we write raw HTML/CSS web design code directly onto the Python window. This code forces the Streamlit structural wrappers to shrink, perfectly aligning the Canvas size with the Output Image size (forcing both to 800x400) and enables mobile responsive stacking using a max-width Media Query.

```python
theme = st.radio("Theme Mode", ["Dark", "Light"], horizontal=True, label_visibility="collapsed")
```
* **Lines 163-180:** Establishes UI elements like title headers, and a radio toggle button for themes. If Dark is chosen, the interface paints the canvas background black and the pen white.

```python
if 'canvas_key' not in st.session_state:
    st.session_state['canvas_key'] = 0
if st.button("🗑️ Clear Canvas", use_container_width=True):
    st.session_state['canvas_key'] += 1
```
* **Lines 182-190:** `session_state` creates variables that don't reset when the screen refreshes. Because the Canvas doesn't naturally let you wipe it clean programmatically without complex javascript, we cheat by assigning an ID key attached to the session state number. Clicking "Clear Canvas" increases the ID by 1, which tricks Streamlit into thinking an entirely new clean canvas just spawned, instantly wiping it clean!

```python
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=400,
    width=800,
    drawing_mode="freedraw",
    key=f"canvas_{st.session_state['canvas_key']}",
)
```
* **Lines 192-201:** The actual call that mounts the third-party drawing package to the interface, feeding it all our dynamic colors, sizes, and the hacking cache-breaking keys.

```python
if st.button("Predict Sequence", type="primary"):
    if canvas_result.image_data is not None:
        sequence_str, annotated_img = predict_image(canvas_result.image_data, is_light)
        
        st.success(sequence_str)
        annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, width=800)
```
* **Lines 203-214:** The very last block. Creates the large red primary button. If clicked, it sends whatever you touched on the canvas into our `predict_image` pipeline. When the pipeline finishes magically predicting all numbers, it prints the success sequence text on screen, converts the debug image back to standard browser colors (RGB instead of openCV BGR), and explicitly stamps it to the screen wrapped at 800 Width so it flawlessly aligns with the drawing board.
