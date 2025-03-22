# MNIST Live Digit Recognition with a Residual Neural Network

A deep learning project for handwritten digit recognition using a custom ResNet-based Convolutional Neural Network architecture trained on the MNIST dataset. This application features an interactive drawing interface that allows users to draw digits and receive real-time predictions.

## Model Architecture

The neural network architecture is implemented as a custom ResNet variant specifically designed for the MNIST digit recognition task. ResNet (Residual Network) architectures are known for their ability to train very deep networks by using skip connections that help mitigate the vanishing gradient problem.

Each residual block contains:
- Two convolutional layers with 3×3 kernels
- Batch normalization after each convolution
- A shortcut connection that either performs identity mapping or a 1×1 convolution when dimensions change
- ReLU activation functions

The shortcut connection allows gradients to flow directly through the network, enabling the training of deeper architectures without degradation.

### Network Architecture

The complete network architecture is as follows:

1. **Input Layer**: Accepts 28×28 grayscale images (1 channel)
2. **Initial Convolution**: Conv2D(1→32, 3×3, stride=1, padding=1) → BatchNorm → ReLU
3. **Downsampling**: MaxPool2D(2×2) reducing dimensions to 14×14
4. **Residual Layer 1**: 2 residual blocks (32→32 channels) at 14×14 resolution
5. **Residual Layer 2**: 2 residual blocks (32→64 channels) with stride=2, reducing dimensions to 7×7
6. **Fully Connected Layers**:
   - Flatten: 64×7×7 = 3136 features
   - FC1: 3136→128 → ReLU
   - Dropout: 50% probability
   - FC2: 128→10 (one per digit class)
7. **Output Layer**: LogSoftmax for class probabilities

The network progressively increases the number of feature maps while reducing spatial dimensions, following the common CNN design pattern. The architecture is compact (compared to modern deep CNNs) but powerful enough to achieve >99% accuracy on the MNIST dataset.

#### Feature Map Dimensions

The feature map dimensions throughout the network are:
- Input: 1×28×28
- After initial conv: 32×28×28
- After max pooling: 32×14×14
- After residual layer 1: 32×14×14
- After residual layer 2: 64×7×7
- After flattening: 3136
- After first FC layer: 128
- Output: 10

# Drawing Normalization Pipeline

A critical component of this system is the preprocessing pipeline that transforms user drawings into MNIST-compatible input. This process ensures that hand-drawn digits of various sizes and positions across the drawing area are normalized to match the characteristics of the MNIST training data.

### Drawing Capture

The drawing interface uses a Pygame-based canvas that:
1. Captures mouse movements to draw lines with a specified thickness
2. Maintains both a visual representation (for display) and a pixel array (for processing)
3. Stores the drawing as a binary image where drawn pixels have a value of 255 and the background is 0

### Preprocessing Steps

The `preprocess_image` function in `DigitRecognizer` implements a normalization pipeline:

1. **Bounding Box Detection**:
   ```python
   rows = np.any(pixels, axis=1)
   cols = np.any(pixels, axis=0)
   rmin, rmax = np.where(rows)[0][[0, -1]]
   cmin, cmax = np.where(cols)[0][[0, -1]]
   ```
   This code identifies the smallest rectangle containing the drawn digit by finding the first and last non-empty rows and columns.

2. **Padding**:
   ```python
   padding = 2
   rmin = max(0, rmin - padding)
   rmax = min(pixels.shape[0] - 1, rmax + padding)
   cmin = max(0, cmin - padding)
   cmax = min(pixels.shape[1] - 1, cmax + padding)
   ```
   A small padding is added around the bounding box to ensure no part of the digit is cut off.

3. **Extraction and Centering**:
   ```python
   digit = pixels[rmin:rmax+1, cmin:cmax+1]
   size = max(digit.shape[0], digit.shape[1])
   square_digit = np.zeros((size, size), dtype=np.uint8)
   y_offset = (size - digit.shape[0]) // 2
   x_offset = (size - digit.shape[1]) // 2
   square_digit[y_offset:y_offset+digit.shape[0], x_offset:x_offset+digit.shape[1]] = digit
   ```
   The digit is extracted and centered in a square canvas, preserving its aspect ratio. This is crucial for maintaining the shape characteristics of the digit regardless of how it was drawn.

4. **MNIST Format Transformation**:
   ```python
   self.transform = transforms.Compose([
       transforms.Resize((28, 28)),
       transforms.ToTensor(),
       transforms.Normalize((0.1307,), (0.3081,))
   ])
   ```
   The final transformation pipeline:
   - Resizes the image to 28×28 pixels (MNIST standard)
   - Converts to a PyTorch tensor
   - Normalizes using the same mean (0.1307) and standard deviation (0.3081) as the MNIST training data

This normalization process is critical for ensuring that user-drawn digits match the distribution of the training data, allowing the model to make accurate predictions regardless of how the user draws the digit (size, position, thickness, etc.).

## Training Methodology

The model is trained using a simple pipeline that includes:

1. **Data Preparation**:
   - Loading the MNIST dataset
   - Applying normalization with mean=0.1307 and std=0.3081
   - Splitting into training (90%) and validation (10%) sets

2. **Optimization**:
   - Loss Function: Cross-Entropy Loss
   - Optimizer: Adam with learning rate of 0.001
   - Training for 15 epochs with batch size of 64

3. **Model Selection**:
   - Monitoring validation accuracy
   - Saving the best-performing model based on validation accuracy
   - Final evaluation on the test set

4. **Performance Metrics**:
   - Training and validation loss/accuracy curves
   - Confusion matrix for error analysis
   - Target accuracy of 99%+ on the MNIST test set

## System Integration

The application integrates several components:

1. **Neural Network** (`nn.py`): Defines the model architecture
2. **Training Pipeline** (`train.py`): Handles model training and evaluation
3. **User Interface** (`ui.py`): Provides the drawing canvas and prediction display
4. **Drawing and Prediction** (`draw_prediction.py`): Connects the UI with the model

The prediction workflow is:
1. User draws a digit on the canvas
2. The drawing is captured as a pixel array
3. The preprocessing pipeline normalizes the drawing
4. The model makes a prediction
5. The UI displays the predicted digit and confidence scores

## Usage
If you've gotten this far, you should be able to figure out how to gather the necessary libraries and run this. Ensure you're in a `venv` ! xd
