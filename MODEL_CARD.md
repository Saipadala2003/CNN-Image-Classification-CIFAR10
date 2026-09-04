# Model Card — CIFAR-10 CNN

## Model purpose
A CNN image-classification model developed for the L&T Edutech Task 1 academic exercise.

## Dataset
CIFAR-10, containing 10 image classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

## Input
32 × 32 RGB images, normalized to the range 0–1.

## Architecture
Three convolution + max-pooling blocks, followed by Flatten, Dense(128), Dropout(0.5), and a 10-unit Softmax output layer.

## Training configuration
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Epochs: 20
- Batch size: 64
- Training images: 45,000
- Validation images: 5,000
- Test images: 10,000

## Recorded evaluation
- Test loss: 0.8892
- Test accuracy: 75.01%
- Final training accuracy: 86.92%
- Final validation accuracy: 75.58%

## Notes
The model is intended for educational demonstration and is not a production-grade image recognition system. Performance varies by class; visually similar CIFAR-10 categories such as cats and dogs are more challenging for this compact CNN.
