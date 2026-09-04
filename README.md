# CNN Image Classification using CIFAR-10

**L&T Edutech – Task 1: Building an Image Classification Model using CNN**

This public repository contains the complete academic implementation of a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset.

## Student
- **Name:** Saikumar Padala
- **Course:** MSc AI Part II
- **Roll No.:** 15
- **PRN / Student ID:** 5711387

## Objective
Develop a CNN for image classification and understand the complete workflow of loading, preprocessing, training, validating, evaluating, visualizing, and saving a deep learning model.

## Dataset
The CIFAR-10 dataset is loaded using `tensorflow.keras.datasets.cifar10`.

- 50,000 training images
- 10,000 test images
- Image size: 32 × 32 × 3
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- Training split: 45,000 images
- Validation split: 5,000 images
- Test set: 10,000 images

## CNN Architecture
- Input: 32 × 32 × 3
- Conv2D: 32 filters, 3 × 3, ReLU, same padding
- MaxPooling2D: 2 × 2
- Conv2D: 64 filters, 3 × 3, ReLU, same padding
- MaxPooling2D: 2 × 2
- Conv2D: 128 filters, 3 × 3, ReLU, same padding
- MaxPooling2D: 2 × 2
- Flatten
- Dense: 128 neurons, ReLU
- Dropout: 0.5
- Output: 10 neurons, Softmax

## Training Configuration
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Metric: Accuracy
- Epochs: 20
- Batch size: 64
- Pixel normalization: division by 255.0

## Recorded Results
| Metric | Result |
|---|---:|
| Test Loss | 0.8892 |
| Test Accuracy | **75.01%** |
| Final Training Accuracy | 86.92% |
| Final Validation Accuracy | 75.58% |

The saved/reloaded model produced the same recorded test accuracy of **75.01%**.

## Class-wise Performance
| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Airplane | 0.79 | 0.77 | 0.78 |
| Automobile | 0.85 | 0.89 | 0.87 |
| Bird | 0.62 | 0.68 | 0.65 |
| Cat | 0.57 | 0.55 | 0.56 |
| Deer | 0.64 | 0.79 | 0.71 |
| Dog | 0.74 | 0.57 | 0.64 |
| Frog | 0.77 | 0.83 | 0.80 |
| Horse | 0.82 | 0.77 | 0.80 |
| Ship | 0.89 | 0.83 | 0.86 |
| Truck | 0.84 | 0.81 | 0.83 |

## Project Files
- `Image_Classification_Model_using_CNN.py` – complete Python implementation extracted from the Jupyter Notebook.
- `CNN_Image_Classification_CIFAR10_Report.pdf` – academic report with methodology, outputs, evaluation and results.
- `requirements.txt` – required Python packages.
- `screenshots/` – selected notebook output screenshots.

## Google Colab
Open the executable notebook in Google Colab:

https://colab.research.google.com/drive/1xtb_VKjqR4l19i0ss65oA9Hle9d20zoa?usp=sharing

## Workflow
1. Load CIFAR-10.
2. Inspect dataset shapes and sample images.
3. Normalize pixel values.
4. Split training data into training and validation sets.
5. Build the CNN.
6. Compile with Adam and sparse categorical crossentropy.
7. Train for 20 epochs.
8. Plot training/validation accuracy and loss.
9. Evaluate on the test set.
10. Generate classification report and confusion matrix.
11. Visualize actual vs predicted images.
12. Save and reload the trained model.

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook
```
Run `Image_Classification_Model_using_CNN.py` in Jupyter or Python, or open the Google Colab notebook linked above.

## Conclusion
The CNN learned useful visual features from CIFAR-10 and achieved a recorded test accuracy of **75.01%**. The project demonstrates the complete CNN image-classification workflow required for the L&T Edutech Task 1 evaluation.

## Review
This repository is organized for trainer/student review and contains the implementation, academic report, results and visual evidence of the experiment.
