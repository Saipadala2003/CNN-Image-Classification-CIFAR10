# Results Summary

## Training and validation
The CNN was trained for 20 epochs with a batch size of 64. Training accuracy increased from 37.88% in epoch 1 to 86.92% in epoch 20. Validation accuracy reached approximately 76.48% at its best point and finished at 75.58%.

## Test evaluation
- Test loss: **0.8892**
- Test accuracy: **75.01%**

## Interpretation
The model learned useful visual representations from CIFAR-10. Automobile, ship, truck, frog, and horse classes were comparatively strong, while cat, dog, and bird were more difficult. The gap between training and validation performance also indicates some overfitting, suggesting that augmentation, batch normalization, regularization, learning-rate scheduling, or a deeper architecture could be explored in future work.

## Required visual outputs
The project notebook contains the accuracy/loss curves, confusion matrix, classification report, and actual-versus-predicted image examples used for evaluation.
