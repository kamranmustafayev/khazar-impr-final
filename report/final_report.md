# Final Report

## Title
**Developing and Tuning a Deep Learning Pipeline for Real-World Image Classification**

## Abstract
This project explores the development and optimization of a deep learning pipeline for image classification using the CIFAR-10 dataset. We implement a Convolutional Neural Network (CNN) from scratch, apply regularization techniques, and tune hyperparameters to improve performance. The pipeline also integrates GitHub for version control. Results are analyzed through various evaluation metrics and visualized using training loss curves.

## Objective
To design a scalable and efficient deep learning pipeline capable of accurately classifying real-world image data. The project emphasizes the full machine learning workflow, including data preprocessing, model development, training, tuning, evaluation, and version control integration.

## Methodology
### Dataset
- **Source**: CIFAR-10 dataset from Kaggle via `torchvision.datasets`
- **Preprocessing**:
  - Normalization
  - Train/validation/test split
  - Data augmentation: random horizontal flip, random crop

### Model
- **Architecture**:
  - Two convolutional layers with ReLU activation and max pooling
  - One fully connected hidden layer
  - Output layer with 10 units (for 10 classes)
- **Regularization**:
  - Dropout (0.25)
  - L2 regularization (weight decay in optimizer)
  - Early stopping considered but not implemented due to project scope

### Training
- **Optimizer**: Adam with learning rate 0.001
- **Loss Function**: CrossEntropyLoss
- **Epochs**: 10
- **Batch Size**: 64

### Hyperparameter Tuning
- Manual tuning performed
- Grid search and random search are left as future improvements

## Results
- **Test Accuracy**: Achieved approximately 70–75% after 10 epochs
- **Loss Visualization**: Training loss decreased consistently across epochs

### Plots
![Training Loss](../results/training_loss.png)

## Challenges and Solutions
- **Challenge**: Overfitting on the training set
  - **Solution**: Implemented dropout and L2 regularization
- **Challenge**: Slow training
  - **Solution**: Used batch size of 64 for efficiency
- **Challenge**: Small image size made classification harder
  - **Solution**: Applied augmentation and experimented with deeper architectures

## Limitations
- Limited number of epochs due to time constraints
- No automated hyperparameter tuning used
- Transfer learning not applied (could be explored for smaller datasets)

## Conclusion
This project successfully developed and tuned a deep learning pipeline for image classification. The system demonstrated strong performance on CIFAR-10, and future improvements could include transfer learning, advanced tuning methods, and deployment options.

## References
1. Krizhevsky, A. (2009). Learning multiple layers of features from tiny images. https://www.cs.toronto.edu/~kriz/cifar.html
2. PyTorch Documentation: https://pytorch.org/docs/
3. CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
4. Deep Learning Book - Ian Goodfellow

---
*Note: This report will be submitted along with the codebase on GitHub and Microsoft Teams.*

