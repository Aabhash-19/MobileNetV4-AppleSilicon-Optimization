# MobileNetV4 Implementation & Optimization
This repository contains PyTorch implementations of MobileNetV4 architecture with comprehensive performance optimization techniques, trained on the **CIFAR-10** dataset.

### CIFAR-10 Dataset Overview
The CIFAR-10 dataset is a widely used benchmark in computer vision and deep learning research, consisting of 60,000 32×32 pixel color images distributed across 10 distinct classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The dataset is perfectly balanced with 6,000 images per category, split into 50,000 training images and 10,000 test images. Despite its small image size, CIFAR-10 presents a challenging multi-class classification problem due to the diversity of objects, varying backgrounds, and different orientations within each category. Its manageable size makes it ideal for rapid prototyping and testing of neural network architectures while still maintaining sufficient complexity to evaluate model performance effectively. The dataset's popularity stems from its well-curated nature, balanced distribution, and the fact that it represents real-world object recognition tasks at a scale suitable for both educational purposes and research experimentation.

---

## 📊 Performance Summary

| Model | Accuracy | Training Time (M4 MacBook Air) | Parameters | Improvement |
|-------|----------|---------------------------------|------------|-------------|
| Baseline | 70.09% | ~2.5 hours (CPU) | ~3.1M | - |
| Enhanced | 83.28% | ~45 minutes (GPU) | ~4.2M | +13.19% |

**Improvement:** +13.19% accuracy with 3× faster training

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run baseline model
python mobilenetv4_baseline.py

# Run enhanced model  
python mobilenetv4_enhanced.py
```

---

## 🏗️ Architecture Improvements

### 1. Squeeze-and-Excitation (SE) Blocks

**Location:** Added after depthwise convolution in inverted residual blocks

```bash
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
```
**Impact:** +3-4% accuracy by enabling feature recalibration and dynamic channel weighting, allowing the model to focus on important features

### 2. Stochastic Depth Regularization

**Implementation:** Random layer dropping during training (10% probability)

```bash
class StochasticDepth(nn.Module):
    def __init__(self, drop_prob=0.1):
        super(StochasticDepth, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        random_tensor = keep_prob + torch.rand()
        return x / keep_prob * random_tensor
```
**Impact:** +2-3% accuracy through better generalization and reduced overfitting by creating an implicit ensemble of sub-networks

### 3. SiLU Activation Function

 **Replacement:** Switched from ReLU6 to SiLU (Swish) activation throughout the network

 ```bash
# Before: nn.ReLU6(inplace=True)
# After: nn.SiLU()  # Swish activation
```
**Impact:** +1-2% accuracy with smoother gradients, better gradient flow, and improved performance in deep networks
