# Deep Learning and Computer Vision — 2026

## About the Course

This repository contains the practice/lab notebooks for **Deep Learning and Computer Vision**, a course taught by **Sergey Nikolenko** at **Harbour.Space University** (HS2026 term, June 2026).

Course page: https://www.sergeynikolenko.ru/en/courses/harbourbcn26

The course consists of 9 lectures covering machine learning and deep learning from probabilistic foundations through modern generative models. This repository holds the **hands-on PyTorch practice notebooks** that accompany a subset of those lectures.

## Course Topics

- Probabilistic/Bayesian foundations of machine learning
- Linear and logistic regression, biological motivation for neural networks
- Computational graphs, backpropagation, gradient descent and its variants
- Convolutional neural networks and architectures (VGG, ResNet)
- Recurrent neural networks
- LSTMs, attention, and Transformers
- Vision Transformers and object detection (R-CNN → DETR)
- Generative Adversarial Networks (GANs, Wasserstein GANs, adversarial autoencoders)
- Variational Autoencoders and diffusion models

## Lectures & Labs

### Lecture 1 — Introduction to Machine Learning
- Topics: ML foundations, history of AI, probabilistic principles, Bayes' rule, prior distributions.

### Lecture 2 — Linear Models and Introduction to Neural Networks
- Topics: Bayesian inference, linear/logistic regression, biological foundations of artificial neural networks.

### Lecture 3 — Backpropagation and Gradient Descent Algorithms
- Topics: computational graphs, backpropagation mechanics, SGD and its variants.
- Practice: [`HS2026. Practice 1.ipynb`](./HS2026.%20Practice%201.ipynb) — PyTorch fundamentals: tensors and tensor operations, building computational graphs, implementing custom differentiable functions via `torch.autograd.Function` (`forward`/`backward`), `nn.Module` basics (parameters, `train`/`eval`, initialization), manual gradient-descent steps, and a first MNIST training loop with `nn.Sequential` + SGD. Graded tasks: a custom `Power` autograd function, root-finding via binary search vs. Newton's method (using autograd for the derivative), MNIST experiments (misclassified examples, confusion matrix, weight-initialization and layer-width ablations), and a custom scaled linear layer (`output = 100 * (x @ W.T + b)`).
- Practice: [`HS2026. Practice 2.ipynb`](./HS2026.%20Practice%202.ipynb) — Practical optimization and debugging: choosing a learning rate by hand and with an LR range test (Smith's cyclical-LR method), learning-rate/batch-size interaction, LR schedulers, SGD vs. Adam, early stopping, augmentation with `torchvision.transforms.v2` (and a note on `albumentations`), gradient inspection/clipping, and one-batch overfitting as a sanity check. Graded tasks: learning-rate search for SGD, learning rate vs. batch size sweep, a custom `TwoOf` augmentation that randomly applies two transforms from a list, and a probability problem on overfitting to a small test set via repeated hyperparameter search.

### Lecture 4 — Convolutional Neural Networks
- Topics: CNN concepts, formalization, pooling operations.
- Practice: see [`HS2026_Practice_3.ipynb`](./HS2026_Practice_3.ipynb) below.

### Lecture 5 — Convolutional Architectures and Recurrent Neural Networks
- Topics: CNN variants, recurrent network designs.
- Practice: [`HS2026_Practice_3.ipynb`](./HS2026_Practice_3.ipynb) — Convolutional architectures and adversarial robustness on CIFAR-10. Covers `nn.Conv2d` basics, loading `torchvision` models, transfer learning with pretrained VGG16 (replacing/adapting the classifier head, freezing/unfreezing the feature extractor), a compact ResNet (`fbresnet20`) with factorized 3×3 convolutions, batch normalization, and skip connections, and `albumentations`-based augmentation. The second half covers adversarial attacks with **Foolbox** — PGD (projected gradient descent, white-box), DeepFool, and the black-box Boundary attack — and defenses: adversarial training and test-time augmentation. Graded tasks: training and comparing three VGG16 variants (plain / augmented / ImageNet-fine-tuned) on CIFAR-10, measuring adversarial-example transferability across models, adversarial training of the ResNet, and evaluating test-time augmentation as a defense — all measured by the smallest PGD budget (ε) that breaks the model, not just accuracy on a fixed attack.

### Lecture 6 — LSTM, Attention Mechanisms, and Transformers
- Topics: sequential models, Transformer architecture fundamentals.
- Practice: to be added to the repo.

### Lecture 7 — Vision Transformers and Object Detection
- Topics: detection evolution from R-CNN to DETR, transformer-based vision approaches.
- Practice: to be added to the repo.

### Lecture 8 — Generative Models and Adversarial Networks
- Topics: GANs, Wasserstein GANs, adversarial autoencoders.
- Practice: to be added to the repo.

### Lecture 9 — VAE and Diffusion Models
- Topics: variational autoencoders, diffusion-based generative models.
- Practice: to be added to the repo.

## Repository Structure

```text
.
├── HS2026. Practice 1.ipynb   # PyTorch basics: tensors, autograd, modules, optimizers, first training loop
├── HS2026. Practice 2.ipynb   # Learning rate, schedulers, augmentation, optimizer choice, debugging
├── HS2026_Practice_3.ipynb    # CNNs (VGG16, ResNet) on CIFAR-10, adversarial attacks & defenses
└── README.md
```

Each notebook is self-contained and designed to run on **Google Colab** (Practice 3 uses a T4 GPU and mounts Google Drive for checkpoint persistence).
