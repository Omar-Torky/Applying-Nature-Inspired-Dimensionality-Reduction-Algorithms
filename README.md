# 🌿 Nature-Inspired Dimensionality Reduction Algorithms

This project explores and compares a variety of **dimensionality reduction techniques**, combining **traditional statistical methods** with **nature-inspired metaheuristic algorithms**. These methods are used to reduce high-dimensional datasets into lower dimensions while preserving their essential structures and patterns.

---

## 🧠 Overview

Dimensionality reduction is a crucial step in preprocessing high-dimensional data. It helps mitigate issues such as the **curse of dimensionality**, overfitting, and computational inefficiency. In this project, we apply several **nature-inspired optimization algorithms** for **feature selection** and compare them with classical reduction techniques like PCA, t-SNE, and Self-Organizing Maps.

---

## 🎯 Objectives

- Apply bio-inspired algorithms for **feature selection** and dimensionality reduction.
- Evaluate and compare the performance of these methods on benchmark datasets.
- Visualize the results and analyze performance metrics like accuracy, computational cost, and cluster preservation.

---

## 🧬 Algorithms

### ✅ Traditional Dimensionality Reduction Methods

- **Principal Component Analysis (PCA)**  
  Linear method that projects data into lower dimensions by maximizing variance.  
  📖 [Jolliffe, 2002](https://link.springer.com/book/10.1007/b98835)

- **t-distributed Stochastic Neighbor Embedding (t-SNE)**  
  Nonlinear method particularly useful for 2D/3D visualization of high-dimensional data.  
  📖 [van der Maaten & Hinton, 2008](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)

- **Self-Organizing Maps (SOM)**  
  Neural-network-based method that preserves the topological properties of the data.  
  📖 [Kohonen, 2001](https://link.springer.com/book/10.1007/978-3-642-56927-2)

---

### 🐜 Nature-Inspired Optimization Algorithms for Feature Selection

- **Ant Colony Optimization (ACO)**  
  Uses pheromone trails to guide a population of artificial ants in searching optimal feature subsets.  
  📖 [Dorigo & Stützle, 2004](https://link.springer.com/book/10.1007/978-3-662-05615-1)

- **Particle Swarm Optimization (PSO)**  
  Simulates social behavior of birds/fish for optimizing search in feature space.  
  📖 [Kennedy & Eberhart, 1995](https://ieeexplore.ieee.org/document/488968)

- **Bat Algorithm**  
  Inspired by bat echolocation to balance local and global search.  
  📖 [Yang, 2010](https://link.springer.com/chapter/10.1007/978-3-642-12538-6_6)

- **Artificial Bee Colony (ABC)**  
  Mimics foraging behavior of honey bees to select optimal feature subsets.  
  📖 [Karaboga, 2005](https://www.researchgate.net/publication/228622735)

- **Firefly Algorithm (FA)**  
  Based on attraction among fireflies based on brightness and distance.  
  📖 [Yang, 2009](https://www.researchgate.net/publication/220902524)

---

## 📊 Datasets

The **Wine dataset** consists of 13 chemical features describing different cultivars of wine. High-dimensional data can lead to increased computational cost and loss of interpretability. This project applies both classical and **bio-inspired feature selection** methods to identify the most informative features.

- **Wine Dataset (UCI Repository)**  
  [https://archive.ics.uci.edu/ml/datasets/wine](https://archive.ics.uci.edu/ml/datasets/wine)
  
**Features:** 13  
**Classes:** 3 wine cultivars  
**Samples:** 178
---

## 🧪 Evaluation Metrics

- **Runtime / computational efficiency**
- **Silhouette Score** for cluster quality
- **Trustworthiness Score** – Measures how well local neighborhoods are preserved after dimensionality reduction.  
- **Visualization** for 2D/3D representation and structure preservation

---

## 💡 Expected Outcomes

- Comparative analysis of traditional vs. nature-inspired dimensionality reduction.
- Visual insights into high-dimensional data structure.
- Practical guidelines on selecting the most suitable reduction method for a given dataset.


