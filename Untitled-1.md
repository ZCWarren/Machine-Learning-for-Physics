# Notes for ML
## Introduction
Machine Learning can be divided into three main categories:
- supervised learning (labeled data)
- unsupervised learning (unlabeled data)
- reinforcement learning (rewards)

## Supervised learning
Spliting the data set into two disjoint part, 
- training set ($80\%$)
- test set ($20\%$)

learning from training set and predicting on the test set.

Data set: ${X_i, y_i}$, $i=1,2,...,n$, where $X_i$ is the set of attributes of the example $i$ and $y_i$ is the target variable we want to predict, which can be a single value or a vector.

When $y \in R$, it indicates a regression problem; if $y \in {C_1,C_2\dots, C_k}$, where $C_j$ is a class, it becomes a classification problem (e.g. Softmax: converts an unstandardized input vector into a normally distributed probability distribution).

ML algorithms aim to fit a model $f$ that maps the attributes to the target variable. Mathmatically, 
$$y = f(x,\theta) + \epsilon,$$
we can refer to $f(x, \theta)$ as the function that generates the data, and the goal of the ML algorithm is to estimate its set of parameters θ so that we can predict new observations as accurately as possible.

### Optimizaton
We use a **loss function** to evaluate how well specific algorithms model the given data. This function can have different forms and **depend** on the type of learning process. In general, the loss function $L$ compares the values of the target variable $y$ and the respective predictions $\hat y$. The average loss of the predictor on the training set is called the cost function, or empirical risk in decision theory, 
$$C(\hat y,y)=\frac{1}{n} \sum_{i=1}^{n}L(y_i,\hat y_i).$$
Example
- Regression: Mean squared error loss:$$MSE(y,\hat y)=\frac{1}{n} \sum_{i=1}^{n}(y_i-\hat y_i)^2$$
- Classification: The 0-1 loss function, and cross-entropy: $$H(y, \hat{y}) = -\sum_{i=1}^{N} y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)$$
Hinge loss:$$L(y, \hat{y}) = \max(0, 1 - y \cdot \hat{y})$$

Thus, the model training process aims to discover a configuration of model parameters denoted as θ, which serves to minimize the cost function when applied to the training dataset








