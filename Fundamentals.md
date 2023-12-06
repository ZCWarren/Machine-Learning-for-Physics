#  Notes for ML
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

Data set: {${X_i, y_i}$}, $i=1,2,...,n$, where $X_i$ is the set of attributes of the example $i$ and $y_i$ is the target variable we want to predict, which can be a single value or a vector.

When $y \in R$, it indicates a regression problem; if $y \in {C_1,C_2\dots, C_k}$, where $C_j$ is a class, it becomes a classification problem (e.g. Softmax: converts an unstandardized input vector into a normally distributed probability distribution).

ML algorithms aim to fit a model $f$ that maps the attributes to the target variable. Mathmatically, 
$$y = f(x,\theta) + \epsilon,$$
where $\epsilon$ is the random error. We can refer to $f(x, \theta)$ as the function that generates the data, and the goal of the ML algorithm is to estimate its set of parameters θ so that we can predict new observations as accurately as possible.

### Optimizaton
We use a *loss function* to evaluate how well specific algorithms model the given data. This function can have different forms and **depend** on the type of learning process. In general, the loss function $L$ compares the values of the target variable $y$ and the respective predictions $\hat y$. The average loss of the predictor on the training set is called the cost function, or empirical risk in decision theory, 
$$C(\hat y,y)=\frac{1}{n} \sum_{i=1}^{n}L(y_i,\hat y_i).$$
Example
- Regression: Mean squared error loss:$$MSE(y,\hat y)=\frac{1}{n} \sum_{i=1}^{n}(y_i-\hat y_i)^2$$
- Classification: The 0-1 loss function, and cross-entropy: $$H(y, \hat{y}) = -\sum_{i=1}^{N} y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)$$
Hinge loss:$$L(y, \hat{y}) = \max(0, 1 - y \cdot \hat{y})$$

Thus, the model training process aims to discover a configuration of model parameters denoted as $θ$, which serves to minimize the cost function when applied to the training dataset.

### Example  of supervised learning
1. Naive Bayes: It's a probabilistic classification algorithm based on Bayes’ theorem. The '$naive$' in Naive Bayes comes from the assumption that each feature (or predictor) in the dataset is independent of all others. 

2. Logistic Regression: It's a linear classification algorithm that models the relationship between the input variables and the probability of belonging to a specific class.

3. Support Vector Machines (SVMs): It aim to find an optimal hyperplane that separates the data points of different classes with the largest margin (i.e., the maximum distance between data points of both classes). $Support$ $vectors$:  Which are the data points that are closest to the hyperplane. $Kernel$ $trick$: The kernel function transforms the data into a higher-dimensional space where a hyperplane can be used to separate classes (i.e., radial basis (RBF), sigmode).

4. Random Forests: The basic idea behind Random Forests is to combine multiple decision trees in determining the final output rather than relying on individual decision trees. Using a random subset of the data and features, RF reduces overfitting. It can handle complex relationships, missing values, and identifying essential features, acting as a feature ranking algorithm.

Unlike traditional methods used in Physics to analyze data, ML does not use curve fitting. The objective is to strike a balance where the model is complex enough to capture the underlying patterns but not overly complex to fit noise or irrelevant details.

Supervised learning methods in Physics are mainly used in problems that involve classification, regression and time series forecasting. For example, supervised learning algorithms have been used to classify particles in particle physics experiments, predict the Higgs boson’s mass, and weather and climate forecasting

## Unsupervised Learning
Unsupervised learning is a type of machine learning where the algorithm learns patterns or structures from unlabeled data without any explicit target variable. The goal is to discover hidden patterns, relationships, or clusters within the data.

There are two main types of unsupervised learning:

1. Clustering: 

    - Objective: Group similar data points together based on certain features or characteristics.
    - Example: K-means clustering, hierarchical clustering, and DBSCAN.

2. Dimensionality Reduction:
    - Objective: Reduce the number of features in a dataset while retaining its essential information and structure.
    - Example: Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE).
    
### Applications:
- Anomaly Detection: Identifying unusual patterns or outliers in data.
- Data Preprocessing: Extracting relevant features or reducing dimensionality before applying supervised learning algorithms.
- Market Basket Analysis: Discovering associations and patterns in customer purchasing behaviors.
- Image and Speech Recognition: Extracting features from unlabeled data to improve the performance of subsequent supervised learning models.

In physics, principal component and clustering analysis can be used to identify phases and phase transitions of many-body systems. Generative models can generate synthetic data that matches the statistical properties of experimental observations, enabling model validation and exploration of new scenarios (https://doi.org/10.1051/0004-6361/201833800).

### Challenges:
- The absence of ground truth labels makes evaluating clustering results subjective and dependent on heuristic (启发式的) measures.
- Determining the optimal number of clusters and selecting appropriate algorithms can be subjective and context-dependent.
- The curse of dimensionality poses difficulties as higher-dimensional spaces make it harder to distinguish meaningful clusters. (关注物理含义)

## Semi-supervised learning (SSL)
SSL can improve the accuracy of machine learning models, because the unlabeled data can be used to regularize the model, preventing overfitting. In Physics, SSL has been used, for example, to classify materials synthesis procedures and detect distinct events in a large dataset of in tokamak discharges.
### Main algorithms:
1. Self-training, which can take any supervised method for classification or regression and modify it to work in a semi-supervised manner.
2. Transductive SVM, which is a variation of SVMs that is specifically designed for semi-supervised learning. The goal of TSVM is to leverage the information from the labeled examples to make predictions on the unlabeled examples, taking into account the structure and distribution of the entire dataset.
3. Label propagation, which assigns labels to unlabeled data by propagating labels from labeled data points to unlabeled data points that are similar to them.
4. Ensemble methods, which combines multiple semi-supervised learning algorithms on different subsets of the data, and then their predictions are combined to make a final prediction.

## Reinforcement Learning (RL)
RL enables the extraction of knowledge from real-world experiences, surpassing the limitations of training data alone. The goal is "*to learn an optimal policy that maximizes cumulative rewards*". RL is used in scenarios without labelled training data and has applications in robotics, game-playing (AlphaGo), and recommendation systems.

The main elements in RL are an agent (make actions) and an environment (provides agent info & feedback) it interacts with. The agent’s primary goal is to maximize the obtained rewards the environment provides.

Main algorithm: **Q-learning**, which continuously learns the optimal action-value function regardless of the policy followed during the training. This algorithm has many versions and can be implemented in neural networks. In physics, reinforcement learning is utilized for tasks such as control of 
- quantum systems (https://doi.org/10.1103/PhysRevX.8.031084), 
- quantum experiment (https://doi.org/10.1073/pnas.17149361).

## Deep Learning
Deep learning is a subfield of machine learning that focuses on artificial neural networks and their application to solve complex problems. It is inspired by the structure and function of the human brain, specifically the way neurons are interconnected to process information. 

Deep learning methods can be used for all the tasks discussed previously.
### Key Concepts


1. Neural Networks

        At the heart of deep learning are "neural networks", which consist of layers of interconnected nodes. These networks can be deep, with multiple hidden layers, allowing them to automatically extract hierarchical features from raw data. The input layer receives data, hidden layers process information, and the output layer produces final results.

2. Activation Functions

        Nodes in neural networks employ "activation functions" to introduce non-linearities into the model. Common activation functions include sigmoid, hyperbolic tangent (tanh), and rectified linear unit (ReLU). These non-linearities enable the network to learn intricate relationships and capture complex patterns in the data.

3. Training and Backpropagation

        Deep learning models undergo a training process facilitated by "backpropagation". During training, the model makes predictions, and the error is backpropagated through the network. This iterative process adjusts the weights of connections, enhancing the model's ability to make accurate predictions.

4. Convolutional Neural Networks (CNNs)
        
        CNNs are a type of deep neural network designed for tasks like image and video analysis. They use convolutional layers to automatically and adaptively learn spatial hierarchies of features from input data.

5. Recurrent Neural Networks (RNNs)

        RNNs are designed for sequential data, such as time series or natural language. They have connections that form cycles, allowing them to capture dependencies over time. Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) are popular types of RNNs.
6. Transfer Learning

        Transfer learning involves using pre-trained models on one task and fine-tuning them for a different but related task. This is especially useful when dealing with limited labeled data for a specific problem.

### Applications

Deep learning has demonstrated remarkable success in various applications, including:
- Image and speech recognition (e.g., self-driving cars, voice assistants)
- Natural language processing (e.g., language translation, chatbots)
- Healthcare (e.g., medical image analysis)

The ability of deep learning models to automatically learn representations from data has eliminated the need for manual feature engineering in many cases, revolutionizing the landscape of machine learning. These models have been used to predict dynamical systems representa- tive of physical phenomena (https://doi.org/10.1016/j.neunet.2021.11.022)

## Physics Informed Neural Networks (PINN)
 In this approach, models are trained on both data and physical principles. More specifically, the cost function is changed to include a term that penalizes the model for violating the physical principles. For example, if we are trying to model the behaviour of a fluid, we might add the Navier-Stokes equations as a constraint to the cost function. 

## Causal Inference:
Only recently, causal machine learning methods have been designed to identify the causal relationships between variables and to use this information to make better predictions.
In physics, causality can be used to infer the connection between variables. For instance, in complex systems, causality methods have been used to infer the structure of the underlying system, like in the brain and climate systems (https://doi.org/10.1038/ncomms9502).










