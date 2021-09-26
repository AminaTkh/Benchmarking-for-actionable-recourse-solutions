
# Benchmarking framework for a number of existing "algorithmic recourse" solutions 

This repository provides a user-oriented software tool. As a benchmarking platform, it offers comparisons and quality evaluations. The software allows a comparison to be made between various approaches to i) counterfactual explanation generation, ii) ranking factor definition, and iii) ranking algorithms, both separately and in combinations. Aside from comparing a variety of methods in a transparent and meaningful manner, this toolkit provides visual representations of fairness measures and algorithmic characteristics derived from the comparisons. It is designed with extensibility in mind. Thus, it is easy to include new counterfactual explanation generation methods, new ranking algorithms, new machine learning models or other datasets. 

### Available Datasets

- German Credit Data Set: [Source](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
- Default of Credit Card Clients Data Set: [Source](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

### Counterfactual Explanation Generation Methods provided

- Actionable Recourse (AR): [Paper](https://arxiv.org/pdf/1809.06514)
- The Model-Agnostic Counterfactual Explanations for Consequential Decisions (MACE): [Paper](https://arxiv.org/abs/1905.11190)
- The Actionable Classification and Fairness project (not published yet)

### Fairness-aware Ranking Algorithms provided

- A Fair Top-k Ranking Algorithm (FA*IR): [Paper](https://arxiv.org/pdf/1706.06368)
- Fairness of Exposure in Rankings (FOEIR): [Paper](https://arxiv.org/abs/1802.07281)
- The Actionable Classification and Fairness project (not published yet)

### Available Ranking factors

- prediction probability of being negatively classified
- distance to a counterfactual explanation 

### Measures

- Average time for generating a counterfactual explanation
- Average distance between the original point and its counterfactual explanation (Proximity) 
- Sparsity 
- Closeness to the training data
- Normalised dis- counted KL-divergence (rKL)
- Normalized discounted difference (rND)
- Normalized discounted ratio (rRD) 
- Average time required for the ranking algorithm
- Action Fairness ratio

### Provided Machine Learning Models

- **Logistic Regression**: Linear Model with no hidden layer and no activation function

## Available Functions

### The following are the functions that can be selected using the “-task” flag:

- Generate counterfactual explanations.
- Calculate the metrics of algorithms for generation of counterfactual explanations.
- Create baseline rankings.
- Rank.
- Measure Fairness in terms of rKL, rRD, rND.
- Get average measures of fairness for each ranking algorithm.
- Get run-time for every ranking algorithm.
- Visualise the results.

### The toolkit also includes the following optional flags, along with the function definitions:

- -d. This flag identifies the data set with which comparative performance is to be evaluated.
- -cg method. The flag specifies a particular method for generation of counterfactual explanations among all those available.
- -factor. It indicates the factor on which the rankings are based. For this project, it can be either the distance to the counterfactual explanation or the probability of being negatively classifier.
- -rank. This flag indicates which ranking algorithm is to be applied.

