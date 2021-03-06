# Gaussian Mixture Model Example
This is a GMM pratice of my pattern recognition course.

### Question
Select a Gaussian Mixture Model as the true probability density function for 2-dimensional real-valued data synthesis. This GMM will have 4 components with different mean vectors, different covariance matrices, and different probability for each Gaussian to be selected as the generator for each sample. Specify the true GMM that generates data.
Conduct the following model order selection exercise multiple times (e.g., M =100), each time using cross-validation based on many (e.g., at least B = 10) independent training-validation sets generated with bootstrapping.

Repeat the following multiple times (e.g., M = 100):

**Step 1:** Generate multiple data sets with independent identically distributed samples using this true GMM; these datasets will have, respectively, 10, 100, 1000 samples.

**Step 2: (needs lots of computations)** For each data set, using maximum likelihood parameter estimation with with the EM algorithm, train and validate GMMs with different model orders (using at least B = 10 bootstrapped training/validation sets). Specifically, evaluate candidate GMMs with 1, 2, 3, 4, 5, 6 Gaussian components. Note that both model parameter estimation and validation performance measures to be used is log-likelihood of the appropriate dataset (training or validation set, depending on whether you are optimizing model parameters or assessing a trained model).

**Step 3:** Report your results for the experiment, indicating details like, how do you initialize your EM algorithm, how many random initializations do you do for each attempt seeking for the global optimum, across many independent experiments how many times each of the six candidate GMM model orders get selected when using different sizes of datasets... Provide a clear description of your experimental procedure and well designed visual and numerical illustrations of your experiment results in the form of tables/figures.

***Note:*** We anticipate that as the number of samples in your dataset increases, the crossvalidation procedure will more frequently lead to the selection of the true model order of 4 in the true data pdf. You may illustrate this by showing that your repeated experiments with different datasets leads to a more concentrated histogram at and around this value.
