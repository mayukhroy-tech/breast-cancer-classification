# Breast Cancer Classification Using Five Machine-Learning Models

## A. Problem Statement

Breast cancer diagnosis is an important classification problem in which
tumour characteristics can help distinguish malignant tumours from
benign tumours. The objective of this project is to develop and compare
five supervised machine-learning classification models that predict
whether a tumour is malignant or benign using measurements from the
Breast Cancer Wisconsin (Diagnostic) dataset.

The five models are trained and evaluated using the same dataset split.
Their performance is compared using Accuracy, AUC, Precision, Recall, F1
Score, and Matthews Correlation Coefficient (MCC). An interactive
Streamlit application is also provided so that a user can upload test
data, select a trained model, and view its evaluation results.

## B. Dataset Description

-   **Dataset name:** Breast Cancer Wisconsin (Diagnostic)
-   **Dataset source:** UCI Machine Learning Repository
-   **UCI dataset page:**
    https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
-   **Scikit-learn dataset documentation:**
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
-   **Total instances:** 569
-   **Number of input features:** 30 numeric features
-   **Target column:** `target`
-   **Target classes:** `0 = malignant`, `1 = benign`
-   **Missing values:** The standard scikit-learn version used for this
    project has no missing values.

The dataset meets the assignment requirements of at least 500 instances
and at least 12 features.

### Data preparation

The dataset is loaded using
`sklearn.datasets.load_breast_cancer(as_frame=True)`. The 30 input
features are separated from the target column. The data is divided into
training and testing subsets using an 80:20 stratified split with
`random_state=42`.

-   **Training records:** 455
-   **Testing records:** 114

Stratification is used to maintain approximately the same class
distribution in the training and test subsets. The held-out test subset
is exported as `test_data.csv`. The saved models are evaluated on this
test data; the Streamlit application does not retrain the models on
uploaded test data.

## C. GitHub Repository Link

**GitHub Repository:** `<PASTE_YOUR_PUBLIC_GITHUB_REPOSITORY_URL_HERE>`

The repository should contain the complete source code,
`requirements.txt`, this `README.md`, `test_data.csv`,
`model_comparison.csv`, and all five saved model files in the `model/`
directory.

Expected repository structure:

``` text
breast_cancer_project/
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model_comparison.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── support_vector_machine.joblib
```

Add the training/preparation notebook or Python source file to the
repository if it is part of the implementation submitted for the
assignment. The filenames above must match the filenames referenced in
`app.py`.

## D. Models Used and Evaluation

All five classifiers are trained using the same training subset and
evaluated using the same held-out test subset.

### 1. Models implemented

1.  **Logistic Regression:** A supervised classification algorithm that
    estimates class probabilities using a linear decision boundary.
2.  **Decision Tree Classifier:** A tree-based algorithm that makes
    predictions using a sequence of feature-based decision rules.
3.  **K-Nearest Neighbour (KNN) Classifier:** A distance-based algorithm
    that predicts a class from the labels of nearby training
    observations.
4.  **Gaussian Naive Bayes:** A probabilistic classifier based on Bayes'
    theorem and the assumption that feature values are conditionally
    independent within each class.
5.  **Support Vector Machine (SVM):** A classifier that learns a
    separating decision boundary and can use a kernel to model
    non-linear boundaries.

### 2. Evaluation metrics

  -----------------------------------------------------------------------
  Metric                              Description
  ----------------------------------- -----------------------------------
  Accuracy                            Proportion of all test observations
                                      classified correctly.

  AUC                                 Area under the ROC curve; measures
                                      how well the model ranks
                                      observations from the two classes.

  Precision                           Proportion of predicted positive
                                      observations that are actually
                                      positive.

  Recall                              Proportion of actual positive
                                      observations correctly identified.

  F1 Score                            Harmonic mean of precision and
                                      recall.

  MCC                                 Matthews Correlation Coefficient; a
                                      balanced measure based on true
                                      positives, true negatives, false
                                      positives, and false negatives.
  -----------------------------------------------------------------------

**Positive-class convention:** The dataset encoding used here is
`0 = malignant` and `1 = benign`. Therefore, unless the evaluation code
explicitly specifies a different positive label, Precision, Recall, and
F1 are calculated with class `1` (benign) as the positive class. Keep
this convention consistent when reporting the results.

### 3. Model comparison table

Enter the actual values generated by your executed evaluation code or
`model_comparison.csv`. Do not use estimated or example values.

  ------------------------------------------------------------------------------------
  ML Model        Accuracy         AUC   Precision      Recall          F1         MCC
  Name                                                                     
  ------------ ----------- ----------- ----------- ----------- ----------- -----------
  Logistic       `<value>`   `<value>`   `<value>`   `<value>`   `<value>`   `<value>`
  Regression                                                               

  Decision       `<value>`   `<value>`   `<value>`   `<value>`   `<value>`   `<value>`
  Tree                                                                     

  KNN            `<value>`   `<value>`   `<value>`   `<value>`   `<value>`   `<value>`

  Naive Bayes    `<value>`   `<value>`   `<value>`   `<value>`   `<value>`   `<value>`

  Support        `<value>`   `<value>`   `<value>`   `<value>`   `<value>`   `<value>`
  Vector                                                                   
  Machine                                                                  
  ------------------------------------------------------------------------------------

### 4. Observations on model performance

Complete the observations below after running all five models. Refer to
the measured metrics and confusion matrices rather than making claims
based only on the algorithm type.

  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  ML Model Name                       Observation about model performance
  ----------------------------------- -----------------------------------------------------------------------------------------------------------------------------------------------------------------
  Logistic Regression                 `<Describe its measured Accuracy, AUC, Precision, Recall, F1 and MCC. Mention any balance or trade-off between identifying malignant and benign cases.>`

  Decision Tree                       `<Describe its measured results. Note whether its performance is comparable to the other models and whether its errors are concentrated in either class.>`

  KNN                                 `<Describe its measured results and comment on how well the distance-based classifier performs on this dataset.>`

  Naive Bayes                         `<Describe its measured results and any difference between its Precision and Recall.>`

  Support Vector Machine              `<Describe its measured results and compare its performance with the other classifiers.>`

  **Overall, winner for this          `<Name the model with the strongest overall test performance and justify the choice using the reported metrics. Consider AUC, F1 and MCC as well as Accuracy.>`
  dataset?**                          
  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### 5. Model selection

The final model should be selected using the actual test results.
Accuracy alone should not automatically determine the winner. AUC, F1
Score, MCC, and the confusion matrix should also be considered. For a
medical classification task, discuss the implications of false negatives
(malignant tumours predicted as benign) as well as false positives.

## Streamlit Application

The Streamlit application (`app.py`) is designed to provide:

-   A CSV upload option for test data.
-   A dropdown to select one of the five trained classifiers.
-   Display of Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
-   A confusion matrix or classification report.
-   Evaluation of the uploaded test data using the saved model, without
    retraining.

**Live Streamlit application:**
`<PASTE_YOUR_STREAMLIT_COMMUNITY_CLOUD_URL_HERE>`

## Running the project locally

1.  Clone the GitHub repository:

    ``` bash
    git clone <YOUR_GITHUB_REPOSITORY_URL>
    cd breast_cancer_project
    ```

2.  Install the dependencies:

    ``` bash
    pip install -r requirements.txt
    ```

3.  Start the application:

    ``` bash
    streamlit run app.py
    ```

4.  Upload `test_data.csv`, select a classifier, and review the
    displayed metrics and confusion matrix.

## Reproducibility and submission notes

-   The train/test split uses `random_state=42`.
-   All five models must use the same train/test split for a fair
    comparison.
-   Save preprocessing together with the fitted estimator when a model
    requires scaling.
-   Keep the feature names and target column consistent between
    training, test data, and the Streamlit app.
-   The app must load saved model files and must not retrain on uploaded
    test data.
-   Replace every placeholder in this README with the actual GitHub URL,
    Streamlit URL, evaluation values, and observations before
    submission.
-   Include this README content in the single PDF required by the
    assignment.

## Student Details

-   **Student name:** MAYUKH ROY
-   **Student ID:** 2026AJ05384
-   **Programme:** M.Tech in Artificial Intelligence and Machine
    Learning, BITS Pilani
-   **Development environment:** Google Colab
-   **Application framework:** Streamlit
