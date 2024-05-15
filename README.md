
# Description

The project aims to analyze the quality of red wine based on its chemical characteristics.
After data loading and preliminary analysis, various machine learning models are investigated to predict wine quality based on its features.
Different classification methods are used for comparison, such as KNN, Logistic Regression, Decision Trees, Random Forests, and Gradient Boosting. Each model is evaluated based on quality metrics such as accuracy and AUC-ROC. The results allow selecting the best model for a given task.

# Dataset Description:

- Number of Records: 1599.
- Number of Features: 18.
- Types of Features: Numerical.

# The dataset includes the following features:
- Fixed acidity: The amount of acid in the wine that does not evaporate during boiling.
- Volatile acidity: The amount of acetic acid in the wine, which can contribute to unpleasant vinegar-like smells.
- Citric acid: The amount of citric acid in the wine, which can add freshness and fruitiness.
- Residual sugar: The amount of sugar left in the wine after fermentation, influencing its sweetness.
- Chlorides: The amount of salt in the wine, which can be derived from the soil or added during production.
- Free sulfur dioxide: The amount of free SO2 in the wine, serving as an antiseptic and antioxidant.
- Total sulfur dioxide: The total amount of SO2 in the wine, including both free and bound SO2, also serving as a preservative and antiseptic.
- Density: The density of the wine, measured in g/cm³.
- pH: The pH level of the wine, influencing its acidity or alkalinity.
- Sulphates: The amount of added sulfates, which can also act as an antiseptic and antioxidant.
- Alcohol: The percentage of alcohol in the wine.
- Quality: The quality rating of the wine, typically on a scale from 1 to 10.
- Loading Libraries

# Result
- From the results, it can be observed that Random forest showed the highest accuracy score among all algorithms (0.808743), as well as the highest area under the ROC curve (AUC score) (0.867880). This may mean that Random forest performs well and best ability to distinguish between positive and negative cases in the classification task.

- GBM (Gradient Boosting Machine) and KNN have comparable accuracy scores to Random Forest, with 0.795082 and 0.751366 respectively. GBM has the second-highest AUC score among all algorithms, indicating it has the best ability to distinguish between positive and negative cases in the classification task.

- Decision Tree has the lowest accuracy score of 0.718579 and the lowest AUC score of 0.773032, suggesting it might not generalize well to unseen data or capture the underlying patterns in the data as effectively as other algorithms.

- While accuracy is an important metric, considering both accuracy and AUC score is crucial. AUC score provides a measure of the classifier's performance across all classification thresholds, making it useful for assessing the overall performance of the model.

- In summary, while Random forest performs better in terms of accuracy and AUC, GBM shows promise with its second-highest which s slightly lower in AUC score than Random forest, indicating it might be better at correctly classifying instances in the dataset. However, the choice of the algorithm also depends on other factors like interpretability, computational efficiency, and the specific requirements of the problem at hand.


# Other ML techniques

# Model Accuracies:
- Random Forest: 0.8142076502732241
- Support Vector Classifier: 0.6338797814207651
- Gradient Boosting: 0.7650273224043715
- Gaussian Naive Bayes: 0.7240437158469946
- Multi-layer Perceptron: 0.6967213114754098
- AdaBoost: 0.7486338797814208
- Extra Trees: 0.8114754098360656
- Linear SVM: 0.5573770491803278

#  Model AUC Scores:
- Random Forest: 0.8123222748815165
- Support Vector Classifier: 0.595138358049228
- Gradient Boosting: 0.7602507261886562
- Gaussian Naive Bayes: 0.7306986699281456
- Multi-layer Perceptron: 0.6898792233603424
- AdaBoost: 0.7417520256841462
- Extra Trees: 0.8099526066350712
- Linear SVM: 0.6109769148448249
