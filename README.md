
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
- Based on the analysis, it's evident that GBM (Gradient Boosting Machine) outperforms all other algorithms with an impressive accuracy score of 92.00% and the highest AUC score of 0.985. This indicates GBM's exceptional ability to both classify instances correctly and distinguish between positive and negative cases in the classification task.

- Following GBM, Random Forest demonstrates strong performance with an accuracy score of 80.87% and an AUC score of 0.868. While Random Forest falls short of GBM in terms of AUC score, it still exhibits robust classification capabilities and stands as a reliable choice for classification tasks.

- KNN (K-Nearest Neighbors) follows with a respectable accuracy score of 75.14% and an AUC score of 0.809. Although KNN's performance is slightly lower compared to GBM and Random Forest, it remains competitive and offers a straightforward approach to classification.

- Ridge and Lasso Logistic Regression, along with Logistic Regression, yield similar accuracy scores around 75%, with AUC scores ranging from 0.783 to 0.787. While logistic regression models provide decent classification performance, they are surpassed by GBM, Random Forest, and KNN in terms of accuracy and AUC score.

- Lastly, Decision Tree trails behind with the lowest accuracy score of 71.86% and an AUC score of 0.773. Decision Tree's comparatively lower performance highlights its limitations in capturing complex relationships within the data and generalizing to unseen instances effectively.

- In conclusion, GBM emerges as the top-performing algorithm, offering the highest accuracy and AUC scores. However, the selection of the most suitable algorithm depends on various factors such as interpretability, computational efficiency, and specific requirements of the classification task.
