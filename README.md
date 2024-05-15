
Description

The project aims to analyze the quality of red wine based on its chemical characteristics.
After data loading and preliminary analysis, various machine learning models are investigated to predict wine quality based on its features.
Different classification methods are used for comparison, such as KNN, Logistic Regression, Decision Trees, Random Forests, and Gradient Boosting. Each model is evaluated based on quality metrics such as accuracy and AUC-ROC. The results allow selecting the best model for a given task.
Dataset Description:

Number of Records: 1599.
Number of Features: 18.
Types of Features: Numerical.
The dataset includes the following features:

Fixed acidity: The amount of acid in the wine that does not evaporate during boiling.
Volatile acidity: The amount of acetic acid in the wine, which can contribute to unpleasant vinegar-like smells.
Citric acid: The amount of citric acid in the wine, which can add freshness and fruitiness.
Residual sugar: The amount of sugar left in the wine after fermentation, influencing its sweetness.
Chlorides: The amount of salt in the wine, which can be derived from the soil or added during production.
Free sulfur dioxide: The amount of free SO2 in the wine, serving as an antiseptic and antioxidant.
Total sulfur dioxide: The total amount of SO2 in the wine, including both free and bound SO2, also serving as a preservative and antiseptic.
Density: The density of the wine, measured in g/cm³.
pH: The pH level of the wine, influencing its acidity or alkalinity.
Sulphates: The amount of added sulfates, which can also act as an antiseptic and antioxidant.
Alcohol: The percentage of alcohol in the wine.
Quality: The quality rating of the wine, typically on a scale from 1 to 10.
Loading Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
Loading Data

df = pd.read_csv('winequality-red.csv')
df.head()
fixed acidity	volatile acidity	citric acid	residual sugar	chlorides	free sulfur dioxide	total sulfur dioxide	density	pH	sulphates	alcohol	quality
0	7.4	0.70	0.00	1.9	0.076	11.0	34.0	0.9978	3.51	0.56	9.4	5
1	7.8	0.88	0.00	2.6	0.098	25.0	67.0	0.9968	3.20	0.68	9.8	5
2	7.8	0.76	0.04	2.3	0.092	15.0	54.0	0.9970	3.26	0.65	9.8	5
3	11.2	0.28	0.56	1.9	0.075	17.0	60.0	0.9980	3.16	0.58	9.8	6
4	7.4	0.70	0.00	1.9	0.076	11.0	34.0	0.9978	3.51	0.56	9.4	5
df.shape
(1599, 12)
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1599 entries, 0 to 1598
Data columns (total 12 columns):
 #   Column                Non-Null Count  Dtype  
---  ------                --------------  -----  
 0   fixed acidity         1599 non-null   float64
 1   volatile acidity      1599 non-null   float64
 2   citric acid           1599 non-null   float64
 3   residual sugar        1599 non-null   float64
 4   chlorides             1599 non-null   float64
 5   free sulfur dioxide   1599 non-null   float64
 6   total sulfur dioxide  1599 non-null   float64
 7   density               1599 non-null   float64
 8   pH                    1599 non-null   float64
 9   sulphates             1599 non-null   float64
 10  alcohol               1599 non-null   float64
 11  quality               1599 non-null   int64  
dtypes: float64(11), int64(1)
memory usage: 150.0 KB
mean_by_quality = df.groupby('quality').mean()
print('Means for Different Values of the Target Variable:')
mean_by_quality
Means for Different Values of the Target Variable:
fixed acidity	volatile acidity	citric acid	residual sugar	chlorides	free sulfur dioxide	total sulfur dioxide	density	pH	sulphates	alcohol
quality											
3	8.360000	0.884500	0.171000	2.635000	0.122500	11.000000	24.900000	0.997464	3.398000	0.570000	9.955000
4	7.779245	0.693962	0.174151	2.694340	0.090679	12.264151	36.245283	0.996542	3.381509	0.596415	10.265094
5	8.167254	0.577041	0.243686	2.528855	0.092736	16.983847	56.513950	0.997104	3.304949	0.620969	9.899706
6	8.347179	0.497484	0.273824	2.477194	0.084956	15.711599	40.869906	0.996615	3.318072	0.675329	10.629519
7	8.872362	0.403920	0.375176	2.720603	0.076588	14.045226	35.020101	0.996104	3.290754	0.741256	11.465913
8	8.566667	0.423333	0.391111	2.577778	0.068444	13.277778	33.444444	0.995212	3.267222	0.767778	12.094444
df.describe().T
count	mean	std	min	25%	50%	75%	max
fixed acidity	1599.0	8.319637	1.741096	4.60000	7.1000	7.90000	9.200000	15.90000
volatile acidity	1599.0	0.527821	0.179060	0.12000	0.3900	0.52000	0.640000	1.58000
citric acid	1599.0	0.270976	0.194801	0.00000	0.0900	0.26000	0.420000	1.00000
residual sugar	1599.0	2.538806	1.409928	0.90000	1.9000	2.20000	2.600000	15.50000
chlorides	1599.0	0.087467	0.047065	0.01200	0.0700	0.07900	0.090000	0.61100
free sulfur dioxide	1599.0	15.874922	10.460157	1.00000	7.0000	14.00000	21.000000	72.00000
total sulfur dioxide	1599.0	46.467792	32.895324	6.00000	22.0000	38.00000	62.000000	289.00000
density	1599.0	0.996747	0.001887	0.99007	0.9956	0.99675	0.997835	1.00369
pH	1599.0	3.311113	0.154386	2.74000	3.2100	3.31000	3.400000	4.01000
sulphates	1599.0	0.658149	0.169507	0.33000	0.5500	0.62000	0.730000	2.00000
alcohol	1599.0	10.422983	1.065668	8.40000	9.5000	10.20000	11.100000	14.90000
quality	1599.0	5.636023	0.807569	3.00000	5.0000	6.00000	6.000000	8.00000
# List of features to create bar plots for
features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 
            'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']

# Create a 4x3 grid of subplots for the bar plots
fig, axes = plt.subplots(4, 3, figsize=(18, 18))

# Iterate over each feature and create a bar plot in the corresponding subplot
for i, feature in enumerate(features):
    row = i // 3
    col = i % 3
    ax = axes[row, col]
    
    sns.barplot(x='quality', y=feature, data=df, palette='inferno', ax=ax)
    ax.set_xlabel('Wine Quality')
    ax.set_ylabel(feature.replace('_', ' ').title())
    ax.set_title(f'Average {feature.replace("_", " ").title()} by Wine Quality')

plt.tight_layout()
plt.show()

outliers_df = pd.DataFrame()
df_out = df.drop(df.columns[-1], axis=1)

aggregation_functions = {}

for column in df_out.columns:
    outliers = df_out[df_out[column] > df_out[column].mean() + 3 * df_out[column].std()].copy()
    outliers['Outlier_Column'] = column
    outliers_df = pd.concat([outliers_df, outliers])
    aggregation_functions[column] = 'max'

aggregation_functions['Outlier_Column'] = lambda x: ', '.join(x)
outliers_df = outliers_df.groupby(outliers_df.index).agg(aggregation_functions)
outliers_df.sort_values(by='Outlier_Column')
fixed acidity	volatile acidity	citric acid	residual sugar	chlorides	free sulfur dioxide	total sulfur dioxide	density	pH	sulphates	alcohol	Outlier_Column
588	5.0	0.420	0.24	2.0	0.060	19.0	50.0	0.99170	3.72	0.74	14.0	alcohol
144	5.2	0.340	0.00	1.8	0.050	27.0	63.0	0.99160	3.68	0.79	14.0	alcohol
1269	5.5	0.490	0.03	1.8	0.044	28.0	87.0	0.99080	3.50	0.82	14.0	alcohol
1270	5.0	0.380	0.01	1.6	0.048	26.0	60.0	0.99084	3.70	0.75	14.0	alcohol
821	4.9	0.420	0.00	2.1	0.048	16.0	42.0	0.99154	3.71	0.74	14.0	alcohol
...	...	...	...	...	...	...	...	...	...	...	...	...
127	8.1	1.330	0.00	1.8	0.082	3.0	12.0	0.99640	3.54	0.48	10.9	volatile acidity
199	6.9	1.090	0.06	2.1	0.061	12.0	31.0	0.99480	3.51	0.43	11.4	volatile acidity
1312	8.0	1.180	0.21	1.9	0.083	14.0	41.0	0.99532	3.34	0.47	10.5	volatile acidity
724	7.5	1.115	0.10	3.1	0.086	5.0	12.0	0.99580	3.54	0.60	11.2	volatile acidity
672	9.8	1.240	0.34	2.0	0.079	32.0	151.0	0.99800	3.15	0.53	9.5	volatile acidity, total sulfur dioxide
136 rows × 12 columns

df = df.drop(outliers_df.index)
df
fixed acidity	volatile acidity	citric acid	residual sugar	chlorides	free sulfur dioxide	total sulfur dioxide	density	pH	sulphates	alcohol	quality
0	7.4	0.700	0.00	1.9	0.076	11.0	34.0	0.99780	3.51	0.56	9.4	5
1	7.8	0.880	0.00	2.6	0.098	25.0	67.0	0.99680	3.20	0.68	9.8	5
2	7.8	0.760	0.04	2.3	0.092	15.0	54.0	0.99700	3.26	0.65	9.8	5
3	11.2	0.280	0.56	1.9	0.075	17.0	60.0	0.99800	3.16	0.58	9.8	6
4	7.4	0.700	0.00	1.9	0.076	11.0	34.0	0.99780	3.51	0.56	9.4	5
...	...	...	...	...	...	...	...	...	...	...	...	...
1594	6.2	0.600	0.08	2.0	0.090	32.0	44.0	0.99490	3.45	0.58	10.5	5
1595	5.9	0.550	0.10	2.2	0.062	39.0	51.0	0.99512	3.52	0.76	11.2	6
1596	6.3	0.510	0.13	2.3	0.076	29.0	40.0	0.99574	3.42	0.75	11.0	6
1597	5.9	0.645	0.12	2.0	0.075	32.0	44.0	0.99547	3.57	0.71	10.2	5
1598	6.0	0.310	0.47	3.6	0.067	18.0	42.0	0.99549	3.39	0.66	11.0	6
1463 rows × 12 columns

EDA

plt.figure(figsize=(22, 18))

for i, column in enumerate(df.columns[:-1]):
    plt.subplot(6, 4, i+1)
    sns.histplot(df[column], kde=True)

plt.suptitle('Histograms of Features', fontsize=20)  
plt.tight_layout()
plt.show()

df_corr = df.drop('quality', axis=1)
df_corr = df_corr.corr()

mask = np.triu(np.ones_like(df_corr, dtype=bool))
plt.figure(figsize=(12, 8))
sns.heatmap(df_corr, mask=mask, annot=True, fmt='.2f', cmap='inferno')
plt.title('Correlation Matrix', fontsize=20)
plt.tight_layout()
plt.show()

df['quality'].value_counts()
5    617
6    589
7    187
4     47
8     16
3      7
Name: quality, dtype: int64
For further work, we will convert the "quality" values into a binary format, where values from 3 to 5 will represent low quality (0), and values from 6 to 8 will represent high quality (1).

df['quality'] = np.where(df['quality'] > 5, 1, 0)
df
fixed acidity	volatile acidity	citric acid	residual sugar	chlorides	free sulfur dioxide	total sulfur dioxide	density	pH	sulphates	alcohol	quality
0	7.4	0.700	0.00	1.9	0.076	11.0	34.0	0.99780	3.51	0.56	9.4	0
1	7.8	0.880	0.00	2.6	0.098	25.0	67.0	0.99680	3.20	0.68	9.8	0
2	7.8	0.760	0.04	2.3	0.092	15.0	54.0	0.99700	3.26	0.65	9.8	0
3	11.2	0.280	0.56	1.9	0.075	17.0	60.0	0.99800	3.16	0.58	9.8	1
4	7.4	0.700	0.00	1.9	0.076	11.0	34.0	0.99780	3.51	0.56	9.4	0
...	...	...	...	...	...	...	...	...	...	...	...	...
1594	6.2	0.600	0.08	2.0	0.090	32.0	44.0	0.99490	3.45	0.58	10.5	0
1595	5.9	0.550	0.10	2.2	0.062	39.0	51.0	0.99512	3.52	0.76	11.2	1
1596	6.3	0.510	0.13	2.3	0.076	29.0	40.0	0.99574	3.42	0.75	11.0	1
1597	5.9	0.645	0.12	2.0	0.075	32.0	44.0	0.99547	3.57	0.71	10.2	0
1598	6.0	0.310	0.47	3.6	0.067	18.0	42.0	0.99549	3.39	0.66	11.0	1
1463 rows × 12 columns

quality = df['quality'].value_counts()
bars = sns.barplot(x=quality.index, y=quality.values)

for bar in bars.patches:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.1, yval, int(yval), va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

plt.figure(figsize=(16, 10)) 

for i, column in enumerate(df.columns[:-1]):
    plt.subplot(3, 4, i+1)
    sns.boxplot(x='quality', y=column, data=df)
    plt.xlabel('Quality', fontsize=12)
    plt.ylabel(f'{column}', fontsize=12)

plt.suptitle('Box Plots of Wine Attribute', fontsize=20)  
plt.tight_layout()
plt.show()

Box Plots provide a graphical summary of the distribution of a dataset, showing the median, quartiles, and outliers. It's useful for comparing distributions across different categories

mean_by_quality = df.groupby('quality').mean()
print('Means for Different Values of the Target Variable:')
mean_by_quality
Means for Different Values of the Target Variable:
fixed acidity	volatile acidity	citric acid	residual sugar	chlorides	free sulfur dioxide	total sulfur dioxide	density	pH	sulphates	alcohol
quality											
0	8.109538	0.578599	0.230089	2.374590	0.083578	15.710879	51.910581	0.996996	3.321177	0.595306	9.929657
1	8.475505	0.476193	0.295694	2.399369	0.079494	14.686869	37.023990	0.996442	3.310896	0.681199	10.844381
median_by_quality = df.groupby('quality').median()
print('Median for Different Values of the Target Variable:')
median_by_quality
Median for Different Values of the Target Variable:
fixed acidity	volatile acidity	citric acid	residual sugar	chlorides	free sulfur dioxide	total sulfur dioxide	density	pH	sulphates	alcohol
quality											
0	7.8	0.58	0.22	2.2	0.080	14.0	43.0	0.9969	3.32	0.57	9.7
1	8.1	0.46	0.31	2.2	0.077	13.0	31.0	0.9964	3.31	0.66	10.8
Data Preprocessing

X = df.drop('quality', axis=1)
y = df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
Due to our data set has a non-normal distribution, we should do scaling using MinMaxScaler.

norm = MinMaxScaler(feature_range = (0, 1))
norm.fit(X_train)
X_train_n = norm.transform(X_train)
X_test_n = norm.transform(X_test)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
(1097, 11)
(366, 11)
(1097,)
(366,)
KNN Classifier

KNN model with top 5 neighbors with minimal train/test accuracy difference

knn_top = pd.DataFrame(columns=['Neighbors', 'Test Accuracies', 'Train Accuracies'])

for i in range(1, 31):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train_n, y_train)
    accuracy_test = knn.score(X_test_n, y_test)
    accuracy_train = knn.score(X_train_n, y_train)
    knn_top.loc[len(knn_top)]=[i, accuracy_test, accuracy_train]

knn_top['Train/Test Accuracy Difference'] = abs(knn_top['Train Accuracies'] - knn_top['Test Accuracies'])
knn_top.sort_values(by='Train/Test Accuracy Difference', ascending=True).head().reset_index(drop=True)
Neighbors	Test Accuracies	Train Accuracies	Train/Test Accuracy Difference
0	13.0	0.740437	0.762990	0.022553
1	10.0	0.751366	0.775752	0.024386
2	16.0	0.734973	0.761167	0.026194
3	30.0	0.740437	0.768459	0.028022
4	25.0	0.734973	0.764813	0.029840
Classification metrics analysis for KNN model

knn_model = KNeighborsClassifier(n_neighbors=10)
knn_model.fit(X_train_n, y_train)
y_pred = knn_model.predict(X_test_n)

accuracy_knn = accuracy_score(y_test, y_pred)
print(f'Accuracy of the KNN Model: {accuracy_knn}')

y_prob = knn_model.predict_proba(X_test_n)[:, 1]
auc_score_knn = roc_auc_score(y_test, y_prob)
print(f'AUC score of the KNN Model: {auc_score_knn}')
Accuracy of the KNN Model: 0.7513661202185792
AUC score of the KNN Model: 0.809356367527901
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=sorted(y_test.unique()), yticklabels=sorted(y_test.unique()))
plt.title('KNN Confusion Matrix', fontsize=16)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('True', fontsize=12)
plt.show()

print(classification_report(y_test, y_pred))
              precision    recall  f1-score   support

           0       0.70      0.74      0.71       155
           1       0.80      0.76      0.78       211

    accuracy                           0.75       366
   macro avg       0.75      0.75      0.75       366
weighted avg       0.75      0.75      0.75       366

y_prob = knn_model.predict_proba(X_test_n)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='#1a5174', lw=2, label='ROC curve(AUC Score={:.4f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='#e57373', lw=2, linestyle='--')
plt.xlabel('False Positive Rate',  fontsize=12)
plt.ylabel('True Positive Rate',  fontsize=12)
plt.title('ROC Curve for KNN Model', fontsize=16)
plt.legend(loc='lower right', fontsize=12)
plt.show()

Logistics Regression

lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_n, y_train)
y_pred = lr_model.predict(X_test_n)

accuracy_lr = accuracy_score(y_test, y_pred)
print(f'Accuracy of the Logistic Regression Model: {accuracy_lr}')

y_prob = lr_model.predict_proba(X_test_n)[:, 1]
auc_score_lr = roc_auc_score(y_test, y_prob)
print(f'AUC score of the Logistic Regression Model: {auc_score_lr}')
Accuracy of the Logistic Regression Model: 0.7404371584699454
AUC score of the Logistic Regression Model: 0.8047087601284209
print(classification_report(y_test, y_pred))
              precision    recall  f1-score   support

           0       0.68      0.73      0.70       155
           1       0.79      0.75      0.77       211

    accuracy                           0.74       366
   macro avg       0.74      0.74      0.74       366
weighted avg       0.74      0.74      0.74       366

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=sorted(y_test.unique()), yticklabels=sorted(y_test.unique()))
plt.title('Logistic Regression Confusion Matrix', fontsize=16)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('True', fontsize=12)
plt.show()

y_prob = lr_model.predict_proba(X_test_n)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='#1a5174', lw=2, label='ROC curve(AUC Score={:.4f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='#e57373', lw=2, linestyle='--')
plt.xlabel('False Positive Rate',  fontsize=12)
plt.ylabel('True Positive Rate',  fontsize=12)
plt.title('ROC Curve for Logistic Regression Model', fontsize=16)
plt.legend(loc='lower right', fontsize=12)
plt.show()

Decision Tree Classifier Best Parameters for Decision Tree model

dt_model = DecisionTreeClassifier(criterion='gini', random_state=42)

parameters = {
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'min_samples_leaf': [2, 1, 3, 4, 6, 10, 20],
    'min_samples_split': [3, 2, 4, 5, 10, 20]
}

bp_dt = GridSearchCV(dt_model, parameters, n_jobs=-1, cv=10)
bp_dt.fit(X_train, y_train)
print('Best Parameters:', bp_dt.best_params_)
Best Parameters: {'max_depth': 6, 'min_samples_leaf': 10, 'min_samples_split': 3}
Classification metrics analysis for Decision Tree model

best_model_dt = bp_dt.best_estimator_
best_model_dt.fit(X_train, y_train)
y_pred = best_model_dt.predict(X_test)

accuracy_dt = accuracy_score(y_test, y_pred)
print(f'Accuracy of the Decision Tree Model: {accuracy_dt}')

y_prob = best_model_dt.predict_proba(X_test)[:, 1]
auc_score_dt = roc_auc_score(y_test, y_prob)
print(f'AUC score of the Decision Tree Model: {auc_score_dt}')
Accuracy of the Decision Tree Model: 0.7185792349726776
AUC score of the Decision Tree Model: 0.7730316465372266
print(classification_report(y_test, y_pred))
              precision    recall  f1-score   support

           0       0.65      0.71      0.68       155
           1       0.77      0.73      0.75       211

    accuracy                           0.72       366
   macro avg       0.71      0.72      0.71       366
weighted avg       0.72      0.72      0.72       366

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=sorted(y_test.unique()), yticklabels=sorted(y_test.unique()))
plt.title('Decision Tree Confusion Matrix', fontsize=16)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('True', fontsize=12)
plt.show()

y_prob = best_model_dt.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='#1a5174', lw=2, label='ROC curve(AUC Score={:.4f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='#e57373', lw=2, linestyle='--')
plt.xlabel('False Positive Rate',  fontsize=12)
plt.ylabel('True Positive Rate',  fontsize=12)
plt.title('ROC Curve for Decision Tree Model', fontsize=16)
plt.legend(loc='lower right', fontsize=12)
plt.show()

Feature importance for Decision Tree model.

feature_importance = best_model_dt.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, color='darkblue')
plt.title('Feature Importance')
plt.show()

Random Forest Classifier

rf_model = RandomForestClassifier(random_state=42)

parameters = {
    'max_depth': [2, 5, 8, 10],
    'max_features': [2, 5, 8],
    'n_estimators': [10, 100, 500],
    'min_samples_split': [2, 5, 10]
}

bp_rf = GridSearchCV(rf_model, parameters, n_jobs=-1, cv=10)
bp_rf.fit(X_train, y_train)
print('Best Parameters:', bp_rf.best_params_)
Best Parameters: {'max_depth': 10, 'max_features': 5, 'min_samples_split': 5, 'n_estimators': 500}
Classification metrics analysis for Random Forest model

best_model_rf = bp_rf.best_estimator_
best_model_rf.fit(X_train, y_train)
y_pred = best_model_rf.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred)
print(f'Test Accuracy of the Random Forest Model: {accuracy_rf}')

y_prob = best_model_rf.predict_proba(X_test)[:, 1]
auc_score_rf = roc_auc_score(y_test, y_prob)
print(f'AUC score of the Random Forest Model: {auc_score_rf}')
Test Accuracy of the Random Forest Model: 0.8087431693989071
AUC score of the Random Forest Model: 0.8678795291239871
print(classification_report(y_test, y_pred))
              precision    recall  f1-score   support

           0       0.76      0.80      0.78       155
           1       0.85      0.82      0.83       211

    accuracy                           0.81       366
   macro avg       0.80      0.81      0.81       366
weighted avg       0.81      0.81      0.81       366

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=sorted(y_test.unique()), yticklabels=sorted(y_test.unique()))
plt.title('Random Forest Confusion Matrix', fontsize=16)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('True', fontsize=12)
plt.show()

y_prob = best_model_rf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='#1a5174', lw=2, label='ROC curve(AUC Score={:.4f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='#e57373', lw=2, linestyle='--')
plt.xlabel('False Positive Rate',  fontsize=12)
plt.ylabel('True Positive Rate',  fontsize=12)
plt.title('ROC Curve for Random Forest Model', fontsize=16)
plt.legend(loc='lower right', fontsize=12)
plt.show()

Feature importance for Random Forest model.

feature_importance = best_model_rf.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, color='darkblue')
plt.title('Feature Importance')
plt.show()

Gradient Boosting Machines (GBM) Classifier Best Parameters for GBM model

gbm_model = GradientBoostingClassifier(random_state=42)

parameters = {'learning_rate': [0.005, 0.008, 0.1, 0.15],
              'n_estimators': [80, 100, 150, 200],
              'max_depth': [2, 3, 4],
              'min_samples_split': [2, 3, 4]}

bp_gbm = GridSearchCV(gbm_model, parameters, n_jobs=-1, cv=10)
bp_gbm.fit(X_train, y_train)
print('Best Parameters:', bp_gbm.best_params_)
Best Parameters: {'learning_rate': 0.15, 'max_depth': 4, 'min_samples_split': 3, 'n_estimators': 150}
Classification metrics analysis for GBM model

best_model_gbm = bp_gbm.best_estimator_
best_model_gbm.fit(X_train, y_train)
y_pred = best_model_gbm.predict(X_test)

accuracy_gbm = accuracy_score(y_test, y_pred)
print(f'Test Accuracy of the GBM Model: {accuracy_gbm}')

y_prob = best_model_gbm.predict_proba(X_test)[:, 1]
auc_score_gbm = roc_auc_score(y_test, y_prob)
print(f'AUC score of the GBM Model: {auc_score_gbm}')
Test Accuracy of the GBM Model: 0.7950819672131147
AUC score of the GBM Model: 0.8674820363858736
print(classification_report(y_test, y_pred))
              precision    recall  f1-score   support

           0       0.74      0.79      0.77       155
           1       0.84      0.80      0.82       211

    accuracy                           0.80       366
   macro avg       0.79      0.79      0.79       366
weighted avg       0.80      0.80      0.80       366

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=sorted(y_test.unique()), yticklabels=sorted(y_test.unique()))
plt.title('GBM Confusion Matrix', fontsize=16)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('True', fontsize=12)
plt.show()

y_prob = best_model_gbm.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='#1a5174', lw=2, label='ROC curve(AUC Score={:.4f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='#e57373', lw=2, linestyle='--')
plt.xlabel('False Positive Rate',  fontsize=12)
plt.ylabel('True Positive Rate',  fontsize=12)
plt.title('ROC Curve for GBM Model', fontsize=16)
plt.legend(loc='lower right', fontsize=12)
plt.show()

Comparing 5 different classification algorithms.

results_df = pd.DataFrame(columns=['Algorithm', 'Recall score'])

results = [
    ('KNN', accuracy_knn, auc_score_knn),
    ('Logistic Regression', accuracy_lr, auc_score_lr),
    ('Decision Tree', accuracy_dt, auc_score_dt),
    ('Random Forest', accuracy_rf, auc_score_rf),
    ('GBM', accuracy_gbm, auc_score_gbm)
]

results_df = pd.DataFrame(results, columns=['Algorithm', 'Accuracy score', 'AUC score'])
results_df.sort_values(by=['Accuracy score', 'AUC score'], ascending=False).reset_index(drop=True)
Algorithm	Accuracy score	AUC score
0	Random Forest	0.808743	0.867880
1	GBM	0.795082	0.867482
2	KNN	0.751366	0.809356
3	Logistic Regression	0.740437	0.804709
4	Decision Tree	0.718579	0.773032
From the results, it can be observed that Random forest showed the highest accuracy score among all algorithms (0.808743), as well as the highest area under the ROC curve (AUC score) (0.867880). This may mean that Random forest performs well and best ability to distinguish between positive and negative cases in the classification task.

GBM (Gradient Boosting Machine) and KNN have comparable accuracy scores to Random Forest, with 0.795082 and 0.751366 respectively. GBM has the second-highest AUC score among all algorithms, indicating it has the best ability to distinguish between positive and negative cases in the classification task.

Decision Tree has the lowest accuracy score of 0.718579 and the lowest AUC score of 0.773032, suggesting it might not generalize well to unseen data or capture the underlying patterns in the data as effectively as other algorithms.

While accuracy is an important metric, considering both accuracy and AUC score is crucial. AUC score provides a measure of the classifier's performance across all classification thresholds, making it useful for assessing the overall performance of the model.

In summary, while Random forest performs better in terms of accuracy and AUC, GBM shows promise with its second-highest which s slightly lower in AUC score than Random forest, indicating it might be better at correctly classifying instances in the dataset. However, the choice of the algorithm also depends on other factors like interpretability, computational efficiency, and the specific requirements of the problem at hand.

Using Unsupervised ML models

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier



models = {
    'Random Forest': RandomForestClassifier(),
    'Support Vector Classifier': SVC(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Gaussian Naive Bayes': GaussianNB(),
    'Multi-layer Perceptron': MLPClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Extra Trees': ExtraTreesClassifier(),
    'Linear SVM': LinearSVC()
}
model_accuracies = {}
from sklearn.metrics import accuracy_score, roc_auc_score

model_accuracies = {}
model_auc_scores = {}  # Dictionary to store AUC scores

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)  # Calculate AUC
    
    model_accuracies[name] = accuracy
    model_auc_scores[name] = auc  # Store AUC score

print("Model Accuracies:")
for name, accuracy in model_accuracies.items():
    print(f"{name}: {accuracy}")

print("\nModel AUC Scores:")
for name, auc in model_auc_scores.items():
    print(f"{name}: {auc}")
Model Accuracies:
Random Forest: 0.8142076502732241
Support Vector Classifier: 0.6338797814207651
Gradient Boosting: 0.7650273224043715
Gaussian Naive Bayes: 0.7240437158469946
Multi-layer Perceptron: 0.6967213114754098
AdaBoost: 0.7486338797814208
Extra Trees: 0.8114754098360656
Linear SVM: 0.5573770491803278

Model AUC Scores:
Random Forest: 0.8123222748815165
Support Vector Classifier: 0.595138358049228
Gradient Boosting: 0.7602507261886562
Gaussian Naive Bayes: 0.7306986699281456
Multi-layer Perceptron: 0.6898792233603424
AdaBoost: 0.7417520256841462
Extra Trees: 0.8099526066350712
Linear SVM: 0.6109769148448249
