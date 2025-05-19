# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("/content/bmi.csv")
df.head()
```
![image](https://github.com/user-attachments/assets/25bfa566-992f-4294-9688-f15fcbafee42)
```
df.dropna()
```
![image](https://github.com/user-attachments/assets/ea55b8be-d569-44b3-9ca1-5e6a6642cfa3)
```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![image](https://github.com/user-attachments/assets/05b10b1d-8d1a-4d74-b92b-cd7832da9d36)
```
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df=pd.read_csv("/content/bmi.csv")
df[['Height', 'Weight']] = sc.fit_transform(df[['Height', 'Weight']])
print(df.head(10))
```
![image](https://github.com/user-attachments/assets/413bbaa4-24d9-4f87-9652-6e2329cfc2e1)
```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/30afbd2b-18b9-40e2-b671-cc1676e0ba82)
```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/38284e6b-fe76-47cc-b55c-a5e05f3e5783)
```
df3=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3
```
![image](https://github.com/user-attachments/assets/6a65d223-51ea-4249-a978-425778d1b1c2)
```
df4=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4.head()
```
![image](https://github.com/user-attachments/assets/beaee8fd-a5d2-475b-a16a-cadd9ed30b91)
```
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/user-attachments/assets/e79dc660-97ad-4f41-852b-c918a5f90aa7)
```
contigency_table=pd.crosstab(tips['sex'],tips['time'])
print(contigency_table)
```
![image](https://github.com/user-attachments/assets/f5cb5085-1c22-432e-bcb4-79af6af6cc38)
```
chi2,p, _, _ = chi2_contingency(contigency_table)
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p}")
```
![image](https://github.com/user-attachments/assets/cb887f8d-68ea-42d2-865c-ce1e45ad1f08)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif

data = {
    'Feature1': [1, 2, 3, 4, 5],
    'Feature2': ['A', 'B', 'C', 'A', 'B'],
    'Feature3': [0, 1, 1, 0, 1],
    'Target':   [0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)
x = df[['Feature1', 'Feature3']]
y = df['Target']
selector = SelectKBest(score_func=mutual_info_classif, k=1)
X_new = selector.fit_transform(x, y)
selected_feature_indices = selector.get_support(indices=True)
selected_features = x.columns[selected_feature_indices]

print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/e8655349-8666-4498-b1da-0dd5f059b309)







# RESULT:
    Thus we performed Feature Scaling and Feature Selection process
