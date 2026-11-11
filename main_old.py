# 0. Import All Required Libraries.
import numpy as np
import pandas as pd 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

# 1. Load the main dataset csv file.
data = pd.read_csv('housing.csv')

# 2. Create a stratified Train & Test set.
data['income_cat'] = pd.cut(data['median_house_value'],
                    bins = [0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                    labels = [1, 2, 3, 4, 5]
)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42 )
for train_index, test_index in split.split(data,data['income_cat']):
    strat_train_set = data.loc[train_index].drop('income_cat', axis=1)
    strat_test_set = data.loc[test_index].drop('income_cat', axis=1)

# Working on copy of training set.
data_copy = strat_train_set.copy()

# 3. Separate Features and Labels.

data_copy_labels = data_copy['median_house_value'].copy()
data_copy = data_copy.drop('median_house_value', axis=1)
# print(data_copy,data_copy_labels)

# 4. Separate numerical and categorical columns.

num_attribute = data_copy.drop('ocean_proximity', axis=1).columns.tolist()
cat_attribute = ['ocean_proximity']

# 5. Make the pipelines.

# For numeric data.
num_pipeline  = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scaler',StandardScaler())
])

# For categoric data.
cat_pipeline = Pipeline([
    ('one_hot', OneHotEncoder(handle_unknown='ignore'))
])

# let's Construct Finall pipeline. 

final_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribute),
    ('cat', cat_pipeline, cat_attribute)
])

# 6. Transform the data. 

data_prepared = final_pipeline.fit_transform(data_copy)
# print(data_prepared.shape)

# 7. Train the Model
# LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(data_prepared, data_copy_labels)
lin_predict = lin_reg.predict(data_prepared)
# rmse = root_mean_squared_error(data_copy_labels, lin_predict)
lin_rmses = -cross_val_score(lin_reg, data_prepared, data_copy_labels, scoring="neg_root_mean_squared_error", cv=10)
# print(f"The Root Mean Square Error for Linear Regression is : {lin_rmses}")
print("Decision Tree CV RMSEs:", lin_rmses)
print("\nCross-Validation Performance (LinearRegression):")
print(pd.Series(lin_rmses).describe())

# Decision Tree Regressor
dec_reg = DecisionTreeRegressor()
dec_reg.fit(data_prepared, data_copy_labels)
dec_predict = dec_reg.predict(data_prepared)
# rmse = root_mean_squared_error(data_copy_labels, dec_predict)
dec_rmses = -cross_val_score(dec_reg, data_prepared, data_copy_labels, scoring="neg_root_mean_squared_error", cv=10)
# print(f"The Root Mean Square Error for Decision Tree Regressor is : {dec_rmses}")
print("Decision Tree CV RMSEs:", dec_rmses)
print("\nCross-Validation Performance (DecisionTree):")
print(pd.Series(dec_rmses).describe())

# Random Forest Regressor
ran_reg = RandomForestRegressor()
ran_reg.fit(data_prepared, data_copy_labels)
ran_predict = ran_reg.predict(data_prepared)
# rmse = root_mean_squared_error(data_copy_labels, ran_predict)
ran_rmses = -cross_val_score(ran_reg, data_prepared, data_copy_labels, scoring="neg_root_mean_squared_error", cv=10)
# print(f"The Root Mean Square Error for Random Forest Regressor is : {ran_rmses}")
print("Decision Tree CV RMSEs:", ran_rmses)
print("\nCross-Validation Performance (RandomForest):")
print(pd.Series(ran_rmses).describe())
