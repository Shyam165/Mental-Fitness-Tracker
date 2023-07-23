import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor

# Read datasets
df1 = pd.read_csv('mental-and-substance-use-as-share-of-disease.csv')
df2 = pd.read_csv('prevalence-by-mental-and-substance-use-disorder.csv')

# Merge datasets
data = pd.merge(df1, df2, on=['Entity', 'Code', 'Year'], how='inner')

# Data cleaning and preprocessing
data.drop(['Entity'], axis=1, inplace=True)
data.rename(columns={'DALYs (Disability-Adjusted Life Years) - Mental disorders - Sex: Both - Age: All Ages (Percent)': 'mental_fitness',
                     'Prevalence - Schizophrenia - Sex: Both - Age: Age-standardized (Percent)': 'Schizophrenia',
                     'Prevalence - Bipolar disorder - Sex: Both - Age: Age-standardized (Percent)': 'Bipolar_disorder',
                     'Prevalence - Eating disorders - Sex: Both - Age: Age-standardized (Percent)': 'Eating_disorder',
                     'Prevalence - Anxiety disorders - Sex: Both - Age: Age-standardized (Percent)': 'Anxiety',
                     'Prevalence - Drug use disorders - Sex: Both - Age: Age-standardized (Percent)': 'drug_usage',
                     'Prevalence - Depressive disorders - Sex: Both - Age: Age-standardized (Percent)': 'depression',
                     'Prevalence - Alcohol use disorders - Sex: Both - Age: Age-standardized (Percent)': 'alcohol'}, inplace=True)

# Convert non-numeric values to NaN
data = data.apply(pd.to_numeric, errors='coerce')

# Drop 'Code' column
data.drop(['Code'], axis=1, inplace=True)

# Impute missing values with the mean
imputer = SimpleImputer(strategy='mean')
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Exploratory analysis
# Correlation heatmap
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Pairplot with scatterplots
sns.pairplot(data=data, corner=True)
plt.show()

# Bar chart for average mental fitness by year
avg_mental_fitness_by_year = data.groupby('Year')['mental_fitness'].mean().reset_index()
fig = px.bar(data_frame=avg_mental_fitness_by_year, x='Year', y='mental_fitness', template='ggplot2')
fig.update_layout(title_text='Average Mental Fitness by Year', xaxis_title='Year', yaxis_title='Average Mental Fitness')
fig.show()

# Line chart for mental fitness over the years
fig = px.line(data_frame=data, x='Year', y='mental_fitness', color='Year', markers=True, template='plotly_dark')
fig.update_layout(title_text='Mental Fitness Over the Years', xaxis_title='Year', yaxis_title='Mental Fitness')
fig.show()

# Select a specific year for analysis
selected_year = 2019

# Filter the data for the selected year
data_selected_year = data[data['Year'] == selected_year]

# Melt the data to create a "mental disorder" column for better visualization
data_selected_year_melted = pd.melt(data_selected_year, id_vars=['Year', 'mental_fitness'],
                                   value_vars=['Schizophrenia', 'Bipolar_disorder', 'Eating_disorder',
                                               'Anxiety', 'drug_usage', 'depression', 'alcohol'],
                                   var_name='Mental_Disorder', value_name='Mental_Disorder_Percent')

# Create a box plot to show the distribution of mental fitness levels across different age groups
fig = px.box(data_frame=data_selected_year_melted, x='Mental_Disorder', y='mental_fitness',
             color='Mental_Disorder', points='all', title=f'Distribution of Mental Fitness Levels by Mental Disorder for {selected_year}',
             labels={'mental_fitness': 'Mental Fitness Level', 'Mental_Disorder_Percent': 'Mental Disorder Percentage'})
fig.show()

# Data preprocessing
df = data.copy()
encoder = LabelEncoder()

for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = encoder.fit_transform(df[column])

X = df.drop('mental_fitness', axis=1)
y = df['mental_fitness']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Linear Regression
lr = LinearRegression()
lr.fit(x_train, y_train)

# Model evaluation for Linear Regression
y_train_pred = lr.predict(x_train)
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_train_pred)

y_test_pred = lr.predict(x_test)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_test_pred)

print("Linear Regression Results:")
print("Train MSE: ", mse_train)
print("Train RMSE: ", rmse_train)
print("Train R2 Score: ", r2_train)
print("\n")
print("Test MSE: ", mse_test)
print("Test RMSE: ", rmse_test)
print("Test R2 Score: ", r2_test)
print("\n")

# Random Forest Regressor
rf = RandomForestRegressor()
rf.fit(x_train, y_train)

# Model evaluation for Random Forest Regressor
y_train_pred = rf.predict(x_train)
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_train_pred)

y_test_pred = rf.predict(x_test)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_test_pred)

print("Random Forest Regressor Results:")
print("Train MSE: ", mse_train)
print("Train RMSE: ", rmse_train)
print("Train R2 Score: ", r2_train)
print("\n")
print("Test MSE: ", mse_test)
print("Test RMSE: ", rmse_test)
print("Test R2 Score: ", r2_test)
print("\n")

# Support Vector Regressor
svr = SVR()
svr.fit(x_train, y_train)

# Model evaluation for Support Vector Regressor
y_train_pred = svr.predict(x_train)
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_train_pred)

y_test_pred = svr.predict(x_test)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_test_pred)

print("Support Vector Regressor Results:")
print("Train MSE: ", mse_train)
print("Train RMSE: ", rmse_train)
print("Train R2 Score: ", r2_train)
print("\n")
print("Test MSE: ", mse_test)
print("Test RMSE: ", rmse_test)
print("Test R2 Score: ", r2_test)
print("\n")

# Decision Tree Regressor
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)

# Model evaluation for Decision Tree Regressor
y_train_pred = dt.predict(x_train)
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_train_pred)

y_test_pred = dt.predict(x_test)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_test_pred)

print("Decision Tree Regressor Results:")
print("Train MSE: ", mse_train)
print("Train RMSE: ", rmse_train)
print("Train R2 Score: ", r2_train)
print("\n")
print("Test MSE: ", mse_test)
print("Test RMSE: ", rmse_test)
print("Test R2 Score: ", r2_test)
print("\n")

# Sample new data for prediction 
new_data = pd.DataFrame({
    'Year': [2022, 2022, 2023, 2023],
    'Schizophrenia': [10.5, 9.8, 8.7, 11.2],
    'Bipolar_disorder': [6.3, 7.1, 5.9, 6.8],
    'Eating_disorder': [3.2, 2.9, 3.5, 3.0],
    'Anxiety': [8.6, 7.9, 8.2, 9.0],
    'drug_usage': [4.7, 5.1, 4.5, 4.8],
    'depression': [12.4, 11.9, 13.2, 12.8],
    'alcohol': [5.8, 5.5, 6.1, 6.3]
})
# Data preprocessing for prediction
encoder = LabelEncoder()

for column in new_data.columns:
    if new_data[column].dtype == 'object':
        new_data[column] = encoder.fit_transform(new_data[column])

# Making predictions using the trained Random Forest Regressor model
predicted_mental_fitness = rf.predict(new_data)

# Create a new DataFrame to store the predictions along with other features (if needed)
predictions_df = pd.DataFrame(data={'Year': new_data['Year'], 'Predicted_Mental_Fitness': predicted_mental_fitness})

# Display the predictions
print(predictions_df)