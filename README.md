# Mental-Fitness-Tracker
Mental Health Fitness Tracker
The Mental Health Fitness Tracker project aims to analyze and predict mental fitness levels of individuals from various countries with different mental disorders. It utilizes regression techniques to provide insights into mental health and make predictions based on the available data. The project also offers a platform for users to track their mental health and fitness levels.

## Table of Contents
Introduction
Installation
Usage
Exploratory Analysis
Data Preprocessing
Model Building and Evaluation
Contributing
License
References
## Introduction
Mental health is a critical aspect of overall well-being, and its significance has been increasingly recognized in recent times. The Mental Health Fitness Tracker project is an initiative to better understand the relationship between mental disorders and mental fitness levels. By analyzing and predicting mental fitness levels using regression techniques, this project aims to provide valuable insights to researchers, healthcare professionals, and individuals seeking to improve their mental health.

The project utilizes two datasets: "mental-and-substance-use-as-share-of-disease.csv" and "prevalence-by-mental-and-substance-use-disorder.csv" sourced from Kaggle. These datasets contain information about mental disorders and their prevalence across different countries and years. By merging and preprocessing the data, we create a comprehensive dataset to perform exploratory analysis and build regression models for prediction.

## Installation
To use the code and run the examples, follow these steps:

Ensure that you have Python 3.x installed on your system.
Install the required libraries by running the following command:
Copy code
pip install pandas numpy seaborn matplotlib plotly scikit-learn
Download the project files and navigate to the project directory.
## Usage
Make sure you have the necessary datasets: "mental-and-substance-use-as-share-of-disease.csv" and "prevalence-by-mental-and-substance-use-disorder.csv".
Modify the code in "mental_health_fitness_tracker.py" to provide the correct file paths for the datasets.
Run the "mental_health_fitness_tracker.py" file to execute the program.
The program will prompt you to select the country, mental disorder, and year range for analysis. It will display the results of the analysis, including visualizations of the data. You can also track your own mental fitness level by providing your data through the console prompts. The program will provide insights and predictions based on the input data.

## Exploratory Analysis
The project begins with exploratory analysis to understand the data and identify patterns. Various visualizations are used to gain insights into the mental health data:

Pairplot: A pairplot with scatterplots is used to visualize the relationships between different features and mental fitness levels. It helps in identifying potential correlations and trends.

Correlation Heatmap: A heatmap is generated to show the correlation between different features. This helps in identifying strong positive or negative correlations between variables.

Box Plot: To understand the distribution of mental fitness levels across different mental disorders for a specific year, a box plot is created. This provides insights into the spread of mental fitness levels and possible outliers.

Bar Chart: The average mental fitness level is plotted against each year to observe trends over time.


Line Chart: A line chart is used to visualize the trend of mental fitness levels over the years. This helps in understanding the overall pattern of mental fitness across time.

## Data Preprocessing
Before building the predictive models, the data is preprocessed to handle missing values and convert non-numeric values to numeric using label encoding. Missing values are imputed with the mean of each feature.

## Model Building and Evaluation
### Model Building
Three regression models are implemented to predict mental fitness levels:

Linear Regression: A simple linear regression model is used as the baseline.

Random Forest Regressor: A powerful ensemble learning model, RandomForestRegressor, is used to capture complex relationships in the data.

Support Vector Regressor: Support Vector Regressor (SVR) is used to handle non-linear relationships between features and the target variable.

Decision Tree Regressor: A Decision Tree Regressor is included as an additional non-gradient boosting model.

### Model Evaluation
The models are evaluated on both the training and test datasets using mean squared error (MSE), root mean squared error (RMSE), and R-squared (R2) score.

### Predictions on New Data
The trained Random Forest Regressor model is used to make predictions on new data. The predictions are stored in a new DataFrame, which includes the year and the corresponding predicted mental fitness level.

## Contributing
Contributions to the Mental Health Fitness Tracker project are welcome. Feel free to open issues or submit pull requests for enhancements or bug fixes.

To contribute:

Fork the repository.
Clone the project to your local machine.
Create a new branch.
Make your changes and enhancements.
Commit your changes.
Push your branch to the forked repository.
Open a pull request.
## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## References
Datasets used in this project provided by mentors of IBM SkillBild and we also take references from Kaggle.
This project was made during my internship period for Edunet Foundation in association with IBM SkillsBuild and AICTE.
Feel free to customize the code snippets or add more explanations to provide specific examples and highlight the important parts of your project. The Mental Health Fitness Tracker aims to contribute to the field of mental health research and provide valuable insights to aid in promoting better mental well-being.
