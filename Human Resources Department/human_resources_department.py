import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf

# Load the dataset
employee_df = pd.read_csv('./Human_Resources.csv')

# Display the first and last few rows of the dataset
print(employee_df.head(5))
print(employee_df.tail(10))

# Display basic information and statistical summary of the dataset
print(employee_df.info())
print(employee_df.describe())

# Convert categorical columns to binary values
employee_df['Attrition'] = employee_df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
employee_df['OverTime'] = employee_df['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)
employee_df['Over18'] = employee_df['Over18'].apply(lambda x: 1 if x == 'Y' else 0)

print(employee_df.head(4))

# Visualize missing values in the dataset
sns.heatmap(employee_df.isnull(), yticklabels=False, cbar=False, cmap="Blues")

# Visualize the distribution of numerical features
employee_df.hist(bins=30, figsize=(20, 20), color='r')

# Drop irrelevant columns
employee_df.drop(['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], axis=1, inplace=True)

# Separate the dataset into employees who left and those who stayed
left_df = employee_df[employee_df['Attrition'] == 1]
stayed_df = employee_df[employee_df['Attrition'] == 0]

print("Total number of employees =", len(employee_df))
print("Number of employees who left the company =", len(left_df))
print("Percentage of employees who left the company = {:.2f}%".format(len(left_df) / len(employee_df) * 100))
print("Number of employees who stayed with the company =", len(stayed_df))
print("Percentage of employees who stayed with the company = {:.2f}%".format(len(stayed_df) / len(employee_df) * 100))

# Display statistical summary for employees who left and those who stayed
print(left_df.describe())
print(stayed_df.describe())

# Compute the correlation matrix
correlations = employee_df.corr()
plt.figure(figsize=(20, 20))
sns.heatmap(correlations, annot=True)

# Visualize the distribution of Attrition by various features
plt.figure(figsize=[25, 12])
sns.countplot(x='Age', hue='Attrition', data=employee_df)

plt.figure(figsize=[20, 20])
plt.subplot(411)
sns.countplot(x='JobRole', hue='Attrition', data=employee_df)
plt.subplot(412)
sns.countplot(x='MaritalStatus', hue='Attrition', data=employee_df)
plt.subplot(413)
sns.countplot(x='JobInvolvement', hue='Attrition', data=employee_df)
plt.subplot(414)
sns.countplot(x='JobLevel', hue='Attrition', data=employee_df)

# KDE plots to visualize the probability density of continuous variables
plt.figure(figsize=(12, 7))
sns.kdeplot(left_df['DistanceFromHome'], label='Employees who left', shade=True, color='r')
sns.kdeplot(stayed_df['DistanceFromHome'], label='Employees who stayed', shade=True, color='b')
plt.xlabel('Distance From Home')

plt.figure(figsize=(12, 7))
sns.kdeplot(left_df['YearsWithCurrManager'], label='Employees who left', shade=True, color='r')
sns.kdeplot(stayed_df['YearsWithCurrManager'], label='Employees who stayed', shade=True, color='b')
plt.xlabel('Years With Current Manager')

plt.figure(figsize=(12, 7))
sns.kdeplot(left_df['TotalWorkingYears'], shade=True, label='Employees who left', color='r')
sns.kdeplot(stayed_df['TotalWorkingYears'], shade=True, label='Employees who stayed', color='b')
plt.xlabel('Total Working Years')

# Box plots to visualize the distribution of Monthly Income by Gender and Job Role
plt.figure(figsize=(15, 10))
sns.boxplot(x='MonthlyIncome', y='Gender', data=employee_df)

plt.figure(figsize=(15, 10))
sns.boxplot(x='MonthlyIncome', y='JobRole', data=employee_df)

print(employee_df.head(3))

# One-hot encode categorical variables
X_cat = employee_df[['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']]
onehotencoder = OneHotEncoder()
X_cat = onehotencoder.fit_transform(X_cat).toarray()
X_cat = pd.DataFrame(X_cat)

# Select numerical features
X_numerical = employee_df[['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 'HourlyRate',
                           'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate',
                           'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike', 'PerformanceRating',
                           'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
                           'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
                           'YearsWithCurrManager']]

# Combine categorical and numerical features
X_all = pd.concat([X_cat, X_numerical], axis=1)

# Normalize the features
scaler = MinMaxScaler()
X = scaler.fit_transform(X_all)

# Define the target variable
y = employee_df['Attrition']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Train and evaluate a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Logistic Regression Accuracy: {:.2f}%".format(100 * accuracy_score(y_pred, y_test)))

cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot=True)
print(classification_report(y_test, y_pred))

# Train and evaluate a Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot=True)
print(classification_report(y_test, y_pred))

# Define and train a Neural Network using TensorFlow
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=500, activation='relu', input_shape=(X_train.shape[1],)))
model.add(tf.keras.layers.Dense(units=500, activation='relu'))
model.add(tf.keras.layers.Dense(units=500, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

model.summary()

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

epochs_hist = model.fit(X_train, y_train, epochs=100, batch_size=50)

# Predict and evaluate the Neural Network model
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

plt.plot(epochs_hist.history['loss'])
plt.title('Model Loss Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.legend(['Training Loss'])

plt.plot(epochs_hist.history['accuracy'])
plt.title('Model Accuracy Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Training Accuracy')
plt.legend(['Training Accuracy'])

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)
print(classification_report(y_test, y_pred))
