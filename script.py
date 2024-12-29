# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('data/train.csv')

# 1. Data Cleaning

# Check for missing values
missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

# Handle missing values (Age: fill with median, Embarked: fill with mode)
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# 2. Data Exploration

# Summary statistics for numerical features
print("\nSummary Statistics:\n", data.describe())

# Distribution of 'Age' feature
plt.figure(figsize=(10, 6))
sns.histplot(data['Age'], kde=True)
plt.title('Distribution of Age')
plt.show()

# 3. Data Visualization

# Count plot of survival rate by sex
sns.countplot(x='Survived', hue='Sex', data=data)
plt.title('Survival Rate by Sex')
plt.show()

# Heatmap of correlations between numerical features
correlation = data.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# 4. Survival Analysis

# Survival rate by Pclass
sns.barplot(x='Pclass', y='Survived', data=data)
plt.title('Survival Rate by Passenger Class')
plt.show()

# Survival rate by Age group
data['AgeGroup'] = pd.cut(data['Age'], bins=[0, 12, 18, 60, 100], labels=['Child', 'Teen', 'Adult', 'Senior'])
sns.countplot(x='AgeGroup', hue='Survived', data=data)
plt.title('Survival Rate by Age Group')
plt.show()

# 5. Feature Engineering

# Create a new feature 'FamilySize'
data['FamilySize'] = data['SibSp'] + data['Parch']

# 6. Handling Categorical Variables

# Convert 'Sex' to numeric (0 = male, 1 = female)
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# Convert 'Embarked' to numeric (C = 0, Q = 1, S = 2)
data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# 7. Display the cleaned dataset
print("\nCleaned Data:\n", data.head())
