import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
titanic_df = pd.read_csv('titanic.csv')

print(titanic_df.head())
print(titanic_df.isnull().sum())

titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)

titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0], inplace=True)

titanic_df.drop('Cabin', axis=1, inplace=True)
titanic_df['FamilySize'] = titanic_df['SibSp'] + titanic_df['Parch'] + 1

titanic_df['IsAlone'] = np.where(titanic_df['FamilySize'] == 1, 1, 0)
sns.histplot(titanic_df['Age'], bins=20, kde=True)
plt.title('Distribution of Age')
plt.show()

sns.countplot(x='Survived', data=titanic_df)
plt.title('Count of Survived and Not Survived')
plt.show()
sns.countplot(x='Pclass', hue='Survived', data=titanic_df)
plt.title('Survival by Pclass')
plt.show()

sns.countplot(x='Sex', hue='Survived', data=titanic_df)
plt.title('Survival by Sex')
plt.show()

sns.boxplot(x='Pclass', y='Fare', data=titanic_df)
plt.title('Fare distribution by Pclass')
plt.show()
correlation_matrix = titanic_df.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()