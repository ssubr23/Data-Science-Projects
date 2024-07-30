import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
reviews_df = pd.read_csv('./amazon_alexa.tsv', sep='\t')

# Display the DataFrame
print(reviews_df)

# Information about the DataFrame
print(reviews_df.info())

# Summary statistics of the DataFrame
print(reviews_df.describe())

# Display the 'verified_reviews' column
print(reviews_df['verified_reviews'])

# Visualize missing values in the DataFrame
sns.heatmap(reviews_df.isnull(), yticklabels=False, cbar=False, cmap="Blues")

# Plot histograms for the DataFrame
reviews_df.hist(bins=30, figsize=(13, 5), color='r')

# Add a column for the length of the reviews
reviews_df['length'] = reviews_df['verified_reviews'].apply(len)
print(reviews_df.head())

# Plot the length of reviews
reviews_df['length'].plot(bins=100, kind='hist')
print(reviews_df['length'].describe())

# Display reviews based on their length
print(reviews_df[reviews_df['length'] == reviews_df['length'].max()]['verified_reviews'].iloc[0])
print(reviews_df[reviews_df['length'] == reviews_df['length'].min()]['verified_reviews'].iloc[0])
print(reviews_df[reviews_df['length'] == reviews_df['length'].mean()]['verified_reviews'].iloc[0])

# Separate positive and negative reviews
positive = reviews_df[reviews_df['feedback'] == 1]
negative = reviews_df[reviews_df['feedback'] == 0]

# Display positive and negative reviews
print(positive)
print(negative)

# Plot feedback count
sns.countplot(reviews_df['feedback'], label="Count")

# Plot rating count
sns.countplot(x='rating', data=reviews_df)

# Plot rating histogram
reviews_df['rating'].hist(bins=5)

# Bar plot for variation vs. rating
plt.figure(figsize=(40, 15))
sns.barplot(x='variation', y='rating', data=reviews_df, palette='deep')

# Convert verified reviews to a list of sentences
sentences = reviews_df['verified_reviews'].tolist()
print(len(sentences))
print(sentences)

# Combine all sentences into one string
sentences_as_one_string = " ".join(sentences)
print(sentences_as_one_string)

# Generate and display a word cloud
plt.figure(figsize=(20, 20))
plt.imshow(WordCloud().generate(sentences_as_one_string))

# Generate a word cloud for negative reviews
negative_list = negative['verified_reviews'].tolist()
negative_sentences_as_one_string = " ".join(negative_list)

plt.figure(figsize=(20, 20))
plt.imshow(WordCloud().generate(negative_sentences_as_one_string))

# Drop unnecessary columns
reviews_df.drop(['date', 'rating', 'length'], axis=1, inplace=True)

# One-hot encode the 'variation' column
variation_dummies = pd.get_dummies(reviews_df['variation'], drop_first=True)
reviews_df.drop(['variation'], axis=1, inplace=True)
reviews_df = pd.concat([reviews_df, variation_dummies], axis=1)

# Function to clean messages by removing punctuation and stopwords
def message_cleaning(message):
    test_punc_removed = [char for char in message if char not in string.punctuation]
    test_punc_removed_join = ''.join(test_punc_removed)
    test_punc_removed_join_clean = [word for word in test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
    return test_punc_removed_join_clean

# Apply the cleaning function to the reviews
reviews_df_clean = reviews_df['verified_reviews'].apply(message_cleaning)

# Display the cleaned and original versions of a review
print(reviews_df_clean[3])
print(reviews_df['verified_reviews'][3])

# Vectorize the cleaned reviews
vectorizer = CountVectorizer(analyzer=message_cleaning)
reviews_countvectorizer = vectorizer.fit_transform(reviews_df['verified_reviews'])

# Display feature names and the vectorized data
print(vectorizer.get_feature_names())
print(reviews_countvectorizer.toarray())
print(reviews_countvectorizer.shape)

# Drop the 'verified_reviews' column and concatenate the vectorized data
reviews_df.drop(['verified_reviews'], axis=1, inplace=True)
reviews = pd.DataFrame(reviews_countvectorizer.toarray())
reviews_df = pd.concat([reviews_df, reviews], axis=1)

# Separate features and target variable
X = reviews_df.drop(['feedback'], axis=1)
y = reviews_df['feedback']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Evaluate the classifier on the training set
y_predict_train = nb_classifier.predict(X_train)
cm_train = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm_train, annot=True)

# Evaluate the classifier on the test set
y_predict_test = nb_classifier.predict(X_test)
cm_test = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm_test, annot=True)
print(classification_report(y_test, y_predict_test))

# Train a Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Predict on the test set
y_pred = logistic_model.predict(X_test)
print(f'Accuracy: {100 * accuracy_score(y_pred, y_test)}%')

# Display confusion matrix and classification report for Logistic Regression
cm_logistic = confusion_matrix(y_pred, y_test)
sns.heatmap(cm_logistic, annot=True)
print(classification_report(y_test, y_pred))
