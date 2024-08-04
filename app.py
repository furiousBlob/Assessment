import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Load the knowledge base
knowledge_base_path = 'ClientABC _ ATB Financial_Knowledge Base.xlsx'
knowledge_base_df = pd.read_excel(knowledge_base_path, sheet_name='Data Sheet', header=5)
knowledge_base_cleaned = knowledge_base_df[['Section Heading', 'Control Heading', 'Question Text', 'Answer']].dropna(subset=['Question Text', 'Answer'])

# Combine features
knowledge_base_cleaned['Combined Features'] = knowledge_base_cleaned['Section Heading'] + ' ' + knowledge_base_cleaned['Control Heading'] + ' ' + knowledge_base_cleaned['Question Text']

knowledge_base_cleaned['Combined Features'] = knowledge_base_cleaned['Combined Features'].fillna('')


# Prepare data for training
X = knowledge_base_cleaned['Combined Features']
y = ['Answerable'] * len(X)  # All these entries are 'Answerable' since they exist in the knowledge base

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline for vectorization and model training
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

# Train the model
model.fit(X_train, y_train)

# Streamlit UI
st.title("Questionnaire Classification")

uploaded_file = st.file_uploader("Upload Industry Standard Questionnaire", type=["xlsx"])

if uploaded_file:
    questions_df = pd.read_excel(uploaded_file, sheet_name='Industry Standard Questionnaire').dropna().reset_index(drop=True)
    questions_df['Combined Features'] = questions_df.iloc[:, 0]  # Assuming the first column contains the questions
    questions_df['Classification'] = model.predict(questions_df['Combined Features'])

    total_questions = len(questions_df)
    answerable_questions = len(questions_df[questions_df['Classification'] == 'Answerable'])
    completion_percentage = (answerable_questions / total_questions) * 100
    unanswerable_questions = questions_df[questions_df['Classification'] == 'Unanswerable']

    st.write(f"Completion Percentage: {completion_percentage}%")
    st.write("Unanswerable Questions:")
    st.write(unanswerable_questions)
