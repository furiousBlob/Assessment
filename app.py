import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to fill missing headings
def fill_missing_headings(df):
    df['Section Heading'].ffill(inplace=True)
    df['Control Heading'].ffill(inplace=True)
    return df

def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Function to load and process the knowledge base
def load_knowledge_base(file):
    df = pd.read_excel(file, header=5)
    df['Section Heading'].ffill(inplace=True)
    df['Control Heading'].ffill(inplace=True)
    df = df[['Section Heading', 'Control Heading', 'Question Text', 'Answer']]
    df['Combined'] = df['Section Heading'] + ' ' + df['Control Heading'] + ' ' + df['Question Text']
    df['Question_Embedding'] = df['Combined'].apply(lambda x: get_embeddings(str(x)))
    df['Answer_Embedding'] = df['Answer'].apply(lambda x: get_embeddings(str(x)))
    return df

def process_knowledge_base(knowledge_base_df):
    knowledge_base = knowledge_base_df.copy()
    knowledge_base['Combined'] = knowledge_base['Section Heading'] + ' ' + knowledge_base['Control Heading'] + ' ' + knowledge_base['Question Text']
    knowledge_base['Question_Embedding'] = knowledge_base['Combined'].apply(lambda x: get_embeddings(str(x)))
    knowledge_base['Answer_Embedding'] = knowledge_base['Answer'].apply(lambda x: get_embeddings(str(x)))
    return knowledge_base

def classify_question(question_embedding, knowledge_base):
    similarities = knowledge_base['Question_Embedding'].apply(lambda x: cosine_similarity(x, question_embedding)[0][0])
    max_index = similarities.idxmax()
    max_similarity = similarities[max_index]
    if max_similarity > 0.85:
        classification = "Answerable"
    elif max_similarity < 0.7:
        classification = "Unanswerable"
    else:
        classification = "Ambiguous"
    return classification, max_similarity, knowledge_base.loc[max_index, 'Section Heading'], knowledge_base.loc[max_index, 'Control Heading']

# Streamlit UI
st.title('Question Classification Interface')

# Upload Knowledge Base
knowledge_base_file = st.file_uploader("Upload Knowledge Base File (Excel)", type=["xlsx"])
if knowledge_base_file:
    knowledge_base_df = load_knowledge_base(knowledge_base_file)
    knowledge_base_df = fill_missing_headings(knowledge_base_df)
    knowledge_base = process_knowledge_base(knowledge_base_df)
    st.write("Knowledge Base loaded successfully.")

    # Upload Questionnaire
    questionnaire_file = st.file_uploader("Upload Industry Standard Questionnaire (Excel)", type=["xlsx"])
    if questionnaire_file:
        questions_df = pd.read_excel(questionnaire_file, header=None)
        questions = questions_df[0].fillna("")
        questions_embeddings = questions.apply(lambda x: get_embeddings(str(x)))

        # Classify the questions
        classifications = questions_embeddings.apply(lambda x: classify_question(x, knowledge_base_df))
        questions_df = pd.DataFrame({
            'Question': questions,
            'Classification': classifications.apply(lambda x: x[0]),
            'Similarity': classifications.apply(lambda x: x[1]),
            'Section Heading': classifications.apply(lambda x: x[2]),
            'Control Heading': classifications.apply(lambda x: x[3])
        })

        # Display Results
        st.write("Classification Results:")
        st.dataframe(questions_df[['Question', 'Classification', 'Similarity', 'Section Heading', 'Control Heading']])
        
        # Calculate and display the percentage of answerable questions
        answerable_percentage = (questions_df['Classification'] == 'Answerable').mean() * 100
        st.write(f"Percentage of answerable questions: {answerable_percentage:.2f}%")

        # List unanswerable questions
        unanswerable_questions = questions_df[questions_df['Classification'] == 'Unanswerable']
        if not unanswerable_questions.empty:
            st.write("Unanswerable questions:")
            st.dataframe(unanswerable_questions[['Question', 'Similarity', 'Section Heading', 'Control Heading']])

    # Allow users to input and classify a random question
    st.write("Classify a New Question")
    user_question = st.text_input("Enter a question:")
    if user_question:
        user_question_embedding = get_embeddings(user_question)
        classification, similarity, section_heading, control_heading = classify_question(user_question_embedding, knowledge_base)
        st.write(f"Classification: {classification}")
        st.write(f"Similarity: {similarity:.2f}")

