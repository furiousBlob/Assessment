Question Classification Interface

This Streamlit app classifies questions based on a provided knowledge base using BERT embeddings and cosine similarity. The app allows users to upload a knowledge base and a set of questions, classify those questions, and display results interactively.

Prerequisites
Before running the app locally, ensure you have the following packages installed:

pandas
streamlit
scikit-learn
transformers
torch
openpyxl (for reading Excel files)

You can install these packages using pip:

pip install pandas streamlit scikit-learn transformers torch openpyxl


Accessing the App
The app is hosted online and can be accessed at:

https://assessment-1v6x.onrender.com/

Using the App
1. Upload Knowledge Base
Click the "Upload Knowledge Base File (Excel)" button.
Select an Excel file containing the knowledge base. The file should have columns for Section Heading, Control Heading, Question Text, and Answer.
The app will automatically process the file and extract relevant information.
2. Upload Industry Standard Questionnaire
Click the "Upload Industry Standard Questionnaire (Excel)" button.
Select an Excel file containing the questions to be classified. The file should have a single column with questions.
The app will classify each question based on its similarity to the questions in the knowledge base.
3. View Results
The app displays the classification results for each question, including:

Question: The original question.
Classification: Whether the question is Answerable, Ambiguous, or Unanswerable.
Similarity: The cosine similarity score indicating how closely the question matches the knowledge base.
Section Heading: The section heading from the knowledge base related to the question.
Control Heading: The control heading from the knowledge base related to the question.
The app also calculates and displays the percentage of answerable questions.

Unanswerable questions are listed separately if any are identified.

4. Classify a New Question
Enter a question in the text input box under "Classify a New Question".
The app will classify the new question and display the result, including its classification and similarity score.
File Format
Knowledge Base File
The knowledge base Excel file should have the following columns:

Section Heading: The heading of the section in the knowledge base.
Control Heading: The control heading in the knowledge base.
Question Text: The question text for classification.
Answer: The answer to the question.
Questionnaire File
The questionnaire Excel file should have a single column with the questions to be classified.

Troubleshooting
Ensure that the Excel files are correctly formatted and do not contain any merged cells or irregularities.
If you encounter errors, check the browser console or the terminal output for error messages, which can provide clues for debugging.