SmartText Studio – AI Text Normalization Tool

SmartText Studio is an AI-powered text normalization tool built with Python and Streamlit.
It converts messy, unstructured text into clean, consistent, and AI-ready text – perfect for Machine Learning, NLP, chatbots, search systems, or text analytics.

What It Does

SmartText Studio processes raw text to make it:

Clean

Consistent

AI-ready

Use Cases:

Preprocessing text for NLP & Machine Learning

Chatbots & conversational AI

Search & text analytics

Data cleaning for analytics pipelines

How It Works – The 4-Step Pipeline

SmartText Studio follows four main steps to clean text:

Case Folding (Lowercasing)
Converts all text to lowercase to ensure consistency.
Example: HELLO World → hello world

Accent / Diacritic Removal
Removes accented letters like é, ñ, ü.
Example: café → cafe, São → Sao

Noise Removal (Text Cleaning)
Removes punctuation, emojis, extra spaces, and symbols.
Example: Hello! Let's meet @ 3pm 😊 → hello lets meet 3pm

Lemmatization (Word Base Form)
Converts words to their root form so different variations are treated the same.
Example: running → run, cats → cat, better → good

Result: Text that is clean, consistent, and AI-ready for NLP or Machine Learning.

Features

Lowercase conversion for text consistency

Accent & diacritic removal

Noise cleaning (punctuation, emojis, symbols)

Lemmatization using spaCy & NLTK

WordCloud visualization of processed text

Simple Streamlit interface – no coding knowledge needed

Tech Stack

Python 3.10+ – Programming language

Streamlit – Web interface

NLTK & spaCy – NLP and lemmatization

Unidecode – Remove accents

WordCloud – Text visualization

NumPy, Scikit-learn, Pillow – Supporting libraries

Project Structure

SmartTextStudio/

app.py – Streamlit application

main.py – Text processing logic

requirements.txt – Python dependencies

README.md – Project documentation

.streamlit/ – Streamlit configuration (config.toml)

venv/ – Python virtual environment (not uploaded)

How to Run the Project (Step-by-Step)

Open the project folder.
cd SmartTextStudio

Create a virtual environment.
python -m venv venv

Activate the virtual environment.
Windows: venv\Scripts\activate

Install required dependencies.
pip install -r requirements.txt

Download spaCy English model (first time only).
python -m spacy download en_core_web_sm

Run the Streamlit app.
streamlit run app.py

Open your browser:
http://localhost:8501

Example Input & Output

Input:
Hello WORLD! café running better cats 😊

Output:
hello world cafe run good cat
