import streamlit as st
import nltk
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import re
from unidecode import unidecode
import string
import io
import zipfile
from datetime import datetime

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)

if 'processing_history' not in st.session_state:
    st.session_state.processing_history = []

if 'custom_stopwords' not in st.session_state:
    st.session_state.custom_stopwords = set()

@st.cache_resource
def load_spacy_model():
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
        nlp = spacy.load("en_core_web_sm")
    return nlp

def case_folding(text):
    return text.lower()

def remove_punctuation_emojis_numbers(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def accent_removal(text):
    return unidecode(text)

def remove_stopwords(text, stopwords):
    if not stopwords:
        return text
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stopwords]
    return ' '.join(filtered_words)

def lemmatize_text(text, nlp):
    doc = nlp(text)
    lemmatized = ' '.join([token.lemma_ for token in doc if not token.is_space])
    return lemmatized

def stem_text(text):
    stemmer = PorterStemmer()
    words = text.split()
    stemmed = ' '.join([stemmer.stem(word) for word in words if word])
    return stemmed

def process_text_pipeline(text, nlp, use_stopwords=False, stopwords=None, use_stemming=False):
    text = case_folding(text)
    text = accent_removal(text)
    text = remove_punctuation_emojis_numbers(text)
    
    if use_stopwords and stopwords:
        text = remove_stopwords(text, stopwords)
    
    if use_stemming:
        text = stem_text(text)
    else:
        text = lemmatize_text(text, nlp)
    
    return text

def process_text_step_by_step(text, nlp, use_stopwords=False, stopwords=None, use_stemming=False):
    steps = []
    
    # Step 1: Original
    steps.append({
        'step': 'Original Text',
        'description': 'Input text as entered by user',
        'result': text
    })
    
    # Step 2: Case Folding
    text = case_folding(text)
    steps.append({
        'step': 'Step 1: Case Folding',
        'description': 'All characters converted to lowercase',
        'result': text
    })
    
    # Step 3: Accent Removal
    text = accent_removal(text)
    steps.append({
        'step': 'Step 2: Accent Removal',
        'description': 'Accented characters normalized (Héllò → hello)',
        'result': text
    })
    
    # Step 4: Text Normalization
    text = remove_punctuation_emojis_numbers(text)
    steps.append({
        'step': 'Step 3: Text Normalization',
        'description': 'Removed punctuation, emojis, numbers, and extra spaces',
        'result': text
    })
    
    # Step 5: Stopword Removal (optional)
    if use_stopwords and stopwords:
        text = remove_stopwords(text, stopwords)
        steps.append({
            'step': 'Step 4: Stopword Removal',
            'description': 'Removed custom stopwords',
            'result': text
        })
    
    # Step 6: Lemmatization or Stemming
    if use_stemming:
        text = stem_text(text)
        steps.append({
            'step': 'Step ' + ('5' if use_stopwords else '4') + ': Stemming',
            'description': 'Words reduced to root form using stemming (running → run)',
            'result': text
        })
    else:
        text = lemmatize_text(text, nlp)
        steps.append({
            'step': 'Step ' + ('5' if use_stopwords else '4') + ': Lemmatization',
            'description': 'Words reduced to base form using lemmatization (running → run)',
            'result': text
        })
    
    # Final result
    steps.append({
        'step': 'Final Result',
        'description': 'Complete preprocessed text ready for analysis',
        'result': text
    })
    
    return steps, text

def get_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    return scores

def create_wordcloud(text, title):
    if len(text.strip()) == 0:
        return None
    wordcloud = WordCloud(width=800, height=400, background_color='white', 
                          colormap='viridis').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold')
    return fig

def create_frequency_chart(text, title, top_n=10):
    if len(text.strip()) == 0:
        return None
    words = text.split()
    word_freq = Counter(words)
    most_common = word_freq.most_common(top_n)
    
    if not most_common:
        return None
    
    df = pd.DataFrame(data=most_common, columns=['Word', 'Frequency'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df['Word'], df['Frequency'], color='skyblue')
    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_ylabel('Words', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    return fig

def create_sentiment_comparison_chart(original_scores, cleaned_scores):
    labels = ['Negative', 'Neutral', 'Positive', 'Compound']
    original_values = [original_scores['neg'], original_scores['neu'], 
                       original_scores['pos'], original_scores['compound']]
    cleaned_values = [cleaned_scores['neg'], cleaned_scores['neu'], 
                      cleaned_scores['pos'], cleaned_scores['compound']]
    
    x = range(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([i - width/2 for i in x], original_values, width, label='Original', color='coral')
    ax.bar([i + width/2 for i in x], cleaned_values, width, label='Cleaned', color='lightblue')
    
    ax.set_xlabel('Sentiment Type', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Sentiment Analysis Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    return fig

def add_to_history(original_text, cleaned_text, method):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.processing_history.insert(0, {
        'timestamp': timestamp,
        'original': original_text[:100] + '...' if len(original_text) > 100 else original_text,
        'cleaned': cleaned_text,
        'method': method
    })
    if len(st.session_state.processing_history) > 10:
        st.session_state.processing_history.pop()

def process_batch_files(uploaded_files, nlp, use_stopwords, stopwords, use_stemming):
    results = []
    for uploaded_file in uploaded_files:
        try:
            text_content = uploaded_file.read().decode('utf-8')
            cleaned = process_text_pipeline(text_content, nlp, use_stopwords, stopwords, use_stemming)
            results.append({
                'filename': uploaded_file.name,
                'original': text_content,
                'cleaned': cleaned
            })
        except Exception as e:
            results.append({
                'filename': uploaded_file.name,
                'original': '',
                'cleaned': f'Error processing file: {str(e)}'
            })
    return results

st.set_page_config(page_title="SmartText Studio", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1F77B4;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .process-step {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        background-color: #f0f2f6;
        border-radius: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-header">🧠 SmartText Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Team 3: Text Normalization & NLP Processing</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("---")
    
    # Processing Mode
    st.subheader("🔧 Processing Mode")
    processing_mode = st.radio("Select Mode:", ["Single Text", "Batch Upload"], index=0)
    
    st.markdown("---")
    st.subheader("🛠️ Advanced Options")
    
    # Stopword removal
    use_stopwords = st.checkbox("Enable Stopword Removal", value=False)
    if use_stopwords:
        stopwords_input = st.text_input(
            "Enter stopwords (comma-separated):",
            value="the,a,an,is,are",
            help="Enter words to remove from text, separated by commas"
        )
        custom_stopwords = set([w.strip().lower() for w in stopwords_input.split(',') if w.strip()])
    else:
        custom_stopwords = set()
    
    # Stemming vs Lemmatization
    use_stemming = st.checkbox("Use Stemming instead of Lemmatization", value=False)
    if use_stemming:
        st.info("ℹ️ Stemming uses Porter Stemmer algorithm")
    else:
        st.info("ℹ️ Lemmatization uses spaCy (more accurate)")
    
    st.markdown("---")
    st.subheader("📊 Visualization Options")
    show_steps = st.checkbox("Show Step-by-Step Processing", value=True)
    show_wordcloud = st.checkbox("Show Word Clouds", value=True)
    show_frequency = st.checkbox("Show Frequency Charts", value=True)
    show_sentiment = st.checkbox("Show Sentiment Analysis", value=True)
    
    if show_frequency:
        top_n_words = st.slider("Top N Words in Frequency Chart", 5, 20, 10)
    else:
        top_n_words = 10
    
    st.markdown("---")
    st.subheader("📜 Processing History")
    if st.session_state.processing_history:
        st.write(f"**{len(st.session_state.processing_history)}** items in history")
        if st.button("Clear History"):
            st.session_state.processing_history = []
            st.rerun()
    else:
        st.write("No history yet")
    
    st.markdown("---")
    st.subheader("ℹ️ Processing Steps")
    st.markdown("""
    **1. Case Folding**: Converts to lowercase
    
    **2. Accent Removal**: Normalizes characters
    
    **3. Text Normalization**: Removes punctuation, emojis, numbers
    
    **4. Stopword Removal**: Removes common words (optional)
    
    **5. Lemmatization/Stemming**: Reduces to base form
    """)

# Main Content Area
if processing_mode == "Single Text":
    st.markdown("### 📝 Enter Your Text")
    user_text = st.text_area(
        "Type or paste your text below:",
        height=150,
        placeholder="Example: Héllooo!!! I AM Running So Fast 😍😂...",
        help="Enter any text you want to process through NLP normalization"
    )

    if st.button("🚀 Preprocess Text", type="primary"):
        if user_text.strip():
            with st.spinner("Loading NLP model..."):
                nlp = load_spacy_model()
            
            with st.spinner("Processing your text..."):
                steps, cleaned_text = process_text_step_by_step(user_text, nlp, use_stopwords, custom_stopwords, use_stemming)
            
            # Add to history
            method = "Stemming" if use_stemming else "Lemmatization"
            if use_stopwords:
                method += " + Stopwords"
            add_to_history(user_text, cleaned_text, method)
            
            st.success("✅ Text processing complete!")
            
            # Show Step-by-Step Processing
            if show_steps:
                st.markdown("### 🔄 Step-by-Step Processing")
                st.markdown("*Watch how your text transforms through each preprocessing stage:*")
                
                for i, step_data in enumerate(steps):
                    with st.expander(f"**{step_data['step']}**", expanded=(i == 0 or i == len(steps) - 1)):
                        st.markdown(f"**Description:** {step_data['description']}")
                        if step_data['step'] == 'Final Result':
                            st.success(f"**Result:** {step_data['result']}")
                        else:
                            st.code(step_data['result'], language=None)
                
                st.markdown("---")
            
            st.markdown("### 📊 Before vs After Comparison")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📄 Original Text")
                st.info(user_text)
            
            with col2:
                st.markdown("#### ✨ Cleaned Text")
                st.success(cleaned_text)
                
                st.download_button(
                    label="⬇️ Download Cleaned Text",
                    data=cleaned_text,
                    file_name="cleaned_text.txt",
                    mime="text/plain"
                )
            
            if show_wordcloud:
                st.markdown("### 🌀 Word Cloud Comparison")
                wc_col1, wc_col2 = st.columns(2)
                
                with wc_col1:
                    st.markdown("#### Original Text Word Cloud")
                    fig_orig = create_wordcloud(user_text, "Original Text")
                    if fig_orig:
                        st.pyplot(fig_orig)
                        plt.close()
                    else:
                        st.warning("Not enough words to generate word cloud")
                
                with wc_col2:
                    st.markdown("#### Cleaned Text Word Cloud")
                    fig_clean = create_wordcloud(cleaned_text, "Cleaned Text")
                    if fig_clean:
                        st.pyplot(fig_clean)
                        plt.close()
                    else:
                        st.warning("Not enough words to generate word cloud")
            
            if show_frequency:
                st.markdown("### 📊 Token Frequency Analysis")
                freq_col1, freq_col2 = st.columns(2)
                
                with freq_col1:
                    st.markdown("#### Original Text Frequencies")
                    fig_freq_orig = create_frequency_chart(user_text, f"Top {top_n_words} Words (Original)", top_n_words)
                    if fig_freq_orig:
                        st.pyplot(fig_freq_orig)
                        plt.close()
                    else:
                        st.warning("Not enough words to generate frequency chart")
                
                with freq_col2:
                    st.markdown("#### Cleaned Text Frequencies")
                    fig_freq_clean = create_frequency_chart(cleaned_text, f"Top {top_n_words} Words (Cleaned)", top_n_words)
                    if fig_freq_clean:
                        st.pyplot(fig_freq_clean)
                        plt.close()
                    else:
                        st.warning("Not enough words to generate frequency chart")
            
            if show_sentiment:
                st.markdown("### 💭 Sentiment Analysis Comparison")
                
                original_sentiment = get_sentiment(user_text)
                cleaned_sentiment = get_sentiment(cleaned_text)
                
                sent_col1, sent_col2 = st.columns(2)
                
                with sent_col1:
                    st.markdown("#### Original Text Sentiment")
                    st.metric("Compound Score", f"{original_sentiment['compound']:.3f}")
                    st.write(f"**Positive:** {original_sentiment['pos']:.3f}")
                    st.write(f"**Neutral:** {original_sentiment['neu']:.3f}")
                    st.write(f"**Negative:** {original_sentiment['neg']:.3f}")
                
                with sent_col2:
                    st.markdown("#### Cleaned Text Sentiment")
                    st.metric("Compound Score", f"{cleaned_sentiment['compound']:.3f}")
                    st.write(f"**Positive:** {cleaned_sentiment['pos']:.3f}")
                    st.write(f"**Neutral:** {cleaned_sentiment['neu']:.3f}")
                    st.write(f"**Negative:** {cleaned_sentiment['neg']:.3f}")
                
                st.markdown("#### Sentiment Comparison Chart")
                fig_sentiment = create_sentiment_comparison_chart(original_sentiment, cleaned_sentiment)
                st.pyplot(fig_sentiment)
                plt.close()
                
                sentiment_explanation = """
                **Note:** VADER (Valence Aware Dictionary and sEntiment Reasoner) provides sentiment scores:
                - **Compound**: Overall sentiment (-1 to +1, where -1 is most negative and +1 is most positive)
                - **Positive, Neutral, Negative**: Individual sentiment components (0 to 1)
                """
                st.info(sentiment_explanation)
        else:
            st.warning("⚠️ Please enter some text to process!")
    
    # Show processing history
    if st.session_state.processing_history:
        st.markdown("---")
        st.markdown("### 📜 Processing History")
        for idx, item in enumerate(st.session_state.processing_history[:5]):
            with st.expander(f"**{item['timestamp']}** - Method: {item['method']}"):
                st.write(f"**Original:** {item['original']}")
                st.write(f"**Cleaned:** {item['cleaned']}")
    
    # Example usage section
    with st.expander("💡 **Tip: Example Usage**"):
        st.code('''Input: "Héllooo!!! I AM Running So Fast 😍😂..."
Output: "hello i be run so fast"
        ''')

else:  # Batch Upload Mode
    st.markdown("### 📁 Batch File Upload")
    st.markdown("*Upload multiple .txt files for batch processing*")
    
    uploaded_files = st.file_uploader(
        "Choose text files (.txt)",
        type=['txt'],
        accept_multiple_files=True,
        help="Upload one or more .txt files to process them all at once"
    )
    
    if st.button("🚀 Process All Files", type="primary"):
        if uploaded_files:
            with st.spinner("Loading NLP model..."):
                nlp = load_spacy_model()
            
            with st.spinner(f"Processing {len(uploaded_files)} files..."):
                results = process_batch_files(uploaded_files, nlp, use_stopwords, custom_stopwords, use_stemming)
            
            st.success(f"✅ Processed {len(results)} files successfully!")
            
            # Display results
            st.markdown("### 📊 Batch Processing Results")
            for result in results:
                with st.expander(f"**{result['filename']}**"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Original:**")
                        st.text_area("", value=result['original'][:500], height=150, key=f"orig_{result['filename']}", disabled=True)
                    with col2:
                        st.markdown("**Cleaned:**")
                        st.text_area("", value=result['cleaned'][:500], height=150, key=f"clean_{result['filename']}", disabled=True)
            
            # Create downloadable zip file
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for result in results:
                    zip_file.writestr(f"cleaned_{result['filename']}", result['cleaned'])
            
            st.download_button(
                label="⬇️ Download All Cleaned Files (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="cleaned_texts.zip",
                mime="application/zip"
            )
        else:
            st.warning("⚠️ Please upload at least one file!")
    
    st.info("💡 **Tip:** Upload multiple .txt files to process them all at once with the same settings.")

st.markdown('<div class="footer">✨ Made with ❤️ by Team 3 – SmartText Studio (NLP Normalization Project)</div>', unsafe_allow_html=True)
