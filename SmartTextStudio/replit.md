# Overview

SmartText Studio is an NLP text normalization and analysis application that demonstrates core natural language processing techniques. The application provides an interactive web interface for users to input text and see it processed through multiple normalization stages including case folding, punctuation/emoji/number removal, accent removal, and lemmatization. It visualizes the results through word clouds, frequency charts, and before/after comparisons, and includes sentiment analysis capabilities.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture

**Framework**: Streamlit  
**Rationale**: Streamlit was chosen for its ability to rapidly create interactive web UIs with minimal code. It natively supports real-time text input, visualization rendering, and state management without requiring separate frontend/backend development.

**Key Components**:
- Interactive text input widgets for user submissions
- Real-time visualization rendering (word clouds, charts)
- Before/after comparison displays
- Integrated sentiment analysis results display

## Backend Architecture

**Processing Pipeline**: Sequential text transformation stages  
**Rationale**: The application uses a pipeline approach where text flows through distinct normalization stages. This modular design allows each transformation to be independently tested and modified.

**Processing Stages**:
1. **Case Folding** - Converts all text to lowercase for standardization
2. **Punctuation/Emoji/Number Removal** - Cleans text using regex patterns and string translation
3. **Accent Removal** - Normalizes Unicode characters using the unidecode library
4. **Lemmatization** - Reduces words to base forms using spaCy's linguistic models

**Design Pattern**: Functional programming approach with pure transformation functions that take text input and return processed output.

## NLP Components

**Primary Libraries**: spaCy and NLTK  
**Rationale**: 
- **spaCy** provides efficient lemmatization and tokenization with pre-trained language models
- **NLTK** offers VADER sentiment analysis, which works well for social media and informal text
- Both libraries are mature, well-documented, and widely adopted in the NLP community

**Model Loading Strategy**: 
- spaCy model (`en_core_web_sm`) is cached using Streamlit's `@st.cache_resource` decorator to prevent redundant loading
- Automatic fallback installation if the model isn't available
- NLTK lexicons are downloaded quietly on first run

**Sentiment Analysis**: VADER (Valence Aware Dictionary and sEntiment Reasoner) from NLTK provides rule-based sentiment scoring suitable for real-time analysis without requiring heavy transformer models during initial processing.

## Visualization Architecture

**Libraries**: Matplotlib and WordCloud  
**Rationale**: 
- **WordCloud** specializes in generating visually appealing word frequency visualizations
- **Matplotlib** provides flexible charting capabilities for frequency distributions
- Both integrate seamlessly with Streamlit's rendering pipeline

**Visualization Types**:
- Word clouds for visual text representation
- Frequency charts showing token distribution
- Comparative displays (before/after processing)

## Data Processing

**Text Cleaning Approach**: Multi-stage regex and string manipulation  
**Rationale**: Breaking cleaning into discrete functions (emoji removal, punctuation removal, whitespace normalization) allows for:
- Individual testing and validation
- Flexible pipeline configuration
- Clear separation of concerns

**Unicode Handling**: The unidecode library handles accent removal and character normalization, converting non-ASCII characters to their closest ASCII equivalents.

# External Dependencies

## NLP Libraries

**spaCy** (`en_core_web_sm` model)
- Purpose: Lemmatization and tokenization
- Integration: Loaded once and cached; auto-installs if missing
- Model size: ~13MB lightweight English model

**NLTK** (VADER lexicon)
- Purpose: Sentiment intensity analysis
- Integration: Lexicon downloaded on application start
- Resource: Pre-trained sentiment dictionary

## Text Processing Libraries

**unidecode**
- Purpose: Convert Unicode text to ASCII equivalents
- Use case: Accent removal and character normalization

**WordCloud**
- Purpose: Generate word frequency visualizations
- Integration: Creates image objects rendered by Matplotlib

## Visualization Libraries

**Matplotlib**
- Purpose: Chart rendering and visualization output
- Integration: Renders plots directly in Streamlit interface

## Web Framework

**Streamlit**
- Purpose: Web application framework and UI layer
- Deployment: Can be hosted on Streamlit Cloud, Replit, or other Python hosting services
- Features used: Caching, widgets, visualization rendering

## Data Handling

**pandas**
- Purpose: Data structure manipulation for frequency analysis
- Use case: Organizing token counts and statistics

**collections.Counter**
- Purpose: Efficient frequency counting of tokens
- Integration: Native Python library for tallying word occurrences

## Potential Future Dependencies

**Transformers (Hugging Face)**
- Status: Mentioned in project documentation but not yet implemented
- Intended purpose: Advanced sentiment analysis using pre-trained transformer models
- Consideration: Would provide more nuanced sentiment detection than VADER but requires more computational resources