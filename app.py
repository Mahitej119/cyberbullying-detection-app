import streamlit as st
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2

def preprocess_text(text):
    """Clean and normalize text"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    words = text.split()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    words = [lemmatizer.lemmatize(word) for word in words if word and word not in stop_words]
    return " ".join(words)

@st.cache_resource
def load_or_train_model():
    """Load or train the cyberbullying detection model"""
    with st.spinner("Loading and training model..."):
        # Load dataset
        df = pd.read_csv("cyberbullying_tweets.csv")
        
        # Rename columns to match dataset
        df.rename(columns={'tweet_text': 'text', 'cyberbullying_type': 'label'}, inplace=True)
        
        # Show class distribution
        st.sidebar.info(f"**Class Distribution:**\n{df['label'].value_counts().to_string()}")
        
        # Preprocess
        df['cleaned_text'] = df['text'].apply(preprocess_text)
        
        # Vectorize
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X = vectorizer.fit_transform(df['cleaned_text'])
        y = df['label']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        # Train model
        model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
        }
        
        return vectorizer, model, metrics

def predict_cyberbullying(text, vectorizer, model):
    """Predict if text contains cyberbullying"""
    cleaned = preprocess_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    confidence = max(model.predict_proba(vectorized)[0])
    return prediction, confidence

# Streamlit UI
st.set_page_config(page_title="Cyberbullying Detection", page_icon="🛡️", layout="wide")
st.title("🛡️ Cyberbullying Detection System")
st.write("Detect potential cyberbullying in text using Machine Learning")

# Load model
vectorizer, model, metrics = load_or_train_model()

# Sidebar: Model Performance
with st.sidebar:
    st.header("📊 Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    col2.metric("F1-Score", f"{metrics['f1']:.2%}")
    
    col1, col2 = st.columns(2)
    col1.metric("Precision", f"{metrics['precision']:.2%}")
    col2.metric("Recall", f"{metrics['recall']:.2%}")
    
    with st.expander("📈 Confusion Matrix"):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Enter Text to Analyze")
    user_input = st.text_area("Paste or type a comment:", height=120, placeholder="Enter text to check for cyberbullying...")

with col2:
    st.subheader("Info")
    st.info("**Cyberbullying includes:**\n- Insults and name-calling\n- Harassment and threats\n- Exclusion or isolation\n- Spreading rumors\n- Mockery of appearance/identity")

# Prediction
if st.button("🔍 Analyze Text", use_container_width=True):
    if not user_input or not user_input.strip():
        st.error("Please enter some text.")
    elif len(user_input) < 5:
        st.error("Text must be at least 5 characters long.")
    elif len(user_input) > 500:
        st.error("Text cannot exceed 500 characters.")
    else:
        try:
            prediction, confidence = predict_cyberbullying(user_input, vectorizer, model)
            
            st.divider()
            st.subheader("📋 Analysis Result")
            
            col1, col2, col3 = st.columns([1.5, 1, 1])
            
            with col1:
                if prediction == "cyberbullying":
                    st.error(f"⚠️ **{prediction.upper()}**")
                else:
                    st.success(f"✓ **{prediction.upper()}**")
            
            with col2:
                st.metric("Confidence", f"{confidence:.1%}")
            
            with col3:
                st.metric("Text Length", f"{len(user_input)} chars")
            
            st.progress(confidence, text=f"Model Confidence: {confidence:.1%}")
            
            if confidence < 0.65:
                st.warning("⚠️ Low confidence prediction - result should be verified manually")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Examples
with st.expander("📝 Try Example Predictions"):
    examples = {
        "Normal Comment": "I really enjoyed the movie, great performance!",
        "Cyberbullying 1": "You're so stupid, nobody likes you anyway",
        "Cyberbullying 2": "Go kill yourself, nobody cares about you",
        "Neutral": "What time is the meeting tomorrow?",
    }
    
    for label, example_text in examples.items():
        if st.button(f"Test: {label}", key=label):
            prediction, confidence = predict_cyberbullying(example_text, vectorizer, model)
            st.info(f"**Text:** {example_text}\n\n**Prediction:** {prediction} ({confidence:.1%})")

st.divider()
st.caption("🔬 Built with TF-IDF + Logistic Regression | Data: ~48K labeled tweets")