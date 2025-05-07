import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Set page configuration
st.set_page_config(
    page_title="News Article Summarizer",
    page_icon="📰",
    layout="centered"  # Changed to centered for better readability
)

# App title and description
st.title("📰 News Article Summarizer")
st.markdown("##### Transform lengthy news articles into concise, informative summaries")

# Create sidebar for model loading status
with st.sidebar:
    st.header("Model Status")
    with st.spinner("Loading model..."):
        @st.cache_resource
        def load_model():
            # Load fine-tuned model
            model_path = "bart_finetuned"
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
            return model, tokenizer

        # Load model
        try:
            model, tokenizer = load_model()
            st.success("✅ Model loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            st.stop()
    
    # Add some helpful tips in the sidebar
    st.markdown("### Tips")
    st.markdown("- For best results, use articles between 300-1000 words")
    st.markdown("- The summary keeps the most important information")
    st.markdown("- You can customize the summary length in Advanced Options")

# Summarization function
def summarize_article(article_text, max_len=150, min_len=50, n_beams=4, len_penalty=2.0):
    if not article_text.strip():
        return "Please enter a valid news article."
        
    # Split into sentences and ensure we don't exceed max length
    sentences = article_text.split('.')
    processed_text = '. '.join(sentences[:10])  # Take first 10 sentences to avoid truncation
        
    inputs = tokenizer(processed_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    output_ids = model.generate(
        **inputs,
        max_length=max_len,
        num_beams=n_beams,
        early_stopping=True,
        no_repeat_ngram_size=3,
        min_length=min_len,
        length_penalty=len_penalty
    )
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
    # Ensure summary ends with proper punctuation
    if not summary.endswith(('.', '!', '?')):
        summary += '.'
        
    return summary

# Create two tabs for different input methods
tab1, tab2 = st.tabs(["✏️ Paste Article", "📄 Upload File"])

with tab1:
    article_text = st.text_area(
        "Paste your news article here:",
        height=250,
        placeholder="Enter your news article text here..."
    )

with tab2:
    uploaded_file = st.file_uploader("Upload a text file:", type=['txt'])
    if uploaded_file is not None:
        article_text = uploaded_file.getvalue().decode("utf-8")
        st.text_area("Article content:", value=article_text, height=250, disabled=True)
    else:
        article_text = ""

# Parameters for advanced users (optional)
with st.expander("⚙️ Advanced Options"):
    st.caption("Customize your summary generation parameters")
    col1, col2 = st.columns(2)
    with col1:
        max_length = st.slider("Maximum summary length", 50, 200, 150, help="The maximum number of tokens in the summary")
        min_length = st.slider("Minimum summary length", 30, 100, 50, help="The minimum number of tokens in the summary")
    with col2:
        num_beams = st.slider("Number of beams", 1, 8, 4, help="Higher values increase quality but slow down generation")
        length_penalty = st.slider("Length penalty", 0.0, 4.0, 2.0, 0.1, help="Higher values favor longer summaries")

# Center the button and make it more prominent
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    generate_button = st.button("Generate Summary", type="primary", use_container_width=True)

# Show a divider before results
st.markdown("---")

# Summarize button action
if generate_button:
    if article_text:
        with st.spinner("📝 Generating your summary..."):
            summary = summarize_article(
                article_text, 
                max_len=max_length, 
                min_len=min_length, 
                n_beams=num_beams, 
                len_penalty=length_penalty
            )
        
        # Display results in a card-like container
        st.subheader("📋 Summary")
        with st.container():
            st.markdown(f"""
            <div style="padding: 20px; border-radius: 5px; border: 1px solid #ddd;">
            {summary}
            </div>
            """, unsafe_allow_html=True)
        
        # Stats section with better visualization
        st.subheader("📊 Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Length", f"{len(article_text)} chars")
        with col2:
            st.metric("Summary Length", f"{len(summary)} chars")
        with col3:
            if len(article_text) > 0:
                compression = round((1 - len(summary)/len(article_text)) * 100, 1)
                st.metric("Compression", f"{compression}%")
                
        # Add copy button for summary
        st.download_button(
            label="Download Summary",
            data=summary,
            file_name="article_summary.txt",
            mime="text/plain"
        )
    else:
        st.warning("⚠️ Please enter or upload an article to summarize.")

# Add footer with cleaner styling
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<div style='text-align: center;'>News Article Summarizer powered by fine-tuned GNN+BART model</div>", unsafe_allow_html=True)