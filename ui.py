import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load fine-tuned model
model_path = "bart_finetuned"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

# Summarization function
def summarize_article(article_text):
    if not article_text.strip():
        return "Please enter a valid news article."
    
    # Split into sentences and ensure we don't exceed max length
    sentences = article_text.split('.')
    processed_text = '. '.join(sentences[:10])  # Take first 10 sentences to avoid truncation
    
    inputs = tokenizer(processed_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    output_ids = model.generate(
        **inputs,
        max_length=150,  # Increased from 128
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
        min_length=50,  # Added minimum length
        length_penalty=2.0  # Added length penalty to encourage longer summaries
    )
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Ensure summary ends with proper punctuation
    if not summary.endswith(('.', '!', '?')):
        summary += '.'
    
    return summary

# UI with Custom CSS
with gr.Blocks() as demo:
    gr.HTML("""
    <style>
        body {
            background: #f5f7fa;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .gradio-container {
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            text-align: center;
            font-size: 2.5rem;
            color: #2c3e50;
            margin-bottom: 10px;
        }
        p.description {
            text-align: center;
            font-size: 1.1rem;
            color: #555;
            margin-bottom: 30px;
        }
        .input-area, .output-area {
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            padding: 20px;
        }
        .gr-button {
            background: #1f6feb;
            color: white;
            border-radius: 8px;
            padding: 10px 24px;
            font-size: 1rem;
            margin-top: 10px;
        }
        .gr-button:hover {
            background: #1158c7;
        }
    </style>
    """)

    gr.Markdown("# 📰 News Summarizer")
    gr.Markdown("<p class='description'>Paste any long news article below and receive a clear, concise summary powered by a fine-tuned BART model.</p>")

    with gr.Row():
        with gr.Column(scale=1, elem_classes="input-area"):
            article_input = gr.Textbox(
                label="News Article",
                placeholder="Paste or type your article here...",
                lines=15,
                max_lines=30
            )
            summarize_button = gr.Button("Summarize ✨")

        with gr.Column(scale=1, elem_classes="output-area"):
            summary_output = gr.Textbox(
                label="Generated Summary",
                placeholder="Summary will appear here...",
                lines=10,
                max_lines=20
            )

    summarize_button.click(fn=summarize_article, inputs=article_input, outputs=summary_output)

demo.launch()
