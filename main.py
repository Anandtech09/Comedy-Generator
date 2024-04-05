import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import subprocess
import sys

# Load your model and tokenizer
def load_model():
    st.spinner('Wait for it... loading the model')
    tokenizer = AutoTokenizer.from_pretrained("Anandappu/bert_comedy")
    model = AutoModelForSeq2SeqLM.from_pretrained("Anandappu/bert_comedy")
    st.success('Done! The model is loaded.')
    return tokenizer, model

tokenizer, model = load_model()

# Function to generate text
def generate(prompt):
    batch = tokenizer(prompt, return_tensors="pt")
    generated_ids = model.generate(batch["input_ids"], max_new_tokens=150)
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return output[0]


# Streamlit interface
def main():
    
    st.title("Comedy Text Generator")
    user_input = st.text_area("Enter your prompt:", "Eg:why banana wear dress")
    if st.button("Generate"):
        with st.spinner("Generating..."):
            generated_text = generate(user_input)
            st.text_area("Generated Text:", generated_text, height=250)

if __name__ == "__main__":
    main()