import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import MarianTokenizer, MarianMTModel
from PIL import Image
import torch

st.title("📸 Image Captioning App with Translation 🌍")

@st.cache_resource
def load_caption_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

@st.cache_resource
def load_translation_models():
    en_to_hi_tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
    en_to_hi_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-hi")

    hi_to_en_tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-hi-en")
    hi_to_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-hi-en")

    return en_to_hi_tok, en_to_hi_model, hi_to_en_tok, hi_to_en_model


processor, model = load_caption_model()
en_hi_tok, en_hi_model, hi_en_tok, hi_en_model = load_translation_models()

uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

lang = st.radio("Choose Caption Language", ["English", "Hindi"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Caption"):
        inputs = processor(images=image, return_tensors="pt")
        output = model.generate(**inputs)
        caption_en = processor.decode(output[0], skip_special_tokens=True)


        if lang == "Hindi":
            translated = en_hi_model.generate(**en_hi_tok(caption_en, return_tensors="pt"))
            hindi_caption = en_hi_tok.decode(translated[0], skip_special_tokens=True)
            st.success("**Hindi Caption:** " + hindi_caption)

        else:
            st.success("**English Caption:** " + caption_en)
