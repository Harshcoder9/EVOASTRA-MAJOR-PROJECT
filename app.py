import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration, MarianTokenizer, MarianMTModel
from PIL import Image
from gtts import gTTS
import tempfile
import torch

st.title("📸 Image Captioning App with Translation 🌍")  

@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()


@st.cache_resource
def load_translator():
    model_name = "Helsinki-NLP/opus-mt-en-hi"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    translator = MarianMTModel.from_pretrained(model_name)
    return tokenizer, translator

trans_tokenizer, translator = load_translator()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])


lang_choice = st.radio("Choose caption language:", ["English", "Hindi"])


def text_to_speech(text, lang_code):
    tts = gTTS(text=text, lang=lang_code)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        return fp.name


def translate_to_hindi(text):
    tokens = trans_tokenizer([text], return_tensors="pt", padding=True)
    output = translator.generate(**tokens)
    hindi = trans_tokenizer.decode(output[0], skip_special_tokens=True)
    return hindi

caption = None

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Caption"):
        inputs = processor(images=image, return_tensors="pt")
        output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)

        # English or Hindi display logic
        if lang_choice == "English":
            st.success("Caption: " + caption)
            st.session_state["caption"] = caption
            st.session_state["lang"] = "en"

        else:
            hindi_cap = translate_to_hindi(caption)
            st.success("कैप्शन: " + hindi_cap)
            st.session_state["caption"] = hindi_cap
            st.session_state["lang"] = "hi"

# READ ALOUD BUTTON
if "caption" in st.session_state:
    if st.button("🔊 Read Aloud"):
        lang = st.session_state["lang"]
        audio_file = text_to_speech(st.session_state["caption"], lang)
        st.audio(audio_file, format="audio/mp3")