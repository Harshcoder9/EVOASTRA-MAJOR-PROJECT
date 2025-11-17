# 🧠 EVOASTRA - Image Captioning using Flickr8k Dataset

## 📘 Overview
This project, built under the **EVOASTRA Internship**, focuses on generating human-like captions for images using Deep Learning.  
The system uses a **CNN encoder (InceptionV3)** to extract visual features and an **LSTM decoder** to generate natural language captions.  
Additionally, the Streamlit interface provides **English → Hindi translation** and **Read-Aloud (Text-to-Speech)** support.

---

## 🎯 Objective
To design and implement a complete deep-learning pipeline that:

- Extracts visual features from images using a pretrained CNN.
- Generates text captions using an LSTM-based decoder.
- Supports **bilingual captioning** (English and Hindi).
- Reads captions aloud using a built-in TTS system.
- Provides an interactive front-end using Streamlit.

---

# 🔄 End-to-End Project Pipeline

## **1. Data Processing**
- Loaded and cleaned the Flickr8k captions dataset.
- Converted raw `captions.txt` into structured CSV format.
- Added `<start>` and `<end>` tokens.
- Tokenized and padded captions for uniform length.
- Prepared final datasets for training.

## **2. Feature Extraction (Encoder)**
- Used **InceptionV3** to extract 2048-dimensional feature vectors.
- Saved features for efficient training.

## **3. Caption Preparation**
- Converted text captions into numerical sequences.
- Applied padding for consistent input shape.

## **4. Model Building (Decoder)**
- Built an Encoder–Decoder architecture:
  - CNN encoder → extract visual features  
  - LSTM decoder → generate captions word-by-word  
- Combined image embeddings with text embeddings.

## **5. Model Training**
- Trained the model with image-caption pairs.
- Used **Adam optimizer** and tuned hyperparameters.
- Saved trained weights for inference.

## **6. Model Evaluation**
- Measured performance using:
  - BLEU
  - METEOR
  - CIDEr
- Performed manual testing on unseen images.

## **7. Translation & Read-Aloud Features**
- Implemented **English → Hindi translation** using MarianMT.
- Added **Text-to-Speech (gTTS)** for both English and Hindi.
- Added a **language selection radio button** in Streamlit.
- Enabled dual-language captioning and voice output.

---

## ⚙️ Tech Stack

| Component        | Technology                 |
| ---------------- | -------------------------- |
| Language         | Python 3                   |
| Deep Learning    | TensorFlow / Keras         |
| Translation      | HuggingFace Transformers   |
| Image Processing | Pillow, OpenCV             |
| Deployment       | Streamlit                  |
| Audio Output     | gTTS                       |
| Utilities        | Pandas, NumPy, Matplotlib  |

---

## 🧩 Folder Structure



```
EVOASTRA/
│
├── convert_to_csv.py
├── captions.txt
├── captions_10k.csv
├── images/
├── feature_extraction.py
├── caption_preprocessing.py
├── caption_sequence_generation.py
├── train_model.py
├── models/
├── app.py
└── README.md
```


---

## 📦 Dataset Details

### **Flickr8k Dataset**
- 8,000 images  
- 5 captions per image (40,000 total)  
- Real, human-written captions  
- Suitable for vision-language tasks

Dataset Source: **Kaggle – Flickr8k**

---

## 🚀 Workflow

### 1. Convert Raw Captions to CSV
python convert_to_csv.py

### 2. Extract CNN Features
python feature_extraction.py

### 3. Preprocess Captions
python caption_preprocessing.py

### 4. Generate Training Sequences
python caption_sequence_generation.py

### 5. Train Model
python train_model.py

### 6. Run Streamlit Interface
streamlit run app.py

## 🔍 Example Outputs

| Image                                   | Generated Caption                   |
| --------------------------------------- | ----------------------------------- |
| ![Dog](https://thumbs.dreamstime.com/b/closeup-brown-white-small-dog-running-grass-small-dog-running-grass-358225772.jpg) | "A brown dog running in the grass." |
| ![Car](https://thumbs.dreamstime.com/b/red-car-parked-slope-outside-old-house-windows-flowers-modern-european-street-285898877.jpg) | "A red car parked beside the road." |

---

## 🔧 Installation

Install all dependencies:

```bash
pip install -r req.txt
```

### Example `req.txt`

```
tensorflow
numpy
pandas
Pillow
tqdm
matplotlib
streamlit
transformers
gTTS
sentencepiece
```

---

## 🔄 Future Enhancements

* [ ] Add Vision Transformer (ViT + GPT) for advanced captioning.
* [ ] Build Flask-based web app for image uploads.
* [ ] Add CLIP-based image-caption retrieval.
* [ ] Implement BLEU and CIDEr evaluation dashboard.

---

## 👨‍💻 Contributors

* **Harsh Pandey** — Data Processing, Caption Cleaning, Project Workflow, English & Hindi Translation Features
* **Anish Mehra** - Image Preprocessing, README, feature additon "Read Aloud" on streamlit using gtts
* **Hitesh** – Worked on training the image captioning model, Worked on training the image captioning model.
* **Om** – Worked on training the image captioning model.
* **Chandrika** - Frontend through streamlit
* **Florence** - Presentation
* **Supriya** - Report
---

## 🏁 License

This project is for **educational and research purposes** only.
Dataset © Flickr8k authors, used under academic usage terms.
