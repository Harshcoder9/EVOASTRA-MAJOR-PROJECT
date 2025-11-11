# 🧠 EVOASTRA - Image Captioning using Flickr8k Dataset

## 📘 Overview

EVOASTRA is an AI-based **Image Captioning System** that generates human-like textual descriptions for images. It leverages the **Flickr8k dataset**, **PostgreSQL** for data storage, and a **CNN-LSTM deep learning model** to map visual features to natural language captions.

The system learns to understand visual features (objects, colors, actions) and describe them in coherent sentences.

---

## 🎯 Objective

To create a deep learning pipeline that:

1. Extracts image features using pretrained **CNN encoders** (InceptionV3/ResNet50).
2. Generates descriptive captions using an **LSTM/Transformer decoder**.
3. Stores and processes data efficiently using **PostgreSQL**.

---

## ⚙️ Tech Stack

| Component        | Technology                 |
| ---------------- | -------------------------- |
| Language         | Python 3                   |
| Database         | PostgreSQL                 |
| Image Processing | Pillow, OpenCV             |
| Deep Learning    | TensorFlow / Keras         |
| Data Handling    | Pandas, NumPy              |
| Visualization    | Matplotlib                 |
| Environment      | Jupyter Notebook / VS Code |

---

## 🧩 Folder Structure

```
EVOASTRA/
│
├── db_connect.py             # Connects to PostgreSQL
├── insert_images.py          # Inserts image metadata into DB
├── insert_captions.py        # Inserts captions into DB
├── convert_to_csv.py         # Converts Flickr8k captions.txt → CSV
├── captions.txt              # Original Flickr8k captions
├── captions_10k.csv          # Cleaned CSV for model training
├── images/                   # Folder of image files (.jpg)
├── image_features.npy        # Extracted CNN features (optional)
├── README.md                 # Documentation file
└── models/                   # (Optional) Trained model weights
```

---

## 📦 Dataset Details

**Flickr8k Dataset**

* Images: 8,000
* Captions per Image: 5
* Total Captions: ~40,000
* Source: [Kaggle - Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k)

Each image includes five human-generated captions describing the scene.

---

## 💾 Database Design

**Database:** `flickr8k_db`

### Tables

#### 🖼️ `images`

| Column    | Type         | Description                    |
| --------- | ------------ | ------------------------------ |
| image_id  | VARCHAR(255) | Image filename (Primary Key)   |
| file_path | TEXT         | Full image path                |
| width     | INT          | Image width                    |
| height    | INT          | Image height                   |
| split     | VARCHAR(10)  | Dataset split (train/val/test) |

#### 💬 `captions`

| Column       | Type         | Description                               |
| ------------ | ------------ | ----------------------------------------- |
| caption_id   | SERIAL       | Primary Key                               |
| image_id     | VARCHAR(255) | Foreign key referencing `images.image_id` |
| caption_text | TEXT         | Original caption text                     |
| cleaned_text | TEXT         | Preprocessed caption text                 |

#### 🧬 `captioning`

| Column       | Type         | Description       |
| ------------ | ------------ | ----------------- |
| image_id     | VARCHAR(255) | Linked image file |
| caption_text | TEXT         | Image caption     |

---

## 🚀 Workflow

### 1. **Prepare Dataset**

* Download Flickr8k dataset.
* Move all `.jpg` files to:

  ```
  D:\evoastra major project\images\
  ```
* Keep `captions.txt` in your project folder.

### 2. **Setup PostgreSQL Database**

Create database and tables:

```sql
CREATE DATABASE flickr8k_db;
CREATE TABLE images (...);
CREATE TABLE captions (...);
```

### 3. **Insert Data**

```bash
python insert_images.py
python insert_captions.py
```

✅ Populates the database with all image metadata and captions.

### 4. **Convert Captions to CSV**

```bash
python convert_to_csv_limit.py
```

✅ Creates `captions_10k.csv` containing 10,000 image-caption pairs for faster training.

### 5. **Feature Extraction (Encoder)**

Use a pretrained CNN such as InceptionV3:

```python
from tensorflow.keras.applications.inception_v3 import InceptionV3
model = InceptionV3(weights='imagenet')
```

Extract 2048-dimensional feature vectors and save them as `.npy`.

### 6. **Model Training (Decoder)**

* Tokenize captions and add `<start>` and `<end>` tokens.
* Pad sequences for uniform input size.
* Train an LSTM or Transformer decoder to generate captions.

### 7. **Evaluation**

Evaluate model performance using:

* BLEU
* METEOR
* CIDEr

---

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
psycopg2
Pillow
tqdm
tensorflow
numpy
pandas
matplotlib
```

---

## 🔄 Future Enhancements

* [ ] Add Vision Transformer (ViT + GPT) for advanced captioning.
* [ ] Build Flask-based web app for image uploads.
* [ ] Add CLIP-based image-caption retrieval.
* [ ] Implement BLEU and CIDEr evaluation dashboard.

---

## 👨‍💻 Contributors

* **Harsh Pandey** — Data Engineer & Model Developer
* **Team EVOASTRA** — Database Architect, ML Engineer, UI Developer

---

## 🏁 License

This project is for **educational and research purposes** only.
Dataset © Flickr8k authors, used under academic usage terms.
