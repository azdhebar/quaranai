from flask import Flask, request, render_template
import pickle
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer
from PIL import Image
import io
import easyocr

app = Flask(__name__)

# Load resources once when the app starts
with open("quran_model.pkl", "rb") as f:
    data = pickle.load(f)
df = data["df"]
model = SentenceTransformer(data["model_path"])
index = faiss.read_index("quran_faiss.index")

# Initialize EasyOCR Reader once (Arabic and English)
reader = easyocr.Reader(['ar', 'en'], gpu=False)  # Set gpu=True if you have CUDA

def search_ayah(query: str, top_k: int = 5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(np.array(query_embedding), top_k)
    results = []

    for score, i in zip(distances[0], indices[0]):
        ayah_index = int(i % len(df))  # ensure it's an int
        row = df.iloc[ayah_index]
        results.append({
            "surah_name_en": str(row["surah_name_en"]),
            "surah_name_roman": str(row["surah_name_roman"]),
            "surah_no": int(row["surah_no"]),
            "ayah_no": int(row["ayah_no_surah"]),
            "juz": int(row["juz_no"]),
            "ayah_en": str(row["ayah_en"]),
            "ayah_ar": str(row["ayah_ar"]),
            "confidence": float(score)
        })
    return results

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error="No file part")

        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', error="No selected file")

        try:
            # Read image file as PIL image
            image = Image.open(file.stream).convert("RGB")

            # Convert image to numpy array (EasyOCR input)
            img_np = np.array(image)

            # Perform OCR with EasyOCR
            ocr_results = reader.readtext(img_np, detail=0)

            # Join detected text chunks into a single string
            query = " ".join(ocr_results).strip()

            if not query:
                return render_template('index.html', error="No text detected in image")

            results = search_ayah(query, top_k=5)

            return render_template('index.html', query=query, results=results)
        except Exception as e:
            return render_template('index.html', error=f"Error processing image: {str(e)}")

    # GET request just renders the form
    return render_template('index.html')


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
