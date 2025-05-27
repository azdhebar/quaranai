from flask import Flask, request, jsonify
import pickle
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Load resources once when the app starts
with open("quran_model.pkl", "rb") as f:
    data = pickle.load(f)
df = data["df"]
model = SentenceTransformer(data["model_path"])
index = faiss.read_index("quran_faiss.index")

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


@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    top_k = int(request.args.get('top_k', 5))

    if not query:
        return jsonify({"error": "Query parameter is required"}), 400

    results = search_ayah(query, top_k)
    return jsonify(results)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use PORT from env, fallback to 5000
    app.run(host="0.0.0.0", port=port)
