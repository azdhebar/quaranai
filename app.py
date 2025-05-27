import streamlit as st
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

@st.cache_resource(show_spinner=False)
def load_resources():
    with open("quran_model.pkl", "rb") as f:
        data = pickle.load(f)
    df = data["df"]
    model = SentenceTransformer(data["model_path"])
    index = faiss.read_index("quran_faiss.index")
    return df, model, index

def search_ayah(query, df, model, index, top_k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(np.array(query_embedding), top_k)
    results = []

    for score, i in zip(distances[0], indices[0]):
        ayah_index = i % len(df)
        row = df.iloc[ayah_index]
        results.append({
            "surah_name_en": row['surah_name_en'],
            "surah_name_roman": row['surah_name_roman'],
            "surah_no": row['surah_no'],
            "ayah_no": row['ayah_no_surah'],
            "juz": row['juz_no'],
            "ayah_en": row['ayah_en'],
            "ayah_ar": row['ayah_ar'],
            "confidence": float(score)
        })
    return results

def main():
    st.title("Quran Ayah Search")
    st.write("Enter an Ayah (in Arabic or English) to find its Surah, Juz, and similar Ayahs with confidence scores.")

    df, model, index = load_resources()

    query = st.text_input("Enter Ayah text here:")
    top_k = st.slider("Number of results to show:", min_value=1, max_value=10, value=5)

    if query:
        with st.spinner("Searching..."):
            results = search_ayah(query, df, model, index, top_k)

        if results:
            for i, res in enumerate(results, 1):
                st.markdown(f"### Result #{i}")
                st.write(f"**Surah Name (EN):** {res['surah_name_en']}  |  **Surah Name (Roman):** {res['surah_name_roman']}")
                st.write(f"**Surah No:** {res['surah_no']}  |  **Ayah No:** {res['ayah_no']}  |  **Juz:** {res['juz']}")
                st.write(f"**Arabic Ayah:** {res['ayah_ar']}")
                st.write(f"**English Ayah:** {res['ayah_en']}")
                st.write(f"**Confidence:** {res['confidence']:.4f}")
                st.markdown("---")
        else:
            st.write("No results found.")

if __name__ == "__main__":
    main()
