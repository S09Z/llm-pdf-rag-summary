import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

# --- ค่าคงที่ (ต้องตรงกับไฟล์ vector_store_builder.py) ---
CHROMA_PERSIST_DIR = "chroma_db"
CHROMA_COLLECTION_NAME = "pdf_collection"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def visualize_vector_db():
    """
    ดึงข้อมูลจาก ChromaDB, ลดมิติด้วย t-SNE, และสร้าง interactive plot ด้วย Plotly
    """
    print("Connecting to the vector store...")
    # 1. โหลด Embedding model และเชื่อมต่อ ChromaDB
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embedding_model,
        collection_name=CHROMA_COLLECTION_NAME
    )

    print("Retrieving all data from the collection...")
    # 2. ดึงข้อมูลทั้งหมด (embeddings, documents, metadatas) จาก collection
    # นี่อาจใช้เวลาสักครู่ถ้ามีข้อมูลเยอะมาก
    data = vector_store.get(include=["embeddings", "documents", "metadatas"])
    
    # ตรวจสอบว่ามีข้อมูลหรือไม่
    if not data or not data.get('ids'):
        print("No data found in the vector store collection.")
        return

    embeddings = np.array(data['embeddings'])
    documents = data['documents']
    metadatas = data['metadatas']
    
    print(f"Retrieved {len(documents)} data points.")
    print("Performing t-SNE dimensionality reduction (this might take a while)...")
    
    # 3. ลดมิติข้อมูลจาก N มิติ เหลือ 2 มิติด้วย t-SNE
    # perplexity ควรน้อยกว่าจำนวน sample (ปรับค่าได้)
    n_samples = len(documents)
    perplexity_value = min(30, n_samples - 1)

    if n_samples <= 1:
        print("Not enough data points to visualize.")
        return
        
    tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42, max_iter=300)
    embeddings_2d = tsne.fit_transform(embeddings)

    print("Data reduction complete. Preparing data for plotting...")
    # 4. สร้าง DataFrame ด้วย Pandas เพื่อจัดการข้อมูล
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'document': documents,
        'source': [meta.get('source_pdf', 'N/A') for meta in metadatas]
    })
    
    # ตัดข้อความ document ให้สั้นลงเพื่อแสดงผลบน hover
    df['hover_text'] = df['document'].apply(lambda x: x[:200] + "..." if len(x) > 200 else x)

    print("Generating interactive plot with Plotly...")
    # 5. สร้าง interactive scatter plot ด้วย Plotly
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='source',  # แยกสีตามไฟล์ PDF ต้นฉบับ
        hover_name='hover_text', # ข้อความที่จะแสดงเมื่อเอาเมาส์ไปชี้
        title="2D Visualization of PDF Document Chunks (t-SNE)",
        labels={'color': 'Source PDF'}
    )

    fig.update_traces(marker=dict(size=8, opacity=0.8))
    fig.update_layout(
        xaxis_title="t-SNE Component 1",
        yaxis_title="t-SNE Component 2",
        legend_title="Source PDF"
    )
    
    # 6. แสดงผลกราฟ
    fig.show()
    print("Plot generated. Your browser should open with the interactive visualization.")

if __name__ == "__main__":
    visualize_vector_db()