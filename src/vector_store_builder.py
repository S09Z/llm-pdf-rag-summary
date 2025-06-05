import os
from typing import List # ไม่จำเป็นต้องใช้ Document ที่นี่แล้วถ้า all_chunks ถูกส่งมาโดยตรง
# from langchain.schema.document import Document # อาจจะไม่จำเป็นต้อง import โดยตรงถ้า all_chunks ถูกส่งมาแล้ว
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
# Import ฟังก์ชันใหม่จาก pdf_processor
from src.pdf_processor import get_all_document_chunks_from_gdrive

# ค่าคงที่ยังคงเดิม
CHROMA_PERSIST_DIR = "chroma_db"
CHROMA_COLLECTION_NAME = "pdf_collection"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def get_embedding_model(): # ฟังก์ชันนี้ยังคงเดิม
    """โหลด Embedding Model."""
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
    )
    print("Embedding model loaded.")
    return embeddings

# def build_or_load_vector_store(...) # ฟังก์ชันนี้ยังคงเดิม (ตรวจสอบโค้ดจากคำตอบก่อนหน้านี้ของคุณ)
# *** คัดลอกฟังก์ชัน build_or_load_vector_store จากคำตอบก่อนหน้านี้มาใส่ตรงนี้ ***
# ให้แน่ใจว่าฟังก์ชันนี้รับ `chunks: List[Document]` และ `embedding_model`
# (โค้ดของ build_or_load_vector_store จากคำตอบก่อนหน้าค่อนข้างยาว ผมขอละไว้เพื่อให้คำตอบนี้ไม่ยาวเกินไป
# กรุณานำโค้ดส่วนนั้นมาใส่เองนะครับ)
# ------ BEGIN COPIED build_or_load_vector_store ------
def build_or_load_vector_store(chunks: List[Document] = None, embedding_model=None, force_rebuild: bool = False):
    """
    สร้าง Vector Store ใหม่จาก Chunks หรือโหลด Vector Store ที่มีอยู่.
    """
    if embedding_model is None:
        embedding_model = get_embedding_model()

    vector_store = None
    if not force_rebuild and os.path.exists(CHROMA_PERSIST_DIR):
        try:
            print(f"Loading existing vector store from: {CHROMA_PERSIST_DIR}")
            vector_store = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=embedding_model,
                collection_name=CHROMA_COLLECTION_NAME
            )
            print(f"Vector store loaded. Collection: '{vector_store._collection.name}', Count: {vector_store._collection.count()}")
            # ถ้ามี chunks ใหม่และต้องการ add เข้าไปใน store ที่มีอยู่ (ไม่ใช่ force_rebuild)
            if chunks and not force_rebuild:
                 # ตรวจสอบว่า chunks ที่จะ add มี content จริงๆ
                 valid_new_chunks = [chk for chk in chunks if hasattr(chk, 'page_content') and chk.page_content]
                 if valid_new_chunks:
                    print(f"Adding {len(valid_new_chunks)} new valid chunks to the existing vector store.")
                    vector_store.add_documents(documents=valid_new_chunks) # ไม่ต้องส่ง embedding model ซ้ำ
                    vector_store.persist()
                    print(f"Vector store updated and persisted. New count: {vector_store._collection.count()}")
                 else:
                    print("No valid new chunks to add to the existing vector store.")

        except Exception as e:
            print(f"Error loading existing vector store: {e}. Will try to build new one if chunks are provided.")
            vector_store = None

    if vector_store is None and chunks: # ถ้าโหลดไม่ได้ หรือ force_rebuild และมี chunks
        # ตรวจสอบว่า chunks ที่จะใช้สร้าง store ใหม่ มี content จริงๆ
        valid_chunks_for_new_store = [chk for chk in chunks if hasattr(chk, 'page_content') and chk.page_content]
        if valid_chunks_for_new_store:
            if force_rebuild and os.path.exists(CHROMA_PERSIST_DIR): # ถ้า force rebuild ให้ลบของเก่า
                import shutil
                shutil.rmtree(CHROMA_PERSIST_DIR)
                print(f"Removed old persist directory for rebuild: {CHROMA_PERSIST_DIR}")

            print(f"Building new vector store with {len(valid_chunks_for_new_store)} valid chunks and persisting to: {CHROMA_PERSIST_DIR}")
            vector_store = Chroma.from_documents(
                documents=valid_chunks_for_new_store,
                embedding=embedding_model,
                persist_directory=CHROMA_PERSIST_DIR,
                collection_name=CHROMA_COLLECTION_NAME
            )
            vector_store.persist()
            print(f"Vector store built and persisted. Count: {vector_store._collection.count()}")
        else:
            print("No valid chunks provided to build a new vector store.")
            vector_store = None # Ensure it's None if no valid chunks

    elif vector_store is None and not chunks:
            print("No chunks provided and no existing vector store found or loadable.")

    return vector_store
# ------ END COPIED build_or_load_vector_store ------


def process_gdrive_pdfs_and_build_store(gdrive_folder_id: str, force_rebuild: bool = False):
    """
    ประมวลผล PDF ทั้งหมดจาก Google Drive Folder ที่กำหนด และสร้าง/อัปเดต Vector Store.
    """
    print(f"--- Starting PDF processing from Google Drive Folder ID: {gdrive_folder_id} ---")
    # 1. ดึง Chunks ทั้งหมดจาก Google Drive
    all_chunks, processed_files = get_all_document_chunks_from_gdrive(gdrive_folder_id)

    if not all_chunks:
        print("No chunks were created from any PDF in Google Drive. Trying to load existing vector store if not forcing rebuild.")
        # ถ้าไม่มี chunks ใหม่ แต่ไม่ได้สั่ง force_rebuild ก็ยังสามารถลองโหลด store เก่าได้
        return build_or_load_vector_store(chunks=None, force_rebuild=force_rebuild)

    print(f"\nTotal {len(all_chunks)} chunks obtained from {len(processed_files)} PDF(s): {', '.join(processed_files)}")
    if all_chunks:
            print(f"Sample metadata of first chunk: {all_chunks[0].metadata}")

    # 2. สร้างหรือโหลด Vector Store โดยใช้ Chunks ที่ได้มา
    vector_store = build_or_load_vector_store(chunks=all_chunks, force_rebuild=force_rebuild)
    return vector_store

if __name__ == '__main__':
    # --- ส่วนนี้สำหรับการทดสอบ ---
    # คุณจะต้องหา Folder ID ของ Google Drive ที่ต้องการดึง PDF
    GOOGLE_DRIVE_FOLDER_ID = "YOUR_GOOGLE_DRIVE_FOLDER_ID_HERE" # <--- แก้ไขตรงนี้!!!

    if GOOGLE_DRIVE_FOLDER_ID == "YOUR_GOOGLE_DRIVE_FOLDER_ID_HERE":
        print("Please update 'GOOGLE_DRIVE_FOLDER_ID' in src/vector_store_builder.py with your actual Google Drive Folder ID.")
    else:
        print("--- Building Vector Store (Indexing) from Google Drive PDFs ---")
        # force_rebuild=True ถ้าต้องการสร้างฐานข้อมูลใหม่ทั้งหมดจาก PDF ใน GDrive
        # force_rebuild=False จะพยายามโหลด store เก่าก่อน ถ้ามี chunks ใหม่จาก GDrive ก็จะ add เข้าไป (ถ้า logic ใน build_or_load_vector_store รองรับ)
        vs = process_gdrive_pdfs_and_build_store(gdrive_folder_id=GOOGLE_DRIVE_FOLDER_ID, force_rebuild=False)

        if vs:
            print("\n--- Vector Store Ready ---")
            print(f"Chroma DB Persist Directory: {os.path.abspath(CHROMA_PERSIST_DIR)}")
            print(f"Collection Name: {vs._collection.name}")
            print(f"Number of items in collection: {vs._collection.count()}")

            print("\n--- Testing Vector Store (Sample Similarity Search) ---")
            try:
                sample_query = "What is a key concept discussed?" # เปลี่ยนเป็นคำถามที่เกี่ยวข้องกับ PDF ของคุณ
                results = vs.similarity_search(sample_query, k=2)
                if results:
                    print(f"\nTop 2 similar chunks for query: '{sample_query}'")
                    for i, doc in enumerate(results):
                        print(f"\n--- Result {i+1} ---")
                        print(f"Source PDF (GDrive): {doc.metadata.get('source_gdrive_pdf', 'N/A')} (ID: {doc.metadata.get('gdrive_file_id', 'N/A')})")
                        print(f"Content snippet: {doc.page_content[:200]}...")
                else:
                    print("No similar chunks found for the sample query.")
            except Exception as e:
                print(f"Error during similarity search test: {e}")
        else:
            print("Vector store could not be built or loaded from Google Drive PDFs.")