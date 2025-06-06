import os
import shutil
import logging
from typing import List # ไม่จำเป็นต้องใช้ Document ที่นี่แล้วถ้า all_chunks ถูกส่งมาโดยตรง
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
# Import ฟังก์ชันใหม่จาก pdf_processor
from pdf_processing import load_pdf, chunk_documents
from logger_config import setup_logger

log = logging.getLogger(__name__)

# ค่าคงที่ยังคงเดิม
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_SOURCE_DIR = os.path.join(SCRIPT_DIR, "temp")

CHROMA_PERSIST_DIR = "chroma_db"
CHROMA_COLLECTION_NAME = "pdf_collection"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def get_embedding_model(): # ฟังก์ชันนี้ยังคงเดิม
    """โหลด Embedding Model."""
    log.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}", extra={"markup": True})
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
    )
    log.info("Embedding model loaded.", extra={"markup": True})
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
            log.info(f"Loading existing vector store from: {CHROMA_PERSIST_DIR}", extra={"markup": True})
            vector_store = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=embedding_model,
                collection_name=CHROMA_COLLECTION_NAME
            )
            log.info(f"Vector store loaded. Collection: '{vector_store._collection.name}', Count: {vector_store._collection.count()}", extra={"markup": True})
            # ถ้ามี chunks ใหม่และต้องการ add เข้าไปใน store ที่มีอยู่ (ไม่ใช่ force_rebuild)
            if chunks and not force_rebuild:
                 # ตรวจสอบว่า chunks ที่จะ add มี content จริงๆ
                 valid_new_chunks = [chk for chk in chunks if hasattr(chk, 'page_content') and chk.page_content]
                 if valid_new_chunks:
                    log.info(f"Adding {len(valid_new_chunks)} new valid chunks to the existing vector store.", extra={"markup": True})
                    vector_store.add_documents(documents=valid_new_chunks) # ไม่ต้องส่ง embedding model ซ้ำ
                    vector_store.persist()
                    log.info(f"Vector store updated and persisted. New count: {vector_store._collection.count()}", extra={"markup": True})
                 else:
                    log.warning("No valid new chunks to add to the existing vector store.", extra={"markup": True})

        except Exception as e:
            log.error(f"Error loading existing vector store: {e}. Will try to build new one if chunks are provided.", extra={"markup": True})
            vector_store = None

    if vector_store is None and chunks: # ถ้าโหลดไม่ได้ หรือ force_rebuild และมี chunks
        # ตรวจสอบว่า chunks ที่จะใช้สร้าง store ใหม่ มี content จริงๆ
        valid_chunks_for_new_store = [chk for chk in chunks if hasattr(chk, 'page_content') and chk.page_content]
        if valid_chunks_for_new_store:
            if force_rebuild and os.path.exists(CHROMA_PERSIST_DIR): # ถ้า force rebuild ให้ลบของเก่า
                import shutil
                shutil.rmtree(CHROMA_PERSIST_DIR)
                log.info(f"Removed old persist directory for rebuild: {CHROMA_PERSIST_DIR}", extra={"markup": True})

            log.info(f"Building new vector store with {len(valid_chunks_for_new_store)} valid chunks and persisting to: {CHROMA_PERSIST_DIR}", extra={"markup": True})
            vector_store = Chroma.from_documents(
                documents=valid_chunks_for_new_store,
                embedding=embedding_model,
                persist_directory=CHROMA_PERSIST_DIR,
                collection_name=CHROMA_COLLECTION_NAME
            )
            vector_store.persist()
            log.info(f"Vector store built and persisted. Count: {vector_store._collection.count()}", extra={"markup": True})
        else:
            log.warning("No valid chunks provided to build a new vector store.", extra={"markup": True})
            vector_store = None # Ensure it's None if no valid chunks

    elif vector_store is None and not chunks:
            log.warning("No chunks provided and no existing vector store found or loadable.", extra={"markup": True})

    return vector_store
# ------ END COPIED build_or_load_vector_store ------


def process_gdrive_pdfs_and_build_store(gdrive_folder_id: str, force_rebuild: bool = False):
    """
    ประมวลผล PDF ทั้งหมดจาก Google Drive Folder ที่กำหนด และสร้าง/อัปเดต Vector Store.
    """
    log.info(f"--- Starting PDF processing from Google Drive Folder ID: {gdrive_folder_id} ---", extra={"markup": True})
    # 1. ดึง Chunks ทั้งหมดจาก Google Drive
    all_chunks, processed_files = get_all_document_chunks_from_gdrive(gdrive_folder_id)

    if not all_chunks:
        log.warning("No chunks were created from any PDF in Google Drive. Trying to load existing vector store if not forcing rebuild.", extra={"markup": True})
        # ถ้าไม่มี chunks ใหม่ แต่ไม่ได้สั่ง force_rebuild ก็ยังสามารถลองโหลด store เก่าได้
        return build_or_load_vector_store(chunks=None, force_rebuild=force_rebuild)

    print(f"\nTotal {len(all_chunks)} chunks obtained from {len(processed_files)} PDF(s): {', '.join(processed_files)}")
    if all_chunks:
            print(f"Sample metadata of first chunk: {all_chunks[0].metadata}")

    # 2. สร้างหรือโหลด Vector Store โดยใช้ Chunks ที่ได้มา
    vector_store = build_or_load_vector_store(chunks=all_chunks, force_rebuild=force_rebuild)
    return vector_store

def process_local_pdfs_and_build_store(pdf_directory: str, force_rebuild: bool = False):
    """
    ประมวลผล PDF ทั้งหมดใน Directory ที่กำหนด และสร้าง/อัปเดต Vector Store.
    """
    all_chunks = []
    if not os.path.exists(pdf_directory):
        log.error(f"PDF source directory not found at '[bold red]{pdf_directory}[/bold red]'", extra={"markup": True})
        return None

    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith(".pdf")]
    if not pdf_files:
        log.warning(f"No PDF files found in '[yellow]{pdf_directory}[/yellow]'", extra={"markup": True})
        return build_or_load_vector_store(chunks=None, force_rebuild=force_rebuild)

    log.info(f"Found {len(pdf_files)} PDF(s) in '[yellow]{pdf_directory}[/yellow]'", extra={"markup": True})
    for pdf_file in pdf_files:
        file_path = os.path.join(pdf_directory, pdf_file)
        log.info(f"\n--- Processing: {file_path} ---", extra={"markup": True})
        loaded_docs = load_pdf(file_path)
        if loaded_docs:
            # เพิ่ม metadata ชื่อไฟล์เข้าไปในแต่ละ document ก่อน chunk
            for doc in loaded_docs:
                doc.metadata["source_pdf"] = pdf_file # เก็บชื่อไฟล์ PDF
            document_chunks = chunk_documents(loaded_docs)
            all_chunks.extend(document_chunks)

    if not all_chunks:
        log.warning("No chunks were created from any PDF. Vector store not built.", extra={"markup": True})
        return build_or_load_vector_store(chunks=None, force_rebuild=force_rebuild)

    log.info(f"\nTotal chunks from all PDFs: {len(all_chunks)}", extra={"markup": True})
    if all_chunks:
         log.info(f"Sample metadata of first chunk: {all_chunks[0].metadata}", extra={"markup": True})

    # สร้างหรือโหลด Vector Store โดยใช้ Chunks ที่ได้มา
    log.info(f"Found {len(pdf_files)} PDF(s) in '[yellow]{pdf_directory}[/yellow]'", extra={"markup": True})
    vector_store = build_or_load_vector_store(chunks=all_chunks, force_rebuild=force_rebuild)
    return vector_store

if __name__ == '__main__':
    # --- ส่วนนี้สำหรับการทดสอบ ---
    # คุณจะต้องหา Folder ID ของ Google Drive ที่ต้องการดึง PDF
    # GOOGLE_DRIVE_FOLDER_ID = "YOUR_GOOGLE_DRIVE_FOLDER_ID_HERE" # <--- แก้ไขตรงนี้!!!

    # if GOOGLE_DRIVE_FOLDER_ID == "YOUR_GOOGLE_DRIVE_FOLDER_ID_HERE":
    #     print("Please update 'GOOGLE_DRIVE_FOLDER_ID' in src/vector_store_builder.py with your actual Google Drive Folder ID.")
    # else:
    #     print("--- Building Vector Store (Indexing) from Google Drive PDFs ---")
        # force_rebuild=True ถ้าต้องการสร้างฐานข้อมูลใหม่ทั้งหมดจาก PDF ใน GDrive
        # force_rebuild=False จะพยายามโหลด store เก่าก่อน ถ้ามี chunks ใหม่จาก GDrive ก็จะ add เข้าไป (ถ้า logic ใน build_or_load_vector_store รองรับ)
    # vs = process_gdrive_pdfs_and_build_store(gdrive_folder_id=GOOGLE_DRIVE_FOLDER_ID, force_rebuild=False)

    # *** เรียกใช้การตั้งค่า logger เป็นอันดับแรก ***
    setup_logger()

    log.info("--- [bold]Starting Vector Store Build Process[/bold] ---", extra={"markup": True})

    if not os.path.exists(PDF_SOURCE_DIR):
         log.error(f"Directory not found: {PDF_SOURCE_DIR}")
         log.warning("Please create a 'temp' folder inside your 'src' directory and add PDF files to it.")
    else:
        vs = process_local_pdfs_and_build_store(pdf_directory=PDF_SOURCE_DIR, force_rebuild=False)

        if vs:
            log.info("[bold green]Vector Store Ready![/bold green]", extra={"markup": True})
            # ... (ส่วนทดสอบ similarity search)
            # สามารถใช้ log.debug() เพื่อแสดงข้อมูลที่ไม่ต้องการให้เห็นในโหมดปกติ
            sample_query = "What is Retrieval Augmented Generation?"
            log.debug(f"Testing similarity search with query: '{sample_query}'")
            results = vs.similarity_search(sample_query, k=2)
            if results:
                # Rich สามารถแสดงผล list/dict สวยๆ ได้เลย
                log.info("Top 2 similar chunks found:")
                # แสดงผล result ที่ 1 แบบสวยๆ
                log.info(results[0])
            else:
                 log.warning("No similar chunks found for the sample query.")
        else:
            log.error("Vector store could not be built or loaded.")