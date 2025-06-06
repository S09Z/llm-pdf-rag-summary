import logging
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from logger_config import setup_logger

# สร้าง logger สำหรับไฟล์นี้
log = logging.getLogger(__name__)

# --- ค่าคงที่ (ต้องตรงกับไฟล์ vector_store_builder.py) ---
CHROMA_PERSIST_DIR = "chroma_db"
CHROMA_COLLECTION_NAME = "pdf_collection"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_MODEL_NAME = "llama3" # หรือชื่อโมเดลที่คุณ pull มา เช่น mistral, gemma:2b

class RAGSystem:
    def __init__(self):
        """
        Initialize the RAG system by setting up the LLM, vector store,
        retriever, and the QA chain.
        """
        log.info("Initializing RAG System...")

        # 1. ตั้งค่า LLM (Ollama)
        log.info(f"Loading LLM: [cyan]{OLLAMA_MODEL_NAME}[/cyan]", extra={"markup": True})
        self.llm = Ollama(model=OLLAMA_MODEL_NAME, temperature=0.1)
        log.info("LLM loaded.")

        # 2. โหลด Vector Store ที่มีอยู่
        log.info("Loading vector store...")
        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.vector_store = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embedding_model,
            collection_name=CHROMA_COLLECTION_NAME
        )
        log.info(f"Vector store loaded with {self.vector_store._collection.count()} items.")

        # 3. สร้าง Retriever
        # Retriever ทำหน้าที่ค้นหาข้อมูลที่เกี่ยวข้องจาก Vector Store
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity", # ประเภทการค้นหา
            search_kwargs={"k": 5}    # ดึงข้อมูลที่เกี่ยวข้องมา 5 chunks
        )
        log.info("Retriever created.")

        # 4. สร้าง Prompt Template (สำคัญมาก!)
        # เราจะสร้าง template เพื่อบอกให้ LLM รู้ว่าต้องตอบคำถามโดยอิงจาก "context" ที่เราป้อนให้เท่านั้น
        # ซึ่งจะช่วยลดการที่ LLM แต่งข้อมูลขึ้นมาเอง (hallucination)
        prompt_template = """
        [INST]
        You are a helpful assistant. Use the following pieces of context to answer the user's question accurately.
        If you don't know the answer from the provided context, just say that you don't know the answer based on the available documents, don't try to make up an answer.
        Provide a concise and to-the-point answer.

        Context:
        {context}

        Question:
        {question}

        Helpful Answer:
        [/INST]
        """
        self.prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        log.info("Prompt template created.")

        # 5. สร้าง RetrievalQA Chain
        # Chain นี้จะรวมทุกอย่างเข้าด้วยกัน: retriever -> prompt -> llm
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff", # "stuff" คือการนำ chunks ทั้งหมดมายัดรวมกันใน prompt
            retriever=self.retriever,
            return_source_documents=True, # ให้แสดงเอกสารอ้างอิงด้วย
            chain_type_kwargs={"prompt": self.prompt}
        )
        log.info("RetrievalQA chain created successfully.")
        log.info("[bold green]RAG System initialized and ready.[/bold green]", extra={"markup": True})

    def answer_question(self, query: str) -> dict:
        """
        รับคำถามจากผู้ใช้, ส่งให้ QA chain, และคืนค่าผลลัพธ์
        """
        if not query:
            return {"error": "Query cannot be empty."}

        log.info(f"Answering question: '[yellow]{query}[/yellow]'", extra={"markup": True})
        try:
            result = self.qa_chain.invoke({"query": query})
            return result
        except Exception as e:
            log.error(f"An error occurred while answering the question: {e}", exc_info=True)
            return {"error": str(e)}

if __name__ == '__main__':
    # --- ส่วนนี้สำหรับการทดสอบ ---
    setup_logger()

    # ตรวจสอบว่า Ollama ทำงานอยู่หรือไม่
    log.info(f"Please make sure the Ollama application is running and the '{OLLAMA_MODEL_NAME}' model is available.")

    try:
        rag_system = RAGSystem()

        # ทดสอบถามคำถาม (เปลี่ยนคำถามให้เกี่ยวกับ PDF ของคุณ)
        test_query = "What is the core idea of an LLM Twin?"
        response = rag_system.answer_question(test_query)

        print("\n" + "="*50)
        if "error" in response:
            log.error(f"Test query failed: {response['error']}")
        else:
            log.info(f"[bold]Query:[/bold] {response.get('query')}", extra={"markup": True})
            log.info(f"[bold cyan]Answer:[/bold cyan]\n{response.get('result')}", extra={"markup": True})

            # แสดง Source documents ที่ใช้ในการตอบ
            log.info("\n[bold]Source Documents:[/bold]", extra={"markup": True})
            for doc in response.get('source_documents', []):
                source_pdf = doc.metadata.get('source_pdf', 'N/A')
                page = doc.metadata.get('page', 'N/A')
                log.info(f"  - Source: {source_pdf}, Page: {page}", extra={"markup": True})
                # log.debug(f"    Content: {doc.page_content[:150]}...")
        print("="*50)

    except Exception as e:
        log.error(f"Failed to initialize or run RAG system: {e}", exc_info=True)