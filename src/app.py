import streamlit as st
from qa_system import RAGSystem # Import คลาสระบบ Q&A ที่เราสร้างไว้
from logger_config import setup_logger

# ตั้งค่า Logger (เพื่อให้ log แสดงผลใน terminal ที่รัน streamlit)
setup_logger()

# ตั้งค่าหน้าเว็บ (ทำเป็นอันดับแรก)
st.set_page_config(
    page_title="PDF Q&A with Llama3",
    page_icon="🤖",
    layout="wide"
)

st.title("📚 ถาม-ตอบข้อมูลจากคลัง PDF ของคุณ")
st.markdown("ขับเคลื่อนโดย Ollama, LangChain, และ ChromaDB")

# --- ส่วนสำคัญ: Caching a Resource ---
@st.cache_resource
def load_rag_system():
    """
    โหลด RAGSystem เพียงครั้งเดียวและเก็บไว้ใน cache
    เพื่อไม่ต้องโหลดโมเดลใหม่ทุกครั้งที่ผู้ใช้ถามคำถาม
    """
    with st.spinner("กำลังเตรียมระบบ Q&A และโหลดโมเดล... (อาจใช้เวลาสักครู่ในครั้งแรก)"):
        system = RAGSystem()
    return system

# เรียกใช้ฟังก์ชันเพื่อโหลดระบบ (Streamlit จะจัดการ cache ให้เอง)
rag_system = load_rag_system()

# --- ส่วนของ User Interface ---

# สร้าง session state สำหรับเก็บประวัติการแชท (ถ้ายังไม่มี)
if "messages" not in st.session_state:
    st.session_state.messages = []

# แสดงประวัติการแชท
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# รับ input จากผู้ใช้
if prompt := st.chat_input("ถามอะไรเกี่ยวกับเอกสารของคุณก็ได้..."):
    # เพิ่มข้อความของผู้ใช้ไปยังประวัติการแชทและแสดงผล
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # สร้างคำตอบจาก RAG system
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("กำลังค้นหาข้อมูลและสร้างคำตอบ..."):
            response = rag_system.answer_question(prompt)

            if "error" in response:
                answer = f"เกิดข้อผิดพลาด: {response['error']}"
                st.error(answer)
            else:
                answer = response.get("result", "ไม่พบคำตอบ")
                message_placeholder.markdown(answer)

                # (Optional) แสดงเอกสารอ้างอิงใน expander
                sources = response.get("source_documents", [])
                if sources:
                    with st.expander("ดูแหล่งข้อมูลอ้างอิง"):
                        for doc in sources:
                            source_pdf = doc.metadata.get('source_pdf', 'N/A')
                            page = doc.metadata.get('page', 'N/A')
                            st.write(f"- **ไฟล์:** {source_pdf} (หน้า: {page})")

                            cleaned_content = doc.page_content[:200].strip().replace('\n', ' ')
                            st.caption(f"> {cleaned_content}...")


    # เพิ่มคำตอบของ assistant ไปยังประวัติการแชท
    st.session_state.messages.append({"role": "assistant", "content": answer})