import os
import io
from typing import List, Tuple # เพิ่ม Tuple
from langchain.schema.document import Document # ตรวจสอบว่า import นี้ยังอยู่
from langchain_community.document_loaders import PyPDFLoader # PyPDFLoader ใช้ path ของไฟล์
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Imports for Google Drive
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

# If modifying these SCOPES, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
CREDENTIALS_FILE = 'credentials.json' # Path to your credentials.json
TOKEN_FILE = 'token.json' # Will be created automatically
TEMP_PDF_DIR = "temp_pdfs" # โฟลเดอร์สำหรับเก็บ PDF ที่ดาวน์โหลดชั่วคราว

def authenticate_google_drive():
    """Authenticates with Google Drive API and returns the service object."""
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0) # เปิด browser ให้ login
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
    try:
        service = build('drive', 'v3', credentials=creds)
        print("Successfully authenticated with Google Drive.")
        return service
    except HttpError as error:
        print(f'An error occurred during Google Drive authentication: {error}')
        return None

def list_pdfs_from_drive_folder(service, folder_id: str) -> List[Tuple[str, str]]: # คืนค่าเป็น List of (id, name)
    """Lists PDF files from a specific Google Drive folder."""
    pdf_files = []
    if not service:
        return pdf_files
    try:
        query = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
        results = service.files().list(
            q=query,
            pageSize=100, # ปรับขนาดได้ตามต้องการ
            fields="nextPageToken, files(id, name)"
        ).execute()
        items = results.get('files', [])
        if not items:
            print(f'No PDF files found in Google Drive folder ID: {folder_id}')
        else:
            for item in items:
                pdf_files.append((item['id'], item['name']))
            print(f"Found {len(pdf_files)} PDF(s) in Google Drive folder: {folder_id}")
    except HttpError as error:
        print(f'An error occurred while listing PDF files from Google Drive: {error}')
    return pdf_files

def download_pdf_from_drive(service, file_id: str, file_name: str) -> str | None:
    """Downloads a PDF file from Google Drive to a temporary local directory."""
    if not service:
        return None
    try:
        if not os.path.exists(TEMP_PDF_DIR):
            os.makedirs(TEMP_PDF_DIR)

        local_file_path = os.path.join(TEMP_PDF_DIR, file_name)
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        print(f"Downloading {file_name} from Google Drive...", end=" ")
        while done is False:
            status, done = downloader.next_chunk()
            # print(F'Download {int(status.progress() * 100)}.') # แสดง % progress ถ้าต้องการ
        print("Done.")

        with open(local_file_path, 'wb') as f:
            fh.seek(0)
            f.write(fh.read())
        print(f"File '{file_name}' downloaded to '{local_file_path}'")
        return local_file_path
    except HttpError as error:
        print(f'An error occurred while downloading {file_name}: {error}')
        return None
    except Exception as e:
        print(f"An unexpected error occurred during download of {file_name}: {e}")
        return None

def cleanup_temp_pdfs():
    """Removes the temporary PDF directory and its contents."""
    if os.path.exists(TEMP_PDF_DIR):
        import shutil
        try:
            shutil.rmtree(TEMP_PDF_DIR)
            print(f"Temporary PDF directory '{TEMP_PDF_DIR}' cleaned up.")
        except Exception as e:
            print(f"Error cleaning up temporary PDF directory: {e}")

def load_pdf(file_path: str) -> List[Document]: # ฟังก์ชันนี้ยังคงเดิม แต่จะใช้กับไฟล์ที่ดาวน์โหลดมา
    """โหลดข้อมูลจากไฟล์ PDF และคืนค่าเป็น List ของ Document."""
    if not os.path.exists(file_path):
        print(f"Error: Local PDF file not found at {file_path}")
        return []
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} page(s) from local file: {os.path.basename(file_path)}")
        return documents
    except Exception as e:
        print(f"Error loading PDF {os.path.basename(file_path)}: {e}")
        return []


def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """แบ่ง Document ออกเป็น Chunks เล็กๆ."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} document(s) into {len(chunks)} chunks.")
    return chunks

# --- ฟังก์ชันใหม่สำหรับดึง PDF จาก GDrive และประมวลผล ---
def get_all_document_chunks_from_gdrive(gdrive_folder_id: str) -> Tuple[List[Document], List[str]]:
    """
    ดึง PDF ทั้งหมดจาก Google Drive folder ที่ระบุ, โหลด, แบ่งเป็น chunks,
    และคืนค่าเป็น list ของ chunks และ list ของชื่อไฟล์ที่ประมวลผลสำเร็จ.
    """
    service = authenticate_google_drive()
    if not service:
        print("Failed to authenticate with Google Drive. Cannot process PDFs.")
        return [], []

    pdf_files_info = list_pdfs_from_drive_folder(service, gdrive_folder_id)
    all_chunks = []
    processed_file_names = []

    if not pdf_files_info:
        cleanup_temp_pdfs() # ลบ temp dir ถ้าไม่มีไฟล์เลย
        return [], []

    for file_id, file_name in pdf_files_info:
        print(f"\n--- Processing: {file_name} (ID: {file_id}) from Google Drive ---")
        local_pdf_path = download_pdf_from_drive(service, file_id, file_name)
        if local_pdf_path:
            loaded_docs = load_pdf(local_pdf_path)
            if loaded_docs:
                # เพิ่ม metadata ชื่อไฟล์ GDrive เข้าไปในแต่ละ document ก่อน chunk
                for doc in loaded_docs:
                    doc.metadata["source_gdrive_pdf"] = file_name # เก็บชื่อไฟล์ GDrive
                    doc.metadata["gdrive_file_id"] = file_id # เก็บ ID ไฟล์ GDrive
                document_chunks = chunk_documents(loaded_docs)
                all_chunks.extend(document_chunks)
                processed_file_names.append(file_name)
            else:
                print(f"Could not load documents from downloaded file: {file_name}")
        else:
            print(f"Failed to download: {file_name}")

    cleanup_temp_pdfs() # ลบไฟล์ PDF ชั่วคราวหลังประมวลผลเสร็จ
    return all_chunks, processed_file_names


if __name__ == '__main__':
    # --- ส่วนนี้สำหรับการทดสอบ ---
    # คุณจะต้องหา Folder ID ของ Google Drive ที่ต้องการดึง PDF
    # วิธีหา Folder ID: เปิด Google Drive ใน browser, เข้าไปในโฟลเดอร์นั้น
    # URL จะมีลักษณะประมาณนี้: https://drive.google.com/drive/folders/THIS_IS_THE_FOLDER_ID
    # ให้ copy ส่วน "THIS_IS_THE_FOLDER_ID" มาใส่
    GOOGLE_DRIVE_FOLDER_ID = "YOUR_GOOGLE_DRIVE_FOLDER_ID_HERE" # <--- แก้ไขตรงนี้!!!

    if GOOGLE_DRIVE_FOLDER_ID == "YOUR_GOOGLE_DRIVE_FOLDER_ID_HERE":
        print("Please update 'GOOGLE_DRIVE_FOLDER_ID' in src/pdf_processor.py with your actual Google Drive Folder ID.")
    else:
        print(f"--- Testing PDF Processor with Google Drive Folder ID: {GOOGLE_DRIVE_FOLDER_ID} ---")
        all_document_chunks, processed_files = get_all_document_chunks_from_gdrive(GOOGLE_DRIVE_FOLDER_ID)

        if all_document_chunks:
            print(f"\nSuccessfully processed {len(processed_files)} PDF(s): {', '.join(processed_files)}")
            print(f"Total chunks created: {len(all_document_chunks)}")
            print("\n--- Sample Chunk (First Chunk) ---")
            print(f"Content: {all_document_chunks[0].page_content[:200]}...")
            print(f"Metadata: {all_document_chunks[0].metadata}")
        elif processed_files: # มีไฟล์แต่สร้าง chunk ไม่ได้
             print(f"\nProcessed {len(processed_files)} PDF(s) but no chunks were created.")
        else:
            print("No PDF files were processed or no chunks were created from Google Drive.")