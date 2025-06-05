import pytest
import os
import io
from unittest.mock import patch, MagicMock, mock_open # mock_open สำหรับ token/credentials

# สมมติว่าโค้ดของคุณอยู่ใน src/pdf_processor.py
# เราต้องมั่นใจว่า Python สามารถหา module นี้เจอ
# วิธีหนึ่งคือการเพิ่ม src เข้าไปใน sys.path ชั่วคราวใน conftest.py หรือตั้งค่า PYTHONPATH
# หรือถ้าคุณรัน pytest จาก root ของ project และมี __init__.py ใน src มันอาจจะหาเจอ
# เพื่อความง่ายในตัวอย่างนี้ จะสมมติว่ามันหาเจอได้ (อาจจะต้องปรับการ import ถ้ามีปัญหา)
from src import pdf_processor
from langchain.schema.document import Document
from googleapiclient.errors import HttpError

# --- Mock Data ---
MOCK_PDF_CONTENT_BYTES = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n4 0 obj\n<< /Length 55 >>\nstream\nBT\n/F1 12 Tf\n100 700 Td\n(Hello Test PDF!) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000010 00000 n \n0000000059 00000 n \n0000000118 00000 n \n0000000198 00000 n \ntrailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n256\n%%EOF"

MOCK_GDRIVE_FILES_RESPONSE_SINGLE_PDF = {
    'files': [{'id': 'fake_pdf_id_1', 'name': 'sample_test_rag_gdrive.pdf'}]
}
MOCK_GDRIVE_FILES_RESPONSE_EMPTY = {'files': []}


@pytest.fixture
def mock_gdrive_service():
    """Fixture เพื่อสร้าง mock Google Drive service object."""
    mock_service = MagicMock()

    # Mock list method
    mock_files_list = MagicMock()
    mock_service.files.return_value.list = mock_files_list

    # Mock get_media method
    mock_files_get_media = MagicMock()
    mock_service.files.return_value.get_media = mock_files_get_media

    return mock_service

@pytest.fixture
def mock_pypdfloader():
    """Fixture เพื่อ mock PyPDFLoader."""
    with patch('src.pdf_processor.PyPDFLoader') as mock_loader_class:
        mock_loader_instance = MagicMock()
        # สมมติว่า .load() คืนค่า list ของ Document 1 หน้า ที่มี content ง่ายๆ
        mock_loader_instance.load.return_value = [
            Document(page_content="This is a mock PDF page content.", metadata={'source': 'mocked.pdf', 'page': 0})
        ]
        mock_loader_class.return_value = mock_loader_instance
        yield mock_loader_class


# --- Test Authentication ---
@patch('src.pdf_processor.os.path.exists')
@patch('src.pdf_processor.Credentials')
@patch('src.pdf_processor.InstalledAppFlow')
@patch('src.pdf_processor.build')
def test_authenticate_google_drive_first_run(mock_build, mock_flow, mock_creds, mock_exists, mocker):
    """Test Case 1.1: การยืนยันตัวตนครั้งแรก."""
    mock_exists.return_value = False # token.json does not exist
    mock_flow_instance = MagicMock()
    mock_flow_instance.run_local_server.return_value = MagicMock(spec=pdf_processor.Credentials) # Return a Credentials-like object
    mock_flow.from_client_secrets_file.return_value = mock_flow_instance

    mock_creds_instance = MagicMock(spec=pdf_processor.Credentials)
    mock_creds_instance.to_json.return_value = '{"fake": "token_content"}'
    mock_flow_instance.run_local_server.return_value = mock_creds_instance # Simulate creds returned

    # Mock open for writing token.json
    m = mock_open()
    with patch('src.pdf_processor.open', m):
        service = pdf_processor.authenticate_google_drive()

    mock_flow.from_client_secrets_file.assert_called_once_with(pdf_processor.CREDENTIALS_FILE, pdf_processor.SCOPES)
    mock_flow_instance.run_local_server.assert_called_once()
    m.assert_called_once_with(pdf_processor.TOKEN_FILE, 'w')
    m().write.assert_called_once_with('{"fake": "token_content"}')
    mock_build.assert_called_once()
    assert service is not None

@patch('src.pdf_processor.os.path.exists')
@patch('src.pdf_processor.Credentials')
@patch('src.pdf_processor.build')
def test_authenticate_google_drive_with_token(mock_build, mock_creds_class, mock_exists):
    """Test Case 1.2: การยืนยันตัวตนครั้งถัดไป (มี token.json)."""
    mock_exists.return_value = True # token.json exists
    mock_creds_instance = MagicMock(spec=pdf_processor.Credentials)
    mock_creds_instance.valid = True # Token is valid
    mock_creds_class.from_authorized_user_file.return_value = mock_creds_instance

    service = pdf_processor.authenticate_google_drive()

    mock_creds_class.from_authorized_user_file.assert_called_once_with(pdf_processor.TOKEN_FILE, pdf_processor.SCOPES)
    mock_build.assert_called_once_with('drive', 'v3', credentials=mock_creds_instance)
    assert service is not None

# --- Test File Listing ---
def test_list_pdfs_from_drive_folder_success(mock_gdrive_service):
    """Test Case 2.1: ลิสต์ไฟล์ PDF จากโฟลเดอร์ที่ถูกต้อง."""
    mock_gdrive_service.files.return_value.list.return_value.execute.return_value = MOCK_GDRIVE_FILES_RESPONSE_SINGLE_PDF
    folder_id = "test_folder_id_with_pdfs"
    pdf_files = pdf_processor.list_pdfs_from_drive_folder(mock_gdrive_service, folder_id)

    mock_gdrive_service.files.return_value.list.assert_called_once_with(
        q=f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false",
        pageSize=100,
        fields="nextPageToken, files(id, name)"
    )
    assert len(pdf_files) == 1
    assert pdf_files[0] == ('fake_pdf_id_1', 'sample_test_rag_gdrive.pdf')

def test_list_pdfs_from_drive_folder_empty(mock_gdrive_service):
    """Test Case 2.2: โฟลเดอร์ Google Drive ไม่มีไฟล์ PDF."""
    mock_gdrive_service.files.return_value.list.return_value.execute.return_value = MOCK_GDRIVE_FILES_RESPONSE_EMPTY
    folder_id = "test_folder_id_empty"
    pdf_files = pdf_processor.list_pdfs_from_drive_folder(mock_gdrive_service, folder_id)
    assert len(pdf_files) == 0

def test_list_pdfs_from_drive_folder_api_error(mock_gdrive_service):
    """Test Case 2.3: Google Drive Folder ID ไม่ถูกต้อง (จำลอง API Error)."""
    # จำลอง HttpError (คุณอาจจะต้อง import HttpError จาก googleapiclient.errors)
    # ในที่นี้เราจะทำให้ execute() raise exception
    mock_gdrive_service.files.return_value.list.return_value.execute.side_effect = HttpError(
        resp=MagicMock(status=404, reason="Not Found"), content=b"Folder not found"
    )
    folder_id = "invalid_folder_id"
    pdf_files = pdf_processor.list_pdfs_from_drive_folder(mock_gdrive_service, folder_id)
    assert len(pdf_files) == 0 # ควรจะคืนค่า list ว่างเมื่อเกิด error

# --- Test File Downloading ---
@patch('src.pdf_processor.io.BytesIO') # Mock BytesIO
@patch('src.pdf_processor.MediaIoBaseDownload') # Mock MediaIoBaseDownload
@patch('src.pdf_processor.os.makedirs') # Mock os.makedirs
def test_download_pdf_from_drive_success(mock_makedirs, mock_media_downloader_class, mock_bytes_io_class, mock_gdrive_service, tmp_path):
    """Test Case 3.1: การดาวน์โหลด PDF ที่ถูกต้อง."""
    # Monkeypatch TEMP_PDF_DIR ให้ใช้ tmp_path ของ pytest เพื่อความสะอาด
    # โดยปกติแล้วไม่ควรแก้ไข global variable ของ module อื่นโดยตรงในเทส
    # แต่เพื่อความง่ายสำหรับโครงสร้างโค้ดเดิม
    with patch('src.pdf_processor.TEMP_PDF_DIR', str(tmp_path)):
        file_id = "fake_pdf_id_1"
        file_name = "test_download.pdf"
        expected_local_path = tmp_path / file_name

        # Mock MediaIoBaseDownload
        mock_downloader_instance = MagicMock()
        # ทำให้ next_chunk() คืนค่า (status, done=True) ในครั้งเดียว
        mock_downloader_instance.next_chunk.return_value = (MagicMock(progress=lambda: 1.0), True)
        mock_media_downloader_class.return_value = mock_downloader_instance

        # Mock BytesIO
        mock_fh = MagicMock(spec=io.BytesIO)
        mock_fh.read.return_value = MOCK_PDF_CONTENT_BYTES # เนื้อหา PDF จำลอง
        mock_bytes_io_class.return_value = mock_fh

        # Mock open for writing the downloaded file
        m_open = mock_open()
        with patch('src.pdf_processor.open', m_open):
            local_file_path = pdf_processor.download_pdf_from_drive(mock_gdrive_service, file_id, file_name)

        mock_gdrive_service.files.return_value.get_media.assert_called_once_with(fileId=file_id)
        mock_media_downloader_class.assert_called_once() # ตรวจสอบว่าถูกเรียกด้วย fh และ request
        mock_fh.seek.assert_called_once_with(0)
        m_open.assert_called_once_with(expected_local_path, 'wb')
        m_open().write.assert_called_once_with(MOCK_PDF_CONTENT_BYTES)
        assert local_file_path == str(expected_local_path)
        # os.makedirs ไม่ควรถูกเรียกถ้า tmp_path (ซึ่งแทน TEMP_PDF_DIR) มีอยู่แล้ว
        # แต่ถ้าโค้ด pdf_processor เช็คและสร้าง TEMP_PDF_DIR เสมอ ก็อาจจะต้อง mock_makedirs.assert_called_once_with(str(tmp_path))


# --- Test End-to-End PDF Processing from GDrive (Mocked) ---
@patch('src.pdf_processor.authenticate_google_drive')
@patch('src.pdf_processor.list_pdfs_from_drive_folder')
@patch('src.pdf_processor.download_pdf_from_drive')
@patch('src.pdf_processor.load_pdf') # Mock การโหลด PDF จริง
@patch('src.pdf_processor.chunk_documents') # Mock การแบ่ง chunks
@patch('src.pdf_processor.cleanup_temp_pdfs') # Mock การลบ temp files
def test_get_all_document_chunks_from_gdrive_e2e_mocked(
    mock_cleanup, mock_chunk_docs, mock_load_pdf,
    mock_download_pdf, mock_list_pdfs, mock_auth_gdrive, tmp_path
):
    """Test Case: ทดสอบกระบวนการทั้งหมดแบบ end-to-end โดย mock ทุกส่วนที่ติดต่อภายนอก."""
    gdrive_folder_id = "mock_folder_id"

    # 1. Mock authenticate_google_drive
    mock_service_instance = MagicMock() # Service object จำลอง
    mock_auth_gdrive.return_value = mock_service_instance

    # 2. Mock list_pdfs_from_drive_folder
    pdf_files_info = [('id_1', 'doc1.pdf'), ('id_2', 'doc2.pdf')]
    mock_list_pdfs.return_value = pdf_files_info

    # 3. Mock download_pdf_from_drive
    # ให้คืนค่า path จำลองที่อยู่ใน tmp_path
    def mock_download_side_effect(service, file_id, file_name):
        # สร้างไฟล์เปล่าๆ ใน tmp_path เพื่อให้ os.path.exists ใน load_pdf (ถ้าไม่ mock load_pdf) ผ่าน
        # หรือถ้า mock load_pdf ก็ไม่จำเป็นต้องสร้างไฟล์จริง
        return str(tmp_path / file_name)
    mock_download_pdf.side_effect = mock_download_side_effect

    # 4. Mock load_pdf
    # ให้คืนค่า Document object จำลอง
    mock_doc1_page1 = Document(page_content="Content from doc1 page 1", metadata={'source_gdrive_pdf': 'doc1.pdf', 'gdrive_file_id': 'id_1', 'page': 0})
    mock_doc2_page1 = Document(page_content="Content from doc2 page 1", metadata={'source_gdrive_pdf': 'doc2.pdf', 'gdrive_file_id': 'id_2', 'page': 0})
    # load_pdf ควรคืนค่า list ของ Documents ต่อ 1 ไฟล์ PDF
    # เราจะทำให้มันคืนค่าตามชื่อไฟล์ที่ถูก