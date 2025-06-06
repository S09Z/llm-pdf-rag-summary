import logging
from rich.logging import RichHandler

def setup_logger():
    """
    ตั้งค่า logger กลางของโปรเจกต์ให้ใช้ RichHandler
    เพื่อให้แสดงผลใน Terminal ได้สวยงาม
    """
    # กำหนดว่าเราจะตั้งค่าสำหรับ root logger
    # การตั้งค่านี้จะส่งผลต่อ logger ทั้งหมดในโปรเจกต์
    logging.basicConfig(
        level="INFO",  # สามารถเปลี่ยนเป็น "DEBUG" เพื่อดูข้อมูลละเอียดขึ้น
        format="%(message)s", # รูปแบบ message (RichHandler จะจัดการส่วนที่เหลือเอง)
        datefmt="[%X]",      # รูปแบบเวลา (ถ้าใช้ใน format)
        handlers=[
            RichHandler(
                show_time=True,          # แสดงเวลา
                show_level=True,         # แสดง Level (INFO, DEBUG)
                show_path=True,          # แสดง path ของไฟล์ที่ log
                rich_tracebacks=True,    # *** ฟีเจอร์เด็ด: แสดง traceback สวยๆ ตอนเกิด error ***
                markup=True,             # เปิดใช้งาน markup เช่น [bold green]ข้อความ[/bold green]
                log_time_format="[%H:%M:%S]" # ปรับรูปแบบเวลา
            )
        ]
    )
    # ปิด logger ของไลบรารีอื่นๆ ที่อาจจะส่ง log เยอะเกินไป (ถ้าต้องการ)
    # logging.getLogger("urllib3").setLevel(logging.WARNING)

# เราสามารถเรียกใช้ logger ได้จากทุกที่โดยใช้ logging.getLogger(__name__)
# หลังจากที่เรียก setup_logger() แล้วหนึ่งครั้ง