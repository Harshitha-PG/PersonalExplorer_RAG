import os
from pypdf import PdfReader
from PIL import Image
import pytesseract

# Windows tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

BASE_DIR = "test_data"

documents = []

# TXT FILES
for folder in ["notes", "texts"]:
    folder_path = os.path.join(BASE_DIR, folder)

    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            path = os.path.join(folder_path, file)

            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            documents.append({
                "content": text,
                "source": file
            })

# PDF FILES
pdf_path = os.path.join(BASE_DIR, "pdfs")

for file in os.listdir(pdf_path):
    if file.endswith(".pdf"):
        path = os.path.join(pdf_path, file)

        reader = PdfReader(path)

        text = ""

        for page in reader.pages:
            text += page.extract_text()

        documents.append({
            "content": text,
            "source": file
        })

# SCREENSHOTS OCR
image_path = os.path.join(BASE_DIR, "screenshots")

for file in os.listdir(image_path):
    if file.endswith((".png", ".jpg", ".jpeg")):
        path = os.path.join(image_path, file)

        image = Image.open(path)

        text = pytesseract.image_to_string(image)

        documents.append({
            "content": text,
            "source": file
        })

print(f"Loaded {len(documents)} documents")

for doc in documents[:5]:
    print("\nSOURCE:", doc["source"])
    print(doc["content"][:300])