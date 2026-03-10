import pdfplumber
import re

def clean_text(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def is_table(chunk):
    numbers = len(re.findall(r'\d', chunk))
    words = len(chunk.split())
    return numbers > words

def load_pdf(file):

    pages = []

    with pdfplumber.open(file) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):

            text = page.extract_text()

            if text:
                cleaned = clean_text(text)

                pages.append({
                    "page": page_num,
                    "text": cleaned
                })

    return pages