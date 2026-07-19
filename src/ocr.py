from PIL import Image
import pytesseract
import docx
import os


def ocr_image(path):
    try:
        img = Image.open(path).convert('RGB')
        text = pytesseract.image_to_string(img)
        return text
    except pytesseract.pytesseract.TesseractNotFoundError:
        # If tesseract is not available, return a placeholder
        # In production, this should use an alternative OCR service (e.g., Cloud Vision, Azure OCR)
        return f"[OCR not available for {os.path.basename(path)}]"
    except Exception as e:
        return f"[Error extracting text: {str(e)}]"


def extract_docx(path):
    doc = docx.Document(path)
    paragraphs = [p.text for p in doc.paragraphs]
    return '\n'.join(paragraphs)


def extract_text(path):
    _, ext = os.path.splitext(path.lower())
    if ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
        return ocr_image(path)
    if ext == '.docx':
        return extract_docx(path)
    # fallback: try to open as image
    try:
        return ocr_image(path)
    except Exception:
        return ''


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--file', required=True)
    args = p.parse_args()
    print(extract_text(args.file)[:1000])
