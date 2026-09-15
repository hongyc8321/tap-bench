import sys

path = "/home/rhap/nkst/基于奈奎斯特采样准则的影视级图文视频大模型训练与生成技术白皮书.pdf"

text = None
lib = None

try:
    import fitz
    doc = fitz.open(path)
    text = "\n".join(page.get_text() for page in doc)
    lib = "pymupdf"
except ImportError:
    pass

if text is None:
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            text = "\n".join((p.extract_text() or "") for p in pdf.pages)
        lib = "pdfplumber"
    except ImportError:
        pass

if text is None:
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        text = "\n".join((p.extract_text() or "") for p in reader.pages)
        lib = "pypdf"
    except ImportError:
        pass

if text is None:
    print("NO_PDF_LIB")
    sys.exit(1)

with open("/home/rhap/nkst/whitepaper.txt", "w", encoding="utf-8") as f:
    f.write(text)
print("OK", lib, len(text))
