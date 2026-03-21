# Load raw PDF(s) + EPUB(s)
# ● Create Chunks
# ● (Next steps: Embeddings + FAISS)

import os
from ebooklib import epub
from bs4 import BeautifulSoup

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document


# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = "DATA/"


# -----------------------------
# EPUB LOADER (Manual - Stable)
# -----------------------------
def load_epub_manual(file_path):
    documents = []

    try:
        book = epub.read_epub(file_path)

        for item in book.get_items():
            if item.get_type() == 9:  # DOCUMENT
                soup = BeautifulSoup(item.get_content(), "html.parser")
                text = soup.get_text(separator=" ", strip=True)

                if text:  # avoid empty chunks
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={"source": file_path}
                        )
                    )

    except Exception as e:
        print(f"Error loading EPUB {file_path}: {e}")

    return documents


# -----------------------------
# LOAD ALL DOCUMENTS
# -----------------------------
def load_all_documents(data_path):
    all_documents = []

    # 🔹 Load PDFs
    try:
        pdf_loader = DirectoryLoader(
            data_path,
            glob="*.pdf",
            loader_cls=PyPDFLoader
        )
        pdf_docs = pdf_loader.load()
        print(f"Loaded PDFs: {len(pdf_docs)}")
        all_documents.extend(pdf_docs)

    except Exception as e:
        print(f"Error loading PDFs: {e}")

    # 🔹 Load EPUBs (manual)
    epub_count = 0
    for file in os.listdir(data_path):
        if file.endswith(".epub"):
            file_path = os.path.join(data_path, file)
            epub_docs = load_epub_manual(file_path)
            epub_count += len(epub_docs)
            all_documents.extend(epub_docs)

    print(f"Loaded EPUB sections: {epub_count}")

    return all_documents


# -----------------------------
# CREATE CHUNKS
# -----------------------------
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)
    return chunks


# -----------------------------
# MAIN EXECUTION // pipenv run python createMemoryForLLM.py1111
# pipenv install langchain langchain-community langchain-text-splitters ebooklib beautifulsoup4
# -----------------------------
if __name__ == "__main__":
    documents_all = load_all_documents(DATA_PATH)

    print("Total raw documents:", len(documents_all))

    chunks = split_documents(documents_all)

    print("Total chunks created:", len(chunks))

    # Preview
    if chunks:
        print("\nSample chunk:\n")
        print(chunks[0].page_content[:500])