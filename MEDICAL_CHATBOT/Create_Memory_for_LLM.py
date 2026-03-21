# Load raw PDF(s)
# ● Create Chunks
# ● Create Vector Embeddings 
# ● Store embeddings in FAISS 
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredEPubLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain.text_splitter import RecursiveCharacterTextSplitter
#1 Load raw PDF(s)
DATA_PATH = "DATA/"
LOADER_MAPPING = {
    "*.pdf": PyPDFLoader,
    "*.epub": UnstructuredEPubLoader,
}
# def load_pdf_files(data): 
#     loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
#     documents=loader.load()
#     return documents

# documents_all = load_pdf_files(data=DATA_PATH)

# def load_all_documents(data_path):
#     all_documents = []

#     # Load PDFs
#     pdf_loader = DirectoryLoader(
#         data_path,
#         glob="*.pdf",
#         loader_cls=PyPDFLoader
#     )
#     all_documents.extend(pdf_loader.load())

#     # Load EPUBs
#     epub_loader = DirectoryLoader(
#         data_path,
#         glob="*.epub",
#         loader_cls=UnstructuredEPubLoader
#     )
#     all_documents.extend(epub_loader.load())

#     return all_documents
def load_all_documents(data_path):
    all_documents = []

    for pattern, loader_cls in LOADER_MAPPING.items():
        loader = DirectoryLoader(
            data_path,
            glob=pattern,
            loader_cls=loader_cls
        )
        try:
            docs = loader.load()
            all_documents.extend(docs)
        except Exception as e:
            print(f"Error loading {pattern}: {e}")

    return all_documents

documents_all = load_all_documents(DATA_PATH)
print('Length_of_pdf_files: ',len(documents_all))
# //run by : pipenv run python Create_Memory_for_LLM.py
# pip install pypandoc
#2 ● Create Chunks
def create_chunks(extacted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extacted_data)
    return text_chunks

text_chunks = create_chunks(documents_all)
print('Legth_of_text_chunks: ',len(text_chunks))
#3 ● Create Vector Embeddings 
# pipenv run python -c "from sentence_transformers import SentenceTransformer; print('OK')"
def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model=get_embedding_model()

#4 ● Store embeddings in FAISS
# DB_FAISS_PATH='D:\Codes\AI\MEDICAL_CHATBOT\db_faiss'
DB_FAISS_PATH='vectorstores/db_faiss'
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)
