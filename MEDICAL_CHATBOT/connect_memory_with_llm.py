# ● Setup LLM (Mistral with HuggingFace)
# ● Connect LLM with FAISS
# ● Create Retrieval Chain (NEW LangChain way)
# ------------------------------------------------

import os
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

# NEW imports (replace RetrievalQA)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


# -------------------------------
# 1. Setup LLM
# -------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")
# huggingface_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
# //RedHatAI/Meta-Llama-3.1-405B-Instruct-quantized.w4a16
# meta-llama/Llama-3.1-405B-Instruct
huggingface_repo_id = "meta-llama/Llama-3.1-405B-Instruct"


def load_llm():
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512
    )
    return llm


# -------------------------------
# 2. Load FAISS Vector DB
# -------------------------------
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2" #//model2 : used to convrt text for vectorembedding
    )

    db = FAISS.load_local(
        "vectorstores/db_faiss",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db


# -------------------------------
# 3. Custom Prompt
# -------------------------------
custom_prompt_template = """
Use the pieces of information provided in the context to answer the user's question.
If you dont know the answer, just say that you dont know.
Dont try to make up an answer.

Context:
{context}

Question:
{question}

Start the answer directly. No small talk please.
"""


def set_custom_prompt():
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )


# -------------------------------
# 4. Create Retrieval Chain (NEW)
# -------------------------------
def create_qa_chain(llm, db):
    prompt = set_custom_prompt()

    # Combines retrieved docs + prompt
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Creates retrieval pipeline
    qa_chain = create_retrieval_chain(
        db.as_retriever(),
        document_chain
    )

    return qa_chain


# -------------------------------
# 5. Read Prompt File (Optional)
# -------------------------------
PROMPT_FILE_PATH = 'custom_prompts.txt'


def read_prompt_file():
    try:
        with open(PROMPT_FILE_PATH, 'r') as file:
            for line in file:
                print(line.strip())
    except FileNotFoundError:
        print(f"Error: The file '{PROMPT_FILE_PATH}' was not found.")


# -------------------------------
# 6. Main Execution
# -------------------------------
if __name__ == "__main__":
    print("Loading LLM...")
    llm = load_llm()

    print("Loading Vector Store...")
    db = load_vectorstore()

    print("Creating QA Chain...")
    qa_chain = create_qa_chain(llm, db)

    # Optional: read prompt file
    read_prompt_file()

    print("\n✅ System Ready! Ask questions (type 'exit' to quit)\n")

    while True:
        user_query = input("Question: ")

        if user_query.lower() == "exit":
            break

        response = qa_chain.invoke({
            "input": user_query
        })

        print("\nAnswer:", response["answer"])
        print("-" * 50)