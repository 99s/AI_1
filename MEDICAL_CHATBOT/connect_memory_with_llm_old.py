# ● Setup LLM (Mistral with HuggingFace)
# ● Connect LLM with FAISS
# ● Create chain
# -------------------------------------------
import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain.chains.retrieval_qa.base import RetrievalQA
#1 ● Setup LLM (Mistral with HuggingFace)
HF_TOKEN = os.environ.get("HF_TOKEN")
huggingface_repo_id="mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={
            "token": HF_TOKEN,
            "max_length": 512
        }
    )
    return llm

# ● Connect LLM with FAISS
custom_prompt_template = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""
PROMPT_FILE_PATH = 'custom_prompts.txt'
def readPrompt():
    file_path = PROMPT_FILE_PATH
    try:
        with open(file_path, 'r') as file:
            lines_array = file.readlines()
        
        print(lines_array)

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")

readPrompt()
# pipenv run python connect_memory_with_llm.py
