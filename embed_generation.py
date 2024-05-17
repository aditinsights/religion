#%%
# general python libraries
import os
import pickle
import numpy as np 
from datetime import datetime
from tqdm import tqdm

#tools for embeddings search
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss

# tools for processing and analyzing text
import nltk
from nltk.tokenize import sent_tokenize
import openai
from openai import OpenAI

# custom classes built for project 
from utils import PDFLoad, BiasGPT, DocumentAnalysis

# commands to access env keys and data 
nltk.download('punkt')
api_key = os.getenv("OPENAI_API_KEY")

#%% file to investigate found here
def embeddings_for_analysis(pdf_file_path, embeddings_dir, api_key, base_file="bias", date_str=datetime.now().strftime('%Y%m%d')):
    pdf_loader = PDFLoad(pdf_file_path)  # check utils.py for class definition
    doc_analysis = DocumentAnalysis(api_key)  # load document analysis class
    documents = pdf_loader.convert_pdfs_to_text()  # Use the instance to call methods
    chunk_embeddings = doc_analysis.generate_embeddings(documents)  # actual embeddings generation 

    base_filename = base_file  # name this something descriptive for the embeddings being created
    pickle_filename = f"{base_filename}_{date_str}.pkl"
    full_path = os.path.join(embeddings_dir, pickle_filename)

    # Ensure the embeddings directory exists
    os.makedirs(embeddings_dir, exist_ok=True)

    with open(full_path, 'wb') as f:
        pickle.dump(chunk_embeddings, f)
        print(f"Saved all embeddings to {full_path}")

    return documents, chunk_embeddings


# #%% testing purposes 
# pdf_dir = os.path.join('.', 'data', 'no_bias')  # replace with the correct directory
# documents, chunk_embeddings = embeddings_for_analysis(pdf_dir, api_key)
# %%
