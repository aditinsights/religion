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
# pdf_dir = os.path.join('.', 'data', 'bias')  # replace with the correct directory

# #%% set upclasses 
#
def text_analysis(pdf_dir, pdf_loader, bias_tool, doc_analysis, search_words): 
    print(f"PDF Directory: {pdf_dir}")

    # Check if the file exists
    if not os.path.exists(pdf_dir):
        print(f"File not found: {pdf_dir}")
        return []

    # Ensure pdf_loader is initialized
    documents = pdf_loader.convert_pdfs_to_text()  
    print(f"Number of documents: {len(documents)}")

    found_sentences = []

    for doc in documents:
        page_content = doc['page_content']
        page_number = doc['page_number']
        sentences = sent_tokenize(page_content)
        for sentence in sentences:
            for word in search_words:
                print(f"Searching for '{word}' in sentence: {sentence.lower()}")
                if word.lower() in sentence.lower():  # Case insensitive search
                    found_sentences.append((page_number, sentence, word))
                    print(f"Found '{word}' in the sentence on page {page_number}: {sentence}")
                    break

    results = []
    print(f"Found sentences: {found_sentences}")
    if found_sentences:  # Only passes to GPT if keywords found 
        bias_answer = bias_tool.bias_detection(found_sentences[0][1])
        print(f"Bias answer: {bias_answer.choices[0].message.content}")
        for page_number, sentence, word in found_sentences:
            results.append({
                'page_number': page_number,
                'text': sentence,
                'keyword': word,
                'bias_answer': bias_answer.choices[0].message.content
            })
    else: 
        results.append("No bias found in the document.")

    return results


#%% check to make sure function working 
#found_sentences = text_analysis(pdf_dir, api_key, bias_tool, doc_analysis)

