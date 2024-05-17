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
from embed_generation import embeddings_for_analysis

# commands to access env keys and data 
nltk.download('punkt')
api_key = os.getenv("OPENAI_API_KEY")

#%%
def keyword_embeddings(query, api_key): 
    bias_tool = BiasGPT()
    keywords = bias_tool.query_creation(query) 
    keyword = keywords.choices[0].message.content.split('\n')
    keyword_docs = [{'page_content': k} for k in keyword]
    doc_analysis = DocumentAnalysis(api_key) # load document analysis class
    keyword_embed = doc_analysis.generate_embeddings(keyword_docs)
    keyword_embed = [embedding for embedding in keyword_embed]
    keyword_embed = np.array(keyword_embed, dtype=np.float32)

    return doc_analysis, keyword, keyword_embed

def embeddings_analysis(chunk_embeddings, documents, doc_analysis, keywords, keyword_embeddings, threshold=0.40):
    chunk_embeddings = [embedding for embedding in chunk_embeddings]
    chunk_embeddings = np.array(chunk_embeddings, dtype=np.float32)

    results = []
    chunks = [doc['page_content'] for doc in documents if 'page_content' in doc]
    print(f"Total documents processed: {len(documents)}")
    print(f"Total chunks created: {len(chunks)}")

    # Find biased chunks using the revised function
    biased_chunks = doc_analysis.find_biased_chunks(chunks, chunk_embeddings, keyword_embeddings, threshold)
    #biased_chunks = doc_analysis.find_bias_faiss(chunks, chunk_embeddings, keywords, keyword_embeddings, threshold)
    if biased_chunks:
        print(f"Found {len(biased_chunks)} biased chunks.")
        # Collect the biased chunks for each document
        for doc in tqdm(documents):
            doc_chunks = [chunk for chunk in biased_chunks if chunk[0] in doc['page_content']]
            for text, bias_answer, max_index, score in doc_chunks:  # Unpack all four values
                results.append({
                    'page_number': doc.get('page_number', 'Unknown'),
                    'text': text[:1000],
                    'bias_answer': bias_answer.choices[0].message.content,
                    #'keyword': keywords[max_index],  # Assign the keyword
                    'score': score
                })

    return results


#%% test working 

# pdf_dir = os.path.join('.', 'data', 'no_bias')  # replace with the correct directory
# documents, chunk_embeddings = embeddings_for_analysis(pdf_dir, api_key, base_file = "bias", date_str = datetime.now().strftime('%Y%m%d'))
# query = ["Does this text mention abstinence only sex education"] 
#     "prohibit comprehensive sex education",
#     "evolution disclaimers", "teaching Intelligent Design",
#     "alternatives to evolution", "homosexuality a detriment to school children",
#     "encouragement of heterosexuality in curricula", "prohibition of homosexual activity",
#     "allowance of group religious practice"

# doc_analysis, keyword, keyword_embeddings = keyword_embeddings(query)
# embeddings_analysis(chunk_embeddings, documents, doc_analysis, keyword, keyword_embeddings)
# %%
