#%%
import pandas 
import numpy as np
import PyPDF2 
import os
import tqdm
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
nltk.download('punkt')

import openai
from openai import OpenAI
import os
import dotenv 

# openai.api_key = os.getenv("OPENAI_API_KEY")
# print(openai.api_key)
client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

#%%
class BiasGPT: 
    def __init__(self): 
        self = self

    def query_creation(self, query): 
        query_str = ', '.join(query)
        completion = client.chat.completions.create(
        model="gpt-4-turbo",  # keep up with latest model - balance with cost 
        messages=[
            {"role": "system", "content": f"You are given this query: {query_str}. You should generate three similar responses from it. Our goal is to pass these queries to a vector database to find close matches to the users intended statement."}, 
            {"role": "user", "content": query_str}
        ]
        )

        return completion#.choices[0].message.content

    def bias_detection(self, sentence): 
        sentence = ', '.join(sentence)
        completion = client.chat.completions.create(
        model="gpt-4-turbo",  # keep up with latest model - balance with cost 
        messages=[
            {"role": "system", "content": f"You are given this information: {sentence}. Do you detect any bias in this statement? If so, answer yes or no. If yes, please answer, without any other explanation, whether there is sexual, religious, or other bias. In addition, return the entire sentence that was provided to you which most closely detected that bias and explain why you thought that."}, 
            {"role": "user", "content": sentence}
        ]
        )

        return completion#.choices[0].message['content']


class PDFLoad: 
    def __init__(self, file_path):
        self.file_path = file_path

    def convert_pdfs_to_text(self):
        all_documents = []
        pdf_path = self.file_path
        if pdf_path.endswith('.pdf'):
            print(f"Processing PDF: {pdf_path}")
            with open(pdf_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page_number, page in enumerate(pdf_reader.pages, start=1):
                    page_text = page.extract_text() if page.extract_text() else ""
                    all_documents.append({
                        'page_number': page_number,
                        'page_content': page_text,
                        'metadata': {'source': pdf_path}
                    })
        print(f"Total documents processed: {len(all_documents)}")
        return all_documents

    def chunk_text(self, text, chunk_type='paragraph', chunk_size=5):
        if chunk_type == 'sentence':
            sentences = sent_tokenize(text)
            chunks = [' '.join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]
        elif chunk_type == 'paragraph':
            paragraphs = text.split('\n\n')  # Assumes two newlines mark a new paragraph
            chunks = paragraphs
        return chunks
    
class DocumentAnalysis:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key
        self.client = OpenAI(api_key=self.openai_api_key)

    def generate_embeddings(self, documents):
        client = OpenAI(api_key=self.openai_api_key)
        embed = []
        for document in tqdm(documents):
            text = document['page_content']  # Assuming document has page_content attribute
            response = client.embeddings.create(
                input=text, 
                model="text-embedding-3-large"  
            )
            embed.append(response.data[0].embedding)
        return embed

    def find_biased_chunks(self, chunks, chunk_embeddings, keyword_embeddings, threshold=0.40):
        results = []
        print(f"Length of chunks: {len(chunks)}")
        print(f"Length of chunk_embeddings: {len(chunk_embeddings)}")
        print(f"Length of keyword_embeddings: {len(keyword_embeddings)}")

        for i, chunk_embedding in enumerate(chunk_embeddings):
            chunk_embedding = chunk_embedding.reshape(1, -1)
            scores = cosine_similarity(chunk_embedding, keyword_embeddings)[0]
            max_score = max(scores)
            # Check if the max score is above the threshold
            if max_score >= threshold:
                max_index = scores.argmax()  # Get the index of the highest score
                # Perform bias detection
                bias_generator = BiasGPT()
                bias_answer = bias_generator.bias_detection([chunks[i]])
                # Append the result with the max score
                results.append((chunks[i], bias_answer, max_index, max_score))
                print(f"Found bias: chunk {i}, score {max_score}")
                print(results)

        return results
    
    def find_bias_faiss(self, chunks, chunk_embeddings, keyword, keyword_embeddings, threshold=0.10):
        # Build the FAISS index
        #print(keyword_embeddings.shape)
        embedding_dim = keyword_embeddings.shape[1]
        index = faiss.IndexFlatL2(embedding_dim)  # Use L2 distance
        faiss.normalize_L2(keyword_embeddings)   # Normalize embeddings for cosine similarity
        index.add(keyword_embeddings)            # Add keyword embeddings to the index

        results = []
        for i, chunk_embedding in enumerate(chunk_embeddings):
            chunk_embedding = chunk_embedding.reshape(1, -1)
            faiss.normalize_L2(chunk_embedding)  # Normalize chunk embedding for cosine similarity
            distances, indices = index.search(chunk_embedding, 10)  # Search for the nearest neighbor
            score = 1 - distances[0][0]  # Convert L2 distance to cosine similarity
            if score >= threshold:
                keyword_index = indices[0][0]
                if keyword_index >= len(keyword):
                    print(f"Skipping keyword_index {keyword_index} as it exceeds keyword list length.")
                    continue
                bias_generator = BiasGPT()
                bias_answer = bias_generator.bias_detection([chunks[i]])
                results.append((chunks[i], bias_answer, keyword[keyword_index], score))
                print(f"Found bias: chunk {i}, keyword {keyword[keyword_index]}, score {score}")
                print(results)
        return results


#%% testing 
# pdf_dir = r'./data/bias'    
# pdf_loader = PDFLoad(pdf_dir)
# documents = pdf_loader.convert_pdfs_to_text()
# len(documents)

# %%
