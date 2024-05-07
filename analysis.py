#%%
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from utils import PDFLoad
import nltk
from nltk.tokenize import sent_tokenize
import openai
from openai import OpenAI

nltk.download('punkt')

class DocumentAnalysis:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key
    
    def generate_embeddings(self, documents):
        client = OpenAI(api_key=self.openai_api_key)
        embed = []
        for document in documents:
            text = document['page_content']  # Assuming document has page_content attribute
            response = client.embeddings.create(
                input=text, 
                model="text-embedding-3-large"  # Using "ada" for demonstration, adjust as needed
            )
            embed.append(response.data[0].embedding)
        return embed

    def find_biased_chunks(self, chunks, keywords, keyword_embeddings):
        formatted_chunks = [{'page_content': chunk} for chunk in chunks if isinstance(chunk, str)]
        chunk_embeddings = self.generate_embeddings(formatted_chunks)
        results = []
        for i, chunk_embedding in enumerate(chunk_embeddings):
            scores = cosine_similarity([chunk_embedding], keyword_embeddings)[0]
            if max(scores) >= 0.5:
                max_index = scores.argmax()
                results.append((chunks[i], keywords[max_index], max(scores)))
        return results

    
#%%
pdf_dir = r'./data/bias'    
pdf_loader = PDFLoad(pdf_dir)  # Use a different variable name for the instance
documents = pdf_loader.convert_pdfs_to_text()  # Use the instance to call methods

api_key = os.getenv("OPENAI_API_KEY")
doc_analysis = DocumentAnalysis(api_key)

# Keywords for the biases
keywords = [
    "abstinence sex education", "prohibit comprehensive sex education",
    "evolution disclaimers", "teaching Intelligent Design",
    "alternatives to evolution", "homosexuality a detriment to school children",
    "encouragement of heterosexuality in curricula", "prohibition of homosexual activity",
    "allowance of group religious practice"
]

keyword_docs = [{'page_content': k} for k in keywords]
# Generate embeddings for these keywords
keyword_embeddings = doc_analysis.generate_embeddings(keyword_docs)

for doc in tqdm(documents):
    # Assume doc is a string or has 'page_content' directly
    if isinstance(doc, dict) and 'page_content' in doc:
        page_content = [doc['page_content']]
    elif isinstance(doc, str):
        page_content = [doc]
    biased_chunks = doc_analysis.find_biased_chunks(page_content, keywords, keyword_embeddings)
    if biased_chunks:
        print(f"Found {len(biased_chunks)} biased chunks on page {doc.get('page_number', 'Unknown')}.")
        for text, keyword, score in biased_chunks:
            print(f"Page {doc.get('page_number', 'Unknown')} Chunk: {text[:1000]}... Keyword: {keyword}... Score: {score}")
#%%
search_word = "homosexual"  # The word you're looking for
found_sentences = []  # To keep track of pages where the word is found

# Loop through each document's pages
for doc in documents:
    page_content = doc['page_content']
    page_number = doc['page_number']
    # Tokenize the content into sentences
    sentences = sent_tokenize(page_content)
    # Search each sentence for the word
    for sentence in sentences:
        if search_word.lower() in sentence.lower():  # Case insensitive search
            found_sentences.append((page_number, sentence))
            print(f"Found '{search_word}' in the sentence on page {page_number}: {sentence}")

# After searching through all the documents
if not found_sentences:
    print(f"No occurrences of '{search_word}' found in the document.")
else:
    print(f"'{search_word}' found in the following sentences:")
    for page, sentence in found_sentences:
        print(f"Page {page}: {sentence}")

# %%
