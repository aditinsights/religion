#%%
import pandas 
import numpy as np
import PyPDF2 
import os
import tqdm
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')


class PDFLoad: 
    def __init__(self, root_directory):
        self.root_directory = root_directory


    def convert_pdfs_to_text(self):
        all_documents = []
        for dirpath, dirnames, filenames in os.walk(self.root_directory):
            for pdf_file in filenames:
                if pdf_file.endswith('.pdf'):
                    pdf_path = os.path.join(dirpath, pdf_file)
                    with open(pdf_path, 'rb') as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        # Iterate over each page in the PDF
                        for page_number, page in enumerate(pdf_reader.pages, start=1):
                            page_text = page.extract_text() if page.extract_text() else ""
                            all_documents.append({
                                'page_number': page_number,
                                'page_content': page_text,
                                'metadata': {'source': pdf_path}
                            })
        return all_documents
    
    def chunk_text(self, text, chunk_type='paragraph', chunk_size=5):
        if chunk_type == 'sentence':
            sentences = sent_tokenize(text)
            chunks = [' '.join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]
        elif chunk_type == 'paragraph':
            paragraphs = text.split('\n\n')  # Assumes two newlines mark a new paragraph
            chunks = paragraphs
        return chunks




# #%%
# pdf_dir = r'./data/bias'    
# pdf_loader = PDFLoad(pdf_dir)
# documents = pdf_loader.convert_pdfs_to_text()
# len(documents)

# %%
