import os
from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
from utils import PDFLoad, BiasGPT, DocumentAnalysis
from embed_generation import embeddings_for_analysis
from embed_analysis import keyword_embeddings, embeddings_analysis
from text_analysis import text_analysis

UPLOAD_FOLDER = 'data/bias'
ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        pdf_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(pdf_file_path)
        pdf_dir = os.path.dirname(pdf_file_path)
        print(f"Uploaded file path: {pdf_file_path}")  # Debugging statement
        print(f"Folder path: {pdf_dir}")  # Debugging statement
        return redirect(url_for('select_keywords', pdf_dir=pdf_dir, pdf_file_path=pdf_file_path))
    return redirect(request.url)

@app.route('/select_keywords', methods=['GET', 'POST'])
def select_keywords():
    pdf_dir = request.args.get('pdf_dir')
    pdf_file_path = request.args.get('pdf_file_path')
    pdf_dir = os.path.normpath(pdf_dir)
    pdf_file_path = os.path.normpath(pdf_file_path)
    embeddings_dir = os.path.join(pdf_dir, 'embeddings')
    print(f"PDF Directory: {pdf_dir}")  # Debugging statement
    print(f"PDF File Path: {pdf_file_path}")  # Debugging statement
    print(f"Embeddings Directory: {embeddings_dir}")  # Debugging statement
    if request.method == 'POST':
        keywords = request.form.get('keywords').split(',')
        analysis_type = request.form.get('analysis_type')
        api_key = os.getenv("OPENAI_API_KEY")
        results = []
        if analysis_type == 'embedding':
            bias_tool = BiasGPT()
            pdf_loader = PDFLoad(pdf_file_path)
            doc_analysis = DocumentAnalysis(api_key)
            documents, chunk_embeddings = embeddings_for_analysis(pdf_file_path, embeddings_dir, api_key)
            doc_analysis, keyword, keyword_embed = keyword_embeddings(keywords, api_key)
            results = embeddings_analysis(chunk_embeddings, documents, doc_analysis, keywords, keyword_embed)
        elif analysis_type == 'text':
            pdf_loader = PDFLoad(pdf_file_path)
            doc_analysis = DocumentAnalysis(api_key)
            bias_tool = BiasGPT()
            print(f"Keywords: {keywords}")  # Debugging statement
            print(f"PDF Directory: {pdf_dir}")  # Debugging statement
            results = text_analysis(pdf_file_path, pdf_loader, bias_tool, doc_analysis, keywords)
        if not results:
            results.append("No results found.")
        return render_template('results.html', results=results)
    return render_template('select_keywords.html', pdf_dir=pdf_dir, pdf_file_path=pdf_file_path)



@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
