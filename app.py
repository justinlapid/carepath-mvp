from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory
import os
from werkzeug.utils import secure_filename
from PIL import Image
import pytesseract
import PyPDF2
import docx
from pdf2image import convert_from_path
import json
import openai
from dotenv import load_dotenv
from weasyprint import HTML
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a strong secret key
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'docx', 'svg'}

def allowed_file(filename, allowed_set):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_set

def extract_text(filepath, filename):
    ext = filename.rsplit('.', 1)[1].lower()
    text = ""

    if ext == 'txt':
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
    elif ext == 'docx':
        doc = docx.Document(filepath)
        for para in doc.paragraphs:
            text += para.text + '\n'
    elif ext == 'pdf':
        with open(filepath, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            page_count = len(reader.pages)

            for page_num in range(page_count):
                extracted_page_text = reader.pages[page_num].extract_text()

                if extracted_page_text and extracted_page_text.strip():
                    text += f"\nPage {page_num + 1}:\n" + extracted_page_text + '\n'
                else:
                    images = convert_from_path(
                        filepath,
                        first_page=page_num + 1,
                        last_page=page_num + 1,
                        dpi=100  # Lower DPI to reduce memory usage
                    )
                    ocr_text = pytesseract.image_to_string(images[0])
                    text += f"\nPage {page_num + 1} (OCR):\n" + ocr_text + '\n'
    elif ext in {'png', 'jpg', 'jpeg', 'gif'}:
        img = Image.open(filepath)
        text = pytesseract.image_to_string(img)
    else:
        text = "Unsupported file format."

    return text

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                flash('No file selected for uploading.')
                return redirect(request.url)
            if file and allowed_file(file.filename, ALLOWED_EXTENSIONS):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                session['uploaded_filename'] = filename
                extracted_text = extract_text(filepath, filename)
                session['extracted_data'] = extracted_text
                return render_template('index.html', extracted_data=extracted_text)
            else:
                flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif, docx, svg.')
                return redirect(request.url)
    return render_template('index.html')

if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
