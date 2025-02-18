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

########################################
# OPTIONAL: If you want to categorize these fields:
FIELDS = {
    "Demographics": [
        "First Name",
        "Last Name",
        "DOB",
        "Patient Address Line 1",
        "Patient Address Line 2",
        "Patient Address City",
        "Patient Address State",
        "Patient Address Zip Code",
        "Patient Address Full Address"
    ],
    "Insurance": [
        "Insurance Carrier",
        "Insurance id",
        "Insurance plan name"
    ],
    "Clinical Details": [
        "Ordering Provider First Name",
        "Ordering Provider Last Name",
        "Ordering Provider Name and Title",
        "Ordering Provider Address Line 1",
        "Ordering Provider Address Line 2",
        "Ordering Provider Address City",
        "Ordering Provider Address State",
        "Ordering Provider Address Zip Code",
        "Ordering Provider Full Address",
        "Ordering Provider Phone Number"
    ],
    "Medical Information": [
        "Medical History",
        "Presenting Symptom",
        "Vitals",
        "Assessments",
        "Medications"
    ]
}
########################################

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
        # Read PDF and handle page-by-page
        with open(filepath, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            page_count = len(reader.pages)

        for page_num in range(page_count):
            # Attempt direct text extraction first
            with open(filepath, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                extracted_page_text = reader.pages[page_num].extract_text()

            if extracted_page_text and extracted_page_text.strip():
                text += f"\nPage {page_num + 1}:\n" + extracted_page_text + "\n"
            else:
                # Convert only this page to image at a lower DPI for OCR
                images = convert_from_path(
                    filepath,
                    first_page=page_num + 1,
                    last_page=page_num + 1,
                    dpi=100  # Lower DPI to reduce memory usage
                )
                ocr_text = pytesseract.image_to_string(images[0])
                text += f"\nPage {page_num + 1} (OCR):\n" + ocr_text + "\n"

    elif ext in {'png', 'jpg', 'jpeg', 'gif'}:
        img = Image.open(filepath)
        text = pytesseract.image_to_string(img)
    else:
        text = "Unsupported file format."

    return text

def extract_fields_with_openai(extracted_text):
    """
    Sends the extracted text to OpenAI to parse out key fields.
    """
    # 1) Limit the text to prevent token/time issues
    MAX_INPUT_CHARACTERS = 3000  # Reduced from 6000
    if len(extracted_text) > MAX_INPUT_CHARACTERS:
        extracted_text = extracted_text[:MAX_INPUT_CHARACTERS]
        extracted_text += "\n\n[Text truncated to fit within token limits.]"

    # 2) Craft the prompt
    prompt = f"""
Extract the following fields from the text:

**Demographics:**
- First Name
- Last Name
- DOB
- Patient Address Line 1
- Patient Address Line 2
- Patient Address City
- Patient Address State
- Patient Address Zip Code
- Patient Address Full Address

**Insurance:**
- Insurance Carrier
- Insurance id
- Insurance plan name

**Clinical Details:**
- Ordering Provider First Name
- Ordering Provider Last Name
- Ordering Provider Name and Title
- Ordering Provider Address Line 1
- Ordering Provider Address Line 2
- Ordering Provider Address City
- Ordering Provider Address State
- Ordering Provider Address Zip Code
- Ordering Provider Full Address
- Ordering Provider Phone Number

**Medical Information:**
- Medical History
- Presenting Symptom
- Vitals
- Assessments
- Medications

Text:
\"\"\"
{extracted_text}
\"\"\"

Please provide the extracted information in JSON format, structured as follows:

{{
    "Demographics": {{
        "First Name": "...",
        "Last Name": "...",
        "DOB": "...",
        "Patient Address Line 1": "...",
        "Patient Address Line 2": "...",
        "Patient Address City": "...",
        "Patient Address State": "...",
        "Patient Address Zip Code": "...",
        "Patient Address Full Address": "..."
    }},
    "Insurance": {{
        "Insurance Carrier": "...",
        "Insurance id": "...",
        "Insurance plan name": "..."
    }},
    "Clinical Details": {{
        "Ordering Provider First Name": "...",
        "Ordering Provider Last Name": "...",
        "Ordering Provider Name and Title": "...",
        "Ordering Provider Address Line 1": "...",
        "Ordering Provider Address Line 2": "...",
        "Ordering Provider Address City": "...",
        "Ordering Provider Address State": "...",
        "Ordering Provider Address Zip Code": "...",
        "Ordering Provider Full Address": "...",
        "Ordering Provider Phone Number": "..."
    }},
    "Medical Information": {{
        "Medical History": ["...", "..."],
        "Presenting Symptom": ["...", "..."],
        "Vitals": ["...", "..."],
        "Assessments": ["...", "..."],
        "Medications": ["...", "..."]
    }}
}}

If a field is not available in the text, please leave it empty, null, or an empty list where appropriate.
"""

    # 3) Call OpenAI
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # or "gpt-3.5-turbo" if you prefer
            messages=[
                {"role": "system", "content": "You are an assistant that extracts structured data from medical documents."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0
        )
    except Exception as e:
        # If you're using a version of openai that has openai.error.OpenAIError, do this instead:
        # except openai.error.OpenAIError as e:
        return {"error": f"OpenAI API error: {str(e)}"}

    # 4) Parse the JSON from OpenAI
    content = response['choices'][0]['message']['content']
    try:
        extracted_data = json.loads(content)
    except json.JSONDecodeError:
        # Attempt to find any JSON in the response
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                extracted_data = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                extracted_data = {"error": "Failed to parse the response from OpenAI (JSON error)."}
        else:
            extracted_data = {"error": "Failed to parse the response from OpenAI (no JSON found)."}

    return extracted_data

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

                # (Optional) Check if file is too large before processing
                if os.path.getsize(filepath) > 5 * 1024 * 1024:  # e.g., 5 MB limit
                    flash("PDF is too large for processing on this free service.")
                    return redirect(request.url)

                # Extract text from the document
                extracted_text = extract_text(filepath, filename)

                # Send the text to OpenAI for structured field extraction
                extracted_data = extract_fields_with_openai(extracted_text)
                
                # Save these in session for later
                session['uploaded_filename'] = filename
                session['extracted_data'] = extracted_data
                
                # Render template with the extracted data
                return render_template('index.html', extracted_data=extracted_data)
            else:
                flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif, docx, svg.')
                return redirect(request.url)
    return render_template('index.html')

if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
