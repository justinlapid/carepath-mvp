# app.py
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
from datetime import datetime  # Imported for print date

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a strong secret key
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Load environment variables from .env file
load_dotenv()

# Fetch OpenAI API Key from environment variables
openai.api_key = os.getenv('OPENAI_API_KEY')

if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'docx', 'svg'}

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
            for page_num, page in enumerate(reader.pages):
                extracted_page_text = page.extract_text()
                if extracted_page_text:
                    text += f"\nPage {page_num + 1}:\n" + extracted_page_text + '\n'
                else:
                    images = convert_from_path(filepath)
                    if page_num < len(images):
                        text += f"\nPage {page_num + 1} (OCR):\n"
                        text += pytesseract.image_to_string(images[page_num]) + '\n'
    elif ext in {'png', 'jpg', 'jpeg', 'gif'}:
        img = Image.open(filepath)
        text = pytesseract.image_to_string(img)
    else:
        text = "Unsupported file format."

    return text

def extract_fields_with_openai(extracted_text):
    # Optional: Limit the extracted_text to a certain number of characters to prevent exceeding token limits
    MAX_INPUT_CHARACTERS = 6000  # Adjust as needed based on average tokens per character
    if len(extracted_text) > MAX_INPUT_CHARACTERS:
        extracted_text = extracted_text[:MAX_INPUT_CHARACTERS]
        extracted_text += "\n\n[Text truncated to fit within token limits.]"

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

    # OpenAI Chat API call
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an assistant that extracts structured data from medical documents."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,  # Reduced from 5000 to prevent exceeding context limit
            temperature=0
        )
    except openai.error.OpenAIError as e:
        return {"error": f"OpenAI API error: {str(e)}"}

    content = response['choices'][0]['message']['content']
    try:
        extracted_data = json.loads(content)
    except json.JSONDecodeError:
        # If JSON parsing fails, attempt to extract JSON from the response
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                extracted_data = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                extracted_data = {"error": "Failed to parse the response from OpenAI."}
        else:
            extracted_data = {"error": "Failed to parse the response from OpenAI."}

    return extracted_data

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Handle Documentation Upload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                flash('No file selected for uploading.')
                return redirect(request.url)
            if file and allowed_file(file.filename, ALLOWED_EXTENSIONS):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Store filename in session for later use (preview/download)
                session['uploaded_filename'] = filename

                extracted_text = extract_text(filepath, filename)
                # Do not remove the file; keep it for preview/download

                extracted_data = extract_fields_with_openai(extracted_text)
                session['extracted_data'] = extracted_data  # Store extracted data in session

                return render_template('index.html', extracted_data=extracted_data)
            else:
                flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif, docx, svg.')
                return redirect(request.url)
    return render_template('index.html')

@app.route('/preview/<filename>')
def preview_file(filename):
    # Security check: ensure the filename is secure
    filename = secure_filename(filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        flash('File not found.')
        return redirect(url_for('upload_file'))
    
    # Determine file type for appropriate handling
    ext = filename.rsplit('.', 1)[1].lower()
    if ext in {'png', 'jpg', 'jpeg', 'gif', 'svg'}:
        # Serve image directly
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    elif ext == 'pdf':
        # Serve PDF with appropriate headers
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    elif ext == 'txt':
        # Read and display text file content
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return f"<pre>{content}</pre>"
    elif ext == 'docx':
        # For simplicity, indicate that preview is not available
        return "<p>Preview not available for DOCX files. Please download to view.</p>"
    else:
        return "<p>Preview not available for this file type.</p>"

@app.route('/download/<filename>')
def download_file(filename):
    # Security check: ensure the filename is secure
    filename = secure_filename(filename)
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/download_summary')
def download_summary():
    if 'uploaded_filename' not in session or 'extracted_data' not in session:
        flash('No extracted data available to download.')
        return redirect(url_for('upload_file'))

    extracted_data = session.get('extracted_data', None)
    if not extracted_data:
        flash('No extracted data available to download.')
        return redirect(url_for('upload_file'))

    if 'error' in extracted_data:
        flash(f"Error in extracted data: {extracted_data['error']}")
        return redirect(url_for('upload_file'))

    # Extract last name, first name, DOB
    last_name = extracted_data['Demographics'].get('Last Name', 'Unknown')
    first_name = extracted_data['Demographics'].get('First Name', 'Unknown')
    dob = extracted_data['Demographics'].get('DOB', 'Unknown')

    # Form the filename
    raw_filename = f"{last_name}, {first_name} ({dob})"
    # Sanitize the filename
    safe_filename = secure_filename(raw_filename) + '.pdf'

    # Get current date and time for print date
    print_date = datetime.now().strftime("%B %d, %Y")

    # Render a separate HTML template for PDF
    rendered = render_template('summary.html', extracted_data=extracted_data, print_date=print_date)

    # Convert HTML to PDF
    try:
        pdf = HTML(string=rendered, base_url=request.base_url).write_pdf()
    except Exception as e:
        flash(f"Error generating PDF: {str(e)}")
        return redirect(url_for('upload_file'))

    # Save PDF with the desired filename
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
    with open(pdf_path, 'wb') as f:
        f.write(pdf)

    # Send PDF as response
    return send_from_directory(directory=app.config['UPLOAD_FOLDER'], path=safe_filename, as_attachment=True, download_name=safe_filename)

if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
