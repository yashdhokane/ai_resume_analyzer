import os
import re
from flask import Flask, render_template, request
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted
    return text.lower()  # convert to lowercase for keyword matching

def calculate_match(resume_text, job_text):
    embeddings = model.encode([resume_text, job_text])
    score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return round(score * 100, 2)

def get_match_status(score):
    if score >= 70:
        return "Matched", "Strong match with job requirements", "green"
    elif score >= 40:
        return "Partially Matched", "Resume needs improvement", "orange"
    else:
        return "Not Matched", "Low relevance to job description", "red"

def extract_keywords(text):
    # simple regex to get words longer than 3 letters
    words = re.findall(r'\b\w{4,}\b', text.lower())
    return set(words)

def get_keyword_feedback(resume_text, job_text):
    job_keywords = extract_keywords(job_text)
    resume_keywords = extract_keywords(resume_text)

    matched_keywords = job_keywords.intersection(resume_keywords)
    missing_keywords = job_keywords - resume_keywords

    return list(matched_keywords), list(missing_keywords)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    keyword_feedback = None

    if request.method == 'POST':
        resume_file = request.files['resume']
        job_desc = request.form['job_description']

        resume_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
        resume_file.save(resume_path)

        resume_text = extract_text_from_pdf(resume_path)
        score = calculate_match(resume_text, job_desc)

        status, message, color = get_match_status(score)
        matched_keywords, missing_keywords = get_keyword_feedback(resume_text, job_desc)

        result = {
            "score": score,
            "status": status,
            "message": message,
            "color": color,
            "matched_keywords": matched_keywords,
            "missing_keywords": missing_keywords
        }

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=False)
