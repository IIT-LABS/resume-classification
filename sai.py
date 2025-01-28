import spacy
import re
import pandas as pd
import streamlit as st
import joblib
from PyPDF2 import PdfReader
from docx import Document
import os
import requests
from io import BytesIO
import base64
from word2number import w2n  # Import word2number to convert word-based numbers to numeric
 
# Load the English language model for spaCy
nlp = spacy.load("en_core_web_sm")
 
# Helper functions to extract text from PDF and DOCX files
def extract_text_from_pdf(file):
    """Extract text from a PDF file."""
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text
 
def extract_text_from_docx(file):
    """Extract text from a DOCX file."""
    doc = Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text
 
# Modified function to extract experience
def extract_experience(text):
    """Extract experience details like '10 years' or 'more than thirty years'."""
    text = text.lower()
    numeric_pattern = r"(?:more than|over|at least|around|approximately|nearly|up to)?\s*(\d+)\+?\s*years?"
    numeric_match = re.search(numeric_pattern, text)
    if numeric_match:
        years = numeric_match.group(1)
        return f"{years} years"
    word_pattern = r"(?:more than|over|at least|around|approximately|nearly|up to)?\s*(\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)\b)\s*years?"
    word_match = re.search(word_pattern, text)
    if word_match:
        try:
            word_years = word_match.group(1)
            numeric_years = w2n.word_to_num(word_years)
            return f"{numeric_years} years"
        except ValueError:
            return "Experience not found"
    return "Not found"
 
def extract_certifications_count(text):
    """Extract the count of certifications mentioned under the 'CERTIFICATIONS' subheading."""
    certifications_section_pattern = r"(?<=\bCERTIFICATIONS\b)(.?)(?=\n\s\n|\Z)"
    certifications_section = re.search(certifications_section_pattern, text, re.IGNORECASE | re.DOTALL)
    if certifications_section:
        certifications_text = certifications_section.group(1).strip()
        certifications = [line.strip() for line in re.split(r"[\n•]", certifications_text) if line.strip()]
        return len(certifications)
    return 0
 
def extract_name_from_text(text):
    text = text.strip()
    lines = text.split("\n")
    for line in lines[:3]:
        line = line.strip()
        if len(line) > 1:
            name = re.sub(r'[^a-zA-Z\s]', '', line)
            if len(name.split()) > 1:
                return name.title()
    return "Name not found"
 
def extract_skills_without_keywords(text):
    doc = nlp(text)
    skills = []
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART"]:
            skills.append(ent.text)
    skill_patterns = re.findall(r"(proficient in|experienced with|skilled in|knowledge of)\s+([a-zA-Z0-9\s\+\-]+)", text, re.IGNORECASE)
    for pattern in skill_patterns:
        skills.append(pattern[1].strip())
    return list(set(skills))
 
def extract_visa_status(text):
    visa_keywords = {
        "H1B": ["h1b"],
        "Green Card": ["green card", "permanent resident"],
        "US Citizen": ["usc", "us citizen", "citizenship: us"],
        "OPT": ["opt"],
        "CPT": ["cpt"],
        "L2": ["l2 visa"],
        "EAD": ["ead"],
        "TN Visa": ["tn visa"],
        "Study Visa": ["study visa"]
    }
    visa_status = []
    for visa, patterns in visa_keywords.items():
        for pattern in patterns:
            if re.search(pattern, text.lower()):
                visa_status.append(visa)
                break
    return ", ".join(visa_status) if visa_status else "Not found"
 
def extract_details(text, job_description):
    name = extract_name_from_text(text)
    experience = extract_experience(text)
    skills = extract_skills_without_keywords(text)
    email = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', text)
    email = email.group(0) if email else "Not found"
    phone = re.search(r'(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\+\d{1,3}\s?\(?\d{1,4}\)?[-.\s]?\d{3}[-.\s]?\d{4})', text)
    phone = phone.group(0) if phone else "Not found"
    location = re.search(r'\b(?:[A-Z][a-z]+(?:\s[A-Z][a-z]+)*,\s(?:TX|CA|NY|FL|WA|IL|PA|GA|NC|OH|NJ|VA|CO|AZ|MA|MD|TN|MO|IN|WI|MN|SC|AL|LA|KY|OR|OK|CT|IA|MS|KS|AR|NV|UT|NM|NE|WV|ID|HI|ME|NH|MT|RI|DE|SD|ND|AK|VT|WY))\b|\b\d{5}(?:-\d{4})?\b', text)
    location = location.group(0) if location else "Not found"
    visa_status = extract_visa_status(text)
    certificates_count = extract_certifications_count(text)
 
    jd_skills = set(re.findall(r'[a-zA-Z0-9\+\-]+', job_description.lower()))
    resume_skills = set([skill.lower() for skill in skills])
    matching_skills = resume_skills.intersection(jd_skills)
    score = len(matching_skills)
 
    return {
        "Name": name,
        "Email": email,
        "Phone": phone,
        "Experience": experience,
        "Location": location,
        "Visa Status": visa_status,
        "Skills": skills,
        "Certificates Count": certificates_count,
        "Score": score,
    }
 
# Streamlit UI
st.markdown(
    """
<style>
    body {
        background-color: #000000;
        color: #FFFFFF;
    }
    .header {
        background-color: #1E90FF;
        color: white;
        padding: 10px 30px;
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 20px;
        position: relative;
    }
    .header .title {
        font-size: 30px;
        font-weight: bold;
    }
    .logo {
        margin-right: 20px;
    }
    .footer {
        background-color: #1E90FF;
        color: white;
        padding: 10px;
        text-align: center;
    }
    .quote-container {
        background-color: #000000;
        color: white;
        padding: 20px;
        text-align: center;
        margin-top: 30px;
        border-radius: 10px;
        font-family: 'Cursive', sans-serif;
    }
    .quote-container p {
        font-size: 24px;
        font-style: italic;
        margin: 8px 0;
    }
</style>
    """,
    unsafe_allow_html=True,
)
  # Define the background image path
background_path = r"C:/Users/shake/Downloads/deployment code/deployment code/background.jpg"

# Initialize encoded_background with a default empty value
encoded_background = ""

# Check if the file exists before using it
if os.path.exists(background_path):
    with open(background_path, "rb") as bg_file:
        encoded_background = base64.b64encode(bg_file.read()).decode()
        print("✅ Background image loaded successfully!")
else:
    print(f"❌ Error: Background image not found at {background_path}")
    # Serve the image via Streamlit's static file serving
    st.markdown(
        f"""
        <style>
        body {{
            background-image: url("file://{background_image_path}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
# Path to your logo
logo_path = "C:/Users/shake/Downloads/deployment code/deployment code/innosoullogo.jpeg"


# Encode the logo image to base64
import os
import base64

# Corrected file path (use raw string `r""` or double slashes)
logo_path = r"C:/Users/shake/Downloads/deployment code/deployment code/innosoullogo.jpeg"

# Initialize encoded_logo with a default empty value
encoded_logo = ""

# Check if the file exists
if os.path.exists(logo_path):
    with open(logo_path, "rb") as logo_file:
        encoded_logo = base64.b64encode(logo_file.read()).decode()
        print("✅ Logo file loaded successfully!")
else:
    print(f"❌ Error: File not found at {logo_path}")


# Add the header with logo and navigation links
st.markdown(
    f"""
    <div style="
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: #1E90FF;
        padding: 10px;
        border-radius: 5px;
    ">
        <div style="display: flex; align-items: center;">
            <img src="data:image/jpeg;base64,{encoded_logo}" alt="Logo" style="width: 100px; height: 100px; margin-right: 15px;">
            <h1 style="color: black; margin: 0;">Resume Shortlisted</h1>
        </div>
        <div style="display: flex; gap: 20px; align-items: center;">
            <a href="#" style="color: white; text-decoration: none; font-size: 15px;">Home</a>
            <a href="#" style="color: white; text-decoration: none; font-size: 15px;">Careers</a>
            <a href="#" style="color: white; text-decoration: none; font-size: 15px;">AboutUs</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
 
# Container with quotes
st.markdown('''
    <div class="quote-container">
        <p>&#128144 With great AI comes great responsibility</p>
        <p>The best way to predict the future is to invent it.&#128526</p>
    </div>
''', unsafe_allow_html=True)
 
# Input fields with custom design
resume_input = st.text_input("Enter the path or URL containing your resume (PDF or DOCX):", key="resume", placeholder="Paste your file path or URL here", help="Enter a valid resume link or local file path.")
job_description_input = st.text_area("Enter the job description:", height=200, placeholder="Paste the job description here")
 
# Stylish "Analyze Resume" Button (Sky Blue)
if st.button("Analyze Resume", key="analyze", help="Click to analyze the resume and job description"):
    st.session_state.results = []
 
    if resume_input:
        if resume_input.lower().startswith("http"):
            try:
                response = requests.get(resume_input)
                response.raise_for_status()
                resume_files = [BytesIO(response.content)]
            except requests.exceptions.RequestException as e:
                resume_files = []
        else:
            if os.path.isdir(resume_input):
                resume_files = [os.path.join(resume_input, f) for f in os.listdir(resume_input) if f.lower().endswith(('.pdf', '.docx'))]
            else:
                resume_files = []
 
        if resume_files:
            for resume_file in resume_files:
                if isinstance(resume_file, BytesIO):
                    file_content = resume_file.read()
                    if resume_input.lower().endswith(".pdf"):
                        resume_text = extract_text_from_pdf(BytesIO(file_content))
                    elif resume_input.lower().endswith(".docx"):
                        resume_text = extract_text_from_docx(BytesIO(file_content))
                else:
                    if resume_file.endswith(".pdf"):
                        with open(resume_file, "rb") as file:
                            resume_text = extract_text_from_pdf(file)
                    elif resume_file.endswith(".docx"):
                        with open(resume_file, "rb") as file:
                            resume_text = extract_text_from_docx(file)
                    else:
                        continue
 
                resume_details = extract_details(resume_text, job_description_input)
                st.session_state.results.append(resume_details)
 
            # Sort results based on matching skills (Score)
            sorted_results = sorted(st.session_state.results, key=lambda x: x['Score'], reverse=True)
 
            for idx, result in enumerate(sorted_results, start=1):
                result['Ranking'] = idx  # Assign Rank
 
            st.session_state.results = sorted_results
            df = pd.DataFrame(st.session_state.results)
            st.dataframe(df)
 
# Footer: Blue background, white text
st.markdown('<div class="footer">© 2025 Resume Analysis Tool. All rights reserved.</div>', unsafe_allow_html=True)
 
 