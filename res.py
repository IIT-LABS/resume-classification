import streamlit as st
import os
import requests
import pandas as pd
from io import BytesIO
from PyPDF2 import PdfReader
from docx import Document
import spacy





# Load the English language model for spaCy
nlp = spacy.load("en_core_web_sm")

# Function to extract text from PDF files
def extract_text_from_pdf(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

# Function to extract text from DOCX files
def extract_text_from_docx(docx_file):
    try:
        doc = Document(docx_file)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {e}")
        return ""

# Define the extract_details function
def extract_details(resume_text, job_description):
    # Preprocess the text by using spaCy to get tokens and entities
    resume_doc = nlp(resume_text)
    job_description_doc = nlp(job_description)

    # Example: Extract skills from both resume and job description (we'll assume a list of skills here)
    skills = ["python", "data analysis", "machine learning", "sql", "excel", "deep learning"]

    # Extracting skills from the resume
    resume_skills = set()
    for skill in skills:
        if skill.lower() in resume_text.lower():
            resume_skills.add(skill.lower())

    # Extracting skills from the job description
    job_skills = set()
    for skill in skills:
        if skill.lower() in job_description.lower():
            job_skills.add(skill.lower())

    # Calculate a score based on the number of matching skills
    matching_skills = resume_skills.intersection(job_skills)
    score = len(matching_skills)

    # Return the resume details including matching skills and score
    resume_details = {
        'Skills Matched': ", ".join(matching_skills),
        'Score': score,
        'Total Skills in Job Description': len(job_skills),
        'Total Skills in Resume': len(resume_skills)
    }

    return resume_details

# Streamlit app UI

st.title("Resume and Job Description Matching Tool")
st.markdown("Upload your resume(s) and enter a job description to analyze the match.")

# File uploader for resumes
uploaded_files = st.file_uploader("Upload Resume(s)", type=["pdf", "docx"], accept_multiple_files=True)

# Job description input box
job_description_input = st.text_area("Enter Job Description")

# Button to process the resumes
if st.button("Process Resumes"):
    if uploaded_files and job_description_input:
        resume_details = []

        # Process each uploaded file
        for uploaded_file in uploaded_files:
            resume_text = ""

            # Check file type and extract text accordingly
            if uploaded_file.name.lower().endswith(".pdf"):
                resume_text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.name.lower().endswith(".docx"):
                resume_text = extract_text_from_docx(uploaded_file)

            # If resume_text is empty after extraction, skip it
            if not resume_text:
                st.warning(f"Could not extract text from {uploaded_file.name}. Skipping this file.")
                continue

            # Extract details from resume and job description
            resume_details.append(extract_details(resume_text, job_description_input))

        # Sort results based on score
        sorted_results = sorted(resume_details, key=lambda x: x['Score'], reverse=True)

        # Display results
        if sorted_results:
            st.write("### Resume Matching Results")
            st.dataframe(pd.DataFrame(sorted_results))

            # Option to download results
            if st.button("Download Results as CSV"):
                df = pd.DataFrame(sorted_results)
                df.to_csv("resume_analysis_results.csv", index=False)
                st.success("CSV file downloaded successfully.")
        else:
            st.warning("No matching results found. Please check the uploaded files or job description.")
    else:
        st.warning("Please upload resume(s) and enter a job description.")
