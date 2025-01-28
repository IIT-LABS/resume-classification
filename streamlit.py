import streamlit as st
import imaplib
import email
from email.header import decode_header
import pandas as pd
from docx import Document
from PyPDF2 import PdfReader
import re
import io
import spacy

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")

# Email credentials
EMAIL_USER = "k12392945@gmail.com"  # Replace with your email
EMAIL_PASS = "xcya gowp wxrd cjav"  # Replace with your app password

# Sanitize filenames (remove problematic characters)
def sanitize_filename(filename):
    if not filename:  # Check if filename is None or empty
        return "unknown_filename"
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    sanitized = sanitized.replace('\r', '').replace('\n', '').replace('\t', '')
    return sanitized

# Extract email body
def extract_email_body(msg):
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition", ""))
            if "attachment" not in content_disposition:
                if content_type == "text/plain":
                    return part.get_payload(decode=True).decode("utf-8", errors="ignore")
                elif content_type == "text/html":
                    return part.get_payload(decode=True).decode("utf-8", errors="ignore")
    else:
        return msg.get_payload(decode=True).decode("utf-8", errors="ignore")

# Extract text from DOCX
def extract_text_from_docx(attachment_content):
    doc = Document(io.BytesIO(attachment_content))
    full_text = [para.text for para in doc.paragraphs]
    return '\n'.join(full_text)

# Extract text from PDF
def extract_text_from_pdf(attachment_content):
    pdf_reader = PdfReader(io.BytesIO(attachment_content))
    text = "".join(page.extract_text() for page in pdf_reader.pages)
    return text

# Extract resume details
def extract_name_from_text(text):
    text = text.strip()
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    lines = text.split("\n")
    irrelevant_words = ["summary", "contact", "education", "experience", "skills", "references", "profile", "resume", "cv"]
    for line in lines[:3]:
        line = line.strip()
        if any(irrelevant_word in line.lower() for irrelevant_word in irrelevant_words):
            continue
        if len(line) > 1:
            name_parts = line.split()
            if len(name_parts) > 1:
                return " ".join([part.title() for part in name_parts])
            elif len(name_parts) == 1:
                return name_parts[0].title()
    return "Name not found"

# Function to extract email from resume text
def extract_email_from_text(text):
    email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return email_match.group(0) if email_match else "Email not found"

# Function to extract phone numbers
def extract_phone_from_text(text):
    phone_pattern = re.compile(r"(?:direct|mobile|phone|ph#|contact|tel|cell)?[:\s-]*"
                               r"(?:\+?\d{1,3}[-.\s]?)?"
                               r"\(?\d{1,4}\)?"
                               r"[-.\s]?\d{1,4}"
                               r"[-.\s]?\d{1,4}"
                               r"[-.\s]?\d{1,9}"
                               r"(?:\s?(?:ext|x|extension)\s?\d{1,5})?")
    matches = phone_pattern.findall(text)
    phones = [re.sub(r"[^+\d\s()-]", "", match).strip() for match in matches if len(re.sub(r"\D", "", match)) >= 10]
    return ", ".join(phones) if phones else "Phone not found"

# Function to extract experience from resume text
def extract_experience(text):
    text = text.lower()
    numeric_pattern = r"(?:more than|over|at least|around|approximately|nearly|up to)?\s*(\d+)\+?\s*years?"
    numeric_match = re.search(numeric_pattern, text)
    if numeric_match:
        years = numeric_match.group(1)
        return f"{int(years)}+ years" if '+' in numeric_match.group(0) else f"{int(years)} years"
    return "Experience not found"

def extract_relevant_skills(resume_text, job_desc_text):
    """
    Extract skills from the resume and match them with skills required in the job description.
    Combines skills from the Skills block and context-based skill extraction.
    """
    # Convert both texts to lowercase for easier matching
    resume_text = resume_text.lower()
    job_desc_text = job_desc_text.lower()
    
    # Tokenize and normalize the text using spaCy
    resume_doc = nlp(resume_text)
    job_desc_doc = nlp(job_desc_text)
    
    # Extract individual words/phrases from the job description
    job_desc_words = set([token.text for token in job_desc_doc if not token.is_stop and not token.is_punct])
    
    # Extract skills explicitly from the Skills block (if exists)
    skills_block_pattern = re.search(
        r"(?:skills|technical skills|competencies|technologies)(.*?)(?:experience|education|summary|projects|certifications|$)", 
        resume_text, 
        re.DOTALL
    )
    
    if skills_block_pattern:
        skills_block = skills_block_pattern.group(1).strip()
        print(f"Skills Block: {skills_block}")  # Debugging output
    else:
        skills_block = ""
    
    skills_from_block = [skill.strip() for skill in re.split(r"[,\n;]", skills_block) if skill.strip()]
    
    # Extract skills based on context phrases (like proficient in, experienced with, etc.)
    contextual_skills_pattern = re.findall(
        r"(?:proficient in|experienced with|skilled in|knowledge of|familiar with)\s+([a-zA-Z0-9\s\+\-]+)", 
        resume_text
    )
    
    skills_from_context = []
    for skill_group in contextual_skills_pattern:
        skills_from_context.extend(skill_group.strip().split(", "))
    
    # Add entities recognized by spaCy (like products, organizations, etc.)
    spaCy_skills = {ent.text.strip() for ent in resume_doc.ents if ent.label_ in ["PRODUCT", "ORG", "WORK_OF_ART"]}
    
    print(f"Contextual Skills: {skills_from_context}")  # Debugging output
    print(f"spaCy Recognized Skills: {spaCy_skills}")  # Debugging output
    
    # Combine all extracted skills (explicit, contextual, and spaCy-based)
    all_skills = set(skills_from_block + skills_from_context).union(spaCy_skills)
    
    # Match skills from the resume to the job description
    matched_skills = all_skills.intersection(job_desc_words)
    
    print(f"Matched Skills: {matched_skills}")  # Debugging output
    
    return list(matched_skills)


# Function to extract certifications
def extract_certifications_count(text):
    certification_keywords = [
        r"certification", r"certifications", r"certified", r"certificate", r"certificates"
    ]
    pattern = r"|".join(certification_keywords)
    matches = re.findall(pattern, text, re.IGNORECASE)
    return len(matches)

# Function to extract location from resume text
def extract_location_from_text(text):
    """Extract location (city, state, or ZIP code) from resume text."""
    location_match = re.search(
        r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*,\s(?:TX|CA|NY|FL|WA|IL|PA|GA|NC|OH|NJ|VA|CO|AZ|MA|MD|TN|MO|IN|WI|MN|SC|AL|LA|KY|OR|OK|CT|IA|MS|KS|AR|NV|UT|NM|NE|WV|ID|HI|ME|NH|MT|RI|DE|SD|ND|AK|VT|WY))\b"  # City, State
        r"|\b\d{5}(?:-\d{4})?\b",  # ZIP code
        text
    )
    if location_match:
        location = location_match.group(0)
        if not any(keyword in location.lower() for keyword in ["assistant", "server", "sql"]):  # Example of filtering out unrelated matches
            return location
    return "Location not found"


# Extract government from resume text
def extract_government_details(text):
    """
    Extract the first current working location from the text using multiple patterns for 'Present', 'Till Date', etc.
    Removes unwanted prefixes and extra whitespace.
    """
    # Patterns to detect blocks with work location details
    patterns = [
        r"(Client:.*?Present|Client:.*?\d{4}|Client:.*?Till Date)",  # Client and its timeframe
        r"(Professional Experience:.*?Present|Professional Experience:.*?\d{4}|Professional Experience:.*?Till Date)",
        r"(EXPERIENCE.*?Present|EXPERIENCE.*?\d{4}|EXPERIENCE.*?Till Date)",
        r"(Past work:.*?Present|Past work:.*?\d{4}|Past work:.*?Till Date)",
        r"(WORK EXPERIENCE:.*?Present|WORK EXPERIENCE:.*?\d{4}|WORK EXPERIENCE:.*?Till Date)",
    ]
   
    # Extract the relevant sections based on the patterns
    extracted_sections = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        extracted_sections.extend(matches)
    
    # Combine all extracted sections into one string for further processing
    combined_text = " ".join(extracted_sections)
    
    # Define the combined pattern to extract location information before keywords like 'Present', 'Till Date', etc.
    location_pattern = re.compile(
        r"""
        # Flexible location matching with optional "Client:" prefix and keywords like 'Present', 'Till Date'
        (?:Client:\s*)?                                      # Optional 'Client:' prefix
        ([A-Za-z\s,.()]+(?:USA|México|Virginia|FL|NJ|Texas|Tallahassee|Reston|New York|U\.S\.A\.|U\.S\.|America))  # Location
        .*?                                                  # Any text in between
        (?=\s*(?:Present|Till Date|to date|current|\d{4}[-–]\d{4}|[\w\s]+))  # Lookahead for keywords or date patterns
        
        |  # OR
        
        # Stricter format where "Client:" is explicitly present and followed by "Present"
        Client:\s*                                            # 'Client:' prefix
        ([A-Za-z\s,]+)                                        # Location
        \s+[A-Z][a-z]+\s\d{4}\s*[-—]\s*Present                # Date range ending with 'Present'
        """,
        re.IGNORECASE | re.VERBOSE
    )
   
    # Find the first match for locations within the extracted sections
    match = location_pattern.search(combined_text)
    
    # Check if a match is found
    if match:
        # Extract the first location
        first_location = match.group(0).strip()
        
        # Remove unwanted parts like "Client:", extra whitespace, and any date/time information
        cleaned_location = re.sub(r"(Client:|Present|EXPERIENCE|Past work:|WORK EXPERIENCE:|\d{4}[-–]\d{4}|[A-Za-z]+\s\d{4}\s*[-—]\s*Present|[\t\n]+)", "", first_location)
        cleaned_location = re.sub(r"\s{2,}", " ", cleaned_location).strip()  # Remove extra spaces
        
        # Format the result
        formatted_location = f"[{cleaned_location}]"
        return formatted_location
    else:
        # If no matches, return a default "Not found"
        return "Not found"


# Function to extract visa status from the resume text
def extract_visa_status(text):
    """Extract visa status from the resume text."""
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

# Calculating resume score
def calculate_resume_score(resume_text, job_desc_text, skills, experience, certifications, visa_status, location):
    corpus = [job_desc_text, resume_text]
    vectorizer = CountVectorizer().fit_transform(corpus)
    vectors = vectorizer.toarray()

    # Cosine Similarity: Measures how closely the resume text aligns with the job description.
    similarity_score = cosine_similarity([vectors[0]], [vectors[1]])[0][0]

    # Skills, experience, and certifications
    skills_count = len(skills)
    experience_years = int(re.search(r"\d+", experience).group(0)) if re.search(r"\d+", experience) else 0
    certifications_count = certifications

    normalized_experience = min(experience_years / 20, 1)
    normalized_skills = min(skills_count / 20, 1)

    # Visa Status Scoring
    visa_priority = {
        "US Citizen": 1.0,
        "Green Card": 0.9,
        "H1B": 0.8,
        "OPT": 0.7,
        "CPT": 0.6,
        "L2": 0.5,
        "EAD": 0.5,
        "TN Visa": 0.6,
        "Study Visa": 0.4,
        "Not found": 0.0
    }
    visa_score = visa_priority.get(visa_status, 0.0)

    # Location Scoring
    location_score = 0.0
    if location.lower() != "location not found":
        location_score = 1.0  # Location provided gets full credit

    # Weighted scoring
    score = (
        similarity_score * 0.4 +  # Adjusted to 40% weight
        normalized_skills * 0.25 +  # Adjusted to 25% weight
        normalized_experience * 0.25 +  # Adjusted to 20% weight
        certifications_count * 0.2 +  # Certifications contribute 10%
        visa_score * 0.05 +  # Visa status contributes 5%
        location_score * 0.05  # Location contributes 5%
    )

    return round(min(score * 100, 100), 2)

# Function to extract job description from email body (for job id)
def extract_job_description_from_email_body(email_body):
    # Here, the job description should be identifiable by specific keywords in the email body.
    # We assume it's generally found after certain keywords or in the initial part of the email.
    job_description_keywords = ["job description", "responsibilities", "job details"]
    for keyword in job_description_keywords:
        if keyword in email_body.lower():
            start_index = email_body.lower().find(keyword)
            return email_body[start_index:]
    return "Job description not found."

# Process resume details
def process_resumes_and_attachments(job_id):
    try:
        mail = imaplib.IMAP4_SSL('imap.gmail.com')
        mail.login(EMAIL_USER, EMAIL_PASS)
        mail.select("inbox")

        # Search emails with Job ID in subject or body
        search_criteria = f'BODY "{job_id}"'
        status, messages = mail.search(None, search_criteria)
        email_ids = messages[0].split()
        st.write(f"Total emails matching Job ID '{job_id}': {len(email_ids)}")

        resume_details = []
        job_description = ""

        for email_id in email_ids:
            status, msg_data = mail.fetch(email_id, "(RFC822)")
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])

                    # Handle missing subject
                    subject, encoding = decode_header(msg.get("Subject", ""))[0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(encoding or "utf-8")
                    subject = subject or "No Subject"

                    # Extract email body
                    email_body = extract_email_body(msg)

                    # Extract job description from email body
                    job_description = extract_job_description_from_email_body(email_body)

                    for part in msg.walk():
                        content_disposition = part.get("Content-Disposition", "")
                        if "attachment" in content_disposition:
                            attachment_filename = sanitize_filename(part.get_filename())
                            attachment_content = part.get_payload(decode=True)

                            # Extract resume text from PDF or DOCX
                            if attachment_filename.lower().endswith('.pdf'):
                                resume_text = extract_text_from_pdf(attachment_content)
                            elif attachment_filename.lower().endswith('.docx'):
                                resume_text = extract_text_from_docx(attachment_content)
                            else:
                                continue

                            # Extract details from resume
                            name = extract_name_from_text(resume_text)
                            email_address = extract_email_from_text(resume_text)
                            phone_number = extract_phone_from_text(resume_text)
                            experience = extract_experience(resume_text)
                            relevant_skills = extract_relevant_skills(resume_text, job_description)
                            certifications_count = extract_certifications_count(resume_text)
                            location = extract_location_from_text(resume_text)
                            government_details = extract_government_details(resume_text)
                            visa_status = extract_visa_status(resume_text)
                            resume_score = calculate_resume_score(
                                resume_text,
                                job_description,
                                relevant_skills,
                                experience,
                                certifications_count,
                                visa_status,
                                location
                            )

                            # Store details in a dictionary
                            details = {
                                "Name": name,
                                "Email Address": email_address,
                                "Phone Number": phone_number,
                                "Experience": experience,
                                "Skills": ", ".join(relevant_skills),
                                "Certifications Count": certifications_count,
                                "Location": location,
                                "Government Details": government_details,
                                "Visa Status": visa_status,
                                "Resume Score": resume_score,
                                
                            }
                            resume_details.append(details)

        mail.logout()

        if resume_details:
            return pd.DataFrame(resume_details)
        else:
            st.write("No resumes found matching the Job ID.")
            return pd.DataFrame()

    except Exception as e:
        st.error(f"Error: {e}")
        return pd.DataFrame()

# Streamlit UI
st.title("Job Resume Processing")

job_id = st.text_input("Enter Job ID:")

if job_id:
    st.write(f"Processing resumes for Job ID: {job_id}")

    df = process_resumes_and_attachments(job_id)

    if not df.empty:
        st.write(f"Found {len(df)} resumes")
        st.dataframe(df)  # Display the dataframe in Streamlit
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv, file_name="resume_details.csv", mime="text/csv")
    else:
        st.write("No resumes found for the given Job ID.")
