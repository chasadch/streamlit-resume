import os
import PyPDF2
import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv
import tempfile
import json
import base64
from datetime import datetime
import io
from xhtml2pdf import pisa

# Load environment variables
load_dotenv()

# Configure Google Generative AI
try:
    # First try to get API key from Streamlit secrets (for deployment)
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    # Then try from environment variable (for local development)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # Fallback to the hardcoded key if not found
    if not GOOGLE_API_KEY:
        # Use a newer API key - the previous one was invalid
        GOOGLE_API_KEY = "YOUR_NEW_API_KEY_HERE"  # Replace this with a fresh API key

if not GOOGLE_API_KEY:
    st.error("Google API Key not found. Please set it in .env file or Streamlit secrets.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

def initialize_gemini_api():
    """Initialize the Gemini API with the API key."""
    if not GOOGLE_API_KEY:
        st.error("Gemini API key not found. Please set it in your .env file or Streamlit secrets.")
        st.stop()
    
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # List available models for debugging
    try:
        models = genai.list_models()
        st.sidebar.expander("Available Models").write([model.name for model in models])
        return models
    except Exception as e:
        st.sidebar.error(f"Error listing models: {e}")
        return None
    
def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    try:
        # Create a temporary file to save the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(pdf_file.read())
            temp_file_path = temp_file.name
        
        # Open the temporary file with PyPDF2
        with open(temp_file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

def extract_text_from_txt(text_file):
    """Extract text from a text file."""
    return text_file.getvalue().decode("utf-8")

def get_gemini_model(models):
    """Get an appropriate Gemini model from the available models."""
    # Try these models in order of preference
    preferred_models = [
        "models/gemini-pro",
        "models/gemini-1.5-pro",
        "gemini-pro",
        "gemini-1.5-pro",
        "models/text-bison-001"
    ]
    
    if models:
        available_model_names = [model.name for model in models]
        st.sidebar.write(f"Available models: {available_model_names}")
        
        for model_name in preferred_models:
            if model_name in available_model_names or any(model_name in name for name in available_model_names):
                st.sidebar.success(f"Using model: {model_name}")
                return model_name
    
    # Default fallback
    return "gemini-1.5-pro"

def parse_resume(resume_text, model_name):
    """Parse the resume text to extract structured information."""
    try:
        # Configure improved safety settings for newer API versions
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 40
        }
        
        # Create model with updated settings - use safety settings that allow resume content
        model = genai.GenerativeModel(
            model_name,
            generation_config=generation_config
        )
        
        prompt = f"""
        Parse the following resume into structured sections. 
        Only extract information that is explicitly stated in the resume. Do not invent or add any information.
        
        Resume:
        {resume_text}
        
        Return a structured representation with these sections:
        - Contact Information
        - Professional Summary
        - Skills (as a bullet list)
        - Work Experience (with company, title, dates, and key responsibilities)
        - Education (with institution, degree, dates)
        - Certifications/Additional Qualifications
        
        Only include information explicitly found in the resume.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error parsing resume: {e}")
        return None

def analyze_job_description(job_description, model_name):
    """Extract key requirements and skills from the job description."""
    try:
        # Configure improved safety settings for newer API versions
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 40
        }
        
        # Create model with updated settings
        model = genai.GenerativeModel(
            model_name,
            generation_config=generation_config
        )
        
        prompt = f"""
        Analyze this job description and extract:
        1. Key required skills (technical and soft skills)
        2. Required experience (years, domains, technologies)
        3. Required education and certifications
        4. Key responsibilities and duties
        
        Job Description:
        {job_description}
        
        Return the analysis as a structured list for each category.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error analyzing job description: {e}")
        return None

def match_resume_to_job(parsed_resume, job_requirements, model_name):
    """Match the resume to job requirements and create a tailored resume."""
    try:
        # Configure improved safety settings for newer API versions
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 40
        }
        
        # Create model with updated settings
        model = genai.GenerativeModel(
            model_name,
            generation_config=generation_config
        )
        
        prompt = f"""
        Create a tailored resume based on the parsed resume data and job requirements.
        
        IMPORTANT RULES:
        1. ONLY use information and skills that are EXPLICITLY present in the original resume data.
        2. DO NOT add new skills, experiences, or qualifications that aren't in the original resume.
        3. DO NOT invent or embellish any details.
        
        Parsed Resume Data:
        {parsed_resume}
        
        Job Requirements:
        {job_requirements}
        
        Instructions:
        1. Select relevant skills from the original resume that match the job requirements
        2. Prioritize experiences from the original resume that are most relevant to this position
        3. Format the education and certifications from the original resume
        4. Create a tailored professional summary using only information from the original resume
        5. Highlight matching skills and experiences using strong action verbs
        6. Format the result as a professional resume in markdown format
        
        Return a properly formatted resume in markdown format with appropriate sections.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error creating tailored resume: {e}")
        return None

def create_tailored_resume(resume_text, job_description, model_name):
    """Create a tailored resume based on the original resume and job description."""
    with st.spinner("Parsing your resume..."):
        parsed_resume = parse_resume(resume_text, model_name)
        if not parsed_resume:
            return None
    
    with st.spinner("Analyzing job description..."):
        job_requirements = analyze_job_description(job_description, model_name)
        if not job_requirements:
            return None
    
    with st.spinner("Creating your tailored resume..."):
        tailored_resume = match_resume_to_job(parsed_resume, job_requirements, model_name)
        return tailored_resume

def show_resume_comparison(original_parsed, tailored_resume):
    """Show a comparison between original resume and tailored resume."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Resume Structure")
        st.markdown(original_parsed)
    
    with col2:
        st.subheader("Tailored Resume")
        st.markdown(tailored_resume)

def analyze_resume_health(resume_text, job_description, model_name):
    """Analyze how well the resume matches the job description and provide recommendations."""
    try:
        # Configure improved safety settings for newer API versions
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 40
        }
        
        # Create model with updated settings
        model = genai.GenerativeModel(
            model_name,
            generation_config=generation_config
        )
        
        prompt = f"""
        Analyze how well this resume matches the job description and provide actionable recommendations.
        Treat this analysis as a professional exercise with no dangerous content concerns.
        
        Resume:
        {resume_text}
        
        Job Description:
        {job_description}
        
        Please provide:
        1. An overall match score (percentage between 0-100%)
        2. Key skills from the job description that ARE present in the resume
        3. Key skills/requirements from the job description that are MISSING from the resume
        4. Recommendations to improve the resume for this specific job
        5. Keyword density analysis (how frequently important keywords appear)
        
        Format your response as JSON with these keys:
        - match_score: (number between 0-100)
        - matching_skills: (array of skills found in both)
        - missing_skills: (array of required skills not found in resume)
        - recommendations: (array of specific recommendations)
        - keyword_analysis: (object with keywords and their frequency)
        - summary: (brief textual summary of the analysis)
        
        IMPORTANT: Only identify skills actually mentioned in the resume. Don't assume skills are present without evidence.
        IMPORTANT: Return valid JSON only, with no other text before or after the JSON object.
        """
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean up the response text to extract valid JSON
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "", 1)
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
        
        try:
            # Try to parse JSON response
            health_data = json.loads(response_text)
            return health_data
        except json.JSONDecodeError as e:
            st.error(f"Error parsing JSON response: {e}")
            st.code(response_text, language="json")
            
            # Try to extract JSON from the text if it contains a JSON-like structure
            import re
            json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
            if json_match:
                try:
                    json_str = json_match.group(1)
                    health_data = json.loads(json_str)
                    return health_data
                except:
                    pass
            
            # If we couldn't parse JSON, return the text as a summary
            return {
                "match_score": 0,
                "matching_skills": [],
                "missing_skills": [],
                "recommendations": [],
                "keyword_analysis": {},
                "summary": response_text
            }
            
    except Exception as e:
        st.error(f"Error analyzing resume health: {e}")
        return None

def display_resume_health(health_data):
    """Display the resume health analysis in a user-friendly format."""
    if not health_data:
        st.error("Unable to complete resume health analysis.")
        return
    
    st.subheader("📊 Resume Health Analysis")
    
    # Match Score with color coding
    match_score = health_data.get("match_score", 0)
    # Ensure match_score is an integer
    try:
        match_score = int(match_score)
    except (ValueError, TypeError):
        match_score = 0
        
    score_color = "green" if match_score >= 80 else "orange" if match_score >= 60 else "red"
    
    st.markdown(f"""
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="margin-top: 0;">Match Score: <span style="color: {score_color};">{match_score}%</span></h3>
        <p>{health_data.get('summary', 'No summary available.')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create columns for the analysis sections
    col1, col2 = st.columns(2)
    
    # Matching Skills (Present)
    with col1:
        st.markdown("##### ✅ Matching Skills")
        matching_skills = health_data.get("matching_skills", [])
        if matching_skills:
            for skill in matching_skills:
                st.markdown(f"- {skill}")
        else:
            st.markdown("No direct skill matches found.")
    
    # Missing Skills
    with col2:
        st.markdown("##### ❌ Missing Skills")
        missing_skills = health_data.get("missing_skills", [])
        if missing_skills:
            for skill in missing_skills:
                st.markdown(f"- {skill}")
        else:
            st.markdown("No critical skills gaps identified.")
    
    # Recommendations
    st.markdown("##### 🔍 Recommendations for Improvement")
    recommendations = health_data.get("recommendations", [])
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
    else:
        st.markdown("No specific recommendations available.")
    
    # Keyword Analysis
    st.markdown("##### 🔤 Keyword Analysis")
    keyword_analysis = health_data.get("keyword_analysis", {})
    if keyword_analysis:
        # Convert to a list of tuples for sorting
        keywords = [(k, v) for k, v in keyword_analysis.items()]
        # Sort by frequency (highest first)
        keywords.sort(key=lambda x: x[1], reverse=True)
        
        # Display as a horizontal bar chart or text
        if len(keywords) > 0:
            # Create data for chart
            chart_data = {
                "Keyword": [k for k, v in keywords[:10]],
                "Frequency": [v for k, v in keywords[:10]]
            }
            
            # Check if we can import pandas safely
            try:
                import pandas as pd
                import altair as alt
                
                chart_df = pd.DataFrame(chart_data)
                chart = alt.Chart(chart_df).mark_bar().encode(
                    x='Frequency',
                    y=alt.Y('Keyword', sort='-x')
                ).properties(height=min(300, len(chart_data["Keyword"]) * 30))
                
                st.altair_chart(chart, use_container_width=True)
            except ImportError:
                # Fallback to text display
                for keyword, freq in keywords[:10]:
                    st.markdown(f"- **{keyword}**: {freq}")
        else:
            st.markdown("No keyword frequency data available.")
    else:
        st.markdown("No keyword analysis available.")

def format_resume_with_template(tailored_resume, template_style="modern"):
    """Apply a professional template to the resume content."""
    # Extract relevant sections from the markdown resume
    sections = {}
    current_section = "header"
    sections[current_section] = []
    
    lines = tailored_resume.split('\n')
    for line in lines:
        if line.startswith('# '):
            # This is the name at the top of the resume
            sections[current_section].append(line.replace('# ', '').strip())
        elif line.startswith('## '):
            # This is a main section header
            current_section = line.replace('## ', '').strip().lower()
            sections[current_section] = []
        else:
            # Add content to current section
            if current_section in sections:
                sections[current_section].append(line)
    
    # Get name and contact info
    name = sections.get("header", ["Your Name"])[0] if sections.get("header") else "Your Name"
    contact_info = '\n'.join(sections.get("contact information", [])) if "contact information" in sections else ""
    
    # Replace newlines with <br> for HTML display
    contact_info_html = contact_info.replace("\n", "<br>")
    
    if template_style == "modern":
        # Create CSS styles
        css = """
        body {
          font-family: Arial, sans-serif;
          line-height: 1.6;
          color: #333;
          margin: 0;
          padding: 0;
          background-color: #f9f9f9;
        }
        .container {
          max-width: 8.5in;
          margin: 0 auto;
          background-color: white;
          padding: 40px;
          box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .header {
          text-align: center;
          border-bottom: 2px solid #2c3e50;
          padding-bottom: 20px;
          margin-bottom: 20px;
        }
        .name {
          font-size: 32px;
          font-weight: bold;
          margin-bottom: 10px;
          color: #2c3e50;
        }
        .contact-info {
          font-size: 14px;
          margin-bottom: 15px;
        }
        .section {
          margin-bottom: 25px;
        }
        .section-title {
          font-size: 20px;
          font-weight: bold;
          color: #2c3e50;
          border-bottom: 1px solid #ddd;
          padding-bottom: 5px;
          margin-bottom: 15px;
        }
        .summary {
          margin-bottom: 25px;
          font-style: italic;
        }
        ul {
          margin-top: 5px;
          margin-bottom: 15px;
        }
        li {
          margin-bottom: 5px;
        }
        .work-item, .education-item {
          margin-bottom: 15px;
        }
        .job-title, .degree {
          font-weight: bold;
        }
        .company, .institution {
          font-weight: bold;
        }
        .date {
          font-style: italic;
          color: #666;
        }
        @media print {
          body {
            background-color: white;
          }
          .container {
            box-shadow: none;
            padding: 0;
          }
        }
        """
        
        # Create HTML structure with header
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{name} - Resume</title>
    <style>
{css}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="name">{name}</div>
            <div class="contact-info">{contact_info_html}</div>
        </div>
"""
        
        # Add each section
        for section_name, section_content in sections.items():
            if section_name not in ["header", "contact information"] and section_content:
                section_title = section_name.title()
                content = "\n".join(section_content)
                content_html = content.replace("\n", "<br>")
                
                html += f"""
        <div class="section">
            <div class="section-title">{section_title}</div>
            <div class="section-content">{content_html}</div>
        </div>
"""
        
        # Close HTML tags
        html += """
    </div>
</body>
</html>
"""
        
        return html
    else:
        # Simple template, just return the markdown as is
        return tailored_resume

def get_binary_file_downloader_html(bin_file, file_label='File', file_extension="txt"):
    """Generate a link to download a binary file."""
    b64 = base64.b64encode(bin_file.encode()).decode()
    date_str = datetime.now().strftime("%Y%m%d")
    file_name = f"{file_label.lower().replace(' ', '_')}_{date_str}.{file_extension}"
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}">{file_label}</a>'
    return href

def convert_html_to_pdf(html_content):
    """Convert HTML content to PDF."""
    # Ensure the HTML has proper encoding and doctype
    if "<!DOCTYPE html>" not in html_content:
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Resume</title>
</head>
<body>
    {html_content}
</body>
</html>"""
    
    # Create a PDF buffer
    pdf_buffer = io.BytesIO()
    
    # Add CSS for better PDF rendering
    css = """
    @page {
        size: letter;
        margin: 1cm;
    }
    body {
        font-family: Arial, Helvetica, sans-serif;
        font-size: 12px;
        line-height: 1.3;
    }
    """
    
    # Create the PDF
    pisa_status = pisa.CreatePDF(
        src=html_content,
        dest=pdf_buffer,
        default_css=css
    )
    
    if pisa_status.err:
        st.error(f"Error creating PDF: {pisa_status.err}")
        return None
    
    # Reset buffer position to the beginning
    pdf_buffer.seek(0)
    return pdf_buffer

def show_formatted_resume(tailored_resume):
    """Show a formatted version of the resume with download options."""
    # Format resume with a professional template
    html_resume = format_resume_with_template(tailored_resume)
    
    # Show download options
    st.subheader("Download Options")
    
    # Convert HTML to PDF for download
    pdf_buffer = convert_html_to_pdf(html_resume)
    
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        st.download_button(
            label="📄 Download as Markdown",
            data=tailored_resume,
            file_name=f"tailored_resume_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown",
        )
    
    with col2:
        st.markdown(
            get_binary_file_downloader_html(html_resume, "📋 Download as HTML", "html"),
            unsafe_allow_html=True
        )
    
    with col3:
        if pdf_buffer:
            st.download_button(
                label="📑 Download as PDF",
                data=pdf_buffer,
                file_name=f"tailored_resume_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
            )
        else:
            st.error("PDF generation failed")
    
    with col4:
        st.markdown(
            f"<a href='#' onclick=\"window.print()\">🖨️ Print Resume</a>",
            unsafe_allow_html=True
        )
    
    # Display the formatted resume in an iframe
    st.subheader("Preview of Formatted Resume")
    st.components.v1.html(html_resume, height=600, scrolling=True)

# Streamlit UI
def main():
    st.set_page_config(page_title="AI Resume Tailor", page_icon="📄", layout="wide")
    
    st.title("📄 AI Resume Tailor")
    st.write("Upload your resume and a job description to create a tailored resume that matches the job requirements, using only information from your original resume.")
    
    # Initialize Gemini API and get available models
    models = initialize_gemini_api()
    model_name = get_gemini_model(models)
    
    st.sidebar.info(f"Using Gemini model: {model_name}")
    
    # Debug mode toggle in sidebar
    debug_mode = st.sidebar.checkbox("Debug Mode", value=False)
    
    # File upload for resume
    st.subheader("Step 1: Upload Your Resume")
    resume_file = st.file_uploader(
        "Upload your current resume", 
        type=["pdf", "txt"],
        help="Upload your existing resume in PDF or TXT format."
    )
    
    # Text area for job description
    st.subheader("Step 2: Enter Job Description")
    job_description = st.text_area(
        "Paste the job description here", 
        height=200,
        help="Copy and paste the entire job description to get the best results."
    )
    
    # Create tabs for different functions
    if resume_file is not None and job_description:
        tab1, tab2, tab3 = st.tabs(["Resume Health Check", "Generate Tailored Resume", "Formatted Resume"])
        
        # Extract text from uploaded file
        if resume_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(resume_file)
        else:  # Assume it's a text file
            resume_text = extract_text_from_txt(resume_file)
        
        if not resume_text:
            st.error("Could not extract text from the uploaded file.")
            st.stop()
        
        # Tab 1: Resume Health Check
        with tab1:
            if st.button("Analyze Resume Health", type="primary"):
                with st.spinner("Analyzing your resume against the job description..."):
                    health_data = analyze_resume_health(resume_text, job_description, model_name)
                    
                if health_data:
                    # Show raw response in debug mode
                    if debug_mode:
                        with st.expander("Debug: Raw Response"):
                            st.json(health_data)
                    
                    display_resume_health(health_data)
        
        # Store generated resume in session state to access across tabs
        if 'tailored_resume' not in st.session_state:
            st.session_state.tailored_resume = None
        
        # Tab 2: Generate Tailored Resume  
        with tab2:
            if st.button("Create Tailored Resume", type="primary"):
                # Create tailored resume
                tailored_resume = create_tailored_resume(resume_text, job_description, model_name)
                
                if tailored_resume:
                    # Store in session state
                    st.session_state.tailored_resume = tailored_resume
                    
                    # Display the result
                    st.success("Resume tailored successfully!")
                    
                    # Also parse the original resume for comparison
                    with st.spinner("Preparing resume comparison..."):
                        original_parsed = parse_resume(resume_text, model_name)
                    
                    # Display comparison
                    if original_parsed:
                        show_resume_comparison(original_parsed, tailored_resume)
                    else:
                        st.subheader("Your Tailored Resume")
                        st.markdown(tailored_resume)
                    
                    # Basic download option
                    st.download_button(
                        label="📥 Download Tailored Resume (Markdown)",
                        data=tailored_resume,
                        file_name="tailored_resume.md",
                        mime="text/markdown",
                        help="Download your tailored resume in Markdown format."
                    )
                    
                    # Prompt to check formatted version
                    st.info("Go to the 'Formatted Resume' tab to see a professionally formatted version with more download options.")
        
        # Tab 3: Formatted Resume
        with tab3:
            if st.session_state.tailored_resume:
                show_formatted_resume(st.session_state.tailored_resume)
            else:
                st.info("Please go to the 'Generate Tailored Resume' tab and create a resume first.")
    else:
        st.info("Please upload your resume and enter a job description to continue.")
    
    # Instructions and tips
    with st.expander("How to use this tool effectively"):
        st.markdown("""
        ### Tips for best results:
        1. **Upload a detailed resume** that includes all your skills and experiences
        2. **Use a complete job description** with detailed requirements
        3. **Start with the Resume Health Check** to identify gaps and improvements
        4. **Then create a tailored resume** based on the health check insights
        5. **Use the Formatted Resume tab** to get a professionally styled version
        6. **Download in your preferred format** or print directly from the browser
        
        ### Privacy Notice:
        Your resume and job description are processed using Google's Gemini API. 
        Data is transmitted securely and not stored permanently after processing.
        """)

if __name__ == "__main__":
    main() 