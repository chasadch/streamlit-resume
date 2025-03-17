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
import markdown
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO
import re

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
        # Use a valid API key
        GOOGLE_API_KEY = "AIzaSyA4QDYfK2zToSHiSDyZPlQtXMvZn1Vrz2M"  # This is a valid API key for this app

if not GOOGLE_API_KEY:
    st.error("Google API Key not found. Please set it in .env file or Streamlit secrets.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# Define a function for API calls with exponential backoff and retry
def api_call_with_retry(func, *args, max_retries=5, **kwargs):
    """Make API calls with exponential backoff and retry logic."""
    retries = 0
    while retries <= max_retries:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_message = str(e)
            
            # If it's a quota error, handle it specially
            if "429" in error_message or "Resource has been exhausted" in error_message or "quota" in error_message.lower():
                if retries == max_retries:
                    raise Exception(f"API quota exceeded. Please try again later or use a different API key. Error: {e}")
                
                # Calculate backoff time: 2^retries + random jitter
                backoff_time = (2 ** retries) + random.uniform(0, 1)
                
                # Show a warning but continue
                st.warning(f"API quota limit hit. Retrying in {backoff_time:.1f} seconds... (Attempt {retries+1}/{max_retries})")
                time.sleep(backoff_time)
                retries += 1
            else:
                # For other errors, just raise them
                raise
    
    # If we get here, all retries failed
    raise Exception("Maximum retry attempts reached. Please try again later.")

def initialize_gemini_api():
    """Initialize the Gemini API with the API key."""
    if not GOOGLE_API_KEY:
        st.error("Gemini API key not found. Please set it in your .env file or Streamlit secrets.")
        st.stop()
    
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # List available models for debugging
    try:
        models = api_call_with_retry(genai.list_models)
        model_names = [model.name for model in models]
        st.sidebar.expander("Available Models").write(model_names)
        if not model_names:
            st.sidebar.warning("No models available. This might indicate an API key issue.")
        
        # Add model selection dropdown
        all_model_options = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro", "gemini-pro-vision"]
        available_options = [m for m in all_model_options if any(m in name for name in model_names)]
        
        if available_options:
            selected_model = st.sidebar.selectbox(
                "Select Model (lower cost models use less quota)",
                options=available_options,
                index=min(1, len(available_options)-1),  # Default to second option (likely flash) if available
                help="Select a model to use. Lower cost models may help avoid quota issues."
            )
            st.sidebar.success(f"User selected: {selected_model}")
            # Store the selection in session state
            st.session_state.user_selected_model = selected_model
        
        return models
    except Exception as e:
        error_message = str(e)
        st.sidebar.error(f"Error listing models: {error_message}")
        
        if "API_KEY_INVALID" in error_message or "key not valid" in error_message.lower():
            st.sidebar.error("""
            🔑 **API Key Invalid**
            
            Please follow these steps to fix:
            1. Get a new API key from [Google AI Studio](https://ai.google.dev/)
            2. Add it to your Streamlit secrets or update the hardcoded key
            """)
        elif "quota" in error_message.lower() or "429" in error_message:
            st.sidebar.error("""
            ⚠️ **API Quota Exceeded**
            
            Your API key has reached its quota limits. Try these solutions:
            1. Wait a few hours for quota to reset
            2. Get a new API key from [Google AI Studio](https://ai.google.dev/)
            3. Try a lower-cost model (select from dropdown if available)
            """)
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
    # Check if user has selected a model
    if hasattr(st.session_state, 'user_selected_model') and st.session_state.user_selected_model:
        user_model = st.session_state.user_selected_model
        st.sidebar.success(f"Using model: {user_model}")
        return user_model
    
    # Try these models in order of preference (from least to most resource intensive)
    preferred_models = [
        "gemini-1.5-flash",
        "gemini-1.0-pro", 
        "models/gemini-1.5-flash",
        "gemini-pro",
        "models/gemini-pro",
        "models/gemini-1.5-pro",
        "gemini-1.5-pro",
    ]
    
    if models:
        available_model_names = [model.name for model in models]
        st.sidebar.write(f"Available models: {available_model_names}")
        
        for model_name in preferred_models:
            if model_name in available_model_names or any(model_name in name for name in available_model_names):
                st.sidebar.success(f"Using model: {model_name}")
                return model_name
    
    # Default fallback
    return "gemini-1.5-flash"  # Use flash instead of pro as default to reduce quota usage

# Optimize text size by chunking for long texts
def optimize_text_length(text, max_chars=10000):
    """Reduce text size if it's too long by keeping the most important parts."""
    if len(text) <= max_chars:
        return text
    
    # Simple approach: keep first third and last third
    third = max_chars // 3
    return text[:third*2] + "\n...[content truncated to reduce API usage]...\n" + text[-third:]

def parse_resume(resume_text, model_name):
    """Parse the resume text to extract structured information."""
    try:
        # Optimize text length to reduce API usage
        optimized_resume = optimize_text_length(resume_text)
        
        # Configure improved safety settings for newer API versions
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048  # Limit output size to reduce token usage
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
        {optimized_resume}
        
        Return a structured representation with these sections:
        - Contact Information
        - Professional Summary
        - Skills (as a bullet list)
        - Work Experience (with company, title, dates, and key responsibilities)
        - Education (with institution, degree, dates)
        - Certifications/Additional Qualifications
        
        Only include information explicitly found in the resume.
        """
        
        # Use the retry wrapper for API calls
        response = api_call_with_retry(model.generate_content, prompt)
        return response.text
    except Exception as e:
        st.error(f"Error parsing resume: {e}")
        return None

def analyze_job_description(job_description, model_name):
    """Extract key requirements and skills from the job description."""
    try:
        # Optimize text length to reduce API usage
        optimized_job_description = optimize_text_length(job_description)
        
        # Configure improved safety settings for newer API versions
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048  # Limit output size to reduce token usage
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
        {optimized_job_description}
        
        Return the analysis as a structured list for each category.
        """
        
        # Use the retry wrapper for API calls
        response = api_call_with_retry(model.generate_content, prompt)
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
            "top_k": 40,
            "max_output_tokens": 2048  # Limit output size to reduce token usage
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
        
        # Use the retry wrapper for API calls
        response = api_call_with_retry(model.generate_content, prompt)
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
        # Optimize text length to reduce API usage
        optimized_resume = optimize_text_length(resume_text)
        optimized_job = optimize_text_length(job_description)
        
        # Configure improved safety settings for newer API versions
        generation_config = {
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048  # Limit output size to reduce token usage
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
        {optimized_resume}
        
        Job Description:
        {optimized_job}
        
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
        
        # Use the retry wrapper for API calls
        response = api_call_with_retry(model.generate_content, prompt)
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

def format_resume_with_template(resume_content):
    """Format the resume content with a professional HTML template."""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Professional Resume</title>
        <style>
            body {
                font-family: 'Arial', sans-serif;
                line-height: 1.6;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
            }
            h1 {
                color: #2c3e50;
                margin-bottom: 10px;
            }
            h2 {
                color: #34495e;
                border-bottom: 2px solid #3498db;
                padding-bottom: 5px;
                margin-top: 20px;
            }
            .section {
                margin-bottom: 25px;
            }
            .contact-info {
                text-align: center;
                color: #7f8c8d;
                margin-bottom: 20px;
            }
            ul {
                list-style-type: none;
                padding-left: 0;
            }
            li {
                margin-bottom: 8px;
            }
            .experience-item {
                margin-bottom: 15px;
            }
            .company-name {
                font-weight: bold;
                color: #2980b9;
            }
            .date {
                color: #7f8c8d;
                font-style: italic;
            }
            @media print {
                body {
                    padding: 20px;
                }
                .page-break {
                    page-break-before: always;
                }
            }
        </style>
    </head>
    <body>
        {{ resume_content | safe }}
    </body>
    </html>
    """
    
    # Convert markdown to HTML
    html_content = markdown.markdown(resume_content)
    
    # Apply template
    template = Template(html_template)
    formatted_html = template.render(resume_content=html_content)
    
    return formatted_html

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

def html_to_pdf(html_content):
    """Convert HTML content to PDF using WeasyPrint."""
    try:
        # Create a PDF from HTML content
        pdf = weasyprint.HTML(string=html_content).write_pdf()
        return pdf
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        return None

def get_pdf_download_link(pdf_content, filename="resume.pdf"):
    """Generate a download link for PDF content."""
    try:
        b64 = base64.b64encode(pdf_content).decode()
        return f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Click here to download PDF</a>'
    except Exception as e:
        st.error(f"Error creating PDF download link: {str(e)}")
        return None

def create_pdf_from_markdown(markdown_content):
    """Convert markdown content to PDF using ReportLab."""
    try:
        # Create a BytesIO buffer to receive PDF data
        buffer = BytesIO()
        
        # Create the PDF document
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Create custom style for better formatting
        custom_style = ParagraphStyle(
            'CustomStyle',
            parent=styles['Normal'],
            fontSize=12,
            leading=14,
            spaceAfter=10
        )
        
        # Convert markdown to basic HTML
        html_content = markdown.markdown(markdown_content)
        
        # Split content into lines and create story
        story = []
        
        # Process the HTML content
        lines = html_content.split('\n')
        for line in lines:
            # Remove HTML tags for simple text processing
            clean_line = re.sub('<[^<]+?>', '', line).strip()
            if clean_line:
                # Create paragraph with custom style
                para = Paragraph(clean_line, custom_style)
                story.append(para)
                story.append(Spacer(1, 12))
        
        # Build PDF document
        doc.build(story)
        
        # Get the value of the BytesIO buffer
        pdf_content = buffer.getvalue()
        buffer.close()
        
        return pdf_content
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        return None

def show_formatted_resume(resume_content):
    """Display the formatted resume and provide download options."""
    try:
        # Convert markdown to HTML for preview
        html_content = markdown.markdown(resume_content)
        
        # Create PDF version
        pdf_content = create_pdf_from_markdown(resume_content)
        
        # Display download options
        st.subheader("Download Options")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Markdown download
            st.download_button(
                "📄 Download as Markdown",
                resume_content,
                file_name="resume.md",
                mime="text/markdown"
            )
        
        with col2:
            # HTML download
            st.download_button(
                "🌐 Download as HTML",
                html_content,
                file_name="resume.html",
                mime="text/html"
            )
        
        with col3:
            if pdf_content:
                # PDF download
                st.download_button(
                    "📑 Download as PDF",
                    pdf_content,
                    file_name="resume.pdf",
                    mime="application/pdf"
                )
            else:
                st.error("PDF generation failed")
        
        with col4:
            # Print option
            st.markdown(
                f'<a href="#" onclick="window.print()">🖨️ Print Resume</a>',
                unsafe_allow_html=True
            )
        
        # Preview
        st.subheader("Preview of Formatted Resume")
        st.markdown(html_content, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error displaying formatted resume: {str(e)}")

# Streamlit UI
def main():
    st.set_page_config(page_title="AI Resume Tailor", page_icon="📄", layout="wide")
    
    st.title("📄 AI Resume Tailor")
    
    # Add information about quota issues
    st.info("""
    📢 **Note:** This application uses the Google Generative AI API which has usage quotas. 
    If you encounter quota errors, try selecting a lower-cost model from the sidebar dropdown or wait a while and try again.
    """)
    
    st.write("Upload your resume and a job description to create a tailored resume that matches the job requirements, using only information from your original resume.")
    
    # Initialize Gemini API and get available models
    models = initialize_gemini_api()
    model_name = get_gemini_model(models)
    
    st.sidebar.info(f"Using Gemini model: {model_name}")
    
    # Quota saving tips in sidebar
    with st.sidebar.expander("💡 Tips to Avoid Quota Issues"):
        st.markdown("""
        1. Use the 'gemini-1.5-flash' model instead of 'pro' models
        2. Keep resume and job descriptions concise
        3. Try during off-peak hours
        4. If you get quota errors, wait a few hours and try again
        """)
    
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