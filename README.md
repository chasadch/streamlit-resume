# AI Resume Builder

An intelligent application that helps you tailor your resume for specific job descriptions using AI.

## Features

- **Resume Parsing:** Upload your existing resume in PDF or TXT format.
- **Job Description Analysis:** Enter job descriptions to analyze requirements.
- **Resume Health Check:** Get a detailed analysis of how well your resume matches the job.
- **AI-Powered Tailoring:** Generate a custom-tailored resume for specific job applications.
- **Professional Formatting:** View and download your resume in a clean, professional format.
- **Multiple Export Options:** Download as Markdown, HTML, or PDF.

## How to Use

1. Upload your resume (PDF or TXT format)
2. Enter the job description
3. Check your resume's match score and get improvement suggestions
4. Generate a tailored resume
5. View and download the professionally formatted resume

## Technology

Built with:
- Streamlit for the web interface
- Google's Generative AI (Gemini) for resume analysis and tailoring
- PDF processing libraries for document handling
- HTML/CSS for professional resume formatting

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your Gemini API key: `GOOGLE_API_KEY=your_api_key_here`
4. Run the app: `streamlit run resume_builder.py`

## Note

You need to obtain a Google Generative AI API key to use this application. Keep your API key confidential. 