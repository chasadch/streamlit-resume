# AI Resume Tailor

An intelligent resume tailoring application that uses Google's Generative AI (Gemini) to help job seekers create targeted resumes for specific job postings.

## Features

- **Resume Health Check**: Analyze how well your resume matches a job description
- **Smart Resume Tailoring**: Create customized versions of your resume for specific jobs
- **Multiple Export Formats**: Download your resume in Markdown, HTML, or PDF formats
- **Professional Formatting**: Clean, professional layout for your tailored resume
- **Quota Management**: Smart handling of API usage with automatic retries and optimization

## Requirements

- Python 3.8+
- Streamlit
- Google Generative AI API key
- Other dependencies listed in requirements.txt

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-resume-tailor.git
cd ai-resume-tailor
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Unix/MacOS:
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your Google Generative AI API key:
   - Get an API key from [Google AI Studio](https://ai.google.dev/)
   - Create a `.env` file in the project root
   - Add your API key: `GOOGLE_API_KEY=your_api_key_here`

5. Run the application:
```bash
streamlit run streamlit_app.py
```

## Usage

1. Upload your existing resume (PDF or TXT format)
2. Paste the job description you're applying for
3. Use the Resume Health Check to analyze your match
4. Generate a tailored version of your resume
5. Download in your preferred format

## Privacy

- Your resume and job description data is processed securely
- No data is stored permanently after processing
- All communication with Google's API is encrypted

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 