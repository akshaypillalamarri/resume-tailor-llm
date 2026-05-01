import gradio as gr
import google.generativeai as genai
import pdfplumber
import docx2txt
import os

# Load Gemini API key from environment
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

print("Gemini model ready")


def extract_text_from_file(file):
    if file is None:
        return ""
    file_path = file if isinstance(file, str) else file.name
    ext = os.path.splitext(file_path)[-1].lower()
    try:
        if ext == ".pdf":
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text.strip()
        elif ext in [".docx", ".doc"]:
            return docx2txt.process(file_path).strip()
        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        else:
            return "Unsupported file type. Please upload PDF, DOCX, or TXT."
    except Exception as e:
        return f"Error reading file: {str(e)}"


def tailor_resume(job_description, resume_file, base_resume_text):
    if resume_file is not None:
        base_resume = extract_text_from_file(resume_file)
        if base_resume.startswith("Error") or base_resume.startswith("Unsupported"):
            return base_resume
    elif base_resume_text.strip():
        base_resume = base_resume_text.strip()
    else:
        return "Please either upload your resume file or paste your resume text below."

    if not job_description.strip():
        return "Please paste the job description."

    prompt = f"""You are an expert resume writer and ATS optimization specialist.
Your task is to tailor the candidate's resume to match the job description provided.
Instructions:
- Reorder and emphasize skills that match the job description
- Incorporate relevant keywords from the job description naturally
- Highlight the most relevant experience and projects
- Keep all information factually accurate — do not invent experience
- Format the output as a clean, professional resume
- Include sections: Summary, Skills, Experience, Projects, Education, Certifications
Job Description:
{job_description}
Candidate's Base Resume:
{base_resume}
Now write the tailored resume:"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"


with gr.Blocks(title="Resume Tailor LLM") as demo:
    gr.Markdown(
        """
        # Resume Tailor LLM
        ### Powered by Gemini AI + Fine-tuned LoRA Research — built by Akshay Pillalamarri
        Paste a job description and upload your resume (PDF or DOCX). Get a tailored,
        ATS-optimized version that emphasizes the right skills and keywords for the role.
        **Tech stack:** Python, Google Gemini API, Hugging Face, PEFT (LoRA), Gradio
        """
    )

    with gr.Row():
        with gr.Column():
            jd_input = gr.Textbox(
                label="Job Description",
                placeholder="Paste the job description here...",
                lines=10
            )
            gr.Markdown("### Your Resume")
            gr.Markdown("**Option 1 — Upload file (PDF or DOCX recommended)**")
            resume_file = gr.File(
                label="Upload Resume",
                file_types=[".pdf", ".docx", ".doc", ".txt"],
                type="filepath"
            )
            gr.Markdown("**Option 2 — Or paste resume text below**")
            resume_text = gr.Textbox(
                label="Paste Resume Text",
                placeholder="Or paste your resume here if you prefer...",
                lines=8
            )
            submit_btn = gr.Button("Tailor My Resume", variant="primary", size="lg")

        with gr.Column():
            output = gr.Textbox(
                label="Tailored Resume Output",
                lines=30,
                placeholder="Your tailored resume will appear here..."
            )

    gr.Markdown(
        """
        ---
        **About this project:** This demo uses Google Gemini for high-quality resume tailoring.
        The underlying research involved fine-tuning TinyLlama-1.1B with LoRA — see the
        [GitHub repo](https://github.com/akshaypillalamarri/resume-tailor-llm) and
        [HuggingFace model](https://huggingface.co/akshayrinku/tinyllama-resume-tailor-lora) for details.
        Connect: [LinkedIn](https://www.linkedin.com/in/akshay-pillalamarri) |
        [GitHub](https://github.com/akshaypillalamarri) |
        [Portfolio](https://akshaypillalamarri.github.io)
        """
    )

    submit_btn.click(
        fn=tailor_resume,
        inputs=[jd_input, resume_file, resume_text],
        outputs=output
    )

demo.launch()
