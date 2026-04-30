import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_REPO = "akshayrinku/tinyllama-resume-tailor-lora"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_REPO)
model.eval()
print("Model ready")


def tailor_resume(job_description, base_resume):
    if not job_description.strip() or not base_resume.strip():
        return "Please provide both a job description and a base resume."

    prompt = f"""### Instruction:
Tailor the following resume to match the job description provided. Optimize keywords, reorder skills by relevance, and emphasize matching experience.

### Job Description:
{job_description}

### Base Resume:
{base_resume}

### Tailored Resume:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    tailored = decoded.split("### Tailored Resume:")[-1].strip()
    return tailored


with gr.Blocks(title="Resume Tailor LLM", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Resume Tailor LLM
        ### Fine-tuned Phi-2 with LoRA — built by Akshay Pillalamarri

        Paste a job description and your base resume. Get a tailored, ATS-optimized version that emphasizes the right skills and keywords for the role.

        **Tech stack:** Python, Hugging Face Transformers, PEFT (LoRA), Microsoft Phi-2, Gradio
        """
    )

    with gr.Row():
        with gr.Column():
            jd_input = gr.Textbox(
                label="Job Description",
                placeholder="Paste the job description here...",
                lines=10,
            )
            resume_input = gr.Textbox(
                label="Base Resume",
                placeholder="Paste your base resume here...",
                lines=10,
            )
            submit_btn = gr.Button("Tailor My Resume", variant="primary")

        with gr.Column():
            output = gr.Textbox(
                label="Tailored Resume",
                lines=22,
                show_copy_button=True,
            )

    gr.Markdown(
        """
        ---
        **About this project:**
        This is a fine-tuned version of Microsoft Phi-2 (2.7B parameters) using LoRA on a custom dataset of resume-JD pairs. Built as part of my AI engineering portfolio.

        Connect: [LinkedIn](https://www.linkedin.com/in/akshay-pillalamarri) | [GitHub](https://github.com/akshaypillalamarri) | [Portfolio](https://akshaypillalamarri.github.io)
        """
    )

    submit_btn.click(fn=tailor_resume, inputs=[jd_input, resume_input], outputs=output)


if __name__ == "__main__":
    demo.launch()
