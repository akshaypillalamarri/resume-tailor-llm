# Resume Tailor LLM

A fine-tuned **Microsoft Phi-2** model that automatically tailors resumes to match specific job descriptions. Built with **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning.

**Live Demo:** [Try it on Hugging Face Spaces](https://huggingface.co/spaces/akshayrinku/tinyllama-resume-tailor-lora)

## Why This Project

After applying to dozens of roles during my job search, I realized resume tailoring was the single most time-consuming and high-impact part of the process. So I built an AI tool to automate it.

This project demonstrates real-world LLM fine-tuning, parameter-efficient training, and end-to-end model deployment.

## What It Does

**Input:** A job description + your base resume
**Output:** A tailored, ATS-optimized resume that:

- Reorders skills by relevance to the JD
- Emphasizes matching experience
- Aligns keyword density with role requirements
- Maintains factual accuracy from your original resume

## Architecture

```
Job Description ─┐
                 ├──→ Phi-2 (Base) + LoRA Adapter ──→ Tailored Resume
Base Resume ─────┘
```

## Tech Stack

| Layer        | Technology                        |
| ------------ | --------------------------------- |
| Base Model   | Microsoft Phi-2 (2.7B parameters) |
| Fine-tuning  | LoRA via PEFT library             |
| Quantization | 4-bit NF4 with bitsandbytes       |
| Training     | Google Colab (T4 GPU)             |
| Serving      | Gradio + Hugging Face Spaces      |
| Language     | Python                            |

## Training Details

- **Method:** LoRA (Low-Rank Adaptation)
- **Trainable params:** ~1% of base model
- **Target modules:** `q_proj`, `k_proj`, `v_proj`, `dense`
- **Rank (r):** 16
- **Alpha:** 32
- **Dropout:** 0.05
- **Epochs:** 3
- **Batch size:** 2 (with gradient accumulation of 4)
- **Learning rate:** 2e-4
- **Dataset:** 15+ curated examples of base resume to tailored resume pairs across roles (AI Engineer, SDET, Salesforce Developer, Data Engineer, etc.)

## Results

The model successfully learns to:

- Restructure resume sections for relevance
- Surface domain-specific keywords from the JD
- Demote unrelated skills without removing them
- Preserve factual content while reformatting

## Project Structure

```
resume-tailor-llm/
├── resume_tailor_finetune.ipynb   # Main fine-tuning notebook (run in Colab)
├── resume_dataset.json             # Training data
├── app.py                          # Gradio demo for HF Spaces
├── requirements.txt                # Dependencies
└── README.md
```

## How To Run

### Step 1 — Fine-tune in Colab

1. Open `resume_tailor_finetune.ipynb` in Google Colab
2. Set runtime to T4 GPU
3. Upload `resume_dataset.json`
4. Run cells sequentially
5. Push adapter to Hugging Face Hub

### Step 2 — Deploy demo

1. Create a new Hugging Face Space (Gradio template)
2. Upload `app.py` and `requirements.txt`
3. The Space auto-deploys

## What I Learned

- **LoRA makes fine-tuning accessible** — full fine-tuning of Phi-2 would require expensive GPUs, but LoRA brings it to free-tier Colab
- **4-bit quantization** dramatically reduces memory while preserving quality
- **Dataset quality matters more than quantity** — 15 carefully crafted examples produce better results than 1000 noisy ones
- **End-to-end deployment** is its own skill — fine-tuning is only half the job

## About Me

I'm Akshay Pillalamarri — an AI/ML Engineer and Software Developer based in Folsom, CA. I hold a Master's in Computer Science from the University of Central Missouri.

Currently open to AI/ML Engineering, SDET, and Software Development roles.

- **Portfolio:** [akshaypillalamarri.github.io](https://akshaypillalamarri.github.io)
- **LinkedIn:** [linkedin.com/in/akshay-pillalamarri](https://www.linkedin.com/in/akshay-pillalamarri)
- **GitHub:** [github.com/akshaypillalamarri](https://github.com/akshaypillalamarri)
- **Email:** pillalamarriakshay01@gmail.com

## License

MIT
