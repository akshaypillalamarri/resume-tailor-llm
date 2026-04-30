# Resume Tailor LLM

A fine-tuned **TinyLlama-1.1B-Chat** model that automatically tailors resumes to match specific job descriptions. Built with **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning.

**Live Demo:** [Try it on Hugging Face Spaces](https://huggingface.co/spaces/akshayrinku/resume-tailor-demo)

**Model on HF Hub:** [akshayrinku/tinyllama-resume-tailor-lora](https://huggingface.co/akshayrinku/tinyllama-resume-tailor-lora)

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
