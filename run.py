# Lab Sheet: Offline Integration with Open Source LLMs using Transformers
# CCS4310 - Deep Learning
# K.A.S.I. Ranaweera | 22UG1-0386

# First, ensure you have the required libraries installed:
# pip install transformers torch

from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
import warnings
import time
warnings.filterwarnings("ignore")

print("=" * 60)
print("Lab: Offline Integration with Open Source LLMs")
print("=" * 60)

# Task 1: Basic Text Generation with GPT-2
print("\n1. Basic Text Generation Example")
print("-" * 60)

# Load model and tokenizer locally
print("Loading GPT-2 model... (this may take a moment on first run)")
generator = pipeline("text-generation", model="gpt2")

# Basic prompt input
prompt = "In the future, artificial intelligence will"
print(f"\nPrompt: '{prompt}'")

# Generate text
output = generator(prompt, max_length=50, num_return_sequences=1, pad_token_id=50256)
print("\nGenerated Text:")
print(output[0]["generated_text"])


# Task 2: Main Activity - Blog Title Generation
print("\n\n2. Main Activity: Climate Change & Technology Blog Titles")
print("-" * 60)

# Modified prompt for the specific scenario
blog_prompt = "Blog post title about climate change and technology:"

print(f"Prompt: '{blog_prompt}'")
print("\nGenerating 3 blog post titles...")

# Generate 3 possible titles
blog_titles = generator(
    blog_prompt,
    max_length=20,
    num_return_sequences=3,
    temperature=0.8,
    pad_token_id=50256,
    do_sample=True,
)

print("\nGenerated Blog Titles:")
for i, title in enumerate(blog_titles, 1):
    # Clean up the output to show just the title part
    generated_text = title["generated_text"]
    # Remove the prompt from the output for cleaner display
    clean_title = generated_text.replace(blog_prompt, "").strip()
    print(f"Blog Title: {i}")
    print(f"{clean_title}\n")


# Task 3: Advanced Usage - Different Models and Parameters
# Load both models for comparison
print("\n\nTask: Advanced Usage - Different Models and Parameters")
print("Loading models...")
gpt2_generator = pipeline("text-generation", model="gpt2")
distilgpt2_generator = pipeline("text-generation", model="distilgpt2")

prompt = "Climate change and technology solutions include"

print(f"Prompt: '{prompt}'")
print("\n" + "=" * 50)

# Test GPT-2
print("GPT-2 Results:")
start_time = time.time()
gpt2_output = gpt2_generator(
    prompt,
    max_length=40,
    num_return_sequences=2,
    temperature=0.8,
    do_sample=True,
    pad_token_id=50256,
)
gpt2_time = time.time() - start_time

for i, result in enumerate(gpt2_output, 1):
    print(f"{i}. {result['generated_text']}")
print(f"Time taken: {gpt2_time:.2f} seconds")

print("\n" + "-" * 50)

# Test DistilGPT-2
print("DistilGPT-2 Results:")
start_time = time.time()
distil_output = distilgpt2_generator(
    prompt,
    max_length=40,
    num_return_sequences=2,
    temperature=0.8,
    do_sample=True,
    pad_token_id=50256,
)
distil_time = time.time() - start_time

for i, result in enumerate(distil_output, 1):
    print(f"{i}. {result['generated_text']}")
print(f"Time taken: {distil_time:.2f} seconds")
