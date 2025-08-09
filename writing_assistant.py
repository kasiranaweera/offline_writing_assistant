import torch
from transformers import pipeline
from datetime import datetime
import os

class WritingAssistant:
    """Simple CLI Writing Assistant using Hugging Face Transformers."""
    
    def __init__(self):
        """Initialize with output file path."""
        self.output_file = "outputs/generated_text.txt"
        self.models = {"1": "gpt2", "2": "distilgpt2"}
    
    def load_model(self, model_name):
        """Load a model with error handling."""
        try:
            print(f"Loading {model_name}...")
            return pipeline("text-generation", model=model_name)
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            return None
    
    def generate_text(self, generator, prompt, max_length=50, num_sequences=1, temperature=0.8):
        """Generate text with given parameters."""
        try:
            return generator(
                prompt,
                max_length=max_length,
                num_return_sequences=num_sequences,
                temperature=temperature,
                do_sample=True,
                pad_token_id=generator.tokenizer.eos_token_id
            )
        except Exception as e:
            print(f"Error generating text: {e}")
            return []
    
    def save_outputs(self, outputs, prompt):
        """Save outputs to a file."""
        os.makedirs("outputs", exist_ok=True)
        with open(self.output_file, "a", encoding="utf-8") as f:
            f.write(f"\n=== Generated on {datetime.now()} ===\n")
            f.write(f"Prompt: {prompt}\n")
            for i, output in enumerate(outputs, 1):
                f.write(f"{i}. {output['generated_text']}\n")
        print(f"Saved to {self.output_file}")
    
    def basic_text_generation(self):
        """Run basic GPT-2 example from lab sheet."""
        print("\n=== Basic Text Generation ===")
        generator = self.load_model("gpt2")
        if not generator:
            return
        prompt = "In the future, artificial intelligence will"
        outputs = self.generate_text(generator, prompt, max_length=50, num_sequences=1)
        for output in outputs:
            print(f"Generated: {output['generated_text']}")
        self.save_outputs(outputs, prompt)
    
    def climate_change_titles(self):
        """Generate blog titles for climate change and technology."""
        print("\n=== Climate Change Blog Titles ===")
        generator = self.load_model("gpt2")
        if not generator:
            return
        prompt = "Blog post title: Climate change and technology -"
        outputs = self.generate_text(generator, prompt, max_length=30, num_sequences=3, temperature=0.8)
        print("Generated Titles:")
        for i, output in enumerate(outputs, 1):
            title = output['generated_text'].replace(prompt, "").strip()
            print(f"{i}. {title}")
        self.save_outputs(outputs, prompt)
    
    def compare_models(self):
        """Compare GPT-2 and DistilGPT-2."""
        print("\n=== Model Comparison ===")
        prompt = "The benefits of renewable energy include"
        
        print("GPT-2:")
        generator_gpt2 = self.load_model("gpt2")
        if generator_gpt2:
            outputs = self.generate_text(generator_gpt2, prompt, max_length=60, num_sequences=1)
            for output in outputs:
                print(output['generated_text'])
            self.save_outputs(outputs, prompt)
        
        print("\nDistilGPT-2:")
        generator_distil = self.load_model("distilgpt2")
        if generator_distil:
            outputs = self.generate_text(generator_distil, prompt, max_length=60, num_sequences=1)
            for output in outputs:
                print(output['generated_text'])
            self.save_outputs(outputs, prompt)
    
    def creativity_demo(self):
        """Demonstrate creativity parameters."""
        print("\n=== Creativity Demo ===")
        generator = self.load_model("gpt2")
        if not generator:
            return
        prompt = "Space exploration is important because"
        
        print("Low Creativity (temperature=0.3):")
        outputs = self.generate_text(generator, prompt, max_length=50, temperature=0.3)
        for output in outputs:
            print(output['generated_text'])
        
        print("\nHigh Creativity (temperature=1.2):")
        outputs = self.generate_text(generator, prompt, max_length=50, temperature=1.2)
        for output in outputs:
            print(output['generated_text'])
        self.save_outputs(outputs, prompt)
    
    def interactive_mode(self):
        """Run interactive CLI."""
        print("\n=== Writing Assistant ===")
        print("Models: 1. GPT-2  2. DistilGPT-2")
        model_choice = input("Choose model (1 or 2): ").strip()
        model_name = self.models.get(model_choice, "gpt2")
        
        generator = self.load_model(model_name)
        if not generator:
            return
        
        print("\nTasks: 1. Titles  2. Outline  3. Paragraph")
        task_choice = input("Choose task (1, 2, or 3): ").strip()
        tasks = {
            "1": ("Titles", "Blog post title: {topic} -", 30, 3),
            "2": ("Outline", "Outline for a blog post about {topic}:", 100, 1),
            "3": ("Paragraph", "A paragraph about {topic}:", 150, 1)
        }
        task = tasks.get(task_choice, tasks["1"])
        
        topic = input("Enter topic (e.g., climate change and technology): ").strip()
        prompt = task[1].format(topic=topic)
        
        try:
            temperature = float(input("Enter temperature (0.1-1.5, default 0.8): ") or 0.8)
        except ValueError:
            temperature = 0.8
        
        outputs = self.generate_text(generator, prompt, max_length=task[2], num_sequences=task[3], temperature=temperature)
        print("\nGenerated Output:")
        for i, output in enumerate(outputs, 1):
            cleaned = output['generated_text'].replace(task[1].format(topic=topic), "").strip() if task_choice == "1" else output['generated_text']
            print(f"{i}. {cleaned}")
        self.save_outputs(outputs, prompt)
    
    def run(self):
        """Run all lab tasks and interactive mode."""
        print("OFFLINE WRITING ASSISTANT\n")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA: {torch.cuda.is_available()}\n")
        
        self.basic_text_generation()
        self.climate_change_titles()
        self.compare_models()
        self.creativity_demo()
        
        while True:
            self.interactive_mode()
            if input("\nAnother task? (y/n): ").lower() != 'y':
                break

if __name__ == "__main__":
    assistant = WritingAssistant()
    assistant.run()