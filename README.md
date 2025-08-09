# Offline Writing Assistant

A simple command-line interface (CLI) application for generating text (blog titles, outlines, or paragraphs) using Hugging Face Transformers with GPT-2 or DistilGPT2 models. Designed for offline use, this tool supports educational purposes, brainstorming, and content creation without internet access after initial model downloads. It fulfills the requirements of the "Offline Integration with Open Source LLMs using Transformers" lab sheet.

## Features

- **Offline Text Generation**: Generate text using pre-trained GPT-2 or DistilGPT2 models without an internet connection after initial setup.
- **Interactive CLI**: Choose model, task (titles, outlines, paragraphs), topic, and creativity settings (temperature).
- **Lab Sheet Compliance**:
  - Basic text generation with GPT-2.
  - Generates three blog titles for "climate change and technology."
  - Compares GPT-2 and DistilGPT2 outputs.
  - Demonstrates creativity parameters (low and high temperature).
- **Output Storage**: Saves generated text to `outputs/generated_text.txt` with timestamps for easy reference.
- **Lightweight and Modular**: Organized folder structure and minimal code for ease of use and extension.

## Folder Structure

```
offline_writing_assistant/
├── outputs/
│   └── generated_text.txt    # Generated text outputs
├── writing_assistant.py      # Main CLI application script
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation (this file)
```

- **src/**: Contains the main Python script (`writing_assistant.py`).
- **outputs/**: Stores generated text files with timestamps.
- **requirements.txt**: Lists dependencies (`transformers`, `torch`).
- **README.md**: This file with setup and usage instructions.

## Prerequisites

- **Python**: Version 3.8 or higher.
- **Hardware**:
  - Minimum 4GB RAM (8GB recommended for GPT-2).
  - ~1.5GB disk space for GPT-2, ~500MB for DistilGPT2.
- **Internet**: Required only for initial model downloads.
- **Optional**: GPU with CUDA support for faster processing.

## Setup Instructions

1. **Clone or Create the Project Directory**:
   - Create the folder structure as shown above:
     ```
     offline_writing_assistant/
     ├── outputs/
     ├── writing_assistant.py
     ├── requirements.txt
     └── README.md
     ```
   - Copy the provided `writing_assistant.py` and `requirements.txt` into their respective directories.

2. **Install Dependencies**:
   - Navigate to the project directory:
     ```bash
     cd offline_writing_assistant
     ```
   - Install required libraries:
     ```bash
     pip install -r requirements.txt
     ```
   - This installs `transformers` and `torch`. The first run will download models (~1.5GB for GPT-2, ~500MB for DistilGPT2).

3. **Verify Setup**:
   - Ensure Python 3.8+ is installed:
     ```bash
     python --version
     ```
   - Check if `torch` supports CUDA (optional):
     ```bash
     python -c "import torch; print(torch.cuda.is_available())"
     ```

## Usage

1. **Run the Application**:
   - Navigate to the `src` directory:
     ```bash
     cd src
     ```
   - Execute the script:
     ```bash
     python writing_assistant.py
     ```

2. **Program Flow**:
   - **Lab Examples**: The script first runs:
     - Basic GPT-2 text generation ("In the future, artificial intelligence will...").
     - Three blog titles for "climate change and technology."
     - Comparison of GPT-2 and DistilGPT2 outputs.
     - Demonstration of creativity settings (low/high temperature).
   - **Interactive Mode**:
     - Choose a model (1: GPT-2, 2: DistilGPT2).
     - Select a task (1: Titles, 2: Outline, 3: Paragraph).
     - Enter a topic (e.g., "climate change and technology").
     - Specify temperature (0.1–1.5, default 0.8).
     - Outputs are displayed and saved to `outputs/generated_text.txt`.
     - Option to run another task or exit.

3. **Example Interaction**:
   ```
   OFFLINE WRITING ASSISTANT

   PyTorch: 2.4.1
   CUDA: False

   === Basic Text Generation ===
   Loading gpt2...
   Model gpt2 loaded successfully.
   Generated: In the future, artificial intelligence will revolutionize industries with smarter automation.
   Saved to outputs/generated_text.txt

   === Climate Change Blog Titles ===
   Loading gpt2...
   Model gpt2 loaded successfully.
   Generated Titles:
   1. AI-Powered Climate Solutions
   2. Sustainable Tech Innovations
   3. Green Energy Through Technology
   Saved to outputs/generated_text.txt

   === Writing Assistant ===
   Models: 1. GPT-2  2. DistilGPT-2
   Choose model (1 or 2): 1
   Tasks: 1. Titles  2. Outline  3. Paragraph
   Choose task (1, 2, or 3): 1
   Enter topic (e.g., climate change and technology): renewable energy
   Enter temperature (0.1-1.5, default 0.8): 0.8
   Loading gpt2...
   Model gpt2 loaded successfully.
   Generated Output:
   1. Solar Power Advancements for a Greener Future
   2. Wind Energy Innovations in Modern Technology
   3. Sustainable Energy Solutions Through AI
   Saved to outputs/generated_text.txt
   Another task? (y/n): n
   ```

4. **Output File**:
   - Generated texts are saved to `outputs/generated_text.txt`. Example:
     ```
     === Generated on 2025-08-10 04:45:00.123456 ===
     Prompt: Blog post title: Climate change and technology -
     1. Blog post title: Climate change and technology - AI-Powered Climate Solutions
     2. Blog post title: Climate change and technology - Sustainable Tech Innovations
     3. Blog post title: Climate change and technology - Green Energy Through Technology
     ```

## Lab Sheet Objectives

This application fulfills the "Offline Integration with Open Source LLMs using Transformers" lab sheet requirements:
- **Objective 1**: Uses open-source LLMs (GPT-2, DistilGPT2) locally without API access.
- **Objective 2**: Installs and uses Hugging Face `transformers` and `torch` libraries.
- **Objective 3**: Performs text generation with GPT-2 (and DistilGPT2) for various tasks.
- **Objective 4**: Demonstrates tokenizers and model pipelines via the `pipeline` API.
- **Activity**: Generates three blog titles for "climate change and technology."
- **Question 5**: Compares GPT-2 and DistilGPT2 outputs.
- **Deliverables**:
  - Python script (`src/writing_assistant.py`).
  - Output file (`outputs/generated_text.txt`).
  - Terminal output for screenshots.

## Troubleshooting

- **Model Download Issues**:
  - Ensure internet access during first run for model downloads.
  - Verify sufficient disk space (~1.5GB for GPT-2, ~500MB for DistilGPT2).
- **Memory Errors**:
  - Use DistilGPT2 (option 2) for low-memory devices.
  - Close other applications to free RAM.
- **CUDA Errors**:
  - If CUDA is unavailable, the script uses CPU (slower but functional).
  - Reinstall `torch` with CUDA support if needed:
    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cu118
    ```
- **No Output**:
  - Check if `outputs/` directory exists (created automatically).
  - Ensure valid inputs for model/task choices.

## Extending the Application

- **Add Models**: Modify `self.models` in `writing_assistant.py` to include models like `gpt2-medium` or `flan-t5-small`.
- **New Tasks**: Update the `tasks` dictionary to support tasks like summarization or Q&A.
- **GUI Interface**: Integrate a GUI using `tkinter` or `streamlit` for a graphical interface.
- **Fine-Tuning**: Fine-tune models locally using the Transformers library for specific domains.

## Notes

- **Offline Operation**: After initial model downloads, no internet is required.
- **Performance**: GPT-2 is more coherent but memory-intensive; DistilGPT2 is faster and lighter.
- **Output Management**: Outputs are appended to `generated_text.txt`; clear the file manually if needed.
- **Memory Management**: To manage conversation history, use the "Data Controls" section in settings or the book icon to forget specific chats.

For issues or contributions, contact the project maintainer or refer to the Hugging Face documentation: https://huggingface.co/docs/transformers
