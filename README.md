# Granite Content Moderation Chatbot

This application is an interactive chatbot powered by **IBM Granite 3.3-8b-instruct**. It features a built-in safety layer that uses the same model to moderate responses before they are displayed to the user, ensuring a safe and helpful conversational experience.

Originally designed for IBM Watsonx.ai, this version has been refactored to run **entirely locally** using the Hugging Face `transformers` library, with specific optimizations for Apple Silicon (MPS).

## Features

- **Local Inference**: Runs entirely on your machine without external API calls.
- **Hardware Acceleration**: Optimized for Apple Silicon (M1/M2/M3) using Metal Performance Shaders (MPS).
- **Dual-Purpose Model**: Uses `granite-3.3-8b-instruct` for both generating answers and performing safety moderation.
- **Flask Web UI**: A simple and clean web interface for interacting with the chatbot.

## Prerequisites

- **Python 3.10+**
- **Disk Space**: Approximately 16GB for model weights.
- **RAM/VRAM**: 16GB+ of unified memory is highly recommended for the 8B model.
- **Hugging Face Account**: You must accept the terms for the [Granite 3.3-8b-instruct](https://huggingface.co/ibm-granite/granite-3.3-8b-instruct) model.

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd granite-guardian
   ```

2. **Install dependencies**:
   This project uses `uv` for fast dependency management.
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```

3. **Authenticate with Hugging Face**:
   Since the Granite models are gated, you need to provide an access token:
   ```bash
   export HF_TOKEN="your_hugging_face_token_here"
   # OR
   huggingface-cli login
   ```

## Running the Application

Start the Flask server:
```bash
python content-moderation-chatbot.py
```

*Note: The first run will download the model weights (approx. 16GB). This may take some time depending on your internet connection.*

Once the server is running, open your browser and navigate to:
[http://127.0.0.1:5000](http://127.0.0.1:5000)

## How it works

1. **User Input**: The user sends a prompt via the web interface.
2. **Generation**: The `granite-3.3-8b-instruct` model generates a potential response.
3. **Moderation**: The application sends a special instruction (prompt) back to the same model, asking it to evaluate if the generated response is harmful.
4. **Filtering**: 
   - If the model labels the response as **Safe**, it is shown to the user.
   - If the model labels it as **Harmful**, the response is blocked and a safety warning is displayed.

## License

This project follows the licensing terms of the IBM Granite models. See the [Hugging Face model card](https://huggingface.co/ibm-granite/granite-3.3-8b-instruct) for more details.
