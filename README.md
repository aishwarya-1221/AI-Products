## Intelligent Meeting Transcription Assistant </br>
This project is an AI-powered application for transcribing audio files and extracting detailed key points using a large language model (LLM). It leverages IBM WatsonX, Hugging Face, and Gradio for seamless speech-to-text processing and analysis.

### Features </br>
Audio Transcription: Converts audio files to text using OpenAI's Whisper model. </br>
Key Point Analysis: Summarizes and extracts detailed insights from the transcription using an IBM WatsonX-powered LLM (LLAMA3). </br>
User-Friendly Interface: Provides an intuitive Gradio-based web interface for easy interaction. </br>
Customizable: The prompt template and models can be easily adjusted for different tasks or use cases. </br>

### Technologies Used </br>
Python: Core language for development. </br>
IBM WatsonX: For LLM-powered text analysis. </br>
Hugging Face: Model hub integration. </br>
Gradio: To create the web-based user interface. </br>
OpenAI Whisper: For automatic speech recognition. </br>
LangChain: For chaining prompts and LLM interactions. </br>

### Prerequisites </br>
Python 3.9 or above </br>
FFmpeg (required for audio processing) </br>
IBM WatsonX credentials </br>
API access to Hugging Face models (if necessary) </br>


## Set Up Environment

### Install required dependencies:
pip install -r requirements.txt


### Install FFmpeg

Linux: </br>
sudo apt install ffmpeg

macOS: </br>
brew install ffmpeg

Windows: </br>
Download from FFmpeg official website.</br>
Add the bin directory to your system's PATH.

### Set Up Credentials

Create a .env file in the root directory with WATSONX_URL, WATSONX_API_KEY, and PROJECT_ID.</br>

### Load Environment Variables </br>
Ensure the .env file is loaded by your script


