"""
Audio Transcription and Analysis App
This script processes audio input, transcribes it using a speech-to-text model,
and generates detailed key point analysis with Llama 3 LLM.
"""

# Import necessary libraries and modules
import os
from dotenv import load_dotenv
import gradio as gr
from transformers import pipeline
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

# Load environment variables from .env file
load_dotenv()

##--LLM Initialization--###

# Retrieve credentials securely from environment variables
my_credentials = {
    "url": os.getenv("WATSONX_URL"),
    "api_key": os.getenv("WATSONX_API_KEY")
}

# Model parameters
params = {
    GenParams.MAX_NEW_TOKENS: 800,
    GenParams.TEMPERATURE: 0.1,
}

# Initialize the LLAMA3 model
# Note: Ensure this model is accessible and properly configured
LLAMA2_model = Model(
    model_id='meta-llama/llama-3-8b-instruct',
    credentials=my_credentials,
    params=params,
    project_id=os.getenv("PROJECT_ID"),
)

# Instantiate the WatsonX LLM
llm = WatsonxLLM(LLAMA2_model)


##--Prompt Template--##

# Define the structured prompt template for key point extraction
# Adjust the tags for other LLMs as necessary
temp = """
<s><<SYS>>
List the key points with details from the context: 
[INST] The context : {context} [/INST] 
<</SYS>>
"""

# Create a prompt template object using LangChain
pt = PromptTemplate(
    input_variables=["context"],  # Input variable expected in the prompt
    template=temp
)

# Combine the LLM and prompt into an LLMChain for processing
prompt_to_LLAMA2 = LLMChain(llm=llm, prompt=pt)


##--Speech-to-Text Function--##

def transcript_audio(audio_file):
    """
    Transcribes an audio file and processes it through the LLM chain for key point extraction.

    Args:
        audio_file (str): Path to the audio file to be transcribed.

    Returns:
        str: Generated analysis from the LLM based on the transcribed text.
    """
    try:
        # Initialize the speech recognition pipeline
        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny.en",  # Replace with your desired Whisper model
            chunk_length_s=30,  # Process audio in 30-second chunks
        )
        
        # Transcribe the audio file
        transcript_txt = pipe(audio_file, batch_size=8)["text"]
        
        # Run the transcription through the LLM chain
        result = prompt_to_LLAMA2.run(transcript_txt)
        
        return result
    except Exception as e:
        return f"Error processing audio file: {str(e)}"


##--Gradio Interface--##

# Define input and output components for Gradio
audio_input = gr.Audio(sources="upload", type="filepath", label="Upload an audio file")
output_text = gr.Textbox(label="Transcription and Analysis")

# Create and configure the Gradio interface
iface = gr.Interface(
    fn=transcript_audio, 
    inputs=audio_input, 
    outputs=output_text, 
    title="Audio Transcription and Analysis App",
    description=(
        "Upload an audio file to transcribe and extract detailed key points using AI. "
        "The transcription is powered by Whisper, and the analysis is performed by LLAMA3."
    ),
    theme="default"
)

# Launch the Gradio app
iface.launch(server_name="0.0.0.0", server_port=7860)
