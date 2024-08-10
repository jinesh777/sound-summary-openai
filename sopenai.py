import openai
import torch
from langchain.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from output_parser import summary_parser,Summary
# Set your API key
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

# Load audio file
audio_file = open('33.mp3', 'rb')
# Transcribe
transcription = openai.audio.transcriptions.create(
  model="whisper-1",
  file=audio_file
)
# Print the transcribed text
information=transcription.text
summary_template = """ given the information {information} about a discussion between two people  I want you to create:
 1. create summary 
 2. identify how many people where discussion
 3. identify what was the second person name
 """


summary_prompt_template = PromptTemplate(
    input_variables=["information"], template=summary_template, partial_variables={
        "format_instructions": summary_parser.get_format_instructions()
    },
)
# llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
llm = ChatOllama(model="llama3")
chain = summary_prompt_template | llm | StrOutputParser()
res = chain.invoke(input={"information": information})
print(res)