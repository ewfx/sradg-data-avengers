from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pandas as pd
import os

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = 

# Load input data
llm_input_df = pd.read_csv('llm_input_data.csv')

# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["prompt"],
    template="{prompt}"
)

# Initialize the OpenAI model
llm = ChatOpenAI(temperature=0)

# Create the LLM chain
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

# Apply the chain to the input data
llm_input_df['llm_output'] = llm_input_df['Prompt'].apply(lambda x: llm_chain.run(prompt=x))

# Display the output
print(llm_input_df.head())