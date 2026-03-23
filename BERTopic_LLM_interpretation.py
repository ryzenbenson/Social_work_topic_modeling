import os
import re

import pandas as pd
from dotenv import load_dotenv
from ollama import chat
from openai import AzureOpenAI
import subprocess

# Pull local Ollama models. Below are the two open source LLMs reported in our study.
subprocess.run(["ollama", "pull", "llama3.1:8b"], check=True)
subprocess.run(["ollama", "pull", "qwen3:4b-instruct-2507-q8_0"], check=True)

load_dotenv('.env')  
API_KEY=os.environ.get('API_KEY') # Replace this endpoint in your .env file.
API_VERSION='2025-04-01-preview'
RESOURCE_ENDPOINT=os.environ.get('RESOURCE_ENDPOINT') # Replace this endpoint in your .env file.

client=AzureOpenAI(
    api_key=API_KEY,
    api_version=API_VERSION,
    azure_endpoint=RESOURCE_ENDPOINT,
)

# Read in csv containing topic clusters, where each topic has it's own row. Columns that must be present include:
# - 'Topic': The number assigned to the cluster by BERTopic (e.g.,Topic1, 2, ...).
# - 'random_sentences': n random representative sentences selected for each cluster, stored as a list of python strings.
# - 'keybert_sentences': The keyBERT keyword set for each topic cluster.
# - 'mmr_keywords': The MMR keyword set for each topic cluster.

df=pd.read_csv('topic_df.csv')
collapsed_df=df.groupby('Topic').agg({
        'random_sentence': lambda x: '. '.join(x.unique()),
        'keybert_keywords': 'first',
        'mmr_keywords': 'first'
    }).reset_index()

gpt5_df=collapsed_df.copy()
llama_df=collapsed_df.copy()
qwen_df=collapsed_df.copy()

previous_labels=[]


def gpt5_interpret(row):
    messages=[
        {"role": "system",
         "content": (
             "TASK:\n"
             "You are a helpful oncology social work research assistant. You will be given sentences from random" 
             "oncology social work notes, along with sets of keywords describing the sentences. Your task is to "
             "generate a unique label describing the sentences and keywords. Additionally, provide a "
             "detailed description of why you chose the label you did. You will be given pre-determined labels " 
             "and descriptions for which yours must differ."
                                  
             "OUTPUT FORMAT:\n"
             "Format all of your responses like below:"
             "Label: (1 sentence), Description: (1-3 sentences)")
        },
        {"role": "user",
         "content": (
             f"Given this data from social work notes:\n"
             f"Sample sentences:\n {row['random_sentence']}\n\n"
             f"keybert_keywords:\n {row['keybert_keywords']}\n\n"
             f"mmr_keywords:\n {row['mmr_keywords']}\n\n"

             "Come up with a thematic label and description for why the label was chosen. Format your response like below:\n"
             "Label: (1 sentence), Description: (1-3 sentences)\n"
             "Do not resue any of the below labels :\n"
             f"Prior Tobic Labels to Avoid: \n {previous_labels}."
             
         )}]

    response=client.chat.completions.create(
        model='gpt-5-2025-08-07',
        messages=messages,
        reasoning_effort="high"
    )
    texts=[choice.message.content for choice in response.choices]
    text_content=texts[0]
    print(text_content)

    label=text_content.split("Label:")[1].split("Description:")[0].strip()
    description=text_content.split("Description:")[1].strip()
    previous_labels.append(text_content)
    
    return pd.Series({'llm_label': label, 'llm_description': description})


previous_labels=[]


def llama_interpret(row):
    previous_labels_str=", ".join(previous_labels)

    messages=[
        {"role": "system",
         "content": (
             "TASK:\n"
             "You are a helpful oncology social work research assistant. You will be given sentences from random" 
             "oncology social work notes, along with sets of keywords describing the sentences. Your task is to "
             "generate a unique label describing the sentences and keywords. Additionally, provide a "
             "detailed description of why you chose the label you did. You will be given pre-determined labels " 
             "and descriptions for which yours must differ."
                                  
             "OUTPUT FORMAT:\n"
             "Format all of your responses like below:"
             "Label: (1 sentence), Description: (1-3 sentences)")
        },
        {"role": "user",
         "content": (
             f"Given this data from social work notes:\n"
             f"Sample sentences:\n {row['random_sentence']}\n\n"
             f"keybert_keywords:\n {row['keybert_keywords']}\n\n"
             f"mmr_keywords:\n {row['mmr_keywords']}\n\n"

             "Come up with a thematic label and description for why the label was chosen. Format your response like below:\n"
             "Label: (1 sentence), Description: (1-3 sentences)\n"
             "Do not resue any of the below labels :\n"
             f"Prior Tobic Labels to Avoid: \n {previous_labels_str}"
             
         )}]

    response=chat('llama3.1:8b', 
                    messages=messages,
                    options={"temperature": 0.7, 
                             "top_k": 20, 
                             "top_p": 0.9, 
                             "min_p": 0})

    content=response.message.content.strip()
    print(content)  

    m=re.search(r'Label:\s*(.*?)\s*Description:\s*(.*)', content, flags=re.S)
    if m:
        label=m.group(1).strip()
        description=m.group(2).strip()
    else:
        lines=[ln.strip() for ln in content.splitlines() if ln.strip()]
        if lines and lines[0].lower().startswith('label:'):
            label=lines[0].split(':', 1)[1].strip()
        else:
            label=lines[0] if lines else ''
        rest=' '.join(lines[1:]) if len(lines) > 1 else ''
        description=rest.replace('Description:', '').strip()

    previous_labels.append(content)

    return pd.Series({'llm_label': label, 'llm_description': description})


previous_labels=[]


def qwen_interpret(row):

    previous_labels_str=", ".join(previous_labels)

    messages=[
        {"role": "system",
         "content": (
             "TASK:\n"
             "You are a helpful oncology social work research assistant. You will be given sentences from random" 
             "oncology social work notes, along with sets of keywords describing the sentences. Your task is to "
             "generate a unique label describing the sentences and keywords. Additionally, provide a "
             "detailed description of why you chose the label you did. You will be given pre-determined labels " 
             "and descriptions for which yours must differ."
                                  
             "OUTPUT FORMAT:\n"
             "Format all of your responses like below:"
             "Label: (1 sentence), Description: (1-3 sentences)")
        },
        {"role": "user",
         "content": (
             f"Given this data from social work notes:\n"
             f"Sample sentences:\n {row['random_sentence']}\n\n"
             f"keybert_keywords:\n {row['keybert_keywords']}\n\n"
             f"mmr_keywords:\n {row['mmr_keywords']}\n\n"

             "Come up with a thematic label and description for why the label was chosen. Format your response like below:\n"
             "Label: (1 sentence), Description: (1-3 sentences)\n"
             "Do not resue any of the below labels :\n"
             f"Prior Tobic Labels to Avoid: \n {previous_labels_str}"
             
         )}]

    response=chat('qwen3:4b-instruct-2507-q8_0',
                    messages=messages,
                    options={"temperature": 0.7, "top_k": 20, "top_p": 0.9, "min_p": 0})

    content=response.message.content.strip()

    m=re.search(r'Label:\s*(.*?)\s*Description:\s*(.*)', content, flags=re.S)
    if m:
        label=m.group(1).strip()
        description=m.group(2).strip()
    else:
        lines=[ln.strip() for ln in content.splitlines() if ln.strip()]
        if lines and lines[0].lower().startswith('label:'):
            label=lines[0].split(':', 1)[1].strip()
        else:
            label=lines[0] if lines else ''
        rest=' '.join(lines[1:]) if len(lines) > 1 else ''
        description=rest.replace('Description:', '').strip()

    previous_labels.append(content)
    return pd.Series({'llm_label': label, 'llm_description': description})


gpt5_df[['llm_label', 'llm_description']]=gpt5_df.apply(gpt5_interpret, axis=1)
gpt5_df=gpt5_df.reset_index(drop=True)
gpt5_df=gpt5_df[['label', 'description']]
gpt5_df.to_excel("path/to/output.xlsx")

llama[['llm_label', 'llm_description']]=llama_df.apply(llama_interpret, axis=1)
llama=llama.reset_index(drop=True)
llama=llama[['llm_label', 'llm_description']]
llama.to_excel("path/to/output.xlsx")

qwen=qwen[['llm_label', 'llm_description']]
qwen=qwen.reset_index(drop=True)
qwen[['llm_label', 'llm_description']]=qwen_df.apply(qwen_interpret, axis=1)
qwen.to_excel("path/to/output.xlsx")