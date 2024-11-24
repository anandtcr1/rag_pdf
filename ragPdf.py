import json
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
# from openai import OpenAI
import openai

api_key = 'key'

# client = OpenAI(
#   api_key = api_key
# )
openai.api_key = api_key

def exract_text_from_path(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text, max_length=300):
    words = text.split()
    chunks = []
    chunk = []
    for word in words:
        if len(' '.join(chunk)) + len(word) + 1 > max_length:
            chunks.append(' '.join(chunk))
            chunk = []

        chunk.append(word)
    if chunk:
        chunks.append(' '.join(chunk))
    
    return chunks

def save_rag_data_to_file(rag_data, filename):
    with open(filename, 'w') as file:
        json.dump(rag_data, file)

def load_data_from_file(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def retrive_relevant_chunk(query, rag_data, model):
    query_embedding = model.encode([query])
    embeddings = np.array([np.array(item['embedding']) for item in rag_data])
    similarities = cosine_similarity(query_embedding, embeddings)
    top_index = np.argmax(similarities)
    return rag_data[top_index]['text']

def generate_openai_response(context, query):
    prompt = f"""You are a helpful assistant. Use the following context to answer the question.
                Context:{context}
                Question:{query}"""
    response = openai.chat.completions.create(
        model="gpt-4",  # or "gpt-3.5-turbo gpt-4" for better quality
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,  # Adjust temperature for creativity (0.7 is moderate)
        max_tokens=300,  # Adjust based on the response length you expect
        top_p=1.0,       # Use nucleus sampling
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    result = response.choices[0].message.content
    return result

def processData():
    # pdf_path = 'smpl.pdf'
    output_rag_file = 'rag_data.json'
    folder_path = 'data'

    for pdf_path in os.listdir(folder_path):
        if pdf_path.lower().endswith('.pdf'):

            print(f"Loading file - {pdf_path}")

            full_path = os.path.join(folder_path, pdf_path)

            text = exract_text_from_path(full_path)
            chunks = chunk_text(text)

            model = SentenceTransformer('all-MiniLM-L6-v2')
            # model = SentenceTransformer('msmarco-distilbert-base-v4')
            embeddings = model.encode(chunks)

            rag_data = [
                {"text": chunks[i], 'embedding': embeddings[i].tolist()}
                for i in range(len(chunks))
            ]

            save_rag_data_to_file(rag_data, output_rag_file)

    print(f"RAG data saved to {output_rag_file}")

    # loaded_rag_data = load_data_from_file(output_rag_file)
    # query = 'list all the company names mentioned in the documents.'
    # most_relevant_text = retrive_relevant_chunk(query, loaded_rag_data, model)

    # response = generate_openai_response(most_relevant_text, query)
    

    # print(f"Query: {query}")
    # print(f"Openai response is : {response}")


# processData()
def chat_with_rag(query) :
    output_rag_file = 'rag_data.json'
    loaded_rag_data = load_data_from_file(output_rag_file)
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    model = SentenceTransformer('msmarco-distilbert-base-v4')

    most_relevant_text = retrive_relevant_chunk(query, loaded_rag_data, model)
    response = generate_openai_response(most_relevant_text, query)
    print(response)

def prompt_user():
    while True:
        user_input = input('Query : ')
        if(user_input == 'exit'):
            print('Exiting..')
            break
        else:
            chat_with_rag(user_input)

# prompt_user()
processData()