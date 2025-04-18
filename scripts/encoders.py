import os
import numpy as np
import torch
import asyncio
from sentence_transformers import SentenceTransformer, util
import aiofiles

# Load the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

async def load_text_file(file_path):
    async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
        return await file.read()

async def encode_text(text, device):
    return model.encode(text, convert_to_tensor=True).to(device)

async def get_textbook_embeddings(user_input: str, current_path: str):
    textbook_folder = os.path.join(current_path, 'data/textbooks/en')
    text_files = [f for f in os.listdir(textbook_folder) if f.endswith('.txt')]

    file_names = []

    if not os.path.exists(os.path.join(current_path, 'encoded/encoded_texts.npy')):
        encoded_texts = []
        for file_name in text_files:
            file_path = os.path.join(textbook_folder, file_name)
            text = await load_text_file(file_path)
            encoded_text = await encode_text(text, device)
            encoded_texts.append(encoded_text)
            file_names.append(file_name)

        # Convert list of tensors to a numpy array
        encoded_array = np.array([encoded_text.cpu().numpy() for encoded_text in encoded_texts])
        
        # Save the encoded values to a .npy file
        np.save(os.path.join(current_path, 'encoded/encoded_texts.npy'), encoded_array)
    else:
        # Load the .npy file if it exists
        encoded_array = np.load(os.path.join(current_path, 'encoded/encoded_texts.npy'))
        # Ensure file_names is populated from the existing files
        file_names = text_files

    # Encode user input
    user_input_encoded = await encode_text(user_input, device)

    # Calculate cosine similarity
    cosine_scores = util.pytorch_cos_sim(user_input_encoded, encoded_array)

    # Find the index of the most similar text
    most_similar_index = cosine_scores.argmax()
    most_similar_file = file_names[most_similar_index]
    similarity_score = cosine_scores[0][most_similar_index].item()

    return most_similar_file, similarity_score
