# DeepSeek R1 Application

## Data Source
[MedQA](https://drive.google.com/file/d/1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw/view)

## 1. Model Fine-Tuning

### Model Used
*unsloth/DeepSeek-R1-Distill-Llama-8B*

### Framework Used for Fine-Tuning
*Unsloth*

This framework allows for fine-tuning with minimal device configuration.

### Fine-Tuning Techniques
Applied `LoRA (Low-Rank Adaptation)` techniques to enhance fine-tuning performance.

Utilized the `Transformers` library to update the optimizer settings, as well as the number of epochs and batch sizes for training the model.

## 2. Building a RAG Application

### Encoding the Documents
Used the Sentence Transformers `all-MiniLM-L6-v2` model to create embeddings.

Currently, embeddings are stored in a local folder as `.npy` files. This can be enhanced to store them in a vector database.

### User Query
Based on the user query, the most relevant document will be retrieved and passed to the LLM for Q&A.

## Steps to Set Up the Project

1. Install Poetry: `pip install poetry`
2. Install dependencies: `poetry install`
3. Run the FastAPI Swagger application: `uvicorn main:app --reload`
4. Manual steps to perform:
   - Create `data`, `encoded`, and `local_model` folders.
   - Download and copy the data from the datasource link and paste it into the `data` folder.
   - The `encoded` folder is to store the embeddings locally.
   - The `local_model` folder is to store the fine-tuned model for local inference.

Pass the user query to the `/query` endpoint.
