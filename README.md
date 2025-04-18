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
Based on the user query, the most matching document will be retrieved and passed to the LLM for Q&A.

## Steps to Set Up the Project

1. `pip install poetry`
2. `poetry install`
3. Run `uvicorn main:app --reload` to initiate the FastAPI Swagger application.
4. Download and copy the datas from the datasource link and paste it in `dataset` folder

Pass the user query in the `/query` endpoint.
