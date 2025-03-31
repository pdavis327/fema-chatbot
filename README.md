# Local RAG Chatbot App

This is a simple open-source local RAG (Retrieval-Augmented Generation) chatbot application. It utilizes LangChain, Chroma for the vector database, and Ollama for the LLM (Large Language Model). The repository includes example PDF documents, and the functionality to embed and store documents in a new Chroma database collection.

## Getting Started

### Prerequisites

#### Creating a Conda Environment

To set up a new Conda environment, run the following commands:

```zsh
conda create --name my_project_env
conda activate my_project_env
pip install -r requirements.txt
```
To add pre-commit hooks, run the following command:
`pre-commit install`

#### Pulling the Model and Running Ollama

To pull the desired model and start Ollama, use the commands:

```zsh
ollama pull llama2
ollama serve
```

If you choose to run the code in Docker, pulling and serving the model will be handled automatically during the build process.

### Installation

1. Clone the repository and navigate to the project directory:

   ```zsh
   git clone <repository-url>
   cd <repository-name>
   ```

2. Rename `.env.example` to  `.env`

3. Specify the environment parameters in the `.env` file.

## Executing the Program

### Creating a Chroma Database and Embedding Documents

You can create a Chroma database and embed documents using `util/chroma.py`. It requires one argument: the filepath to the documents you wish to embed and store.

Run the following command:

```zsh
python util/chroma.py ./assets/library
```

The results will be stored using your environment variables in a new Chroma database defined by `CHROMA_COLLECTION_NAME` and `CHROMA_PERSIST_PATH`.

### Running the Application

To run the app locally, use:

```zsh
streamlit run app.py
```

#### Running the App in containers

If you prefer to run the app in containers, use the following command:

```zsh
docker-compose up
```

You should be able to view the app in your browser at the following URL:

```
http://0.0.0.0:8501
```
