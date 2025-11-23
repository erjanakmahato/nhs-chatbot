from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# 1. Keep Pinecone API Key
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable is not set")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Check if data directory exists
data_dir = 'data/'
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Data directory '{data_dir}' does not exist")

extracted_data = load_pdf_file(data=data_dir)
if not extracted_data:
    raise ValueError(f"No PDF files found in '{data_dir}' directory")

filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)

# This uses local embeddings, so no API key is needed for this part
embeddings = download_hugging_face_embeddings()

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)


index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384, # Matches the dimension of the HuggingFace MiniLM model
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)