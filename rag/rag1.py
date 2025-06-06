from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from langchain_qdrant import QdrantVectorStore



def load_pdf():
    pdf_path = Path(__file__).parent / "nodejs.pdf"

    loader = PyPDFLoader(file_path=pdf_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    split_docs = text_splitter.split_documents(documents=docs)

    embedder = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=""
    )

    # Inserting data only needed to be executed once
    vector_store = QdrantVectorStore.from_documents(
        documents=[],
        url="http://localhost:6333",
        collection_name="learning_langchain",
        embedding=embedder
    )

    vector_store.add_documents(documents=split_docs)
    print("Injection Done")


def search_data():
    embedder = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=""
        )


    # Getting data needed to be executed whenever data is read from db
    retriver = QdrantVectorStore.from_existing_collection(
        url="http://localhost:6333",
        collection_name="learning_langchain",
        embedding=embedder
    )

    search_result = retriver.similarity_search(
        query="What is FS Module?"
    )

    print("Relevant Chunks", search_result)
    
# load_pdf()
search_data()