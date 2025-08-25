from mcp.server.fastmcp import FastMCP
from pypdf import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()
mcp = FastMCP("knowledge_base")
path = r"C:\Users\HP\OneDrive\Desktop\mcpl\documents"


@mcp.tool()
def chunks(name: str) -> str:
    """
    Processes and indexes the content of a local PDF file specified by its filename.
    This tool must be called to index a new document before you can ask questions about it.

    :param name: The filename of the PDF to be indexed (e.g., 'my_document.pdf').
    :return: A string confirming that the document has been successfully indexed.
    """
    raw_text = get_doc(name)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(raw_text)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key="AIzaSyDyEIgF_PXaowPbyTaowqkEu8Q-um3ZhN8",
    )
    vectordb = Chroma(
        collection_name=name,
        embedding_function=embeddings,
        persist_directory=r"C:\Users\HP\OneDrive\Desktop\mcpl\db",
    )
    vectordb.add_texts(chunks)
    return f"Document '{name}' has been successfully indexed."

@mcp.resource("documents://{name}")
def get_doc(name: str) -> str:
    p = f"{path}/{name}"
    reader = PdfReader(p)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text.strip()


@mcp.tool()
def rag(name:str,query: str) -> str:
    """
    Searches the indexed documents to find information relevant to the user's query.
    Use this to answer questions after a document has been indexed with the 'chunks' tool.
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key="AIzaSyDyEIgF_PXaowPbyTaowqkEu8Q-um3ZhN8",
    )
    vector_db = Chroma(
        collection_name=name,
        embedding_function=embeddings,
        persist_directory=r"C:\Users\HP\OneDrive\Desktop\mcpl\db",
    )
    docs = vector_db.similarity_search(query, k=3)
    formatted = "\n\n".join(
        [f"==Chunk {i + 1}==\n{doc.page_content}" for i, doc in enumerate(docs)]
    )
    return formatted

@mcp.tool()
def is_indexed(name:str)->str:
    """
    Check if the given PDF has already been indexed in the vector database.
    Returns 'true' or 'false'.
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key="AIzaSyDyEIgF_PXaowPbyTaowqkEu8Q-um3ZhN8",
    )
    vector_db = Chroma(
        collection_name=name,
        embedding_function=embeddings,
        persist_directory=r"C:\\Users\\HP\\OneDrive\\Desktop\\mcpl\\db",
    )
    count = vector_db._collection.count()
    return "true" if count > 0 else "false"

if __name__ == "__main__":
    mcp.run(transport="stdio")
