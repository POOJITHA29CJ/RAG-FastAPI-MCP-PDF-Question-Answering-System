import os
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from agents import Agent, Runner
from agents.mcp import MCPServerStdio
import shutil
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(
    title="RAG API with FastAPI and MCP",
    description="An API to upload PDF documents and ask questions about them.",
    version="1.0.0",
)
DOCS_PATH = r"C:\Users\HP\OneDrive\Desktop\mcpl\documents"
server: MCPServerStdio | None
agent: Agent | None
last_uploaded_file: str | None = None


@app.on_event("startup")
async def startup_event():
    """
    This function runs when the FastAPI application starts.
    It initializes the document directory, the MCP server, and the agent.
    """
    global server, agent
    logger.info("Starting MCP server......")
    server = MCPServerStdio(
        params={
            "command": "python",
            "args": [r"C:\Users\HP\OneDrive\Desktop\mcpl\retrieval.py"],
            "env": os.environ,
        },
        client_session_timeout_seconds=30.0,
    )
    try:
        await server.__aenter__()
        logger.info("MCP SERVER STARTED SUCESSFULLY")
    except Exception as e:
        logger.error(f"falied to start mcp server:{e}")
    logger.info("Initializing agent.....")
    agent = Agent(
        name="Assistant",
        model="litellm/gemini/gemini-2.0-flash-lite",
        instructions=(
            "You are an expert system that uses tools to answer questions about PDF documents.\n"
            "\n"
            "Workflow:\n"
            "1. When a document is uploaded, always check first with `is_indexed(name)`.\n"
            "   - If it returns 'false', call `chunks(name)` to index the document.\n"
            "   - If it returns 'true', confirm that the document is already indexed and do NOT re-index.\n"
            "\n"
            "2. When the user asks a question, you must directly call the `rag(name, query)` tool, "
            "where `name` is the last uploaded file, and `query` is the user’s question."
            "\n"
            "3. Never rely on your own knowledge. Only use the tools.\n"
            "4. Always respond with the result from the tool calls, not with promises or explanations."
        ),
        mcp_servers=[server],
    )


@app.on_event("shutdown")
async def shutdown_event():
    """Stop MCP server cleanly"""
    global server
    if server:
        await server.__aexit__(None, None, None)


@app.post("/upload")
async def upload_file(file: UploadFile):
    """
    Upload and index a PDF
    """
    global last_uploaded_file
    filename = file.filename
    logger.info(filename)
    if filename.lower().endswith(".pdf.pdf"):
        filename = filename[:-4]
    elif not filename.lower().endswith(".pdf"):
        filename = f"{filename}.pdf"
    filepath = os.path.join(DOCS_PATH, filename)
    logger.info(f"Uploading file: {filename}")
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)
    last_uploaded_file = filename
    prompt = f"Use the `chunks` tool to index the document '{filename}'"
    res = await Runner.run(agent, prompt)
    return JSONResponse(content={"message": res.final_output})


@app.post("/ask")
async def question(query: str):
    """Ask a question after PDF has been indexed"""
    prompt = (
        f"Use the `rag` tool with document '{last_uploaded_file}' and query '{query}'"
    )
    res = await Runner.run(agent, prompt)
    return JSONResponse(content={"answer": res.final_output})
