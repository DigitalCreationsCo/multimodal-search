from mcp.server.fastmcp import FastMCP

# 1. Import the core logic directly from the repository's modules.
# (Note: Adjust these import paths based on the exact internal structure 
# of the 'semantic-multimodal' package).
from semantic_multimodal.search_engine import perform_search
from semantic_multimodal.ingestion_pipeline import process_video_url

# 2. Initialize the MCP server
mcp = FastMCP("SemanticMultimodalSearch")

# 3. Define tools using the @mcp.tool() decorator.
# The docstrings and type hints automatically generate the MCP Tool Schema 
# for the agent to understand how to use them.

@mcp.tool()
async def search_video(query: str, max_results: int = 5) -> str:
    """
    Searches the Qdrant vector database using a semantic multimodal query.
    Returns matched video chunks, timestamps, and metadata.
    """
    # Call the underlying Python module directly
    results = await perform_search(query, top_k=max_results)
    
    # MCP expects string/text returns for standard tool content
    return str(results)

@mcp.tool()
async def ingest_url(url: str) -> str:
    """
    Ingests a video from a URL. Detects scenes, transcribes audio, 
    generates embeddings, and stores them in the local Qdrant database.
    """
    # Call the underlying Python module directly
    status = await process_video_url(url)
    return f"Ingestion complete. Status: {status}"

if __name__ == "__main__":
    # 4. Run the server using standard input/output
    # This takes over stdin/stdout to communicate securely with the host agent.
    mcp.run()