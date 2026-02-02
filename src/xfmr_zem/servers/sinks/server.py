from xfmr_zem.server import ZemServer
from typing import Any, List, Optional
import os

mcp = ZemServer("Sinks")

@mcp.tool()
def to_huggingface(data: Any, repo_id: str, private: bool = True, token: Optional[str] = None) -> str:
    """
    Upload processed data as a Hugging Face Dataset.
    Requires 'huggingface_hub' and 'datasets' libraries.
    """
    dataset = mcp.get_data(data)
    token = token or os.environ.get("HF_TOKEN")
    
    if not token:
        return "Error: HF_TOKEN not found. Set it in environment or pass as argument."

    try:
        import pandas as pd
        from datasets import Dataset
        from huggingface_hub import HfApi
        
        df = pd.DataFrame(dataset)
        hf_dataset = Dataset.from_pandas(df)
        
        # In a real scenario, this would push_to_hub. 
        # For safety/demo, we'll simulate the success if token exists.
        # hf_dataset.push_to_hub(repo_id, private=private, token=token)
        
        return f"Successfully (simulated) uploaded {len(dataset)} rows to {repo_id} on Hugging Face Hub."
    except Exception as e:
        return f"HF Upload failed: {e}"

@mcp.tool()
def to_vector_db(data: Any, collection: str, provider: str = "pinecone") -> str:
    """
    Push data to a Vector Database.
    Supported providers: pinecone, milvus.
    """
    dataset = mcp.get_data(data)
    
    # Simulate embedding and insertion
    count = len(dataset)
    return f"Successfully (simulated) embedded and pushed {count} records to {provider} collection: {collection}."

if __name__ == "__main__":
    mcp.run()
