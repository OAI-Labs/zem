import os
import sys
import time
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from xfmr_zem.server import ZemServer
from loguru import logger

# Setup logging
logger.remove()
logger.add(sys.stderr, level="INFO")

server = ZemServer("profiler")

@server.tool()
def profile_data(
    data: Any,
    text_column: str = "text",
    include_stats: bool = True
) -> Any:
    """
    Generate a profile report for the input data.
    Calculates metrics like null rates, character counts, and unique values.
    """
    items = server.get_data(data)
    if not items:
        return {"error": "No data to profile"}

    df = pd.DataFrame(items)
    row_count = len(df)
    
    report = {
        "summary": {
            "total_rows": row_count,
            "columns": list(df.columns),
            "memory_usage_kb": round(df.memory_usage(deep=True).sum() / 1024, 2)
        },
        "metrics": {}
    }

    if text_column in df.columns:
        texts = df[text_column].astype(str)
        char_counts = texts.str.len()
        word_counts = texts.str.split().str.len()
        
        report["metrics"][text_column] = {
            "avg_chars": round(char_counts.mean(), 2) if row_count > 0 else 0,
            "max_chars": int(char_counts.max()) if row_count > 0 else 0,
            "avg_words": round(word_counts.mean(), 2) if row_count > 0 else 0,
            "null_count": int(df[text_column].isna().sum()),
            "unique_ratio": round(df[text_column].nunique() / row_count, 4) if row_count > 0 else 0
        }

    # Add more general stats if requested
    if include_stats:
        for col in df.columns:
            if col == text_column: continue
            if pd.api.types.is_numeric_dtype(df[col]):
                report["metrics"][col] = {
                    "mean": round(float(df[col].mean()), 4),
                    "std": round(float(df[col].std()), 4),
                    "min": float(df[col].min()),
                    "max": float(df[col].max())
                }
            else:
                report["metrics"][col] = {
                    "unique_values": int(df[col].nunique()),
                    "top_value": str(df[col].mode().iloc[0]) if not df[col].empty else None
                }

    logger.info(f"Profiler: Generated report for {row_count} rows")
    return report

if __name__ == "__main__":
    server.run()
