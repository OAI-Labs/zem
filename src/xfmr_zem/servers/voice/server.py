import os
import pandas as pd
from xfmr_zem.server import ZemServer
from xfmr_zem.servers.voice.engines import VoiceEngineFactory
from loguru import logger

# Initialize ZemServer for Voice
mcp = ZemServer("voice")

@mcp.tool()
async def transcribe(
    file_path: str, 
    engine: str = "whisper", 
    model_size: str = "base"
) -> pd.DataFrame:
    """
    Transcribes an audio file using the specified voice engine.
    
    Args:
        file_path: Path to the audio file (wav, mp3, m4a, etc.).
        engine: The voice engine to use (currently only "whisper"). Defaults to "whisper".
        model_size: Whisper model size ("tiny", "base", "small", "medium", "large"). Defaults to "base".
    """
    logger.info(f"Voice Transcription: {file_path} using {engine} ({model_size})")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        # Get engine from factory
        voice_engine = VoiceEngineFactory.get_engine(engine, model_size=model_size)
        
        # Transcribe
        result = voice_engine.transcribe(file_path)
        
        # Format as DataFrame
        df = pd.DataFrame([{
            "text": result["text"].strip(),
            "language": result["language"],
            "engine": result["engine"],
            "metadata": result["metadata"]
        }])
        
        logger.info(f"Successfully transcribed {file_path}")
        return df.to_dict(orient="records")
        
    except Exception as e:
        logger.error(f"Voice Error with {engine}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Transcription failed: {str(e)}")

if __name__ == "__main__":
    mcp.run()
