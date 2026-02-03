import os
import urllib.request
from tqdm import tqdm
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def download_file(url, file_path):
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    if os.path.exists(file_path):
        logging.info(f"File already exists: {file_path}")
        return

    logging.info(f"Downloading {url} to {file_path}")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    try:
        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, filename=file_path, reporthook=t.update_to)
        logging.info("Download completed.")
    except Exception as e:
        logging.error(f"Failed to download: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        raise e

def main():
    base_dir = Path(__file__).resolve().parent / "deepdoc_vietocr"
    
    models = [
        # Detection
        ("https://huggingface.co/monkt/paddleocr-onnx/resolve/main/detection/v5/det.onnx", 
         base_dir / "onnx" / "det.onnx"),
        
        # Layout Analysis
        ("https://huggingface.co/monkt/paddleocr-onnx/resolve/main/layout/v1/layout.onnx", 
         base_dir / "onnx" / "layout.onnx"),
        
        # Table Structure
        ("https://huggingface.co/monkt/paddleocr-onnx/resolve/main/tsr/v1/tsr.onnx", 
         base_dir / "onnx" / "tsr.onnx"),
         
        # VietOCR Recognition
        ("https://github.com/p_nhm/vietocr-weights/raw/main/vgg_seq2seq.pth", 
         base_dir / "vietocr" / "weight" / "vgg_seq2seq.pth")
    ]
    
    logging.info("Starting OCR model installation...")
    for url, path in models:
        try:
            download_file(url, str(path))
        except Exception as e:
            logging.error(f"Skipping {path} due to error: {e}")

if __name__ == "__main__":
    main()
