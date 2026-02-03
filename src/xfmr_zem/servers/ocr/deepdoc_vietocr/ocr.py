#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import logging
import copy
import time
import os
import sys
from pathlib import Path

# External libs
import numpy as np
import cv2
import onnxruntime as ort
import torch
from PIL import Image

# Internal modules
from . import operators
from .postprocess import build_post_process
from .vietocr.tool.config import Cfg

# Handle VietOCR import
try:
    from .vietocr.tool.predictor import Predictor
except ImportError:
    from .vietocr.tool.translate import Predictor

def get_project_base_directory():
    return Path(__file__).resolve().parent

loaded_models = {}

def load_model(model_dir, nm, device_id: int | None = None):
    model_file_path = os.path.join(model_dir, nm + ".onnx")
    model_cached_tag = model_file_path + str(device_id) if device_id is not None else model_file_path

    global loaded_models
    loaded_model = loaded_models.get(model_cached_tag)
    if loaded_model:
        logging.info(f"load_model {model_file_path} reuses cached model")
        return loaded_model

    if not os.path.exists(model_file_path):
        raise ValueError("not find model file path {}".format(
            model_file_path))

    def cuda_is_available():
        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.device_count() > (device_id if device_id else 0):
                return True
        except Exception:
            return False
        return False

    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 2

    run_options = ort.RunOptions()
    
    providers = ['CPUExecutionProvider']
    if cuda_is_available():
        cuda_provider_options = {
            "device_id": device_id if device_id else 0,
            "gpu_mem_limit": 512 * 1024 * 1024,
            "arena_extend_strategy": "kNextPowerOfTwo",
        }
        providers = [('CUDAExecutionProvider', cuda_provider_options)]
        run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:" + str(device_id if device_id else 0))
        logging.info(f"load_model {model_file_path} uses GPU")
    else:
        run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "cpu")
        logging.info(f"load_model {model_file_path} uses CPU")
        
    sess = ort.InferenceSession(model_file_path, options=options, providers=providers)
    
    loaded_model = (sess, run_options)
    loaded_models[model_cached_tag] = loaded_model
    return loaded_model

def download_file(url, file_path):
    import urllib.request
    from tqdm import tqdm
    
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    logging.info(f"Downloading {url} to {file_path}")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    try:
        with DownloadProgressBar(unit='B', unit_scale=True,
                                miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, filename=file_path, reporthook=t.update_to)
        logging.info("Download completed.")
    except Exception as e:
        logging.error(f"Failed to download model: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        raise e

def create_operators(op_param_list, global_config=None):
    ops = []
    for operator in op_param_list:
        assert isinstance(operator, dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if global_config is not None:
            param.update(global_config)
        op = getattr(operators, op_name)(**param)
        ops.append(op)
    return ops

class TextRecognizer:
    def __init__(self, model_dir=None, device_id: int | None = None):
        # VietOCR Configuration
        config = Cfg.load_config_from_name('vgg-seq2seq')
        weights_path = os.path.join(get_project_base_directory(), "vietocr", "weight", "vgg_seq2seq.pth")
        config['weights'] = weights_path
        config['cnn']['pretrained'] = True
        config['device'] = 'cpu' # Optimized with Quantization in translate.py
        
        logging.info("Initializing VietOCR (Text Recognition)...")
        self.detector = Predictor(config)

    def __call__(self, img_list):
        results = []
        for img in img_list:
            # Ensure PIL Image
            if isinstance(img, np.ndarray):
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            try:
                text = self.detector.predict(img)
                results.append((text, 1.0))
            except Exception as e:
                logging.warning(f"VietOCR prediction failed: {e}")
                results.append(("", 0.0))
        return results, 0.0


class TextRecognizerPaddleOCR:
    """
    PaddleOCR Recognition Model (Python API)
    Thay thế VietOCR với model accuracy cao hơn

    Uses PaddleOCR library directly instead of ONNX for simpler setup
    """
    def __init__(self, model_dir=None, device_id: int | None = None):
        from paddleocr import PaddleOCR

        # Initialize PaddleOCR với English language (supports Latin alphabet + Vietnamese)
        logging.info("Initializing PaddleOCR (English/Latin Recognition)...")

        # Initialize with only lang parameter (most compatible)
        self.ocr = PaddleOCR(lang='en')

    def __call__(self, img_list):
        """Recognize text from image crops"""
        results = []

        for img in img_list:
            try:
                # Convert to numpy if PIL Image
                if isinstance(img, Image.Image):
                    img = np.array(img)

                # Convert to RGB if needed
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

                # Use PaddleOCR's text recognizer directly (skip detection)
                # Format image for recognition (convert to correct format)
                if img.shape[0] < 32:  # Too small height
                    img = cv2.resize(img, (int(img.shape[1] * 32 / img.shape[0]), 32))

                # Call recognizer directly
                rec_result = self.ocr.rec(img)

                # Extract text and confidence from recognizer output
                if rec_result and len(rec_result) > 0:
                    # rec_result format: [(text, confidence), ...]
                    if isinstance(rec_result[0], (tuple, list)) and len(rec_result[0]) >= 2:
                        text = str(rec_result[0][0])
                        confidence = float(rec_result[0][1])
                    else:
                        text = str(rec_result[0])
                        confidence = 1.0
                else:
                    text = ""
                    confidence = 0.0

                results.append((text, float(confidence)))

            except Exception as e:
                logging.warning(f"PaddleOCR recognition failed: {e}")
                results.append(("", 0.0))

        return results, 0.0


class TextDetector:
    def __init__(self, model_dir, device_id: int | None = None):
        # ONNX Model Path
        self.model_path = os.path.join(model_dir, "det.onnx")

        # Download if not exists
        if not os.path.exists(self.model_path):
            logging.info("ONNX detection model not found. Downloading from monkt/paddleocr-onnx...")
            url = "https://huggingface.co/monkt/paddleocr-onnx/resolve/main/detection/v5/det.onnx"
            download_file(url, self.model_path)

        # Initialize ONNX Session
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 2
        
        providers = ['CPUExecutionProvider']
        # Enable CUDA if requested/available
        if device_id is not None and torch.cuda.is_available():
             pass # Stick to CPU as requested, or add 'CUDAExecutionProvider' if needed.

        logging.info(f"Loading ONNX Text Detector from {self.model_path}")
        self.session = ort.InferenceSession(self.model_path, sess_options, providers=providers)
        self.input_tensor_name = self.session.get_inputs()[0].name

        # Preprocess Configuration (Standard DBNet / PP-OCR)
        pre_process_list = [{"DetResizeForTest": {"limit_side_len": 960, "limit_type": "max"}},
                          {"NormalizeImage": {"std": [0.229, 0.224, 0.225], "mean": [0.485, 0.456, 0.406], "scale": "1./255.", "order": "hwc"}},
                          {"ToCHWImage": None},
                          {"KeepKeys": {"keep_keys": ["image", "shape"]}}]
        self.preprocess_op = create_operators(pre_process_list)

        # Postprocess Configuration
        postprocess_params = {
            "name": "DBPostProcess", 
            "thresh": 0.3, 
            "box_thresh": 0.6, 
            "max_candidates": 1000,
            "unclip_ratio": 1.5, 
            "use_dilation": False, 
            "score_mode": "fast", 
            "box_type": "quad"
        }
        self.postprocess_op = build_post_process(postprocess_params)

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if isinstance(box, list):
                box = np.array(box)
            box = self.order_points_clockwise(box)
            box[:, 0] = np.clip(box[:, 0], 0, img_width - 1)
            box[:, 1] = np.clip(box[:, 1], 0, img_height - 1)
            
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        return np.array(dt_boxes_new)

    def transform(self, data, ops=None):
        if ops is None:
            ops = []
        for op in ops:
            data = op(data)
            if data is None:
                return None
        return data

    def __call__(self, img):
        ori_im = img.copy()
        data = {'image': img}

        st = time.time()
        
        # Preprocess
        data = self.transform(data, self.preprocess_op)
        img, shape_list = data
        
        if img is None:
            return None, 0
            
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)

        # Inference
        input_dict = {self.input_tensor_name: img}
        outputs = self.session.run(None, input_dict)

        # Postprocess
        post_result = self.postprocess_op({"maps": outputs[0]}, shape_list)
        dt_boxes = post_result[0]['points']
        
        if dt_boxes is None or len(dt_boxes) == 0:
            return None, time.time() - st
            
        dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)

        return dt_boxes, time.time() - st


class OCR:
    def __init__(self, model_dir=None, use_paddleocr_rec=True):
        """
        Initialize OCR pipeline

        Args:
            model_dir: Directory for model files
            use_paddleocr_rec: If True, use PaddleOCR recognition (recommended)
                              If False, use VietOCR recognition (legacy)
        """
        if not model_dir:
            model_dir = os.path.join(get_project_base_directory(), "onnx")

        os.makedirs(model_dir, exist_ok=True)

        # Detect parallel devices (optional, mostly for GPU)
        try:
            import torch.cuda
            parallel_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:
            parallel_devices = 0

        # Log which recognizer is being used
        recognizer_name = "PaddleOCR" if use_paddleocr_rec else "VietOCR"
        logging.info(f"Initializing OCR with {recognizer_name} recognizer")

        # Initialize Detector and Recognizer
        if parallel_devices > 0:
            self.text_detector = []
            self.text_recognizer = []
            for device_id in range(parallel_devices):
                self.text_detector.append(TextDetector(model_dir, device_id))

                # Choose recognizer based on flag
                if use_paddleocr_rec:
                    self.text_recognizer.append(TextRecognizerPaddleOCR(model_dir, device_id))
                else:
                    self.text_recognizer.append(TextRecognizer(model_dir, device_id))
        else:
            self.text_detector = [TextDetector(model_dir, 0)]

            # Choose recognizer based on flag
            if use_paddleocr_rec:
                self.text_recognizer = [TextRecognizerPaddleOCR(model_dir, 0)]
            else:
                self.text_recognizer = [TextRecognizer(model_dir, 0)]

        self.drop_score = 0.5

    def get_rotate_crop_image(self, img, points):
        assert len(points) == 4, "shape of points must be 4*2"
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def sorted_boxes(self, dt_boxes):
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        for i in range(num_boxes - 1):
            for j in range(i, -1, -1):
                if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                        (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                    tmp = _boxes[j]
                    _boxes[j] = _boxes[j + 1]
                    _boxes[j + 1] = tmp
                else:
                    break
        return _boxes

    def detect(self, img, device_id: int | None = None):
        if device_id is None: device_id = 0
        
        # Handle list if accessed by index
        if isinstance(self.text_detector, list):
             detector = self.text_detector[device_id if device_id < len(self.text_detector) else 0]
        else:
             detector = self.text_detector

        dt_boxes, elapse = detector(img)

        if dt_boxes is None or len(dt_boxes) == 0:
            return []

        return zip(self.sorted_boxes(dt_boxes), [("", 0) for _ in range(len(dt_boxes))])

    def recognize(self, ori_im, box, device_id: int | None = None):
        if device_id is None: device_id = 0
        
        if isinstance(self.text_recognizer, list):
             recognizer = self.text_recognizer[device_id if device_id < len(self.text_recognizer) else 0]
        else:
             recognizer = self.text_recognizer

        img_crop = self.get_rotate_crop_image(ori_im, box)
        rec_res, elapse = recognizer([img_crop])
        text, score = rec_res[0]
        if score < self.drop_score:
            return ""
        return text

    def recognize_batch(self, img_list, device_id: int | None = None):
        if device_id is None: device_id = 0
        if isinstance(self.text_recognizer, list):
             recognizer = self.text_recognizer[device_id if device_id < len(self.text_recognizer) else 0]
        else:
             recognizer = self.text_recognizer

        rec_res, elapse = recognizer(img_list)
        texts = []
        for i in range(len(rec_res)):
            text, score = rec_res[i]
            if score < self.drop_score:
                text = ""
            texts.append(text)
        return texts

    def __call__(self, img, device_id=0, cls=True):
        if device_id is None: device_id = 0
        
        # Access detector/recognizer safely
        if isinstance(self.text_detector, list):
             detector = self.text_detector[device_id if device_id < len(self.text_detector) else 0]
             recognizer = self.text_recognizer[device_id if device_id < len(self.text_recognizer) else 0]
        else:
             detector = self.text_detector
             recognizer = self.text_recognizer

        # Detection
        start = time.time()
        ori_im = img.copy()
        dt_boxes, det_time = detector(img)

        if dt_boxes is None or len(dt_boxes) == 0:
            return []

        # Crop
        img_crop_list = []
        dt_boxes = self.sorted_boxes(dt_boxes)
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)

        # Recognition
        rec_res, rec_time = recognizer(img_crop_list)

        # Filter
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)

        return list(zip([a.tolist() for a in filter_boxes], filter_rec_res))
