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

import os

import re

from collections import Counter

from copy import deepcopy

from pathlib import Path



try:

    from doclayout_yolo import YOLOv10

except ImportError:

    YOLOv10 = None



import cv2

import numpy as np



from .recognizer import Recognizer

from .operators import nms



def get_project_base_directory():

    return Path(__file__).resolve().parent





class LayoutRecognizer(Recognizer):

    labels = [

        "_background_",

        "Text",

        "Title",

        "Figure",

        "Figure caption",

        "Table",

        "Table caption",

        "Header",

        "Footer",

        "Reference",

        "Equation",

    ]



    def __init__(self, domain):

        # Base init that doesn't load ONNX model automatically

        # Subclasses should handle their own model loading

        self.domain = domain

        self.garbage_layouts = ["footer", "header", "reference"]

        self.client = None

        if os.environ.get("TENSORRT_DLA_SVR"):

            from deepdoc.vision.dla_cli import DLAClient

            self.client = DLAClient(os.environ["TENSORRT_DLA_SVR"])



    def get_layouts_from_model(self, image_list, thr, batch_size):

        if self.client:

            return self.client.predict(image_list)

        raise NotImplementedError("Subclasses must implement get_layouts_from_model")



    def __call__(self, image_list, ocr_res, scale_factor=3, thr=0.2, batch_size=16, drop=True):

        def __is_garbage(b):

            patt = [r"^•+$", "^[0-9]{1,2} / ?[0-9]{1,2}$",

                    r"^[0-9]{1,2} of [0-9]{1,2}$", "^http://[^ ]{12,}",

                    "\\(cid *: *[0-9]+ *\\)"

                    ]

            return any([re.search(p, b["text"]) for p in patt])



        layouts = self.get_layouts_from_model(image_list, thr, batch_size)

        # save_results(image_list, layouts, self.labels, output_dir='output/', threshold=0.7)

        assert len(image_list) == len(ocr_res)

        # Tag layout type

        boxes = []

        assert len(image_list) == len(layouts)

        garbages = {}

        page_layout = []

        for pn, lts in enumerate(layouts):

            bxs = ocr_res[pn]

            lts = [{"type": b["type"],

                    "score": float(b["score"]),

                    "x0": b["bbox"][0] / scale_factor, "x1": b["bbox"][2] / scale_factor,

                    "top": b["bbox"][1] / scale_factor, "bottom": b["bbox"][-1] / scale_factor,

                    "page_number": pn,

                    } for b in lts if float(b["score"]) >= 0.4 or b["type"] not in self.garbage_layouts]

            lts = self.sort_Y_firstly(lts, np.mean(

                [lt["bottom"] - lt["top"] for lt in lts]) / 2)

            lts = self.layouts_cleanup(bxs, lts)

            page_layout.append(lts)



            # Tag layout type, layouts are ready

            def findLayout(ty):

                nonlocal bxs, lts, self

                lts_ = [lt for lt in lts if lt["type"] == ty]

                i = 0

                while i < len(bxs):

                    if bxs[i].get("layout_type"):

                        i += 1

                        continue

                    if __is_garbage(bxs[i]):

                        bxs.pop(i)

                        continue



                    ii = self.find_overlapped_with_threashold(bxs[i], lts_,

                                                              thr=0.4)

                    if ii is None:  # belong to nothing

                        bxs[i]["layout_type"] = ""

                        i += 1

                        continue

                    lts_[ii]["visited"] = True

                    keep_feats = [

                        lts_[

                            ii]["type"] == "footer" and bxs[i]["bottom"] < image_list[pn].size[1] * 0.9 / scale_factor,

                        lts_[

                            ii]["type"] == "header" and bxs[i]["top"] > image_list[pn].size[1] * 0.1 / scale_factor,

                    ]

                    if drop and lts_[

                            ii]["type"] in self.garbage_layouts and not any(keep_feats):

                        if lts_[ii]["type"] not in garbages:

                            garbages[lts_[ii]["type"]] = []

                        garbages[lts_[ii]["type"]].append(bxs[i]["text"])

                        bxs.pop(i)

                        continue



                    bxs[i]["layoutno"] = f"{ty}-{ii}"

                    bxs[i]["layout_type"] = lts_[ii]["type"] if lts_[

                        ii]["type"] != "equation" else "figure"

                    i += 1



            for lt in ["footer", "header", "reference", "figure caption",

                       "table caption", "title", "table", "text", "figure", "equation"]:

                findLayout(lt)



            # add box to figure layouts which has not text box

            for i, lt in enumerate(

                    [lt for lt in lts if lt["type"] in ["figure", "equation"]]):

                if lt.get("visited"):

                    continue

                lt = deepcopy(lt)

                del lt["type"]

                lt["text"] = ""

                lt["layout_type"] = "figure"

                lt["layoutno"] = f"figure-{i}"

                bxs.append(lt)



            boxes.extend(bxs)



        ocr_res = boxes



        garbag_set = set()

        for k in garbages.keys():

            garbages[k] = Counter(garbages[k])

            for g, c in garbages[k].items():

                if c > 1:

                    garbag_set.add(g)



        ocr_res = [b for b in ocr_res if b["text"].strip() not in garbag_set]

        return ocr_res, page_layout



    def forward(self, image_list, thr=0.7, batch_size=16):

        return self.get_layouts_from_model(image_list, thr, batch_size)





class LayoutRecognizerDocLayoutYOLO(LayoutRecognizer):





    def __init__(self, domain):





        # DocLayout-YOLO handles loading via from_pretrained





        self.labels = LayoutRecognizer.labels





        self.domain = domain





        self.garbage_layouts = ["footer", "header", "reference"]





        self.client = None





        





        if YOLOv10 is None:





             raise ImportError("Could not import YOLOv10 from doclayout_yolo. Please run 'pip install doclayout-yolo'.")





             





        # Load YOLOv10 model (DocStructBench) using the official library





        try:





             # Use hf_hub_download explicitly for robustness





             from huggingface_hub import hf_hub_download





             model_path = hf_hub_download(





                repo_id="juliozhao/DocLayout-YOLO-DocStructBench",





                filename="doclayout_yolo_docstructbench_imgsz1024.pt"





             )





             self.model = YOLOv10(model_path)
             
             # OPTIMIZATION: Try to use ONNX model for CPU acceleration
             try:
                 onnx_path = model_path.replace(".pt", ".onnx")
                 import logging
                 
                 # If ONNX doesn't exist, try to export it
                 if not os.path.exists(onnx_path):
                     logging.info(f"⚡ Generating optimized ONNX model for Layout Detection (First run only)...")
                     try:
                        self.model.export(format="onnx", imgsz=1024)
                        logging.info(f"✅ Exported ONNX model to: {onnx_path}")
                     except Exception as e:
                        logging.warning(f"⚠️ Could not export ONNX (using PyTorch fallback): {e}")

                 # Load ONNX model if available
                 if os.path.exists(onnx_path):
                     logging.info(f"🚀 Loading optimized ONNX model: {onnx_path}")
                     # Re-initialize with ONNX
                     self.model = YOLOv10(onnx_path, task='detect') 
             except Exception as e:
                 logging.warning(f"⚠️ ONNX Optimization failed, using standard PyTorch: {e}")
                 self.model = YOLOv10(model_path)





        except Exception as e:





             # Fallback if download fails or other issue, though from_pretrained handles cache





             raise RuntimeError(f"Failed to load DocLayout-YOLO model: {e}")











    def get_layouts_from_model(self, image_list, thr, batch_size):

        # Use batch processing as suggested in the guide

        # image_list is expected to be a list of numpy arrays (cv2 images)

        

        results = self.model.predict(

            image_list, 

            imgsz=1024, 

            conf=thr, 

            verbose=False,

            device="cuda" if np.mod(1,1)==0 and False else "cpu" # Auto-detect device or use CPU for safety if no torch.cuda

        )

        

        layouts = []

        for res in results:

            page_layout = []

            if res.boxes:

                for i in range(len(res.boxes)):

                    box = res.boxes[i]

                    # box.xyxy: [x1, y1, x2, y2]

                    coords = box.xyxy[0].cpu().numpy().tolist()

                    score = float(box.conf[0].item())

                    cls_id = int(box.cls[0].item())

                    label = res.names[cls_id]

                    

                    page_layout.append({

                        "type": label.lower(), # Ensure lowercase for compatibility

                        "bbox": coords,

                        "score": score

                    })

            layouts.append(page_layout)

            

        return layouts

    

    def forward(self, image_list, thr=0.7, batch_size=16):

        return self.get_layouts_from_model(image_list, thr, batch_size)



        