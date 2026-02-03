import os
import torch
import numpy as np
import cv2
from ..model.vocab import Vocab
from ..model.transformerocr import VietOCR
import math
from PIL import Image


def translate(img, model, max_seq_length=128, sos_token=1, eos_token=2):
    """data: BxCxHxW"""
    model.eval()
    device = img.device

    with torch.no_grad():
        src = model.cnn(img)
        memory = model.transformer.forward_encoder(src)

        translated_sentence = [[sos_token] * len(img)]
        max_length = 0

        while max_length <= max_seq_length and not all(np.any(np.asarray(translated_sentence).T == eos_token, axis=1)):
            tgt_inp = torch.LongTensor(translated_sentence).to(device)
            output, memory = model.transformer.forward_decoder(tgt_inp, memory)
            output = output.to('cpu')

            values, indices = torch.topk(output, 1)
            indices = indices[:, -1, 0]
            indices = indices.tolist()

            translated_sentence.append(indices)
            max_length += 1

            del output

        translated_sentence = np.asarray(translated_sentence).T

    return translated_sentence


def build_model(config):
    vocab = Vocab(config['vocab'])
    device = config['device']

    model = VietOCR(len(vocab),
            config['backbone'],
            config['cnn'],
            config['transformer'],
            config['seq_modeling'])

    model = model.to(device)

    if 'weights' in config and config['weights']:
        weights = config['weights']
        if weights.startswith('http'):
            # Logic for downloading could go here, but we assume local path for now
            pass

        if os.path.exists(weights):
            model.load_state_dict(torch.load(weights, map_location=device))
            model.eval()
        else:
            import logging
            logging.warning(f"Weight file not found: {weights}")

    return model, vocab

def resize(w, h, expected_height, image_min_width, image_max_width):
    new_w = int(expected_height * float(w) / float(h))
    round_to = 10
    new_w = math.ceil(new_w/round_to)*round_to
    new_w = max(new_w, image_min_width)
    new_w = min(new_w, image_max_width)

    return new_w, expected_height

def process_image(image, image_height, image_min_width, image_max_width):
    img = image.convert('RGB')

    w, h = img.size
    new_w, image_height = resize(w, h, image_height, image_min_width, image_max_width)

    img = img.resize((new_w, image_height), Image.LANCZOS)

    img = np.asarray(img).transpose(2,0, 1)
    img = img/255
    return img


def process_input(image, image_height, image_min_width, image_max_width):
    img = process_image(image, image_height, image_min_width, image_max_width)
    img = img[np.newaxis, ...]
    img = torch.FloatTensor(img)
    return img


class Predictor:
    def __init__(self, config):
        self.model, self.vocab = build_model(config)
        self.config = config

    def predict(self, img):
        img_input = process_input(img, self.config['dataset']['image_height'],
                                  self.config['dataset']['image_min_width'],
                                  self.config['dataset']['image_max_width'])
        img_input = img_input.to(self.config['device'])
        s = translate(img_input, self.model)
        s = s[0].tolist()
        s = self.vocab.decode(s)
        return s
