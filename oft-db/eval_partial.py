#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import csv
import hashlib
import logging
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers

# Silence transformers/torch warnings that write to stderr mid-loop and break tqdm bars.
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
transformers.logging.set_verbosity_error()
from packaging import version
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer, PretrainedConfig, ViTFeatureExtractor, ViTModel

import lpips
import json
from PIL import Image
import requests
from transformers import AutoProcessor, AutoTokenizer, CLIPModel
import torchvision.transforms.functional as TF
from torch.nn.functional import cosine_similarity
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, ToPILImage


class progress:
    def __init__(self, iterable, desc=None, **kwargs):
        import time as _time
        self._iterable = iterable
        self._desc = desc
        self._total = len(iterable) if hasattr(iterable, "__len__") else None
        self._time = _time

    def set_postfix(self, **kw):
        pass

    def __iter__(self):
        label = f"[{self._desc}] " if self._desc else ""
        last = 0.0
        total = self._total
        for i, item in enumerate(self._iterable):
            now = self._time.monotonic()
            if now - last >= 5.0 or i == 0:
                if total:
                    print(f"{label}{i}/{total} ({100*i/total:.0f}%)", flush=True)
                else:
                    print(f"{label}{i}", flush=True)
                last = now
            yield item
        if total:
            print(f"{label}{total}/{total} (100%)", flush=True)


subject_names = [
    "backpack", "backpack_dog", "bear_plushie", "berry_bowl", "can",
    "candle", "cat", "cat2", "clock", "colorful_sneaker",
    "dog", "dog2", "dog3", "dog5", "dog6",
    "dog7", "dog8", "duck_toy", "fancy_boot", "grey_sloth_plushie",
    "monster_toy", "pink_sunglasses", "poop_emoji", "rc_car", "red_cartoon",
    "robot_toy", "shiny_sneaker", "teapot", "vase", "wolf_plushie"
]

class_tokens = [
    "backpack", "backpack", "stuffed animal", "bowl", "can",
    "candle", "cat", "cat", "clock", "sneaker",
    "dog", "dog", "dog", "dog", "dog",
    "dog", "dog", "toy", "boot", "stuffed animal",
    "toy", "glasses", "toy", "toy", "cartoon",
    "toy", "sneaker", "teapot", "vase", "stuffed animal",
]

class_token_by_subject = dict(zip(subject_names, class_tokens))

object_prompt_subjects = {
    "backpack", "backpack_dog", "bear_plushie", "berry_bowl", "can",
    "candle", "clock", "colorful_sneaker", "duck_toy", "fancy_boot",
    "grey_sloth_plushie", "monster_toy", "pink_sunglasses", "poop_emoji",
    "rc_car", "red_cartoon", "robot_toy", "shiny_sneaker", "teapot",
    "vase", "wolf_plushie",
}

object_prompt_templates = [
    "a {unique_token} {class_token} in the jungle",
    "a {unique_token} {class_token} in the snow",
    "a {unique_token} {class_token} on the beach",
    "a {unique_token} {class_token} on a cobblestone street",
    "a {unique_token} {class_token} on top of pink fabric",
    "a {unique_token} {class_token} on top of a wooden floor",
    "a {unique_token} {class_token} with a city in the background",
    "a {unique_token} {class_token} with a mountain in the background",
    "a {unique_token} {class_token} with a blue house in the background",
    "a {unique_token} {class_token} on top of a purple rug in a forest",
    "a {unique_token} {class_token} with a wheat field in the background",
    "a {unique_token} {class_token} with a tree and autumn leaves in the background",
    "a {unique_token} {class_token} with the Eiffel Tower in the background",
    "a {unique_token} {class_token} floating on top of water",
    "a {unique_token} {class_token} floating in an ocean of milk",
    "a {unique_token} {class_token} on top of green grass with sunflowers around it",
    "a {unique_token} {class_token} on top of a mirror",
    "a {unique_token} {class_token} on top of the sidewalk in a crowded street",
    "a {unique_token} {class_token} on top of a dirt road",
    "a {unique_token} {class_token} on top of a white rug",
    "a red {unique_token} {class_token}",
    "a purple {unique_token} {class_token}",
    "a shiny {unique_token} {class_token}",
    "a wet {unique_token} {class_token}",
    "a cube shaped {unique_token} {class_token}",
]

live_prompt_templates = [
    "a {unique_token} {class_token} in the jungle",
    "a {unique_token} {class_token} in the snow",
    "a {unique_token} {class_token} on the beach",
    "a {unique_token} {class_token} on a cobblestone street",
    "a {unique_token} {class_token} on top of pink fabric",
    "a {unique_token} {class_token} on top of a wooden floor",
    "a {unique_token} {class_token} with a city in the background",
    "a {unique_token} {class_token} with a mountain in the background",
    "a {unique_token} {class_token} with a blue house in the background",
    "a {unique_token} {class_token} on top of a purple rug in a forest",
    "a {unique_token} {class_token} wearing a red hat",
    "a {unique_token} {class_token} wearing a santa hat",
    "a {unique_token} {class_token} wearing a rainbow scarf",
    "a {unique_token} {class_token} wearing a black top hat and a monocle",
    "a {unique_token} {class_token} in a chef outfit",
    "a {unique_token} {class_token} in a firefighter outfit",
    "a {unique_token} {class_token} in a police outfit",
    "a {unique_token} {class_token} wearing pink glasses",
    "a {unique_token} {class_token} wearing a yellow shirt",
    "a {unique_token} {class_token} in a purple wizard outfit",
    "a red {unique_token} {class_token}",
    "a purple {unique_token} {class_token}",
    "a shiny {unique_token} {class_token}",
    "a wet {unique_token} {class_token}",
    "a cube shaped {unique_token} {class_token}",
]


def infer_prompt_from_data_dir(data_dir, unique_token="qwe"):
    try:
        subject, prompt_idx = data_dir.rsplit("-", 1)
        prompt_idx = int(prompt_idx)
    except ValueError:
        return None

    class_token = class_token_by_subject.get(subject)
    if class_token is None or prompt_idx < 0 or prompt_idx >= len(object_prompt_templates):
        return None

    templates = object_prompt_templates if subject in object_prompt_subjects else live_prompt_templates
    return templates[prompt_idx].format(unique_token=unique_token, class_token=class_token)


def list_image_files(data_dir, suffixes, strict=False):
    if not os.path.isdir(data_dir):
        if strict:
            raise FileNotFoundError(f"Missing image directory: {data_dir}")
        return []

    return sorted(
        os.path.join(data_dir, filename)
        for filename in os.listdir(data_dir)
        if filename.lower().endswith(suffixes)
    )


def mean_or_raise(values, criterion):
    if not values:
        raise ValueError(f"No valid images were found for {criterion}; check image_dir, epoch, and metadata.")
    return torch.tensor(values).mean().item()



def parse_run_name(data_dir_name):
    try:
        subject, prompt_idx = data_dir_name.rsplit("-", 1)
        return subject, int(prompt_idx)
    except ValueError:
        return None, None


def image_is_empty(image):
    extrema = image.getextrema()
    if not extrema:
        return True
    if isinstance(extrema[0], tuple):
        return all(min_val == max_val == 0 for min_val, max_val in extrema)
    min_val, max_val = extrema
    return min_val == max_val == 0


def mean_list(values):
    if not values:
        return None
    return float(torch.tensor(values).mean().item())


def parse_epoch_selection(value):
    if value is None or value == "" or value == "all":
        return None

    epochs = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            start = int(start)
            end = int(end)
            if end < start:
                raise ValueError(f"Invalid epoch range: {item}")
            epochs.update(range(start, end + 1))
        else:
            epochs.add(int(item))
    return sorted(epochs)


def list_available_epochs(run_dir):
    if not os.path.isdir(run_dir):
        return []

    epochs = []
    for name in os.listdir(run_dir):
        epoch_dir = os.path.join(run_dir, name)
        if name.isdigit() and os.path.isdir(epoch_dir) and list_image_files(epoch_dir, (".png",)):
            epochs.append(int(name))
    return sorted(epochs)


def prompt_records(image_dir, metadata_path="metadata.json", unique_token="qwe"):
    records = []
    image_dir = os.path.normpath(image_dir)

    single_name = os.path.basename(image_dir)
    single_subject, _ = parse_run_name(single_name)
    if os.path.isdir(image_dir) and single_subject in class_token_by_subject:
        prompt = None
        if metadata_path and os.path.isfile(metadata_path):
            with open(metadata_path, "r") as json_data:
                metadata_dict = json.load(json_data)
            for value in metadata_dict.values():
                if value.get("data_dir") == single_name:
                    prompt = (
                        value.get("validation_prompt")
                        or value.get("prompt")
                        or value.get("instance_prompt")
                    )
                    break
        prompt = prompt or infer_prompt_from_data_dir(single_name, unique_token)
        if prompt:
            return [{"data_dir": single_name, "subject": single_subject, "prompt": prompt, "run_dir": image_dir}]

    if metadata_path and os.path.isfile(metadata_path):
        with open(metadata_path, "r") as json_data:
            metadata_dict = json.load(json_data)
        for value in metadata_dict.values():
            data_dir_name = value["data_dir"]
            subject, _ = parse_run_name(data_dir_name)
            prompt = (
                value.get("validation_prompt")
                or value.get("prompt")
                or value.get("instance_prompt")
                or infer_prompt_from_data_dir(data_dir_name, unique_token)
            )
            if subject and prompt:
                records.append({
                    "data_dir": data_dir_name,
                    "subject": subject,
                    "prompt": prompt,
                    "run_dir": os.path.join(image_dir, data_dir_name),
                })
    else:
        if metadata_path:
            print(f"metadata file not found: {metadata_path}; inferring prompts from output folder names")
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Missing image directory: {image_dir}")
        for data_dir_name in sorted(os.listdir(image_dir)):
            run_dir = os.path.join(image_dir, data_dir_name)
            if not os.path.isdir(run_dir):
                continue
            subject, _ = parse_run_name(data_dir_name)
            prompt = infer_prompt_from_data_dir(data_dir_name, unique_token)
            if subject and prompt:
                records.append({"data_dir": data_dir_name, "subject": subject, "prompt": prompt, "run_dir": run_dir})
    return records


def generated_run_dirs(image_dir, subject):
    image_dir = os.path.normpath(image_dir)
    basename = os.path.basename(image_dir)
    if os.path.isdir(image_dir) and basename.startswith(subject + "-"):
        return [image_dir]
    if not os.path.isdir(image_dir):
        return []

    return [
        os.path.join(image_dir, subfolder)
        for subfolder in sorted(os.listdir(image_dir))
        if subfolder.startswith(subject + "-") and os.path.isdir(os.path.join(image_dir, subfolder))
    ]

def load_rgb_image(image_path):
    return Image.open(image_path).convert("RGB")


def to_device(inputs, device):
    return {key: value.to(device) for key, value in inputs.items()}


def clip_text_features(context, prompt):
    cache = context["text_features"]
    if prompt not in cache:
        inputs = context["tokenizer"]([prompt], padding=True, return_tensors="pt")
        inputs = to_device(inputs, context["device"])
        with torch.no_grad():
            cache[prompt] = context["model"].get_text_features(**inputs)
    return cache[prompt]


def clip_image_feature(context, image_path):
    image = load_rgb_image(image_path)
    if image_is_empty(image):
        return None
    inputs = context["processor"](images=image, return_tensors="pt")
    inputs = to_device(inputs, context["device"])
    with torch.no_grad():
        features = context["model"].get_image_features(**inputs)
    return features / features.norm(p=2, dim=-1, keepdim=True)


def dino_image_feature(context, image_path):
    image = load_rgb_image(image_path)
    if image_is_empty(image):
        return None
    inputs = context["feature_extractor"](images=image, return_tensors="pt")
    inputs = to_device(inputs, context["device"])
    with torch.no_grad():
        outputs = context["model"](**inputs)
    features = outputs.last_hidden_state[:, 0, :]
    return features / features.norm(p=2, dim=-1, keepdim=True)


def reference_features(context, subject, feature_fn):
    if subject not in context["reference_features"]:
        subject_dir = os.path.join(context["reference_dir"], subject)
        features = []
        for image_path in list_image_files(subject_dir, (".jpg", ".jpeg", ".png")):
            feature = feature_fn(context, image_path)
            if feature is not None:
                features.append(feature)
        if not features:
            context["reference_features"][subject] = None
        else:
            context["reference_features"][subject] = torch.cat(features, dim=0)
    return context["reference_features"][subject]


def lpips_tensor(context, image_path):
    image = load_rgb_image(image_path)
    if image_is_empty(image):
        return None
    return context["transform"](image).unsqueeze(0).to(context["device"])


def reference_tensors(context, subject):
    if subject not in context["reference_tensors"]:
        subject_dir = os.path.join(context["reference_dir"], subject)
        tensors = []
        for image_path in list_image_files(subject_dir, (".jpg", ".jpeg", ".png")):
            tensor = lpips_tensor(context, image_path)
            if tensor is not None:
                tensors.append(tensor)
        context["reference_tensors"][subject] = tensors
    return context["reference_tensors"][subject]


def build_metric_context(metric, reference_dir=None):
    reference_dir = reference_dir or default_reference_dir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {metric} model on {device}...", flush=True)
    if metric in ("clip_text", "clip_image"):
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        context = {"device": device, "model": model, "processor": processor}
        if metric == "clip_text":
            context["tokenizer"] = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            context["text_features"] = {}
        else:
            context["reference_dir"] = reference_dir
            context["reference_features"] = {}
        return context

    if metric == "dino":
        return {
            "device": device,
            "model": ViTModel.from_pretrained("facebook/dino-vits16").to(device),
            "feature_extractor": ViTFeatureExtractor.from_pretrained("facebook/dino-vits16"),
            "reference_dir": reference_dir,
            "reference_features": {},
        }

    if metric == "lpips_image":
        return {
            "device": device,
            "loss_fn": lpips.LPIPS(net="alex").to(device),
            "reference_dir": reference_dir,
            "reference_tensors": {},
            "transform": Compose([
                Resize((512, 512)),
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]),
        }

    raise ValueError(f"Unknown metric: {metric}")


def score_clip_text_epoch(context, prompt, image_files):
    text_features = clip_text_features(context, prompt)
    scores = []
    for image_path in image_files:
        image_features = clip_image_feature(context, image_path)
        if image_features is None:
            continue
        sim = cosine_similarity(image_features, text_features)
        scores.append(sim.item())
    return mean_list(scores), len(scores)


def score_feature_epoch(context, subject, image_files, feature_fn):
    refs = reference_features(context, subject, feature_fn)
    if refs is None:
        return None, 0

    generated_features = []
    for image_path in image_files:
        feature = feature_fn(context, image_path)
        if feature is not None:
            generated_features.append(feature)
    if not generated_features:
        return None, 0

    generated = torch.cat(generated_features, dim=0)
    scores = torch.matmul(generated, refs.t())
    return scores.mean().item(), scores.numel()


def score_lpips_epoch(context, subject, image_files):
    refs = reference_tensors(context, subject)
    if not refs:
        return None, 0

    generated_tensors = []
    for image_path in image_files:
        tensor = lpips_tensor(context, image_path)
        if tensor is not None:
            generated_tensors.append(tensor)
    if not generated_tensors:
        return None, 0

    scores = []
    with torch.no_grad():
        for generated in generated_tensors:
            for reference in refs:
                scores.append(context["loss_fn"](reference, generated).item())
    return mean_list(scores), len(scores)


def score_prompt_epoch(metric, context, record, run_dir, epoch, max_images=0):
    epoch_dir = run_dir if epoch is None else os.path.join(run_dir, str(epoch))
    image_files = list_image_files(epoch_dir, (".png",))
    if max_images and max_images > 0:
        image_files = image_files[:max_images]
    if not image_files:
        return None, 0

    if metric == "clip_text":
        return score_clip_text_epoch(context, record["prompt"], image_files)
    if metric == "clip_image":
        return score_feature_epoch(context, record["subject"], image_files, clip_image_feature)
    if metric == "dino":
        return score_feature_epoch(context, record["subject"], image_files, dino_image_feature)
    if metric == "lpips_image":
        return score_lpips_epoch(context, record["subject"], image_files)
    raise ValueError(f"Unknown metric: {metric}")



class PromptDatasetCLIP(Dataset):
    def __init__(self, image_dir, json_file, tokenizer, processor, epoch=None, partial=True, unique_token="qwe"):
        self.image_dir = image_dir
        self.image_lst = []
        self.prompt_lst = []
        self.skipped_dirs = []
        self.scored_dirs = 0

        records = prompt_records(image_dir, json_file, unique_token)

        for record in records:
            data_dir_name = record["data_dir"]
            prompt = record["prompt"]
            run_dir = record.get("run_dir", os.path.join(self.image_dir, data_dir_name))
            if epoch is not None:
                data_dir = os.path.join(run_dir, str(epoch))
            else:
                data_dir = run_dir

            image_files = list_image_files(data_dir, (".png",), strict=not partial)
            if not image_files:
                self.skipped_dirs.append(data_dir)
                continue

            selected_files = image_files
            self.image_lst.extend(selected_files)
            self.prompt_lst.extend([prompt] * len(selected_files))
            self.scored_dirs += 1

        print('data_list', len(self.image_lst), len(self.prompt_lst), 'scored_dirs', self.scored_dirs, 'skipped_dirs', len(self.skipped_dirs))
        self.tokenizer = tokenizer
        self.processor = processor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.image_lst)

    def __getitem__(self, idx):
        image_path = self.image_lst[idx]
        image = Image.open(image_path)
        prompt = self.prompt_lst[idx]

        extrema = image.getextrema()
        if all(min_val == max_val == 0 for min_val, max_val in extrema):
            return None, None
        else:
            prompt_inputs = self.tokenizer([prompt], padding=True, return_tensors="pt")
            image_inputs = self.processor(images=image, return_tensors="pt")

            return image_inputs, prompt_inputs



class PairwiseImageDatasetCLIP(Dataset):
    def __init__(self, subject, data_dir_A, data_dir_B, processor, epoch):
        self.data_dir_A = data_dir_A
        self.data_dir_B = data_dir_B
        
        self.data_dir_A = os.path.join(self.data_dir_A, subject)
        self.image_files_A = list_image_files(self.data_dir_A, (".jpg", ".jpeg", ".png"))

        self.image_files_B = []
        for run_dir in generated_run_dirs(self.data_dir_B, subject):
            if epoch is not None:
                run_dir = os.path.join(run_dir, str(epoch))
            image_files_b = list_image_files(run_dir, (".png",))
            self.image_files_B.extend(image_files_b)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = processor

    def __len__(self):
        return len(self.image_files_A) * len(self.image_files_B)

    def __getitem__(self, index):
        index_A = index // len(self.image_files_B)
        index_B = index % len(self.image_files_B)
        
        image_A = Image.open(self.image_files_A[index_A]) # .convert("RGB")
        image_B = Image.open(self.image_files_B[index_B]) # .convert("RGB")

        extrema_A = image_A.getextrema()
        extrema_B = image_B.getextrema()
        if all(min_val == max_val == 0 for min_val, max_val in extrema_A) or all(min_val == max_val == 0 for min_val, max_val in extrema_B):
            return None, None
        else:
            inputs_A = self.processor(images=image_A, return_tensors="pt")
            inputs_B = self.processor(images=image_B, return_tensors="pt")

            return inputs_A, inputs_B


class PairwiseImageDatasetDINO(Dataset):
    def __init__(self, subject, data_dir_A, data_dir_B, feature_extractor, epoch):
        self.data_dir_A = data_dir_A
        self.data_dir_B = data_dir_B
        
        self.data_dir_A = os.path.join(self.data_dir_A, subject)
        self.image_files_A = list_image_files(self.data_dir_A, (".jpg", ".jpeg", ".png"))

        self.image_files_B = []
        for run_dir in generated_run_dirs(self.data_dir_B, subject):
            if epoch is not None:
                run_dir = os.path.join(run_dir, str(epoch))
            image_files_b = list_image_files(run_dir, (".png",))
            self.image_files_B.extend(image_files_b)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.image_files_A) * len(self.image_files_B)

    def __getitem__(self, index):
        index_A = index // len(self.image_files_B)
        index_B = index % len(self.image_files_B)
        
        image_A = Image.open(self.image_files_A[index_A]) # .convert("RGB")
        image_B = Image.open(self.image_files_B[index_B]) # .convert("RGB")

        extrema_A = image_A.getextrema()
        extrema_B = image_B.getextrema()
        if all(min_val == max_val == 0 for min_val, max_val in extrema_A) or all(min_val == max_val == 0 for min_val, max_val in extrema_B):
            return None, None
        else:
            inputs_A = self.feature_extractor(images=image_A, return_tensors="pt")
            inputs_B = self.feature_extractor(images=image_B, return_tensors="pt")

            return inputs_A, inputs_B


class SelfPairwiseImageDatasetCLIP(Dataset):
    def __init__(self, subject, data_dir, processor):
        self.data_dir_A = data_dir
        self.data_dir_B = data_dir
        
        self.data_dir_A = os.path.join(self.data_dir_A, subject)
        self.image_files_A = list_image_files(self.data_dir_A, (".jpg", ".jpeg", ".png"))

        self.data_dir_B = os.path.join(self.data_dir_B, subject)
        self.image_files_B = [os.path.join(self.data_dir_B, f) for f in os.listdir(self.data_dir_B) if f.endswith(".jpg")]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = processor

    def __len__(self):
        return len(self.image_files_A) * (len(self.image_files_B) - 1)

    def __getitem__(self, index):
        index_A = index // (len(self.image_files_B) - 1)
        index_B = index % (len(self.image_files_B) - 1)

        # Ensure we don't have the same index for A and B
        if index_B >= index_A:
            index_B += 1

        image_A = Image.open(self.image_files_A[index_A]) # .convert("RGB")
        image_B = Image.open(self.image_files_B[index_B]) # .convert("RGB")
        
        inputs_A = self.processor(images=image_A, return_tensors="pt")
        inputs_B = self.processor(images=image_B, return_tensors="pt")

        return inputs_A, inputs_B


class SelfPairwiseImageDatasetDINO(Dataset):
    def __init__(self, subject, data_dir, feature_extractor):
        self.data_dir_A = data_dir
        self.data_dir_B = data_dir
        
        self.data_dir_A = os.path.join(self.data_dir_A, subject)
        self.image_files_A = list_image_files(self.data_dir_A, (".jpg", ".jpeg", ".png"))

        self.data_dir_B = os.path.join(self.data_dir_B, subject)
        self.image_files_B = [os.path.join(self.data_dir_B, f) for f in os.listdir(self.data_dir_B) if f.endswith(".jpg")]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.image_files_A) * (len(self.image_files_B) - 1)

    def __getitem__(self, index):
        index_A = index // (len(self.image_files_B) - 1)
        index_B = index % (len(self.image_files_B) - 1)

        # Ensure we don't have the same index for A and B
        if index_B >= index_A:
            index_B += 1

        image_A = Image.open(self.image_files_A[index_A]) # .convert("RGB")
        image_B = Image.open(self.image_files_B[index_B]) # .convert("RGB")
        
        inputs_A = self.feature_extractor(images=image_A, return_tensors="pt")
        inputs_B = self.feature_extractor(images=image_B, return_tensors="pt")

        return inputs_A, inputs_B


class SelfPairwiseImageDatasetLPIPS(Dataset):
    def __init__(self, subject, data_dir):
        self.data_dir_A = data_dir
        self.data_dir_B = data_dir
        
        self.data_dir_A = os.path.join(self.data_dir_A, subject)
        self.image_files_A = list_image_files(self.data_dir_A, (".jpg", ".jpeg", ".png"))

        self.data_dir_B = os.path.join(self.data_dir_B, subject)
        self.image_files_B = [os.path.join(self.data_dir_B, f) for f in os.listdir(self.data_dir_B) if f.endswith(".jpg")]

        self.transform = Compose([
            Resize((512, 512)),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.image_files_A) * (len(self.image_files_B) - 1)

    def __getitem__(self, index):
        index_A = index // (len(self.image_files_B) - 1)
        index_B = index % (len(self.image_files_B) - 1)

        # Ensure we don't have the same index for A and B
        if index_B >= index_A:
            index_B += 1
        
        image_A = Image.open(self.image_files_A[index_A]) # .convert("RGB")
        image_B = Image.open(self.image_files_B[index_B]) # .convert("RGB")

        extrema_A = image_A.getextrema()
        extrema_B = image_B.getextrema()
        if all(min_val == max_val == 0 for min_val, max_val in extrema_A) or all(min_val == max_val == 0 for min_val, max_val in extrema_B):
            return None, None
        else:
            if self.transform:
                image_A = self.transform(image_A)
                image_B = self.transform(image_B)

            return image_A, image_B


class PairwiseImageDatasetLPIPS(Dataset):
    def __init__(self, subject, data_dir_A, data_dir_B, epoch):
        self.data_dir_A = data_dir_A
        self.data_dir_B = data_dir_B
        
        self.data_dir_A = os.path.join(self.data_dir_A, subject)
        self.image_files_A = list_image_files(self.data_dir_A, (".jpg", ".jpeg", ".png"))

        self.image_files_B = []
        for run_dir in generated_run_dirs(self.data_dir_B, subject):
            if epoch is not None:
                run_dir = os.path.join(run_dir, str(epoch))
            image_files_b = list_image_files(run_dir, (".png",))
            self.image_files_B.extend(image_files_b)

        self.transform = Compose([
            Resize((512, 512)),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.image_files_A) * len(self.image_files_B)

    def __getitem__(self, index):
        index_A = index // len(self.image_files_B)
        index_B = index % len(self.image_files_B)
        
        image_A = Image.open(self.image_files_A[index_A]) # .convert("RGB")
        image_B = Image.open(self.image_files_B[index_B]) # .convert("RGB")

        extrema_A = image_A.getextrema()
        extrema_B = image_B.getextrema()
        if all(min_val == max_val == 0 for min_val, max_val in extrema_A) or all(min_val == max_val == 0 for min_val, max_val in extrema_B):
            return None, None
        else:
            if self.transform:
                image_A = self.transform(image_A)
                image_B = self.transform(image_B)

            return image_A, image_B


def clip_text(image_dir, epoch=None, metadata_path="metadata.json", partial=True, unique_token="qwe"):
    criterion = 'clip_text'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    # Get the text features
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    # Get the image features
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    dataset = PromptDatasetCLIP(image_dir, metadata_path, tokenizer, processor, epoch, partial, unique_token)
    dataloader = DataLoader(dataset, batch_size=32)

    similarity = []
    for i in progress(range(len(dataset))):
        image_inputs, prompt_inputs = dataset[i]
        if image_inputs is not None and prompt_inputs is not None:
            image_inputs['pixel_values'] = image_inputs['pixel_values'].to(device)
            prompt_inputs['input_ids'] = prompt_inputs['input_ids'].to(device)
            prompt_inputs['attention_mask'] = prompt_inputs['attention_mask'].to(device)
            # print(prompt_inputs)
            image_features = model.get_image_features(**image_inputs)
            text_features = model.get_text_features(**prompt_inputs)

            sim = cosine_similarity(image_features, text_features)

            #image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            #text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            #logit_scale = model.logit_scale.exp()
            #sim = torch.matmul(text_features, image_features.t()) * logit_scale
            similarity.append(sim.item())

    mean_similarity = mean_or_raise(similarity, criterion)
    print(criterion, 'mean_similarity', mean_similarity, 'num_scores', len(similarity))

    return mean_similarity, criterion


def clip_image(image_dir, epoch=None, reference_dir=None):
    reference_dir = reference_dir or default_reference_dir()
    criterion = 'clip_image'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    # Get the image features
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    similarity = []
    for subject in subject_names:
        dataset = PairwiseImageDatasetCLIP(subject, reference_dir, image_dir, processor, epoch)
        # dataset = SelfPairwiseImageDatasetCLIP(subject, './data', processor)

        for i in progress(range(len(dataset))):
            inputs_A, inputs_B = dataset[i]
            if inputs_A is not None and inputs_B is not None:
                inputs_A['pixel_values'] = inputs_A['pixel_values'].to(device)
                inputs_B['pixel_values'] = inputs_B['pixel_values'].to(device) 

                image_A_features = model.get_image_features(**inputs_A)
                image_B_features = model.get_image_features(**inputs_B)

                image_A_features = image_A_features / image_A_features.norm(p=2, dim=-1, keepdim=True)
                image_B_features = image_B_features / image_B_features.norm(p=2, dim=-1, keepdim=True)
            
                logit_scale = model.logit_scale.exp()
                sim = torch.matmul(image_A_features, image_B_features.t()) # * logit_scale
                similarity.append(sim.item())

    mean_similarity = mean_or_raise(similarity, criterion)
    print(criterion, 'mean_similarity', mean_similarity, 'num_scores', len(similarity))

    return mean_similarity, criterion


def dino(image_dir, epoch=None, reference_dir=None):
    reference_dir = reference_dir or default_reference_dir()
    criterion = 'dino'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTModel.from_pretrained('facebook/dino-vits16').to(device)
    feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vits16')

    similarity = []
    for subject in subject_names:
        dataset = PairwiseImageDatasetDINO(subject, reference_dir, image_dir, feature_extractor, epoch)
        # dataset = SelfPairwiseImageDatasetDINO(subject, './data', feature_extractor)

        for i in progress(range(len(dataset))):
            inputs_A, inputs_B = dataset[i]
            if inputs_A is not None and inputs_B is not None:
                inputs_A['pixel_values'] = inputs_A['pixel_values'].to(device)
                inputs_B['pixel_values'] = inputs_B['pixel_values'].to(device) 

                outputs_A = model(**inputs_A)
                image_A_features = outputs_A.last_hidden_state[:, 0, :]

                outputs_B = model(**inputs_B)
                image_B_features = outputs_B.last_hidden_state[:, 0, :]

                image_A_features = image_A_features / image_A_features.norm(p=2, dim=-1, keepdim=True)
                image_B_features = image_B_features / image_B_features.norm(p=2, dim=-1, keepdim=True)

                sim = torch.matmul(image_A_features, image_B_features.t()) # * logit_scale
                similarity.append(sim.item())

    mean_similarity = mean_or_raise(similarity, criterion)
    print(criterion, 'mean_similarity', mean_similarity, 'num_scores', len(similarity))

    return mean_similarity, criterion


def lpips_image(image_dir, epoch=None, reference_dir=None):
    reference_dir = reference_dir or default_reference_dir()
    criterion = 'lpips_image'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = lpips.LPIPS(net='alex').to(device)

    similarity = []
    for subject in subject_names:
        dataset = PairwiseImageDatasetLPIPS(subject, reference_dir, image_dir, epoch)
        # dataset = SelfPairwiseImageDatasetLPIPS(subject, './data')
        
        for i in progress(range(len(dataset))):
            image_A, image_B = dataset[i]
            if image_A is not None and image_B is not None:
                image_A = image_A.to(device)
                image_B = image_B.to(device)
                if image_A.dim() == 3:
                    image_A = image_A.unsqueeze(0)
                if image_B.dim() == 3:
                    image_B = image_B.unsqueeze(0)

                # Calculate LPIPS between the two images
                distance = loss_fn(image_A, image_B)

                similarity.append(distance.item())

    mean_similarity = mean_or_raise(similarity, criterion)
    print(criterion, 'LPIPS distance', mean_similarity, 'num_scores', len(similarity))

    return mean_similarity, criterion


def default_image_dir():
    return str(Path(__file__).resolve().parent / "log_cara")


def default_reference_dir():
    return str(Path(__file__).resolve().parents[1] / "data" / "dreambooth")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generated DreamBooth images.")
    parser.add_argument("--image-dir", default=default_image_dir(), help="Directory containing generated image folders.")
    parser.add_argument("--epoch", type=int, default=8, help="Single epoch subfolder to evaluate. Use -1 for no epoch subfolder.")
    parser.add_argument(
        "--epochs",
        default="0-50",
        help="Epochs for --all-epochs, e.g. 0-50, all, or 0,2,4,6,8.",
    )
    parser.add_argument(
        "--metric",
        choices=("clip_text", "clip_image", "dino", "lpips_image", "all"),
        default="all",
        help="Metric to calculate.",
    )
    parser.add_argument(
        "--all-epochs",
        action="store_true",
        help="Score every selected epoch and write one CSV row per epoch.",
    )
    parser.add_argument("--metadata", default="metadata.json", help="Optional metadata JSON for clip_text.")
    parser.add_argument("--reference-dir", default=default_reference_dir(), help="Reference subject image directory for image-image metrics.")
    parser.add_argument("--results-file", default=None, help="CSV file to write/merge results to. Defaults to <image-dir-name>_results.csv.")
    parser.add_argument(
        "--final-results-file",
        default=None,
        help="CSV file with one selected best-epoch row per run. Defaults to <image-dir-name>_final_results.csv.",
    )
    parser.add_argument("--strict", action="store_true", help="Fail on missing folders instead of skipping them in single-epoch mode.")
    parser.add_argument("--unique-token", default="qwe", help="Unique token used in generated prompts when metadata is absent.")
    parser.add_argument("--max-images", type=int, default=0, help="Maximum generated images per prompt/epoch to score. Use 0 for all images.")
    return parser.parse_args()


def run_single_epoch_metric(metric, args, epoch):
    if metric == "clip_text":
        return clip_text(
            args.image_dir,
            epoch,
            metadata_path=args.metadata,
            partial=not args.strict,
            unique_token=args.unique_token,
        )
    if metric == "clip_image":
        return clip_image(args.image_dir, epoch, reference_dir=args.reference_dir)
    if metric == "dino":
        return dino(args.image_dir, epoch, reference_dir=args.reference_dir)
    if metric == "lpips_image":
        return lpips_image(args.image_dir, epoch, reference_dir=args.reference_dir)
    raise ValueError(f"Unknown metric: {metric}")


metric_columns = ["clip_text", "clip_image", "dino", "lpips_image"]
image_image_metrics = {"clip_image", "dino", "lpips_image"}
image_suffixes = (".jpg", ".jpeg", ".png")


def epoch_row(record, image_dir, epoch, epochs_label, max_images):
    return {"epoch": "" if epoch is None else epoch, "run": record["data_dir"]}


def csv_epoch(value):
    return "" if value is None else str(value)


def parse_metric_float(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def merge_epoch_rows(results_file, rows):
    fieldnames = ["run", "epoch"] + metric_columns
    rows_by_key = {}
    incoming_runs = {row.get("run", "") for row in rows if row.get("run")}
    fallback_run = next(iter(incoming_runs)) if len(incoming_runs) == 1 else ""

    if os.path.isfile(results_file) and os.path.getsize(results_file) > 0:
        with open(results_file, newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                run = row.get("run") or fallback_run
                epoch = csv_epoch(row.get("epoch", ""))
                merged = {"run": run, "epoch": epoch}
                for metric in metric_columns:
                    value = row.get(metric, "")
                    if value not in (None, ""):
                        merged[metric] = value
                rows_by_key[(run, epoch)] = merged

    for row in rows:
        run = row.get("run", "")
        epoch = csv_epoch(row.get("epoch", ""))
        merged = rows_by_key.setdefault((run, epoch), {"run": run, "epoch": epoch})
        for metric in metric_columns:
            value = row.get(metric, "")
            if value not in (None, ""):
                merged[metric] = value

    with open(results_file, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        merged_rows = sort_epoch_rows(rows_by_key.values())
        for row in merged_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    return merged_rows


def sort_epoch_rows(rows):
    def key(row):
        epoch = row.get("epoch", "")
        try:
            epoch_key = int(epoch)
        except (TypeError, ValueError):
            epoch_key = -1
        return (row.get("run", ""), epoch_key)

    return sorted(rows, key=key)


def normalize_metric_values(values):
    if not values:
        return []
    min_value = min(values)
    max_value = max(values)
    denom = max_value - min_value
    if denom == 0:
        return [0.0 for _ in values]
    return [value / denom for value in values]


def select_best_epoch_rows(rows):
    rows_by_run = {}
    for row in rows:
        run = row.get("run")
        epoch = row.get("epoch")
        if not run or epoch in (None, ""):
            continue
        rows_by_run.setdefault(run, []).append(row)

    selected_rows = []
    skipped_runs = []
    for run in sorted(rows_by_run):
        run_rows = sort_epoch_rows(rows_by_run[run])
        complete_rows = [
            row
            for row in run_rows
            if all(parse_metric_float(row.get(metric)) is not None for metric in metric_columns)
        ]
        if not complete_rows:
            skipped_runs.append(run)
            continue

        combined_scores = [0.0 for _ in complete_rows]
        for metric in metric_columns:
            raw_values = [parse_metric_float(row.get(metric)) for row in complete_rows]
            normalized_values = normalize_metric_values(raw_values)
            combined_scores = [
                score + normalized
                for score, normalized in zip(combined_scores, normalized_values)
            ]

        best_idx = max(range(len(complete_rows)), key=lambda idx: combined_scores[idx])
        best_row = complete_rows[best_idx]
        selected_row = {
            "run": run,
            "best_epoch": best_row["epoch"],
            "normalized_score": combined_scores[best_idx],
        }
        for metric in metric_columns:
            selected_row[metric] = parse_metric_float(best_row.get(metric))
        selected_rows.append(selected_row)

    return selected_rows, skipped_runs


def print_best_epoch_summary(selected_rows, skipped_runs):
    if skipped_runs:
        examples = ", ".join(skipped_runs[:5])
        if len(skipped_runs) > 5:
            examples += f", and {len(skipped_runs) - 5} more"
        print(
            "Skipped best-epoch summary for "
            f"{len(skipped_runs)} run(s) without all four metrics: {examples}",
            flush=True,
        )

    if selected_rows:
        print(f"Printed best normalized epoch for {len(selected_rows)} run(s).", flush=True)
    for row in selected_rows:
        run = row["run"]
        best_epoch = row["best_epoch"]
        normalized_score = row["normalized_score"]
        print(
            f"{run} best_epoch {best_epoch} normalized_score {normalized_score:.6f}",
            flush=True,
        )


def write_final_results(final_results_file, selected_rows):
    fieldnames = ["run", "best_epoch", "normalized_score"] + metric_columns
    with open(final_results_file, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print(f"Wrote {len(selected_rows)} final row(s) to {final_results_file}", flush=True)


def validate_reference_images(metrics, records, reference_dir):
    if not any(metric in image_image_metrics for metric in metrics):
        return

    missing = []
    for subject in sorted({record["subject"] for record in records}):
        subject_dir = os.path.join(reference_dir, subject)
        if not list_image_files(subject_dir, image_suffixes):
            missing.append((subject, subject_dir))

    if missing:
        examples = ", ".join(f"{subject} ({subject_dir})" for subject, subject_dir in missing[:5])
        if len(missing) > 5:
            examples += f", and {len(missing) - 5} more"
        raise FileNotFoundError(
            "No reference images found for image-image metrics. "
            f"Missing subject folders/images: {examples}. "
            "Pass --reference-dir pointing to DreamBooth instance images with one subfolder per subject."
        )


def infer_results_file(image_dir):
    image_dir = os.path.normpath(image_dir)
    name = os.path.basename(image_dir) or "results"
    return f"{name}_results.csv"


def infer_final_results_file(image_dir):
    image_dir = os.path.normpath(image_dir)
    name = os.path.basename(image_dir) or "results"
    return f"{name}_final_results.csv"


def evaluate_all_epochs_wide(metrics, args, epochs, epochs_label):
    records = prompt_records(args.image_dir, metadata_path=args.metadata, unique_token=args.unique_token)
    if not records:
        raise ValueError(f"No prompt folders found under {args.image_dir}")

    validate_reference_images(metrics, records, args.reference_dir)

    rows_by_key = {}
    print(
        f"Writing per-epoch CSV: {len(records)} prompt folder(s), metrics={','.join(metrics)}, epochs={epochs_label}",
        flush=True,
    )

    for record in records:
        run_dir = record.get("run_dir", os.path.join(args.image_dir, record["data_dir"]))
        candidate_epochs = epochs if epochs is not None else list_available_epochs(run_dir)
        if len(candidate_epochs) < 51:
            print(f"--- {record['data_dir']} skipped ({len(candidate_epochs)}/51 epochs) ---", flush=True)
            continue
        print(f"--- {record['data_dir']} ({len(candidate_epochs)} epochs) ---", flush=True)

        for metric in metrics:
            context = build_metric_context(metric, reference_dir=args.reference_dir)
            print(f"  {metric}...", flush=True)
            epoch_iter = progress(candidate_epochs, desc=f"{record['data_dir']} {metric}")
            for epoch in epoch_iter:
                score, count = score_prompt_epoch(metric, context, record, run_dir, epoch, max_images=args.max_images)
                key = (record["data_dir"], epoch)
                row = rows_by_key.setdefault(
                    key,
                    epoch_row(record, args.image_dir, epoch, epochs_label, args.max_images),
                )
                if score is not None:
                    row[metric] = score
                    epoch_iter.set_postfix(score=f"{score:.4f}")
            del context
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Write results after finishing all metrics for this instance
        instance_rows = [v for k, v in rows_by_key.items() if k[0] == record["data_dir"]]
        merged_rows = merge_epoch_rows(args.results_file, instance_rows)
        print(f"  wrote {len(instance_rows)} row(s) to {args.results_file}", flush=True)
        # Simultaneously write best rows for all completed instances so far
        partial_selected, _ = select_best_epoch_rows(merged_rows)
        write_final_results(args.final_results_file, partial_selected)

    rows = sort_epoch_rows(rows_by_key.values())
    selected_rows, skipped_runs = select_best_epoch_rows(rows)
    print_best_epoch_summary(selected_rows, skipped_runs)
    return rows


def evaluate_single_epoch_wide(metrics, args, epoch):
    records = prompt_records(args.image_dir, metadata_path=args.metadata, unique_token=args.unique_token)
    if not records:
        raise ValueError(f"No prompt folders found under {args.image_dir}")

    validate_reference_images(metrics, records, args.reference_dir)

    rows_by_key = {}
    epoch_label = "" if epoch is None else str(epoch)
    print(
        f"Writing single-epoch CSV: {len(records)} prompt folder(s), metrics={','.join(metrics)}, epoch={epoch_label}",
        flush=True,
    )

    for record in records:
        run_dir = record.get("run_dir", os.path.join(args.image_dir, record["data_dir"]))
        available = list_available_epochs(run_dir)
        if len(available) < 51:
            print(f"--- {record['data_dir']} skipped ({len(available)}/51 epochs) ---", flush=True)
            continue
        print(f"--- {record['data_dir']} ---", flush=True)

        for metric in metrics:
            context = build_metric_context(metric, reference_dir=args.reference_dir)
            print(f"  {metric}...", flush=True)
            score, count = score_prompt_epoch(metric, context, record, run_dir, epoch, max_images=args.max_images)
            key = (record["data_dir"], epoch)
            row = rows_by_key.setdefault(
                key,
                epoch_row(record, args.image_dir, epoch, epoch_label, args.max_images),
            )
            if score is not None:
                row[metric] = score
                print(f"  {metric} = {score:.4f}", flush=True)
            del context
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Write results after finishing all metrics for this instance
        instance_rows = [v for k, v in rows_by_key.items() if k[0] == record["data_dir"]]
        merged_rows = merge_epoch_rows(args.results_file, instance_rows)
        print(f"  wrote to {args.results_file}", flush=True)
        # Simultaneously write best rows for all completed instances so far
        partial_selected, _ = select_best_epoch_rows(merged_rows)
        write_final_results(args.final_results_file, partial_selected)

    rows = sort_epoch_rows(rows_by_key.values())
    selected_rows, skipped_runs = select_best_epoch_rows(rows)
    print_best_epoch_summary(selected_rows, skipped_runs)
    return rows



if __name__ == "__main__":
    args = parse_args()
    if args.results_file is None:
        args.results_file = infer_results_file(args.image_dir)
    if args.final_results_file is None:
        args.final_results_file = infer_final_results_file(args.image_dir)
    metrics = metric_columns if args.metric == "all" else [args.metric]

    if args.all_epochs:
        print(f"Starting all-epochs evaluation: image_dir={args.image_dir}, metric={args.metric}, epochs={args.epochs}", flush=True)
        epochs = parse_epoch_selection(args.epochs)
        evaluate_all_epochs_wide(metrics, args, epochs, args.epochs)
    else:
        print(f"Starting single-epoch evaluation: image_dir={args.image_dir}, metric={args.metric}, epoch={args.epoch}", flush=True)
        epoch = None if args.epoch < 0 else args.epoch
        evaluate_single_epoch_wide(metrics, args, epoch)
