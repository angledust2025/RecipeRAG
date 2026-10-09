# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
# limitations under the License.

import math
import os
import re
from typing import Any

from rouge import Rouge
import sacrebleu
from sentence_transformers import SentenceTransformer, util


rouge_scorer = Rouge()
model = SentenceTransformer(
    os.environ.get("TITLE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
)

# Metadata
REWARD_NAME = "reward"
REWARD_TYPE = "batch"


def format_reward(response: str) -> float:
    pattern = re.compile(
        r"<thinking>.*?</thinking>.*?<title>.*?</title>.*?<ingredients>.*?</ingredients>.*?<instructions>.*?</instructions>",
        re.DOTALL,
    )
    return 1.0 if re.search(pattern, response) else 0.0


def title_score(response: str, ground_truth: str) -> float:
    try:
        gt_match = re.search(r"<title>(.*?)</title>", ground_truth, re.DOTALL)
        pred_match = re.search(r"<title>(.*?)</title>", response, re.DOTALL)
        ref_title = gt_match.group(1).strip() if gt_match else ""
        pred_title = pred_match.group(1).strip() if pred_match else ""
        if not ref_title or not pred_title:
            return 0.0

        ref_embedding = model.encode(ref_title, convert_to_tensor=True)
        pred_embedding = model.encode(pred_title, convert_to_tensor=True)
        return float(util.cos_sim(ref_embedding, pred_embedding)[0][0])
    except Exception as exc:
        print(f"Title Score Exception: {exc}")
        return 0.0


def ingredients_score(response: str, ground_truth: str) -> float:
    try:
        gt_match = re.search(r"<ingredients>(.*?)</ingredients>", ground_truth, re.DOTALL)
        pred_match = re.search(r"<ingredients>(.*?)</ingredients>", response, re.DOTALL)
        gt = gt_match.group(1).strip() if gt_match else ""
        pred = pred_match.group(1).strip() if pred_match else ""
        gt_set = {item.strip().lower() for item in gt.split(",") if item.strip()}
        pred_set = {item.strip().lower() for item in pred.split(",") if item.strip()}
        if not gt_set or not pred_set:
            return 0.0

        intersection = len(gt_set & pred_set)
        precision = intersection / len(pred_set)
        recall = intersection / len(gt_set)
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    except Exception:
        return 0.0


def instructions_score_rouge(response: str, ground_truth: str) -> float:
    try:
        gt_match = re.search(r"<instructions>(.*?)</instructions>", ground_truth, re.DOTALL)
        pred_match = re.search(r"<instructions>(.*?)</instructions>", response, re.DOTALL)
        ref_instructions = gt_match.group(1).strip() if gt_match else ""
        pred_instructions = pred_match.group(1).strip() if pred_match else ""
        if not ref_instructions or not pred_instructions:
            return 0.0
        return rouge_scorer.get_scores(pred_instructions, ref_instructions)[0]["rouge-l"]["p"]
    except Exception:
        return 0.0


def instructions_score_sacrebleu(response: str, ground_truth: str) -> float:
    try:
        gt_match = re.search(r"<instructions>(.*?)</instructions>", ground_truth, re.DOTALL)
        pred_match = re.search(r"<instructions>(.*?)</instructions>", response, re.DOTALL)
        ref_instructions = gt_match.group(1).strip() if gt_match else ""
        pred_instructions = pred_match.group(1).strip() if pred_match else ""
        if not ref_instructions or not pred_instructions:
            return 0.0
        return float(sacrebleu.sentence_bleu(pred_instructions, [ref_instructions]).score)
    except Exception:
        return 0.0


def title_reward_sigmoid_01(score: float, center: float = 0.60, sharpness: float = 12) -> float:
    score = max(0.0, min(1.0, float(score)))
    return 1.0 / (1.0 + math.exp(-sharpness * (score - center)))


def ingredient_reward_sigmoid_01(score: float, center: float = 0.50, sharpness: float = 10) -> float:
    score = max(0.0, min(1.0, float(score)))
    return 1.0 / (1.0 + math.exp(-sharpness * (score - center)))


def bleu_reward_exp(bleu: float, scale: float = 12.0) -> float:
    bleu = max(0.0, float(bleu))
    return 1.0 - math.exp(-bleu / scale)


def rouge_precision_reward_sigmoid_01(score: float, center: float = 0.45, sharpness: float = 10) -> float:
    score = max(0.0, min(1.0, float(score)))
    return 1.0 / (1.0 + math.exp(-sharpness * (score - center)))


def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1) -> list[dict[str, float]]:
    """Compute the R1V recipe reward for a batch of generated responses."""
    scores = []
    for reward_input in reward_inputs:
        ground_truth = reward_input["ground_truth"]
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        format_ok = format_reward(response)

        title_reward = 0.0
        ingredient_reward = 0.0
        rouge_reward = 0.0
        bleu_reward = 0.0
        if format_ok:
            raw_title = title_score(response, ground_truth)
            raw_ingredients = ingredients_score(response, ground_truth)
            raw_rouge = instructions_score_rouge(response, ground_truth)
            raw_bleu = instructions_score_sacrebleu(response, ground_truth)

            title_reward = title_reward_sigmoid_01(raw_title, center=0.60, sharpness=12)
            ingredient_reward = ingredient_reward_sigmoid_01(raw_ingredients, center=0.50, sharpness=10)
            rouge_reward = rouge_precision_reward_sigmoid_01(raw_rouge, center=0.45, sharpness=10)
            bleu_reward = bleu_reward_exp(raw_bleu, scale=10.0)

        overall = title_reward + ingredient_reward + rouge_reward + bleu_reward
        scores.append(
            {
                "overall": overall,
                "title": title_reward,
                "ingredients": ingredient_reward,
                "instructions_bleu": bleu_reward,
                "instructions_rouge": rouge_reward,
            }
        )
    return scores
