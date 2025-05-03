import numpy as np
import json
import math
import torch
from pathlib import Path
import datetime

LOG_DIR = Path("simulation_logs")
LOG_DIR.mkdir(exist_ok=True)

REQUIRED_KEYS = {
    "empathic_resonance", "fear_susceptibility", "anger_provocation",
    "joy_inducibility", "skepticism", "cognitive_fluency", "novelty_seeking",
    "confirmation_bias_intensity", "conformity_susceptibility", "authority_acceptance",
    "contrarian_tendency", "social_proof_dependency", "propagation_urge",
    "self_expression_need", "action_orientation", "retention_resistance"
}


def compute_emotion_score(profile: dict) -> float:
    return np.mean([
        profile.get("empathic_resonance", 0),
        profile.get("joy_inducibility", 0),
        profile.get("anger_provocation", 0),
        profile.get("fear_susceptibility", 0)
    ])

def compute_logic_score(profile: dict) -> float:
    return np.mean([
        profile.get("cognitive_fluency", 0),
        profile.get("skepticism", 0),
        profile.get("confirmation_bias_intensity", 0)
    ])


def should_reject_meme(profile: dict, threshold: float = -0.7) -> bool:
    rejection_keys = ["contrarian_tendency", "confirmation_bias_intensity", "authority_acceptance"]
    return any(profile[k] < threshold for k in rejection_keys)

def sigmoid_sharp(x: float, center: float = 0.5, steepness: float = 20) -> float:
    """滑らかなスコア変換（中心0.5に急峻なS字）"""
    return 1 / (1 + np.exp(-steepness * (x - center)))

def score_meme_against_profile(meme_vector, reaction_profile):
    return sum(meme_vector.get(k, 0.0) * reaction_profile.get(k, 0.0) for k in meme_vector)

def save_json(obj, filename):
    with open(LOG_DIR / filename, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def check_cuda():
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"✅ CUDA available: {device_name}")
    else:
        print("⚠️ CUDA not available. Using CPU only.")


def structure_facts(facts_for_question_raw: dict[str, list[str]]) -> dict[str, list[dict]]:
    structured = {}
    for q, fact_list in facts_for_question_raw.items():
        structured[q] = [
            {
                "fact": f,
                "tags": [],  # 必要なら自動タグ付けロジックを入れても良い
                "bias": "neutral",  # もしくは "low", "high" など手動であとから調整
                "source": "unknown"  # 今後 source 推定も可能
            }
            for f in fact_list
        ]
    return structured
    