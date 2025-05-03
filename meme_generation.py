import random
import numpy as np
import json
from collections import defaultdict
from llm_interface import query_ollama_async as query_ollama_async
from propmts import generate_initial_memes_async_prompt as generate_initial_memes_async_prompt
from llm_interface import prompt_noise as prompt_noise
from propmts import reformulate_memes_async_prompt as reformulate_memes_async_prompt
from config import n_agent, top_k, num_iter, top_n,layers,meme_categories,meme_vectors_by_category,initial_questions,facts_for_question
from typing import List, Dict, Tuple
from collections import defaultdict


def generate_random_concept_vector(std: float = 0.05) -> dict:
    """ミーム属性テンプレートからランダムなベースベクトルを選び、少しノイズを加える"""
    base = random.choice(list(meme_vectors_by_category.values()))
    noisy_vector = {
        k: round(max(0.0, min(1.0, v + np.random.normal(0, std))), 3)
        for k, v in base.items()
    }
    return noisy_vector


def generate_concept_hint(concept_vec: dict[str, float]) -> str:
    """概念ベクトルを人間向けのヒントとして整形する"""
    if not concept_vec:
        return "No conceptual bias provided."
    
    sorted_items = sorted(concept_vec.items(), key=lambda x: -abs(x[1]))[:5]  # 上位5件
    lines = [f"{key}: {value:.2f}" for key, value in sorted_items]
    return "\nConceptual Tendencies:\n" + "\n".join(lines)


async def generate_initial_memes_async(
    question: str,
    n_per_category: int = 3,
    vector_noise_std: float = 0.05
) -> dict[str, list[Tuple[str, dict]]]:
    """
    各カテゴリごとに複数のミームと対応するノイズ付き属性ベクトルを生成する。
    ベクトルはLLMプロンプトにも埋め込まれる。
    戻り値: Dict[カテゴリ名, List[Tuple[ミーム文, 属性ベクトル]]]
    """
    tasks = []
    metadata = []

    for category, prompt in meme_categories.items():
        base_vector = meme_vectors_by_category.get(category, {})
        if not base_vector:
            print(f"⚠️ No base vector found for category: {category}")
            continue

        for _ in range(n_per_category):
            # ノイズ付きベクトルを構築
            noisy_vector = {
                k: round(max(0.0, min(1.0, v + np.random.normal(0, vector_noise_std))), 3)
                for k, v in base_vector.items()
            }

            # ベクトルを自然言語的に埋め込む（JSON表現）
            concept_hint = (
                f"\nThis meme should reflect the following conceptual properties (values in [0.0–1.0]):\n"
                f"{json.dumps(noisy_vector, indent=2)}"
            )

            full_prompt = generate_initial_memes_async_prompt(question,concept_hint)
            
            temp = 0.7 + 0.1 * random.uniform(-1, 1)
            tasks.append(query_ollama_async(full_prompt + prompt_noise(), temperature=temp))
            metadata.append((category, noisy_vector))  # ← vectorをメタにも保持
    results = await asyncio.gather(*tasks, return_exceptions=True)

    memes = defaultdict(list)
    for (cat, vector), res in zip(metadata, results):
        if isinstance(res, Exception) or not isinstance(res, str):
            print(f"❌ Meme generation failed for category: {cat}")
            continue
        line = res.strip().split("\n")[0].strip().strip('"').strip("'")
        #if not (5 < len(line) < 150):
        #    print(f"⚠️ Invalid meme length for category: {cat} → {repr(line)}")
        #    continue

        memes[cat].append((line, vector))

        print(f"=== LLM Response for {cat} ===")
        print(res)

    return memes  # Dict[str, List[Tuple[meme_str, vector_dict]]]


async def reformulate_memes_async(agent, internal_model: str) -> list[str]:
    memes = [m for m, _ in agent.memory]
    if not memes:
        return []
    # 交配プロンプト：既存ミームを基に進化的表現を作る
    prompt = reformulate_memes_async_prompt.strip()
    content = await query_ollama_async(prompt)
    # 分解
    new_memes = [line.strip('- ').strip().strip('"') for line in content.strip().split('\n') if line.strip()]
    return [m for m in new_memes if 5 < len(m) < 150]


async def mutate_meme_async(meme: str, prob: float = 0.1) -> str:
    if np.random.rand() > prob:
        return meme
    prompt = f"Slightly rephrase this meme, keeping its spirit: '{meme}'. Change the wording just a little."
    return (await query_ollama_async(prompt)).strip()


async def infer_vector_from_meme_async(meme: str, meme_vectors: dict, model: str = "dolphin-mistral") -> dict:
    if meme in infer_cache:
        return infer_cache[meme]

    prompt = f"Given the following meme: '{meme}', which of these categories best describes it? {list(meme_vectors.keys())}. Respond with one category name."
    content = await query_ollama_async(prompt)
    category = content.strip().lower().replace(' ', '_')

    vector = meme_vectors.get(category)
    if vector is None:
        print(f"⚠️ Unknown category '{category}' inferred. Using random fallback.")
        vector = random.choice(list(meme_vectors.values()))
    infer_cache[meme] = vector
    return vector