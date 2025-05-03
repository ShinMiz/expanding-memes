import asyncio
import re
import random
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple
from utils import compute_emotion_score, compute_logic_score, score_meme_against_profile, should_reject_meme, sigmoid_sharp
from meme_generation import mutate_meme_async, reformulate_memes_async
from llm_interface import query_ollama_async


async def evaluate_and_propagate_async(
    agents: List[Agent],
    G: nx.DiGraph,
    memes: Dict[str, Dict[str, float]],
    meme_vectors: Dict[str, Dict[str, float]],
    top_n: int = top_n
) -> List[Tuple[int, int, str, float]]:

    agent_meme_scores: Dict[int, List[Tuple[str, float]]] = {agent.id: [] for agent in agents}
    eval_tasks = []
    task_mapping = []

    for agent in agents:
        for meme_text, vector in memes.items():
            if not meme_text.strip():
                continue
            eval_tasks.append(agent.evaluate_meme_async(meme_text))
            task_mapping.append((agent.id, meme_text, vector))

    results = await asyncio.gather(*eval_tasks, return_exceptions=True)

    for (agent_id, meme_text, vector), reaction in zip(task_mapping, results):
        if not isinstance(reaction, dict):
            continue

        if should_reject_meme(reaction): 
            print(f"🚫 Meme rejected by Agent {agent_id}: {meme_text}")
            continue

        base_score = score_meme_against_profile(vector, reaction)
        emotion = compute_emotion_score(reaction)
        logic = compute_logic_score(reaction)
        hybrid_score = 0.6 * base_score + 0.2 * emotion + 0.2 * logic

        agent_meme_scores[agent_id].append((meme_text, hybrid_score))

    transmissions = []
    for agent in agents:
        agent.update_memory(agent_meme_scores[agent.id], top_n=top_n)
        for meme in agent.get_high_propagation_memes():
            for neighbor in G.successors(agent.id):
                weight = G[agent.id][neighbor]['weight']
                transmissions.append((agent.id, neighbor, meme, weight))

    return transmissions


async def propagate_with_persuasion_async(
    agents,
    G,
    meme_vectors: Dict[str, Dict[str, float]],
    mutation_prob=0.1
) -> list[tuple[int, int, str, float]]:
    
    tasks = []
    id2agent = {agent.id: agent for agent in agents}

    async def process_task(aid: int, nid: int, meme: str, strength: float):
        agent = id2agent[aid]
        prob = get_mutation_prob(agent)
        try:
            meme_mutated = await mutate_meme_async(meme, prob=prob)
            prompt = f"Try to convince the receiver of this meme with high persuasion strength. Meme: '{meme_mutated}'"
            response = await query_ollama_async(prompt)
            return (aid, nid, response.strip(), strength)
        except Exception as e:
            print(f"❌ Persuasion error {aid}->{nid}: {e}")
            return (aid, nid, meme, strength)

    for agent in agents:
        memes = agent.get_high_propagation_memes()
        strengths = compute_persuasion_strengths(agent, G)
        for meme in memes:
            for neighbor, strength in strengths.items():
                tasks.append((agent.id, neighbor, meme, strength))

    results = await asyncio.gather(*[process_task(*args) for args in tasks])

    threshold = 0.5
    for aid, nid, meme_text, strength in results:
        receiver = next((a for a in agents if a.id == nid), None)
        if receiver:
            reaction = await receiver.evaluate_meme_async(meme_text)
            if isinstance(reaction, dict):
                vector = meme_vectors.get(meme_text)
                if vector is None:
                    continue
                score = score_meme_against_profile(vector, reaction)
                if score > threshold:
                    sender = id2agent.get(aid)
                    if sender:
                        sender.propagation_history[nid] = sender.propagation_history.get(nid, 0) + 1

    return results

async def evaluate_final_answer_async(answer: str, question: str) -> float:
    prompt = evaluate_final_answer_async_prompt(meme, question).strip()

    content = await query_ollama_async(prompt)
    
    # 柔軟な抽出: "Score: 0.72" 形式
    match = re.search(r"Score:\s*([01](?:\.\d+)?)", content)
    raw_score = float(match.group(1)) if match else 0.5

    # ロジスティック補正（中心に収束しすぎないよう調整可）
    return sigmoid_sharp(raw_score, center=0.5, steepness=1)


async def update_agent_memories_async(
    agents,
    received_memes_by_agent,
    G,
    meme_vectors,
    top_n=top_n
):
    agent_scores = {}
    id_to_agent = {agent.id: agent for agent in agents}
    all_tasks = []
    task_info = []

    for agent in agents:
        for meme_text, from_id in received_memes_by_agent.get(agent.id, []):
            all_tasks.append(agent.evaluate_meme_async(meme_text))
            task_info.append((agent.id, meme_text, from_id))

    reactions = await asyncio.gather(*all_tasks, return_exceptions=True)

    new_meme_scores = {agent.id: [] for agent in agents}

    for (agent_id, meme_text, from_id), reaction in zip(task_info, reactions):
        if not isinstance(reaction, dict):
            continue

        vector = meme_vectors.get(meme_text)
        if vector is None:
            continue  # fallback防止：未知のミームは無視

        score = score_meme_against_profile(vector, reaction)
        weight = G[from_id][agent_id]['weight']
        adjusted = score * weight
        new_meme_scores[agent_id].append((meme_text, adjusted))
        agent_scores.setdefault(agent_id, []).append((from_id, adjusted))

    for agent in agents:
        combined = agent.memory + new_meme_scores.get(agent.id, [])
        agent.memory = sorted(combined, key=lambda x: -x[1])[:top_n]

        for src_id, _ in agent_scores.get(agent.id, []):
            if src_id in id_to_agent:
                id_to_agent[src_id].discussion_point += 1

        relevant_profiles = [
            r for ((aid, _, _), r) in zip(task_info, reactions)
            if aid == agent.id and isinstance(r, dict)
        ]
        agent.update_tendencies(relevant_profiles)

    return agent_scores