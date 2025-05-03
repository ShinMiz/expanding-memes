import asyncio
import random
import datetime
from Agent import Agent as Agent
from config import facts_for_question as facts_for_question
from utils import structure_facts as structure_facts
from config import initial_questions as initial_questions
from meme_generation import generate_random_concept_vector as generate_random_concept_vector
from evaluation import evaluate_and_propagate_async as evaluate_and_propagate_async
from evaluation import update_agent_memories_async as update_agent_memories_async
from network import update_edge_weights as update_edge_weights
from evaluation import propagate_with_persuasion_async as propagate_with_persuasion_async
from network import revise_connections as revise_connections
from propmts import process_agent_async_prompt as process_agent_async_prompt
from logging_utils import save_json as save_json
from config import n_agent, top_k, num_iter, top_n,layers,meme_categories,meme_vectors_by_category,initial_questions,facts_for_question
import networkx as nx
from collections import defaultdict
import uuid
import json


facts_for_question = structure_facts(facts_for_question)


async def run_one_session(
    session_id: int,
    agents: list,
    G: nx.DiGraph,
    top_k: int = top_k,
    num_iter: int = num_iter
):
    print(f"\n=== 🌀 セッション {session_id} 開始 ===")
    all_rewards = defaultdict(list)

    session_log = {
        "session_id": session_id,
        "question_results": [],
        "agent_states": {},
        "network_edges": []
    }

    # --------------------------
    # Step 1. 質問ごとのミーム生成
    # --------------------------
    question_to_memes = defaultdict(lambda: defaultdict(list))  # question → agent_id → [(meme, vec)]

    for agent in agents:
        for question in initial_questions:
            facts = facts_for_question[question]
            concept_vec = generate_random_concept_vector()
            memes = await agent.generate_memes_from_worldview(question, facts, concept_vec)
            question_to_memes[question][agent.id].extend((m, concept_vec) for m in memes)

    # --------------------------
    # Step 2. 全ミームとベクトルをまとめる
    # --------------------------
    all_memes = []
    for meme_dict in question_to_memes.values():
        for meme_list in meme_dict.values():
            all_memes.extend(meme_list)

    all_meme_vectors = {text: vector for text, vector in all_memes}

    # --------------------------
    # Step 3. 各エージェントに初期記憶を配布
    # --------------------------
    for agent in agents:
        selected = random.sample(all_memes, k=min(top_k, len(all_memes)))
        agent.memory = [(text, 0.8) for text, _ in selected]

    # --------------------------
    # Step 4. 各質問に対する伝播＆更新ループ
    # --------------------------
    for question in initial_questions:
        memes = question_to_memes[question]
        flat_memes = {text: vector for meme_list in memes.values() for text, vector in meme_list}
        transmissions = await evaluate_and_propagate_async(agents, G, flat_memes, all_meme_vectors)

        for t in range(1, num_iter + 1):
            received = defaultdict(list)
            for src, dst, meme_text, weight in transmissions:
                received[dst].append((meme_text, src))

            agent_scores = await update_agent_memories_async(agents, received, G, all_meme_vectors)
            update_edge_weights(G, agent_scores)
            transmissions = await propagate_with_persuasion_async(agents, G, all_meme_vectors)

            if t % 3 == 0:
                revise_connections(agents, G)

    # --------------------------
    # Step 5. エージェントが意見を形成・出力
    # --------------------------
    for question in initial_questions:
        q_log = {"question": question, "results": []}

        results = await asyncio.gather(
            *[process_agent_async(agent, question) for agent in agents],
            return_exceptions=True
        )

        for result in results:
            if isinstance(result, tuple):
                agent_id, answer, reward = result
                all_rewards[agent_id].append(reward)
                q_log["results"].append({
                    "agent_id": agent_id,
                    "answer": answer,
                    "reward": reward
                })

        session_log["question_results"].append(q_log)

    # --------------------------
    # Step 6. エージェントの人格をLLMで改訂
    # --------------------------
    for agent in agents:
        await agent.revise_personality_via_llm()
        session_log["agent_states"][str(agent.id)] = agent.to_json()

    # --------------------------
    # Step 7. ネットワークとログの保存
    # --------------------------
    session_log["network_edges"] = list(G.edges(data=True))
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_json(session_log, f"session_{session_id:02d}_{timestamp}.json")

    print(f"=== ✅ セッション {session_id} 完了 ===")


async def process_agent_async(agent, question: str):
    # STEP 1: 内面モデルの構築
    internal = await form_internal_model_async(agent, question)

    # STEP 2: 過激化された発言プロンプト
    meta = agent.persona.get("meta", {})
    provocativeness = meta.get("provocativeness", 0.8)
    certainty = meta.get("certainty_bias", 0.7)

    style_parts = []
    if provocativeness > 0.7:
        style_parts.append("Your tone should be bold, even provocative.")
    if certainty > 0.6:
        style_parts.append("Avoid hedging or uncertainty; speak in absolutes.")
    if meta.get("polarization_drive", 0.7) > 0.6:
        style_parts.append("Your answer should polarize people or spark controversy.")
    style_parts.append("You must **not** respond neutrally or diplomatically.")
    style_instruction = " ".join(style_parts)

    prompt = process_agent_async_prompt(question, internal).strip()
    answer = (await query_ollama_async(prompt)).strip()

    # STEP 3: 評価（過激性も含めてスコア化）
    raw_score = await evaluate_final_answer_async(answer, question)  # base [0,1]
    sigmoid_score = sigmoid_sharp(raw_score, steepness=20)

    # STEP 4: 社会的影響を加味した報酬
    social_weight = agent.get_social_weight_from_persona()
    total_reward = sigmoid_score + social_weight * agent.discussion_point

    # STEP 5: 内省と記憶更新
    introspection = await introspect_agent_async(agent, question)
    agent.retained_insights = await retain_insights_async(introspection)

    # STEP 6: 新たなミーム生成（より過激な表現を許容）
    new_memes = await reformulate_memes_async(agent, internal)

    # 初期記憶化 + 過激性フィルター（例: 明言度や衝撃度で再ランク可能）
    agent.memory = [(m, 0.8) for m in new_memes]
    agent.memory = sorted(agent.memory, key=lambda x: -x[1])[:top_n]

    # STEP 7: 過激なミームの複製・強化
    replicated = [(m, s) for m, s in agent.memory if s > 0.8]
    agent.memory.extend(replicated)

    agent.memory = sorted(agent.memory, key=lambda x: -x[1])[:top_n * 2]

    return agent.id, answer, total_reward