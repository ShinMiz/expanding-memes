import random
import numpy as np
import networkx as nx
from typing import List, Tuple
from Agent import Agent as Agent
from Agent import generate_persona as generate_persona
from config import n_agent, top_k, num_iter, top_n,layers,meme_categories,meme_vectors_by_category,initial_questions,facts_for_question
from collections import defaultdict


def create_agents_and_graph(n_agents: int = n_agent, seed: int = 42) -> Tuple[List[Agent], nx.DiGraph]:
    random.seed(seed)
    np.random.seed(seed)
    agents = [Agent(i, generate_persona(seed=i)) for i in range(n_agents)]
    graph = nx.barabasi_albert_graph(n_agents, m=2, seed=seed).to_directed()

    for u, v in graph.edges():
        graph[u][v]['weight'] = np.random.uniform(0.1, 1.0)

    return agents, graph


def compute_persuasion_strengths(agent, G, total_strength=1.0):
    neighbors = list(G.successors(agent.id))
    if not neighbors:
        return {}
    base = total_strength / len(neighbors)
    return {nbr: base for nbr in neighbors}


def revise_connections(agents, G, top_k=top_k, min_score=1):
    for agent in agents:
        # 最も伝播成功が多かった上位 k を残す
        scores = agent.propagation_history
        best_neighbors = sorted(scores.items(), key=lambda x: -x[1])[:top_k]

        current_neighbors = list(G.successors(agent.id))
        for neighbor in current_neighbors:
            if neighbor not in dict(best_neighbors):
                G.remove_edge(agent.id, neighbor)

        # 新たに成功経験のあるが未接続の相手に追加する
        for nbr, score in scores.items():
            if score >= min_score and not G.has_edge(agent.id, nbr):
                G.add_edge(agent.id, nbr)
                G[agent.id][nbr]['weight'] = 0.5  # 初期重み


def update_edge_weights(G, agent_scores, alpha=0.05):
    for dst_id, evaluations in agent_scores.items():
        for src_id, score in evaluations:
            if G.has_edge(src_id, dst_id):
                delta = alpha * (score - 0.5)
                current = G[src_id][dst_id]['weight']
                G[src_id][dst_id]['weight'] = max(0.01, min(1.0, current + delta))
