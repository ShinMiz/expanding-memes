import random
import numpy as np
from typing import List, Tuple, Dict, Optional
from .llm_interface import query_ollama_async
from .utils import compute_emotion_score, compute_logic_score, generate_description


# ==============================
# Agent クラス（記憶・評価・伝播）
# ==============================
from typing import List, Dict, Tuple

class Agent:
    def __init__(self, agent_id: int, persona: dict):
        self.id = agent_id
        self.persona = persona
        self.prompt = generate_description(persona)
        self.memory: List[Tuple[str, float]] = []
        self.retained_insights = ""
        self.propagation_history = {} 
        self.emotion_tendency = 0.0  # -1.0〜+1.0
        self.logic_tendency = 0.0

    def is_serious(self) -> bool:
        return self.persona.get("seriousness to the task", {}).get("seriousness", 0) > 0

    def __repr__(self):
        return f"Agent(id={self.id}, memory_size={len(self.memory)})"

    async def evaluate_meme_async(self, meme_text: str):
        if not self.is_serious():
            print(f"😒 Agent {self.id} is not serious enough to evaluate meme: {meme_text}")
            return None  # ミーム評価スキップ
        try:
            return await limited_generate_reaction_profile(self.prompt, meme_text)
        except Exception as e:
            print(f"❌ Error evaluating meme for Agent {self.id}: {e}")
            return None
    def update_tendencies(self, reaction_profiles: List[dict]):
        if not reaction_profiles:
            self.emotion_tendency = 0.0
            self.logic_tendency = 0.0
            return

        emotions = [compute_emotion_score(p) for p in reaction_profiles]
        logics = [compute_logic_score(p) for p in reaction_profiles]
        self.emotion_tendency = float(np.mean(emotions))
        self.logic_tendency = float(np.mean(logics))

    def update_memory(self, meme_scores: List[Tuple[str, float]], top_n: int = top_k):
        self.memory = sorted(meme_scores, key=lambda x: -x[1])[:top_n]

    def get_high_propagation_memes(self, threshold: float = 0.7) -> List[str]:
        return [m for m, score in self.memory if score > threshold]
    
    def get_social_weight_from_persona(self) -> float:
        traits = self.persona.get("group_behavior", {})
        keys = ["conformity_tendency", "emotional_contagion", "mobility_readiness"]
        values = [traits.get(k, 0.0) for k in keys]
        avg = np.mean(values)
        # scale [-1, 1] → [0, 0.3] （最大で 0.3 倍の重み）
        return max(0.0, min(0.3, (avg + 1) / 2 * 0.3))
        
    def select_facts_for_meme(self, question: str, fact_list: list[dict], k: int = 3) -> list[str]:
        # 自身の性格に基づいてスコアリング
        def score(f):
            score = 0.0
            traits = self.persona["group_behavior"]
            if traits["polarization_tendency"] > 0.5 and f["bias"] == "high":
                score += 1
            if traits["obedience_to_authority"] < 0 and f["source"] == "official":
                score -= 1
            if traits["emotional_contagion"] > 0.3 and "fear" in f["tags"]:
                score += 1
            return score + random.uniform(0, 1)

        top_facts = sorted(fact_list, key=score, reverse=True)[:k]
        return [f["fact"] for f in top_facts]

    async def generate_memes_from_worldview(agent, question: str, facts: list[dict], concept_vec: dict) -> list[str]:
        internal = await form_internal_model_async(agent, question)
        selected_facts = agent.select_facts_by_worldview(internal, facts)

        fact_section = (
            "\nRelevant factual triggers:\n" +
            "\n".join(f"- {fact}" for fact in selected_facts)
        )
        concept_hint = generate_concept_hint(concept_vec)

        prompt = f"""
        You are an unapologetic, highly opinionated civic strategist.
        You have a unique personality shaped by strong meme exposure.

        You are now being interviewed on the topic:
        "{question}"

        Based on your internal worldview:
        "{internal}"

        {fact_section}

        The following conceptual preferences describe the rhetorical force, emotional tone, and ideological stance you should adopt:
        {concept_hint}

        You must provide a **concrete, implementable proposal** or radical stance.
        Do NOT summarize. Do NOT hedge. Do NOT use abstract language.

        Your answer must:
        - Describe a specific policy, mechanism, or action
        - Avoid vague words like “should consider” or “might”
        - Include a social domain, method, or consequence
        - Be polarizing, if necessary

        Respond in 1–3 direct sentences only. No explanations or disclaimers.
        """.strip()

        response = await query_ollama_async(prompt)
        return [line.strip() for line in response.strip().split("\n") if line.strip()]

    def select_facts_by_worldview(self, worldview: str, facts: list[dict], k: int = 3) -> list[str]:
        # optional: embed worldview and facts using an embedding model and compute similarity
        scored = []
        for f in facts:
            score = 0
            if any(tag in worldview.lower() for tag in f.get("tags", [])):
                score += 1
            if f.get("bias") == "high" and self.persona["group_behavior"]["polarization_tendency"] > 0.5:
                score += 1
            if f.get("source") == "official" and self.persona["group_behavior"]["obedience_to_authority"] > 0.3:
                score += 1
            scored.append((f["fact"], score + random.uniform(0, 1)))

        return [fact for fact, _ in sorted(scored, key=lambda x: -x[1])[:k]]


    async def revise_personality_via_llm(self, model_name="dolphin-mistral"):
        EDITABLE_TRAITS = {
            "individual_traits": [
                "extraversion", "neuroticism", "openness", "conscientiousness",
                "agreeableness", "self_efficacy", "intrinsic_motivation", "social_need"
            ],
            "group_behavior": [
                "conformity_tendency", "norm_acceptance", "polarization_tendency",
                "obedience_to_authority", "emotional_contagion", "mobility_readiness"
            ]
        }

        # 現在の性格（編集可能な部分のみ）
        editable_layers = {
            layer: {k: self.persona[layer][k] for k in keys}
            for layer, keys in EDITABLE_TRAITS.items()
        }

        # 採択ミーム
        adopted_memes = [m for m, _ in self.memory]

        prompt = f"""
            You are a neutral and thoughtful psychiatrist.
            Your task is to make very small, evidence-based adjustments to the psychological traits of an agent,
            based on their current personality parameters and the set of meme-like statements they have recently adopted.

            The goal is to gently reflect how exposure to these ideas might shape the agent's inner tendencies,
            without introducing bias or extreme changes.

            Here is the agent's current editable personality profile (only the traits listed are allowed to be modified):
            {json.dumps(editable_layers, indent=2, ensure_ascii=False)}

            And here are the meme-like messages the agent has recently adopted:
            {json.dumps(adopted_memes, indent=2, ensure_ascii=False)}

            Rules:
            - Modify only the numeric values shown above, by no more than ±0.05
            - All final values must remain within the range [-1.0, 1.0]
            - Keep the JSON structure exactly the same (no new traits, no removals)
            - Return only a single valid JSON object, with no explanation or formatting (no markdown, no prose, no code block)
            """.strip()

        response = await query_ollama_async(prompt, model=model_name)

        try:
            match = re.search(r'{[\s\S]*}', response)
            if not match:
                print("❌ Personality update failed: no JSON detected")
                return

            updated_json = json.loads(match.group(0))

            for layer in EDITABLE_TRAITS:
                if layer in updated_json:
                    for trait in EDITABLE_TRAITS[layer]:
                        if trait in updated_json[layer]:
                            new_val = float(updated_json[layer][trait])
                            self.persona[layer][trait] = max(-1.0, min(1.0, new_val))

            self.prompt = generate_description(self.persona)

        except Exception as e:
            print("❌ Failed to parse or apply personality update:", e)
            print("=== Raw response ===")
            print(response)


    def to_json(self) -> dict:
        return {
            "id": self.id,
            "persona": self.persona,
            "prompt": self.prompt,
            "memory": [{"meme": m, "score": s} for m, s in self.memory],
            "retained_insights": self.retained_insights
        }

    @staticmethod
    def from_json(data: dict) -> 'Agent':
        agent = Agent(data["id"], data["persona"])
        agent.prompt = data.get("prompt", "")
        agent.memory = [(item["meme"], item["score"]) for item in data.get("memory", [])]
        agent.retained_insights = data.get("retained_insights", "")
        return agent