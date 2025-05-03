import aiohttp
import asyncio
import hashlib
import random
import datetime
import json
import json5
import re

ollama_cache = {}
sem = asyncio.Semaphore(1)  # 最大同時に1つまで実行（変更可）


def prompt_noise() -> str:
    noise_options = [
        f"<!-- UUID:{str(uuid.uuid4())[:8]} -->",
        f"<!-- TS:{datetime.datetime.now().isoformat()} -->",
        f"<!-- hash_seed:{random.randint(1000, 9999)} -->",
        f"<!-- model_hint:{random.choice(['v1', 'v2', 'beta', 'edge'])} -->",
        f"<!-- generation_note:{random.choice(['variation A', 'variation B', 'exploratory', 'no bias'])} -->",
        f"<!-- entropy_boost:{random.random():.4f} -->"
    ]
    # ランダムに1〜2個選んで結合（順番もシャッフル）
    selected = random.sample(noise_options, k=random.randint(1, 2))
    random.shuffle(selected)
    return "\n" + "\n".join(selected)



def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()

reaction_cache = {}

def hash_reaction_key(persona_prompt: str, meme_text: str) -> str:
    return hashlib.sha256(f"{persona_prompt}|{meme_text}".encode()).hexdigest()

REQUIRED_KEYS = {
    "empathic_resonance", "fear_susceptibility", "anger_provocation",
    "joy_inducibility", "skepticism", "cognitive_fluency", "novelty_seeking",
    "confirmation_bias_intensity", "conformity_susceptibility", "authority_acceptance",
    "contrarian_tendency", "social_proof_dependency", "propagation_urge",
    "self_expression_need", "action_orientation", "retention_resistance"
}

def fix_json_like_text(text: str) -> str:
    """
    LLMから壊れた形式で返ってきた"擬似JSON"を補正して、
    有効なJSONとして構成し直す。
    """
    cleaned = {}
    for line in text.strip().splitlines():
        if ":" not in line:
            continue
        try:
            key, val = line.split(":", 1)
            key = key.strip().strip('"\'- ')
            key = key.replace(" ", "_").replace("/", "_").lower()

            # ()内コメントを除去
            val = re.sub(r'\([^)]*\)', '', val).strip()

            # 値が不正な場合は強制 0.0
            if "n/a" in val.lower() or val.strip() == "":
                val = "0.0"
            val = val.strip().rstrip(',')

            # 数値としてキャスト可能なものだけ
            try:
                num = float(val)
            except ValueError:
                num = 0.0

            # キーがREQUIRED_KEYSに近いものだけ拾う（厳格）
            if key in REQUIRED_KEYS:
                cleaned[key] = num
        except Exception:
            continue
    return json.dumps(cleaned, indent=2)

# グローバルで定義（ファイル冒頭などに置く）
sem = asyncio.Semaphore(1)  # 最大同時に3つまで実行

# ラッパー関数を定義（別名でもよい）
async def limited_generate_reaction_profile(persona_prompt: str, meme_text: str, model_name: str = 'dolphin-mistral'):
    async with sem:
        return await generate_reaction_profile_llm_async(persona_prompt, meme_text, model_name)


ollama_cache = {}

# 実際のリクエスト実行を行う内部関数（システムプロンプト込み）
async def query_ollama_async_(prompt: str, model: str = "dolphin-mistral", temperature: float = 1.0) -> str:
    #prompt = prompt + "\n" + prompt_noise(agent.id)
    prompt = prompt + prompt_noise()
    key = hash_prompt(prompt + str(temperature))
    if key in ollama_cache:
        return ollama_cache[key]

    system_prompt =  query_ollama_async_system_promopt()

    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model,
                "system": system_prompt,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "stream": False
            }
        ) as resp:
            try:
                data = await resp.json()
                content = data.get("message", {}).get("content", "")
                print("=== Raw LLM Content ===")
                print(content)
                ollama_cache[key] = content
                return content
            except Exception as e:
                print("❌ Failed to parse JSON from ollama:")
                text = await resp.text()
                print("=== Raw response text ===")
                print(text)
                raise e

# 外部から使うときはこちら：非同期セマフォ付き＋ラッパー
async def query_ollama_async(prompt: str, model: str = "dolphin-mistral", temperature: float = 0.7) -> str:
    async with sem:
        return await query_ollama_async_(prompt, model=model, temperature=temperature)


async def generate_reaction_profile_llm_async(
    persona_prompt: str,
    meme_text: str,
    model_name: str = 'dolphin-mistral'
) -> dict | None:
    key = hash_reaction_key(persona_prompt, meme_text)
    if key in reaction_cache:
        return reaction_cache[key]

    REQUIRED_KEYS = {
        "empathic_resonance", "fear_susceptibility", "anger_provocation",
        "joy_inducibility", "skepticism", "cognitive_fluency", "novelty_seeking",
        "confirmation_bias_intensity", "conformity_susceptibility", "authority_acceptance",
        "contrarian_tendency", "social_proof_dependency", "propagation_urge",
        "self_expression_need", "action_orientation", "retention_resistance"
    }

    base_system_prompt = generate_reaction_profile_llm_asyn_prompt_system()

    base_user_prompt = generate_reaction_profile_llm_asyn_prompt_user.strip()

    for attempt in range(3):
        # 🛠 retry時にはプロンプトをさらに強調して矯正
        system_prompt = base_system_prompt
        user_prompt = base_user_prompt
        if attempt > 0:
            system_prompt += " If you previously failed, this time you MUST ONLY output valid JSON with no additional text."
            user_prompt += "\nThis is a retry. You MUST strictly follow the JSON-only format. Do NOT add explanation."

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=90)) as session:
                async with session.post(
                    "http://localhost:11434/api/chat",
                    json={
                        "model": model_name,
                        "system": system_prompt,
                        "messages": [{"role": "user", "content": user_prompt}],
                        "stream": False
                    }
                ) as resp:
                    data = await resp.json()
                    content = data.get("message", {}).get("content", "").strip()

                    print("=== Full LLM Response ===")
                    print(json.dumps(data, ensure_ascii=False, indent=2))

                    if not content:
                        print(f"⚠️ Attempt {attempt+1}: Empty content received.")
                        continue

                    if any(w in content.lower() for w in ["i'm sorry", "i cannot", "not allowed", "拒否"]):
                        print(f"❌ Attempt {attempt+1}: LLM refused to answer.")
                        return None

                    if not content.startswith("{"):
                        if all('"' in line and ":" in line for line in content.splitlines()):
                            content = "{\n" + content.strip().rstrip(',') + "\n}"

                    match = re.search(r'\{[\s\S]*?\}', content)
                    json_str = match.group(0).strip() if match else content

                    try:
                        profile = json.loads(json_str)
                    except json.JSONDecodeError:
                        print(f"⚠️ Attempt {attempt+1}: Standard JSON parsing failed. Trying fix...")
                        fixed = fix_json_like_text(content)
                        try:
                            profile = json.loads(fixed)
                        except json.JSONDecodeError:
                            try:
                                import json5
                                profile = json5.loads(fixed)
                            except Exception:
                                print(f"❌ Attempt {attempt+1}: Final fallback failed. JSON irrecoverable.")
                                traceback.print_exc()
                                continue

                    # 🔎 key validation
                    if not isinstance(profile, dict) or not REQUIRED_KEYS.issubset(profile):
                        if profile == {}:
                            print(f"⚠️ Attempt {attempt+1}: Empty JSON object — LLM likely failed silently.")
                        else:
                            missing = REQUIRED_KEYS - profile.keys()
                            print(f"⚠️ Attempt {attempt+1}: Parsed but missing keys: {missing}")
                        continue  # next attempt
                    else:
                        reaction_cache[key] = profile
                        return profile

        except Exception:
            print(f"❌ Attempt {attempt+1}: Unexpected error during LLM call.")
            traceback.print_exc()
            continue

    print("❌ Failed to obtain valid JSON after 3 attempts.")
    return None

async def generate_reaction_profile_llm_async(persona_prompt: str, meme_text: str, model_name: str = 'dolphin-mistral') -> dict | None:
    key = hash_reaction_key(persona_prompt, meme_text)
    if key in reaction_cache:
        return reaction_cache[key]

    system_prompt = generate_reaction_profile_llm_asyn_prompt_system()
    user_prompt = generate_reaction_profile_llm_asyn_prompt_user.strip()

    for attempt in range(3):
        if attempt > 0:
            system_prompt += " Retry with STRICT JSON format."
            user_prompt += "\nRETRY. Output ONLY valid JSON, nothing else."

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=90)) as session:
                async with session.post(
                    "http://localhost:11434/api/chat",
                    json={
                        "model": model_name,
                        "system": system_prompt,
                        "messages": [{"role": "user", "content": user_prompt}],
                        "stream": False
                    }
                ) as resp:
                    data = await resp.json()
                    content = data.get("message", {}).get("content", "").strip()

                    if not content:
                        continue
                    if any(w in content.lower() for w in ["i'm sorry", "i cannot", "not allowed"]):
                        return None

                    if not content.startswith("{"):
                        if all('"' in line and ":" in line for line in content.splitlines()):
                            content = "{\n" + content.strip().rstrip(',') + "\n}"

                    match = re.search(r'\{[\s\S]*?\}', content)
                    json_str = match.group(0).strip() if match else content

                    try:
                        profile = json.loads(json_str)
                    except json.JSONDecodeError:
                        fixed = fix_json_like_text(content)
                        try:
                            profile = json.loads(fixed)
                        except json.JSONDecodeError:
                            try:
                                profile = json5.loads(fixed)
                            except Exception:
                                continue

                    if not isinstance(profile, dict) or not REQUIRED_KEYS.issubset(profile):
                        continue

                    reaction_cache[key] = profile
                    return profile
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            continue

    return None

