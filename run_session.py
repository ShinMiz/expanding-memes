import asyncio
import nest_asyncio
from session import run_one_session
from utils import check_cuda
from network import create_agents_and_graph

# 外部変数として想定:
# - initial_questions
# - meme_vectors_by_category
# - facts_for_question

nest_asyncio.apply()

async def main_async(n_sessions=10):
    print("=== Phase 0: CUDAチェック ===")
    check_cuda()

    print("=== Phase 1: エージェント・ネットワーク初期化 ===")
    agents, G = create_agents_and_graph(n_agents=n_agent)

    for session_id in range(1, n_sessions + 1):
        for agent in agents:
            agent.discussion_point = 0
        await run_one_session(session_id, agents, G)

    print("\n=== ✅ 全セッション完了 ===")
    print("=== 📂 ログ保存先:", LOG_DIR.absolute())


if __name__ == "__main__":
    asyncio.run(main_async())
