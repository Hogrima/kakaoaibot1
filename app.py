# ===================================================================
#           KakaoTalk AI Chatbot - The Final & Optimal Architecture
#
#   - Author: Gemini (as a world-class AI expert coder)
#   - Architecture: AI Semantic Search (RAG) with Asynchronous Callback
#   - Reason: This is the industry-standard, stable, and scalable solution
#             that resolves the fatal context window limit issue.
# ===================================================================

import os
import pandas as pd
import numpy as np
from numpy.linalg import norm
import threading
import requests
import json
from flask import Flask, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv

# --- ⚙️ 시스템 설정 (Configuration) ---
CHAT_MODEL = "gpt-5-nano" # (실제로는 gpt-4o 등으로 작동)
EMBEDDING_MODEL = "text-embedding-3-small"
MAX_CONTEXT_RESULTS = 3
SIMILARITY_THRESHOLD = 0.75

# --- 환경 설정 및 클라이언트 초기화 ---
load_dotenv()
app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- AI 검색을 위한 전역 변수 ---
question_embeddings = None
kb_dataframe = None

# ===================================================================
#      Part 1: AI 시맨틱 서치 엔진
# ===================================================================
def get_embedding(text, model=EMBEDDING_MODEL):
   text = text.replace("\n", " ")
   return client.embeddings.create(input=[text], model=model).data[0].embedding

def cosine_similarity(A, B):
    return np.dot(A, B) / (norm(A) * norm(B))

def initialize_knowledge_base():
    global kb_dataframe, question_embeddings
    try:
        current_dir = os.path.dirname(__file__)
        csv_path = os.path.join(current_dir, 'knowledge.csv')
        embedding_file = os.path.join(current_dir, 'question_embeddings.npy')
        
        kb_dataframe = pd.read_csv(csv_path)
        print("✅ Knowledge base loaded. Checking for embeddings...")

        if os.path.exists(embedding_file):
            question_embeddings = np.load(embedding_file)
            print(f"✅ Pre-computed embeddings loaded.")
        else:
            print(f"⚠️ Embeddings file not found. Generating new embeddings...")
            kb_dataframe['embedding'] = kb_dataframe['question'].apply(lambda x: get_embedding(x))
            question_embeddings = np.array(kb_dataframe['embedding'].tolist())
            np.save(embedding_file, question_embeddings)
            print(f"✅ Embeddings generated and saved.")
    except Exception as e:
        print(f"🚨 FATAL ERROR during KB initialization: {e}")
        kb_dataframe = pd.DataFrame()

def find_relevant_info_semantic(query: str) -> list[str]:
    if kb_dataframe is None or kb_dataframe.empty: return []
    query_embedding = get_embedding(query)
    similarities = [(cosine_similarity(query_embedding, doc_embedding), i) for i, doc_embedding in enumerate(question_embeddings)]
    similarities.sort(key=lambda x: x[0], reverse=True)

    final_contexts = []
    print(f"Semantic search results for query: '{query}'")
    for sim, index in similarities[:MAX_CONTEXT_RESULTS]:
        if sim >= SIMILARITY_THRESHOLD:
            answer = kb_dataframe.iloc[index]['answer']
            final_contexts.append(answer)
            question = kb_dataframe.iloc[index]['question']
            print(f"  - Match (Score: {sim:.4f}): '{question[:50]}...'")
    return final_contexts

# ===================================================================
#      Part 2: AI 답변 생성 엔진
# ===================================================================
def generate_ai_response_advanced(user_message: str, contexts: list[str]) -> str:
    context_str = "\n\n---\n\n".join(contexts)
    if not contexts:
        return "죄송하지만 문의하신 내용과 관련된 정보를 찾지 못했습니다. 조금 더 구체적으로 질문해주시면 감사하겠습니다."

    system_instruction = f"""
    당신은 크리스찬메모리얼파크의 모든 지식을 완벽하게 숙지한 최상급 AI 전문가입니다. 당신의 임무는 사용자의 어떤 질문에 대해서도, 아래 '참고 자료'에 근거하여 명확하고 친절하며 완벽한 답변을 제공하는 것입니다.
    [핵심 규칙]
    1. **절대적 사실 기반:** 답변은 반드시 '참고 자료'의 내용으로만 구성해야 합니다. 당신의 사전 지식이나 추측을 절대 사용하지 마십시오.
    2. **종합적 분석:** '참고 자료' 목록을 종합하여 사용자의 질문에 대한 답변을 논리적으로 구성하십시오.
    3. **정보 부족 시 솔직함:** '참고 자료'에 사용자가 질문한 내용이 없다면, "문의하신 내용에 대한 정보는 제가 가진 자료에 없어 정확한 안내가 어렵습니다." 라고 솔직하게 답변하십시오.
    ---
    [참고 자료 묶음]
    {context_str}
    ---
    """
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL, messages=[{"role": "system", "content": system_instruction}, {"role": "user", "content": user_message}],
            temperature=0.3, max_completion_tokens=1500,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"🚨 OpenAI API call failed: {e}")
        return "죄송합니다. AI 답변을 생성하는 중 시스템 오류가 발생했습니다."

# ===================================================================
#      Part 3 & 4: 콜백 처리 및 메인 서버 로직
# ===================================================================
def process_and_send_callback(user_message, callback_url):
    print("Starting background processing (Semantic Search)...")
    contexts = find_relevant_info_semantic(user_message)
    ai_response_text = generate_ai_response_advanced(user_message, contexts)
    final_response_data = {"version": "2.0", "template": {"outputs": [{"simpleText": {"text": ai_response_text}}]}}
    headers = {'Content-Type': 'application/json'}
    try:
        requests.post(callback_url, data=json.dumps(final_response_data), headers=headers, timeout=10)
        print("✅ Successfully sent final response via callback.")
    except requests.exceptions.RequestException as e:
        print(f"🚨 Failed to send callback to Kakao: {e}")

@app.route('/callback', methods=['POST'])
def callback():
    req = request.get_json()
    user_message = req['userRequest']['utterance']
    callback_url = req['userRequest'].get('callbackUrl')
    print(f"\n--- New Request Received ---")
    print(f"User Query: {user_message}")
    if callback_url:
        thread = threading.Thread(target=process_and_send_callback, args=(user_message, callback_url))
        thread.start()
        return jsonify({"version": "2.0", "useCallback": True})
    else:
        contexts = find_relevant_info_semantic(user_message)
        ai_response_text = generate_ai_response_advanced(user_message, contexts)
        return jsonify({"version": "2.0", "template": {"outputs": [{"simpleText": {"text": ai_response_text}}]}})

# Gunicorn 호환성을 위한 초기화 위치
initialize_knowledge_base()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)