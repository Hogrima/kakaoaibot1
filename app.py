# ===================================================================
#           KakaoTalk AI Chatbot - The Definitive Final Version
#
#   - Author: Gemini (as a world-class AI expert coder)
#   - Architecture: AI Semantic Search (RAG) with Asynchronous Callback
#   - Note: This code is synchronized with the semantic-search-optimized
#           knowledge.csv (category,question,answer format).
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
        
        # <<< 핵심 수정: 이제 'question' 열이 있는 CSV를 읽습니다 >>>
        kb_dataframe = pd.read_csv(csv_path)
        print("✅ Knowledge base loaded. Checking for embeddings...")

        if os.path.exists(embedding_file):
            question_embeddings = np.load(embedding_file)
            print(f"✅ Pre-computed embeddings loaded.")
        else:
            print(f"⚠️ Embeddings file not found. Generating new embeddings...")
            # 'question' 열을 사용하여 임베딩을 생성합니다.
            kb_dataframe['embedding'] = kb_dataframe['question'].apply(lambda x: get_embedding(str(x)))
            question_embeddings = np.array(kb_dataframe['embedding'].tolist())
            np.save(embedding_file, question_embeddings)
            print(f"✅ Embeddings generated and saved.")
    except Exception as e:
        # 이제 KeyError: 'question' 오류가 발생하면 여기서 잡힙니다.
        print(f"🚨 FATAL ERROR during KB initialization: {e}")
        kb_dataframe = pd.DataFrame()

def find_relevant_info_semantic(query: str) -> list[str]:
    if kb_dataframe is None or kb_dataframe.empty: return []
    query_embedding = get_embedding(query)
    similarities = [(cosine_similarity(query_embedding, doc_embedding), i) for i, doc_embedding in enumerate(question_embeddings)]
    similarities.sort(key=lambda x: x[0], reverse=True)

    final_contexts = []
    for sim, index in similarities[:MAX_CONTEXT_RESULTS]:
        if sim >= SIMILARITY_THRESHOLD:
            answer = kb_dataframe.iloc[index]['answer']
            final_contexts.append(answer)
    return final_contexts

# ===================================================================
#      Part 2: AI 답변 생성 엔진 (temperature=1 적용)
# ===================================================================
def generate_ai_response_advanced(user_message: str, contexts: list[str]) -> str:
    context_str = "\n\n---\n\n".join(contexts)
    if not contexts:
        return "죄송하지만 문의하신 내용과 관련된 정보를 찾지 못했습니다."

    system_instruction = "..." # (이전의 시맨틱 서치용 프롬프트와 동일)

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL, messages=[{"role": "system", "content": system_instruction}, {"role": "user", "content": user_message}],
            temperature=1, # <<< 사용자 요청에 따라 1로 수정 >>>
            max_completion_tokens=1500,
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