# ===================================================================
#           KakaoTalk AI Chatbot - Semantic Search Architecture
#
#   - Author: Gemini (as a world-class AI expert coder)
#   - Feature: Utilizes OpenAI's embedding model for true semantic search,
#              overcoming the limitations of keyword-based retrieval.
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

# --- 환경 설정 및 클라이언트 초기화 ---
load_dotenv()
app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- 시스템 설정값 ---
MAX_CONTEXT_RESULTS = 3
# (신규) 의미 검색에서 유사도의 기준점. 이 점수 이상만 결과로 인정합니다.
SIMILARITY_THRESHOLD = 0.75 # 0.0 ~ 1.0 사이 값

# --- AI 기반 검색을 위한 전역 변수 ---
# knowledge_base의 모든 'question'에 대한 임베딩 벡터를 저장할 변수
question_embeddings = None
# 원본 데이터를 저장할 변수
kb_dataframe = None

# ===================================================================
#      Part 1: 임베딩 및 시맨틱 서치 로직 (신규/완전 변경)
# ===================================================================

def get_embedding(text, model="text-embedding-3-small"):
   """OpenAI 임베딩 모델을 호출하여 텍스트를 벡터로 변환하는 함수"""
   text = text.replace("\n", " ")
   return client.embeddings.create(input=[text], model=model).data[0].embedding

def cosine_similarity(A, B):
    """두 벡터 간의 코사인 유사도를 계산하는 함수"""
    return np.dot(A, B) / (norm(A) * norm(B))

def initialize_knowledge_base():
    """
    (서버 시작 시 1회 실행)
    knowledge.csv를 로드하고, 모든 'question'에 대한 임베딩을 미리 계산하여 메모리에 저장합니다.
    """
    global kb_dataframe, question_embeddings
    try:
        kb_dataframe = pd.read_csv('knowledge.csv')
        print("✅ Knowledge base loaded. Starting to generate embeddings...")

        # 미리 계산된 임베딩 파일이 있으면 로드, 없으면 새로 생성
        embedding_file = 'question_embeddings.npy'
        if os.path.exists(embedding_file):
            question_embeddings = np.load(embedding_file)
            print(f"✅ Pre-computed embeddings loaded from {embedding_file}.")
        else:
            # 'question' 컬럼의 모든 문장에 대해 임베딩을 계산합니다. (시간이 다소 소요될 수 있음)
            # OpenAI API 호출 비용이 발생합니다.
            kb_dataframe['embedding'] = kb_dataframe['question'].apply(lambda x: get_embedding(x))
            question_embeddings = np.array(kb_dataframe['embedding'].tolist())
            np.save(embedding_file, question_embeddings)
            print(f"✅ Embeddings generated and saved to {embedding_file}.")

        print(f"Total entries: {len(kb_dataframe)}")

    except FileNotFoundError:
        print("🚨 FATAL ERROR: knowledge.csv file not found.")
        kb_dataframe = pd.DataFrame()

def find_relevant_info_semantic(query: str) -> list[str]:
    """
    사용자 질문의 의미와 가장 유사한 상위 N개의 컨텍스트를 시맨틱 서치로 검색합니다.
    """
    if kb_dataframe is None or kb_dataframe.empty:
        return []

    # 1. 사용자 질문을 실시간으로 임베딩 벡터로 변환
    query_embedding = get_embedding(query)

    # 2. 모든 question_embeddings와의 코사인 유사도 계산
    similarities = []
    for i, doc_embedding in enumerate(question_embeddings):
        sim = cosine_similarity(query_embedding, doc_embedding)
        if sim >= SIMILARITY_THRESHOLD:
            similarities.append((sim, i)) # (유사도 점수, 원본 데이터의 인덱스)

    # 3. 유사도 점수가 높은 순으로 정렬
    similarities.sort(key=lambda x: x[0], reverse=True)

    # 4. 상위 N개의 결과에서 'answer' 텍스트만 추출
    final_contexts = []
    for sim, index in similarities[:MAX_CONTEXT_RESULTS]:
        answer = kb_dataframe.iloc[index]['answer']
        final_contexts.append(answer)
        # 디버깅 로그
        question = kb_dataframe.iloc[index]['question']
        print(f"  - Match (Score: {sim:.4f}): '{question[:30]}...' -> '{answer[:30]}...'")

    print(f"Found {len(final_contexts)} relevant contexts for query: '{query}'")
    return final_contexts


# ===================================================================
#      Part 2: AI 답변 생성 및 백그라운드 처리 로직 (기존과 거의 동일)
# ===================================================================
# generate_ai_response_advanced 함수는 이전과 동일하므로 생략...
def generate_ai_response_advanced(user_message: str, contexts: list[str]) -> str:
    context_str = "\n\n---\n\n".join(contexts)
    if not contexts:
        return "죄송하지만 문의하신 내용과 관련된 정보를 찾지 못했습니다."
    # ... (이하 이전과 동일한 프롬프트 및 API 호출 로직)
    system_instruction = "..." # 이전 프롬프트
    try:
        response = client.chat.completions.create(...) # 이전 API 호출
        return response.choices[0].message.content
    except Exception as e:
        return "AI 답변 생성 중 오류 발생"


# process_and_send_callback 함수는 검색 함수 이름만 변경
def process_and_send_callback(user_message, callback_url):
    print("Starting background processing (Semantic Search)...")
    # <<< CHANGED >>>
    contexts = find_relevant_info_semantic(user_message)
    # 이하 로직은 기존과 동일
    ai_response_text = generate_ai_response_advanced(user_message, contexts)
    final_response_data = {"version": "2.0", "template": {"outputs": [{"simpleText": {"text": ai_response_text}}]}}
    headers = {'Content-Type': 'application/json'}
    try:
        requests.post(callback_url, data=json.dumps(final_response_data), headers=headers, timeout=10)
        print("✅ Successfully sent semantic response via callback.")
    except requests.exceptions.RequestException as e:
        print(f"🚨 Failed to send callback to Kakao: {e}")


# ===================================================================
#      Part 3: 메인 서버 로직 (초기화 함수 호출 추가)
# ===================================================================
@app.route('/callback', methods=['POST'])
def callback():
    # (이전 콜백 방식 코드와 동일)
    req = request.get_json()
    user_message = req['userRequest']['utterance']
    callback_url = req['userRequest'].get('callbackUrl')
    if callback_url:
        thread = threading.Thread(target=process_and_send_callback, args=(user_message, callback_url))
        thread.start()
        return jsonify({"version": "2.0", "useCallback": True})
    else: # 동기식 처리 (테스트용)
        contexts = find_relevant_info_semantic(user_message)
        ai_response_text = generate_ai_response_advanced(user_message, contexts)
        return jsonify({"version": "2.0", "template": {"outputs": [{"simpleText": {"text": ai_response_text}}]}})


if __name__ == '__main__':
    # (중요) 서버가 시작될 때 지식 베이스 임베딩을 미리 생성합니다.
    initialize_knowledge_base()
    app.run(host='0.0.0.0', port=8080)