# ===================================================================
#           KakaoTalk AI Chatbot - Robust Final Version
#
#   - Author: Gemini (as a world-class AI expert coder)
#   - Architecture: Total Knowledge Ingestion (Robust & Stable)
#   - Features:
#       - Absolute pathing for file access, ensuring stability in any environment.
#       - Enhanced error logging for easier debugging.
#       - Initialization logic moved for better compatibility with Gunicorn.
# ===================================================================

import os
import pandas as pd
import threading
import requests
import json
from flask import Flask, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv

# --- ⚙️ 시스템 설정 (Configuration) ---
CHAT_MODEL = "gpt-5-nano"

# --- 환경 설정 및 클라이언트 초기화 ---
load_dotenv()
app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- 전체 지식 베이스를 저장할 전역 변수 ---
KNOWLEDGE_TEXTBOOK = ""

# ===================================================================
#      Part 1: 지식 베이스 컴파일 엔진 (수정됨)
# ===================================================================

def load_and_format_knowledge_base():
    """
    (서버 시작 시 1회 실행)
    knowledge.csv를 로드하고, AI가 이해하기 쉬운 Markdown 형식의
    거대한 텍스트 '교과서'를 생성합니다.
    """
    global KNOWLEDGE_TEXTBOOK
    try:
        # <<< CHANGED #1: 절대 경로 사용 >>>
        # app.py 파일이 있는 위치를 기준으로 knowledge.csv 파일의 절대 경로를 계산합니다.
        # 이렇게 하면 어떤 환경에서 실행되더라도 항상 정확한 위치의 파일을 찾을 수 있습니다.
        current_dir = os.path.dirname(__file__)
        csv_path = os.path.join(current_dir, 'knowledge.csv')
        
        print(f"Attempting to load knowledge base from: {csv_path}")

        # <<< CHANGED #2: 인코딩 지정 >>>
        # CSV 파일의 인코딩 문제를 방지하기 위해 'utf-8-sig'를 명시적으로 지정합니다.
        kb_dataframe = pd.read_csv(csv_path, encoding='utf-8-sig')
        
        print("✅ Knowledge base CSV file loaded successfully. Compiling into a single textbook...")

        formatted_texts = []
        for category, group in kb_dataframe.groupby('category'):
            formatted_texts.append(f"## {category}\n")
            for index, row in group.iterrows():
                formatted_texts.append(f"### {row['topic']}\n{row['content']}\n")
        
        KNOWLEDGE_TEXTBOOK = "\n".join(formatted_texts)
        
        print("✅ Knowledge textbook successfully compiled.")

    # <<< CHANGED #3: 포괄적인 오류 처리 >>>
    # FileNotFoundError 뿐만 아니라, Pandas 파싱 오류 등 모든 종류의 예외를 잡아냅니다.
    except Exception as e:
        # 어떤 종류의 오류가 발생했는지 정확히 로그에 남깁니다.
        print(f"🚨 FATAL ERROR during knowledge base initialization: {e}")
        KNOWLEDGE_TEXTBOOK = "오류: 지식 베이스 파일을 초기화하는 중 심각한 오류가 발생했습니다."


# ===================================================================
#      Part 2: AI 답변 생성 엔진 (기존과 동일)
# ===================================================================
def generate_ai_response_total_knowledge(user_message: str) -> str:
    if not KNOWLEDGE_TEXTBOOK or "오류:" in KNOWLEDGE_TEXTBOOK:
        # 사용자에게 전달되는 오류 메시지를 조금 더 구체적으로 변경합니다.
        return f"죄송합니다. 현재 챗봇의 지식 베이스에 문제가 발생하여 답변할 수 없습니다. (오류 원인: {KNOWLEDGE_TEXTBOOK})"
    
    system_instruction = f"""
    당신은 크리스찬메모리얼파크의 모든 규정과 정보를 완벽하게 암기한 최상급 AI 전문가입니다.
    (이하 프롬프트 내용은 이전과 동일)
    ---
    [크리스찬메모리얼파크 공식 지식 베이스]
    {KNOWLEDGE_TEXTBOOK}
    ---
    """
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "system", "content": system_instruction}, {"role": "user", "content": user_message}],
            temperature=0.2,
            max_completion_tokens=2000,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"🚨 OpenAI API call failed: {e}")
        return "죄송합니다. AI 답변을 생성하는 중 시스템 오류가 발생했습니다. 잠시 후 다시 시도해주세요."


# ===================================================================
#      Part 3 & 4: 콜백 처리 및 메인 서버 로직 (초기화 위치 변경)
# ===================================================================
# (process_and_send_callback 함수는 이전과 동일)
def process_and_send_callback(user_message, callback_url):
    print("Starting background processing (Total Knowledge Ingestion)...")
    ai_response_text = generate_ai_response_total_knowledge(user_message)
    final_response_data = {"version": "2.0", "template": {"outputs": [{"simpleText": {"text": ai_response_text}}]}}
    headers = {'Content-Type': 'application/json'}
    try:
        requests.post(callback_url, data=json.dumps(final_response_data), headers=headers, timeout=10)
        print("✅ Successfully sent final response via callback.")
    except requests.exceptions.RequestException as e:
        print(f"🚨 Failed to send callback to Kakao: {e}")

# (callback 함수는 이전과 동일)
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
        ai_response_text = generate_ai_response_total_knowledge(user_message)
        return jsonify({"version": "2.0", "template": {"outputs": [{"simpleText": {"text": ai_response_text}}]}})


# <<< CHANGED #4: Gunicorn 호환성을 위한 초기화 위치 변경 >>>
# if __name__ == '__main__': 블록 밖으로 초기화 함수를 이동시킵니다.
# 이렇게 하면 gunicorn이 앱을 실행할 때도 이 함수가 확실하게 호출됩니다.
initialize_knowledge_base()

if __name__ == '__main__':
    # 로컬 테스트를 위한 서버 실행
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)