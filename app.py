# ===================================================================
#           KakaoTalk AI Chatbot - Total Knowledge Ingestion Architecture
#
#   - Author: Gemini (as a world-class AI expert coder)
#   - Target Model: gpt-5-nano (as of Sep 2025)
#   - Platform: Optimized for any serverful environment (e.g., Fly.io, Render)
#   - Core Engine: Full knowledge base injection for holistic reasoning.
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
# 사용자님께서 지정하신 2025년 9월 출시 모델을 사용합니다.
CHAT_MODEL = "gpt-5-nano"

# --- 환경 설정 및 클라이언트 초기화 ---
load_dotenv()
app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- 전체 지식 베이스를 저장할 전역 변수 ---
# 이 변수에 CSV의 모든 내용이 하나의 텍스트로 저장됩니다.
KNOWLEDGE_TEXTBOOK = ""


# ===================================================================
#      Part 1: 지식 베이스 컴파일 엔진
# ===================================================================

def load_and_format_knowledge_base():
    """
    (서버 시작 시 1회 실행)
    knowledge.csv를 로드하고, AI가 이해하기 쉬운 Markdown 형식의
    거대한 텍스트 '교과서'를 생성합니다.
    """
    global KNOWLEDGE_TEXTBOOK
    try:
        kb_dataframe = pd.read_csv('knowledge.csv')
        print("✅ Knowledge base loaded. Compiling into a single textbook...")

        formatted_texts = []
        # 'category'를 기준으로 데이터를 그룹화합니다.
        for category, group in kb_dataframe.groupby('category'):
            # 대분류 제목을 추가합니다.
            formatted_texts.append(f"## {category}\n")
            
            # 각 소주제와 내용을 추가합니다.
            for index, row in group.iterrows():
                formatted_texts.append(f"### {row['topic']}\n{row['content']}\n")
        
        KNOWLEDGE_TEXTBOOK = "\n".join(formatted_texts)
        
        print("✅ Knowledge textbook successfully compiled.")
        # print(KNOWLEDGE_TEXTBOOK[:500]) # 생성된 텍스트 일부를 확인하고 싶을 때 주석 해제

    except FileNotFoundError:
        print("🚨 FATAL ERROR: knowledge.csv file not found.")
        KNOWLEDGE_TEXTBOOK = "오류: 지식 베이스 파일을 찾을 수 없습니다."


# ===================================================================
#      Part 2: AI 답변 생성 엔진 (전체 지식 주입 방식)
# ===================================================================
def generate_ai_response_total_knowledge(user_message: str) -> str:
    """
    지식 베이스 전체를 컨텍스트로 주입하여 전문가 수준의 최종 답변을 생성합니다.
    """
    if not KNOWLEDGE_TEXTBOOK or "오류:" in KNOWLEDGE_TEXTBOOK:
        return "죄송합니다. 현재 챗봇의 지식 베이스에 문제가 발생하여 답변할 수 없습니다."

    # gpt-5-nano 모델에게 매우 강력하고 명확한 지시를 내립니다.
    system_instruction = f"""
    당신은 크리스찬메모리얼파크의 모든 규정과 정보를 완벽하게 암기한 최상급 AI 전문가입니다.
    당신의 유일한 정보 출처는 아래에 제공되는 '[크리스찬메모리얼파크 공식 지식 베이스]'입니다.

    [매우 중요한 핵심 규칙]
    1.  **절대적 사실 기반:** 당신의 답변은 반드시 아래 '[크리스찬메모리얼파크 공식 지식 베이스]'에 명시된 내용에만 100% 근거해야 합니다. 당신의 사전 지식, 추측, 외부 정보는 단 한 글자도 사용해서는 안 됩니다.
    2.  **종합적 추론:** 사용자의 질문 의도를 파악하고, 지식 베이스 전체를 종합적으로 검토하여 질문과 관련된 모든 정보를 논리적으로 연결하여 하나의 완벽한 답변을 생성해야 합니다.
    3.  **정보 부재 시 대응:** 만약 지식 베이스에 사용자가 질문한 내용이 없다면, 절대로 답변을 지어내지 말고 "문의하신 내용에 대한 정보는 저희 공식 자료에 명시되어 있지 않아 정확한 안내가 어렵습니다." 라고 솔직하게 답변하십시오.
    4.  **전문가적이고 친절한 말투:** 복잡한 규정도 사용자가 이해하기 쉽도록, 전문가적이면서도 친절한 말투로 설명해야 합니다.

    ---
    [크리스찬메모리얼파크 공식 지식 베이스]
    {KNOWLEDGE_TEXTBOOK}
    ---
    """

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_message}
            ],
            temperature=0.2, # 사실 기반의 정확한 답변을 위해 온도를 매우 낮게 설정
            max_completion_tokens=2000, # 복합적인 답변을 위해 충분한 토큰 할당
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"🚨 OpenAI API call failed: {e}")
        return "죄송합니다. AI 답변을 생성하는 중 시스템 오류가 발생했습니다. 잠시 후 다시 시도해주세요."


# ===================================================================
#      Part 3: 비동기 콜백 처리 로직
# ===================================================================
def process_and_send_callback(user_message, callback_url):
    """백그라운드에서 전체 AI 로직을 실행하고 최종 답변을 카카오 서버로 전송합니다."""
    print("Starting background processing (Total Knowledge Ingestion)...")
    
    # 검색 과정 없이 바로 답변 생성 함수를 호출합니다.
    ai_response_text = generate_ai_response_total_knowledge(user_message)
    
    final_response_data = {"version": "2.0", "template": {"outputs": [{"simpleText": {"text": ai_response_text}}]}}
    headers = {'Content-Type': 'application/json'}
    try:
        requests.post(callback_url, data=json.dumps(final_response_data), headers=headers, timeout=10)
        print("✅ Successfully sent final response via callback.")
    except requests.exceptions.RequestException as e:
        print(f"🚨 Failed to send callback to Kakao: {e}")


# ===================================================================
#      Part 4: 메인 서버 애플리케이션 (Flask)
# ===================================================================
@app.route('/callback', methods=['POST'])
def callback():
    req = request.get_json()
    user_message = req['userRequest']['utterance']
    callback_url = req['userRequest'].get('callbackUrl')

    print(f"\n--- New Request Received ---")
    print(f"User Query: {user_message}")

    if callback_url:
        print(f"Callback mode enabled. Responding immediately and processing in background.")
        thread = threading.Thread(target=process_and_send_callback, args=(user_message, callback_url))
        thread.start()
        return jsonify({"version": "2.0", "useCallback": True})
    else:
        # 콜백 기능이 비활성화된 경우(테스트 등)를 위한 동기식 처리
        print("Synchronous mode enabled (no callbackUrl).")
        ai_response_text = generate_ai_response_total_knowledge(user_message)
        return jsonify({"version": "2.0", "template": {"outputs": [{"simpleText": {"text": ai_response_text}}]}})


if __name__ == '__main__':
    # 서버가 시작될 때 지식 베이스를 '교과서'로 미리 만들어 둡니다.
    load_and_format_knowledge_base()
    # 로컬 테스트 또는 서버 환경(e.g., Fly.io)을 위한 포트 설정
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)