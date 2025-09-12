# ===================================================================
#           KakaoTalk AI Chatbot - Production Ready Version
#
#   - Author: Gemini (as a world-class AI expert coder)
#   - Architecture: Total Knowledge Ingestion with Live Slack Monitoring
#   - Features:
#       - Robust file loading and error handling.
#       - Asynchronous callback for seamless user experience.
#       - Real-time logging of all user/bot interactions to a Slack channel.
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
        current_dir = os.path.dirname(__file__)
        csv_path = os.path.join(current_dir, 'knowledge.csv')
        print(f"Attempting to load knowledge base from: {csv_path}")
        kb_dataframe = pd.read_csv(csv_path, encoding='utf-8-sig')
        print("✅ Knowledge base CSV file loaded successfully. Compiling into a single textbook...")
        formatted_texts = []
        for category, group in kb_dataframe.groupby('category'):
            formatted_texts.append(f"## {category}\n")
            for index, row in group.iterrows():
                formatted_texts.append(f"### {row['topic']}\n{row['content']}\n")
        KNOWLEDGE_TEXTBOOK = "\n".join(formatted_texts)
        print("✅ Knowledge textbook successfully compiled.")
    except Exception as e:
        print(f"🚨 FATAL ERROR during knowledge base initialization: {e}")
        KNOWLEDGE_TEXTBOOK = "오류: 지식 베이스 파일을 초기화하는 중 심각한 오류가 발생했습니다."


# ===================================================================
#      Part 2: AI 답변 생성 엔진
# ===================================================================
def generate_ai_response_total_knowledge(user_message: str) -> str:
    if not KNOWLEDGE_TEXTBOOK or "오류:" in KNOWLEDGE_TEXTBOOK:
        return f"죄송합니다. 현재 챗봇의 지식 베이스에 문제가 발생하여 답변할 수 없습니다. (오류 원인: {KNOWLEDGE_TEXTBOOK})"
    
    system_instruction = f"""
    당신은 '크리스찬메모리얼파크 AI 상담원'입니다. 당신의 단 하나의 임무는, 아래 제공되는 '[공식 지식 베이스]'의 내용에만 근거하여 사용자의 질문에 가장 정확하고 도움이 되는 답변을 제공하는 것입니다.

    [답변 생성 가이드라인]
    1.  **지식의 근원:** 당신의 모든 답변은 반드시 '[공식 지식 베이스]'에서만 나와야 합니다. 당신의 외부 지식이나 추측은 절대 허용되지 않습니다.
    2.  **답변의 형식:** 답변은 카카오톡의 단순한 말풍선에 표시됩니다. 따라서, 굵은 글씨(**), 목록 기호(-,*)와 같은 특수 서식(마크다운)을 절대 사용하지 말고, 오직 순수한 텍스트(Plain Text)로만 답변을 구성해야 합니다.
    3.  **답변의 분량:** 사용자가 읽기 편하도록, 답변은 핵심 내용 위주로 간결하게 요약해야 합니다. 가급적 500자 이내로 답변하는 것을 목표로 하십시오.
    4.  **정보가 없을 경우:** 만약 지식 베이스에서 질문에 대한 답을 명확히 찾을 수 없다면, "문의하신 내용에 대한 정보는 저희 공식 자료에 명시되어 있지 않아 정확한 안내가 어렵습니다. 관리사무소로 문의해 주시기 바랍니다." 라고 일관되게 답변하십시오.
    ---
    [크리스찬메모리얼파크 공식 지식 베이스]
    {KNOWLEDGE_TEXTBOOK}
    ---
    """
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "system", "content": system_instruction}, {"role": "user", "content": user_message}],
            temperature=1,
            max_completion_tokens=500,
        )
        
        sanitized_text = response.choices[0].message.content
        sanitized_text = sanitized_text.replace("**", "")
        sanitized_text = sanitized_text.replace("*", "")
        
        return sanitized_text

    except Exception as e:
        print(f"🚨 OpenAI API call failed: {e}")
        return "죄송합니다. AI 답변을 생성하는 중 시스템 오류가 발생했습니다. 잠시 후 다시 시도해주세요."


# ===================================================================
#      Part 3: 모니터링 및 콜백 처리 로직 (수정/통합됨)
# ===================================================================

def send_to_slack(message: str):
    """주어진 메시지를 슬랙 웹훅으로 보냅니다."""
    # 서버 환경 변수에서 슬랙 웹훅 URL을 가져옵니다.
    slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not slack_webhook_url:
        # 슬랙 URL이 설정되지 않았으면 조용히 종료합니다.
        return

    payload = {"text": message}
    try:
        requests.post(slack_webhook_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'}, timeout=5)
        print("✅ Slack notification sent.")
    except requests.exceptions.RequestException as e:
        # 슬랙 전송 실패가 챗봇의 핵심 기능에 영향을 주지 않도록 경고만 기록합니다.
        print(f"⚠️ Failed to send Slack notification: {e}")

def process_and_send_callback(user_message, callback_url):
    print("Starting background processing (Total Knowledge Ingestion)...")
    ai_response_text = generate_ai_response_total_knowledge(user_message)

    # 서버 로그 기록 (기본 모니터링)
    log_message = (
        f"{'='*50}\n"
        f"[AI RESPONSE PREVIEW & LOG]\n"
        f"  - User Query: {user_message}\n"
        f"  - AI Generated Answer:\n---\n{ai_response_text}\n---\n"
        f"{'='*50}"
    )
    print(log_message)

    # <<< CHANGED: 최종 답변 검증 및 폴백(Fallback) 로직 추가 >>>
    # =================================================================
    # AI가 빈 답변을 생성했는지 최종적으로 확인합니다.
    # .strip()은 공백 문자만 있는 경우도 비어있는 것으로 처리합니다.
    if not ai_response_text or not ai_response_text.strip():
        print("🚨 CRITICAL: AI returned an empty response. Sending a fallback message.")
        # AI가 답변 생성에 실패했을 때 사용자에게 보낼 표준 오류 메시지
        ai_response_text = "죄송합니다. AI가 답변을 생성하는 데 실패했습니다. 질문을 조금 더 구체적으로 해주시거나, 잠시 후 다시 시도해주세요."
    # =================================================================

    # 슬랙으로 실시간 알림 전송 (고급 모니터링)
    slack_message = f"💬 **New Chat Interaction**\n\n*User asked:*\n`{user_message}`\n\n*Bot answered:*\n```{ai_response_text}```"
    send_to_slack(slack_message)
    
    # 최종 답변을 카카오톡 서버로 전송
    final_response_data = {"version": "2.0", "template": {"outputs": [{"simpleText": {"text": ai_response_text}}]}}
    headers = {'Content-Type': 'application/json'}
    try:
        requests.post(callback_url, data=json.dumps(final_response_data), headers=headers, timeout=10)
        print("✅ Successfully sent final response via callback.")
    except requests.exceptions.RequestException as e:
        print(f"🚨 Failed to send callback to Kakao: {e}")


# ===================================================================
#      Part 4: 메인 서버 로직 (Flask)
# ===================================================================
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
        # 콜백 기능이 비활성화된 경우(테스트 등)를 위한 동기식 처리
        ai_response_text = generate_ai_response_total_knowledge(user_message)
        # 동기식 처리 시에도 로그 및 슬랙 알림을 보내고 싶다면 아래 두 줄의 주석을 해제하세요.
        # print(f"AI Response (Sync): {ai_response_text}")
        # send_to_slack(f"💬 **New Chat (Sync)**\n\n*User:* {user_message}\n\n*Bot:*\n{ai_response_text}")
        return jsonify({"version": "2.0", "template": {"outputs": [{"simpleText": {"text": ai_response_text}}]}})


# Gunicorn이 앱을 실행할 때 이 부분이 가장 먼저 실행됩니다.
load_and_format_knowledge_base()

if __name__ == '__main__':
    # 로컬 테스트를 위한 서버 실행
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)