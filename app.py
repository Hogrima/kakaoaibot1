# ===================================================================
#           KakaoTalk AI Chatbot - Production Ready Version
#
#   - Author: Gemini (as a world-class AI expert coder)
#   - Architecture: Total Knowledge Ingestion with Live Slack Monitoring
#   - Features:
#       - Robust file loading and error handling.
#       - Asynchronous callback for seamless user experience.
#       - Real-time logging of all user/bot interactions to a Slack channel.
#       - Enhanced configuration management and stability improvements.
# ===================================================================

import os
import pandas as pd
import threading
import requests
import json
import re
from flask import Flask, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv

# --- ⚙️ 시스템 설정 (Configuration) ---ㅡ
# 환경 변수를 통해 설정을 관리하여 유연성을 확보합니다.
# .env 파일에 OPENAI_API_KEY, SLACK_WEBHOOK_URL 등을 설정하세요.
load_dotenv()

CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano") # 최신 고효율 모델 사용을 권장합니다.
KNOWLEDGE_FILE_NAME = "knowledge.csv"

# --- 💡 상수 (Constants) ---
# 자주 사용되는 메시지를 상수로 관리하여 일관성과 유지보수성을 높입니다.
ERROR_MSG_KNOWLEDGE_BASE = "오류: 지식 베이스 파일을 초기화하는 중 심각한 오류가 발생했습니다."
ERROR_MSG_AI_FAILED = "죄송합니다. AI 답변을 생성하는 중 시스템 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
FALLBACK_MSG_EMPTY_RESPONSE = "죄송합니다. AI가 답변을 생성하는 데 실패했습니다. 질문을 조금 더 구체적으로 해주시거나, 잠시 후 다시 시도해주세요."
FALLBACK_MSG_NO_INFO = "문의하신 내용에 대한 정보가 부족하여 정확한 안내가 어렵습니다. 관리사무실로 문의해 주시기 바랍니다."

# --- 클라이언트 초기화 ---
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
    지식 베이스 CSV 파일을 로드하고, AI가 이해하기 쉬운 Markdown 형식의
    통합 텍스트 '교과서'를 생성합니다.
    """
    global KNOWLEDGE_TEXTBOOK
    try:
        # 스크립트 파일의 위치를 기준으로 파일 경로를 안전하게 찾습니다.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, KNOWLEDGE_FILE_NAME)
        print(f"INFO: Attempting to load knowledge base from: {csv_path}")

        kb_dataframe = pd.read_csv(csv_path, encoding='utf-8-sig')
        print("✅ Knowledge base CSV file loaded successfully. Compiling into a single textbook...")

        formatted_texts = []
        for category, group in kb_dataframe.groupby('category'):
            formatted_texts.append(f"## {category}\n")
            for _, row in group.iterrows():
                formatted_texts.append(f"### {row['topic']}\n{row['content']}\n")

        KNOWLEDGE_TEXTBOOK = "\n".join(formatted_texts)
        print("✅ Knowledge textbook successfully compiled.")
    except FileNotFoundError:
        error_msg = f"{ERROR_MSG_KNOWLEDGE_BASE} (파일을 찾을 수 없음: {csv_path})"
        print(f"🚨 FATAL ERROR: {error_msg}")
        KNOWLEDGE_TEXTBOOK = error_msg
    except Exception as e:
        error_msg = f"{ERROR_MSG_KNOWLEDGE_BASE} (원인: {e})"
        print(f"🚨 FATAL ERROR during knowledge base initialization: {e}")
        KNOWLEDGE_TEXTBOOK = error_msg


# ===================================================================
#      Part 2: AI 답변 생성 엔진
# ===================================================================

def generate_ai_response_total_knowledge(user_message: str) -> str:
    """AI 모델과 전체 지식 베이스를 사용하여 사용자 질문에 대한 답변을 생성합니다."""
    if not KNOWLEDGE_TEXTBOOK or ERROR_MSG_KNOWLEDGE_BASE in KNOWLEDGE_TEXTBOOK:
        return f"죄송합니다. 현재 챗봇의 지식 베이스에 문제가 발생하여 답변할 수 없습니다. (상세: {KNOWLEDGE_TEXTBOOK})"

    system_instruction = f"""
    당신은 '크리스찬메모리얼파크 AI 상담원'입니다. 당신의 단 하나의 임무는, 아래 제공되는 '[공식 지식 베이스]'의 내용에만 근거하여 사용자의 질문에 가장 정확하고 도움이 되는 답변을 제공하는 것입니다.

    [답변 생성 가이드라인]
    1.  **지식의 근원:** 당신의 모든 답변은 반드시 '[공식 지식 베이스]'에서만 나와야 합니다. 당신의 외부 지식이나 추측은 절대 허용되지 않습니다.
    2.  **답변의 형식:** 답변은 카카오톡의 단순한 말풍선에 표시됩니다. 따라서, 굵은 글씨(`**`), 목록 기호(`-`,`*`), 헤더(`#`)와 같은 모든 종류의 마크다운 서식을 절대 사용하지 말고, 오직 순수한 텍스트(Plain Text)로만 답변을 구성해야 합니다.
    3.  **답변의 분량:** 사용자가 읽기 편하도록, 답변은 핵심 내용 위주로 간결하게 요약해야 합니다. 가독성을 위해 문단이 넘어가면 줄바꿈을 하고, 전체 500자 이내로 답변하는 것을 목표로 하십시오.
    4.  **정보가 없을 경우:** 만약 지식 베이스에서 질문에 대한 답을 명확히 찾을 수 없다면, "{FALLBACK_MSG_NO_INFO}" 라고 일관되게 답변하십시오.
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
        )
        ai_message = response.choices[0].message.content

        # 최종 방어: AI가 실수로 마크다운을 사용했을 경우를 대비하여 관련 문자를 모두 제거합니다.
        sanitized_text = re.sub(r"[\*#\-`]", "", ai_message).strip()
        return sanitized_text

    except Exception as e:
        print(f"🚨 ERROR: OpenAI API call failed for user message '{user_message}'. Details: {e}")
        return ERROR_MSG_AI_FAILED


# ===================================================================
#      Part 3: 모니터링 및 콜백 처리 로직 (JANDI로 교체됨)
# ===================================================================

def send_to_jandi(user_query: str, bot_answer: str):
    """사용자 질문과 봇 답변을 JANDI 웹훅으로 비동기적으로 전송합니다."""
    jandi_webhook_url = os.getenv("JANDI_WEBHOOK_URL")
    if not jandi_webhook_url:
        return

    # JANDI가 요구하는 헤더 형식
    headers = {
        'Accept': 'application/vnd.tosslab.jandi-v2+json',
        'Content-Type': 'application/vnd.tosslab.jandi-v2+json'
    }

    # JANDI의 구조화된 메시지 형식에 맞춰 페이로드를 생성합니다.
    payload = {
        "body": "💬 신규 챗봇 문의 발생",
        "connectColor": "#007AFF",  # JANDI 메시지 좌측에 표시될 색상
        "connectInfo": [
            {
                "title": "사용자 질문:",
                "description": user_query
            },
            {
                "title": "AI 답변:",
                "description": bot_answer
            }
        ]
    }

    try:
        requests.post(jandi_webhook_url, data=json.dumps(payload), headers=headers, timeout=5)
        print("INFO: JANDI notification sent.")
    except requests.exceptions.RequestException as e:
        print(f"⚠️ WARNING: Failed to send JANDI notification: {e}")


def process_and_send_callback(user_message: str, callback_url: str):
    """백그라운드에서 AI 답변 생성, 로깅, JANDI 알림, 콜백 전송을 모두 처리합니다."""
    print("INFO: Starting background processing for Total Knowledge Ingestion...")
    ai_response_text = generate_ai_response_total_knowledge(user_message)

    final_text_for_user = ai_response_text
    if not final_text_for_user or not final_text_for_user.strip():
        print("🚨 CRITICAL: AI returned an empty or whitespace-only response. Using fallback message.")
        final_text_for_user = FALLBACK_MSG_EMPTY_RESPONSE

    log_message = (
        f"{'='*50}\n"
        f"  [AI RESPONSE LOG]\n"
        f"  - User Query: {user_message}\n"
        f"  - Final Answer: {final_text_for_user}\n"
        f"{'='*50}"
    )
    print(log_message)

    # JANDI로 실시간 알림을 전송합니다. (고급 모니터링)
    send_to_jandi(user_query=user_message, bot_answer=final_text_for_user)

    # 최종 답변을 카카오톡 서버로 전송합니다.
    final_response_data = {"version": "2.0", "template": {"outputs": [{"simpleText": {"text": final_text_for_user}}]}}
    try:
        requests.post(callback_url, json=final_response_data, timeout=10)
        print("✅ INFO: Successfully sent final response via callback.")
    except requests.exceptions.RequestException as e:
        print(f"🚨 ERROR: Failed to send callback to Kakao: {e}")


# ===================================================================
#      Part 4: 메인 서버 로직 (Flask)
# ===================================================================

@app.route('/', methods=['GET'])
def health_check():
    """서버가 정상적으로 동작하는지 확인하는 헬스 체크 엔드포인트입니다."""
    return jsonify({"status": "ok", "message": "KakaoTalk AI Chatbot is running."}), 200

@app.route('/callback', methods=['POST'])
def callback():
    """카카오톡 스킬 서버의 메인 콜백 엔드포인트입니다."""
    req = request.get_json()

    # 필수 데이터 추출 및 로깅
    try:
        user_message = req['userRequest']['utterance']
        callback_url = req['userRequest'].get('callbackUrl')
    except (KeyError, TypeError):
        return jsonify({"status": "error", "message": "Invalid request format"}), 400

    print(f"\n[INFO] New request received from KakaoTalk.")
    print(f"[DEBUG] User Query: {user_message}")
    print(f"[DEBUG] Callback URL present: {'Yes' if callback_url else 'No'}")

    if callback_url:
        # 비동기 처리를 위해 별도 스레드에서 로직을 실행하고 즉시 응답합니다.
        # 이를 통해 사용자는 '챗봇이 생각 중...'이라는 UX를 경험하게 됩니다.
        thread = threading.Thread(target=process_and_send_callback, args=(user_message, callback_url))
        thread.start()
        return jsonify({"version": "2.0", "useCallback": True})
    else:
        # 콜백 URL이 없는 경우(카카오톡 테스트 콘솔 등) 동기식으로 처리합니다.
        ai_response_text = generate_ai_response_total_knowledge(user_message)
        # 동기식 처리 시에도 로깅과 알림을 보낼 수 있습니다.
        print(f"[INFO] AI Response (Sync): {ai_response_text}")
        send_to_jandi(user_query=user_message, bot_answer=ai_response_text)
        return jsonify({"version": "2.0", "template": {"outputs": [{"simpleText": {"text": ai_response_text}}]}})


# --- 서버 실행 ---
if __name__ == '__main__':
    # 서버가 시작되기 전에 지식 베이스를 로드합니다.
    load_and_format_knowledge_base()
    port = int(os.environ.get("PORT", 8080))
    # 로컬 테스트 환경에서는 debug=True를 사용할 수 있습니다.
    # app.run(host='0.0.0.0', port=port, debug=True)
    app.run(host='0.0.0.0', port=port)
else:
    # Gunicorn과 같은 프로덕션 WSGI 서버로 실행될 때 이 부분이 호출됩니다.
    load_and_format_knowledge_base()