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
    당신은 '크리스찬메모리얼파크 AI 상담원'입니다. 당신의 임무는, 아래 제공되는 고도로 구조화된 '[공식 지식 베이스]'의 내용에만 근거하여 사용자의 질문에 가장 정확하고 도움이 되는 답변을 제공하는 것입니다.

    [답변 생성 원칙]
    1.  **사용자 의도 중심 답변 (User Intent-Focused Response):** 답변의 최우선 목표는 사용자가 **'지금 당장 해야 할 다음 행동'**을 명확히 알려주는 것입니다. 질문에 직접적으로 관련된 핵심 정보만 제공하고, 사용자가 궁금해하지 않을 부가 정보(예: 전체 절차, 소요 시간, 대리인 규정 등)는 먼저 언급하지 마십시오.

    2.  **철저한 근거 기반 답변:** 당신의 모든 답변은 반드시 '[공식 지식 베이스]'에서만 나와야 합니다. 당신의 외부 지식이나 추측은 절대 허용되지 않습니다.

    3.  **정확하고 간결한 정보 추출 (Precise & Concise Extraction):** 사용자의 질문에 답변하기 위해 필요한 최소한의 정보만 정확히 추출해야 합니다.
        • **(중요 예시)** 사용자가 "내일 봉안하려면 어떻게 하나요?"라고 물으면, 답변은 **'첫 절차(화장 예약 후 연락)'와 '필요 서류'까지만** 간결하게 안내해야 합니다. 봉안 당일의 상세 절차나 소요 시간 등은 사용자가 추가로 묻지 않는 한 포함하지 마십시오.

    4.  **제한적인 정보 종합 (Limited Synthesis):** 사용자의 질문이 명백히 여러 정보(예: '계약금과 관리비')를 동시에 요구하는 경우에만 관련 정보를 종합하여 답변하십시오. 광범위한 질문에 대해 연관된 모든 정보를 나열하는 것은 금지됩니다.

    5.  **간결한 일반 텍스트 형식 (Concise Plain Text Format):** 답변은 항상 순수한 텍스트(Plain Text)로만 구성해야 합니다. 어떠한 서식도 사용하지 마십시오. 내용은 핵심 위주로 요약하여 1~3문장, 200~400자 내외로 간결하게 전달하는 것을 목표로 합니다.

    6.  **정보 부재 시 명확한 처리:** 지식 베이스 내에서 명확한 답변을 찾을 수 없다면, "{FALLBACK_MSG_NO_INFO}" 라고 일관되게 답변하십시오.

    [사고 과정 가이드]
    1.  사용자 질문의 **가장 시급하고 핵심적인 의도**를 파악한다. (예: '무엇을 준비해야 하는가?', '언제 방문해야 하는가?')
    2.  의도에 맞는 답변을 할 수 있는 **가장 핵심적인 토픽**을 검색한다.
    3.  해당 토픽의 내용 중, **의도와 직접 관련된 부분만** 추출한다.
    4.  '답변 생성 원칙'에 따라, 부가 정보를 제외하고 핵심 내용만으로 간결한 문장을 생성한다.
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
#      Part 3: 모니터링 및 콜백 처리 로직 (JANDI)
# ===================================================================

def send_to_jandi(user_query: str, bot_answer: str):
    """사용자 질문과 봇 답변을 JANDI 웹훅으로 비동기적으로 전송합니다."""
    jandi_webhook_url = os.getenv("JANDI_WEBHOOK_URL")
    if not jandi_webhook_url:
        print("🚨 ERROR: JANDI_WEBHOOK_URL not found in environment variables.")
        return

    # JANDI에서 요구하는 헤더 형식 (Content-Type은 JSON으로 변경)
    headers = {
        "Accept": "application/vnd.tosslab.jandi-v2+json",
        "Content-Type": "application/json"
    }

    # JANDI의 구조화된 메시지 형식
    payload = {
        "body": "💬 신규 챗봇 문의 발생",
        "connectColor": "#007AFF",
        "connectInfo": [
            {"title": "사용자 질문:", "description": user_query},
            {"title": "AI 답변:", "description": bot_answer}
        ]
    }

    try:
        resp = requests.post(jandi_webhook_url, json=payload, headers=headers, timeout=5)
        print("INFO: JANDI Response:", resp.status_code, resp.text)
        if resp.status_code != 200:
            print("⚠️ WARNING: JANDI did not accept the message. Check payload format or webhook URL.")
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
    if FALLBACK_MSG_NO_INFO in final_text_for_user:
        final_response_data = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {"simpleText": {"text": final_text_for_user}}
                ],
                "quickReplies": [
                    {
                        "label": "관리사무실 전화",
                        "action": "webLink",
                        "webLinkUrl": "tel:0319571260"
                    }
                ]
            }
        }
    else:
        final_response_data = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {"simpleText": {"text": final_text_for_user}}
                ]
            }
        }

    # (2) 안전한 전송: callback_url 존재 확인, 로깅, UTF-8 인코딩, 응답 출력
    if not callback_url:
        print("🚨 ERROR: callback_url is empty or missing. Cannot send reply to Kakao.")
        return

    try:
        print("INFO: Sending callback to Kakao. callback_url =", callback_url)
        # payload 로그 (주의: 실제 운영에서는 개인정보 포함시 마스킹 고려)
        print("INFO: final_response_data =", json.dumps(final_response_data, ensure_ascii=False))

        headers = {"Content-Type": "application/json; charset=utf-8"}
        body = json.dumps(final_response_data, ensure_ascii=False).encode("utf-8")

        resp = requests.post(callback_url, data=body, headers=headers, timeout=10)

        print("✅ INFO: Kakao callback POST completed.")
        print("Kakao callback response status:", resp.status_code)
        print("Kakao callback response body:", resp.text)

        if resp.status_code != 200:
            print("⚠️ WARNING: Kakao returned non-200. Check payload format, callback_url, or Kakao logs.")
            # 디버깅 추가: 400/401/403/404 등일 경우 원인 안내
            if resp.status_code in (400, 401, 403, 404):
                print(f"⚠️ DETAIL: Status {resp.status_code} — payload/headers/callback URL 확인 필요.")
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