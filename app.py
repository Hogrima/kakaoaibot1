# ===================================================================
#           KakaoTalk AI Chatbot - Production Ready Version
#
#   - Author: Gemini (as a world-class AI expert coder)
#   - Architecture: Total Knowledge Ingestion with Ephemeral Conversation Context
#   - Database: Local SQLite for temporary memory
# ===================================================================
import os
import pandas as pd
import threading
import requests
import json
import re
import sqlite3 # <--- psycopg2 대신 sqlite3로 다시 변경
from flask import Flask, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv

# --- ⚙️ 시스템 설정 (Configuration) ---ㅡ
# 환경 변수를 통해 설정을 관리하여 유연성을 확보합니다.
# .env 파일에 OPENAI_API_KEY, SLACK_WEBHOOK_URL 등을 설정하세요.
load_dotenv()

CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano") # 최신 고효율 모델 사용을 권장합니다.
KNOWLEDGE_FILE_NAME = "knowledge.csv"
DB_NAME = "local_conversation.db" # <--- 로컬 파일 DB 이름 지정

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
#      Part 0: 데이터베이스 및 지식 베이스 초기화 (SQLite 버전)
# ===================================================================

def get_db_connection():
    """SQLite 데이터베이스 연결 객체를 반환합니다."""
    conn = sqlite3.connect(DB_NAME)
    return conn

def init_db():
    """(서버 시작 시 1회 실행) SQLite 테이블을 생성합니다."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # SQLite에 맞는 테이블 생성 쿼리
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
        print("✅ SQLite Database table initialized successfully.")
    except Exception as e:
        print(f"🚨 FATAL ERROR during DB initialization: {e}")

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
    except Exception as e:
        error_msg = f"{ERROR_MSG_KNOWLEDGE_BASE} (원인: {e})"
        print(f"🚨 FATAL ERROR during knowledge base initialization: {e}")
        KNOWLEDGE_TEXTBOOK = error_msg

# ===================================================================
#      Part 1.5: 대화 기록 관리 (SQLite Interaction)
# ===================================================================

def get_conversation_history(user_id: str, limit: int = 10) -> list:
    """DB에서 특정 사용자의 최근 대화 기록을 가져옵니다."""
    history = []
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # SQLite 쿼리 (placeholder가 ? 로 변경됨)
        cursor.execute(
            "SELECT role, content FROM conversations WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
            (user_id, limit)
        )
        history = [{"role": row[0], "content": row[1]} for row in cursor.fetchall()]
        conn.close()
    except Exception as e:
        print(f"🚨 ERROR getting conversation history: {e}")
    return list(reversed(history)) # 시간 순서대로 다시 뒤집어서 반환

def add_to_conversation_history(user_id: str, role: str, content: str):
    """DB에 새로운 대화 내용을 추가합니다."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # SQLite 쿼리 (placeholder가 ? 로 변경됨)
        cursor.execute(
            "INSERT INTO conversations (user_id, role, content) VALUES (?, ?, ?)",
            (user_id, role, content)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"🚨 ERROR adding to conversation history: {e}")

# ===================================================================
#      Part 2: AI 답변 생성 엔진
# ===================================================================

def generate_ai_response_total_knowledge(user_message: str, history: list) -> str:
    """AI 모델과 지식 베이스, 이전 대화 기록을 사용하여 답변을 생성합니다."""
    if not KNOWLEDGE_TEXTBOOK or ERROR_MSG_KNOWLEDGE_BASE in KNOWLEDGE_TEXTBOOK:
        return f"죄송합니다. 현재 챗봇의 지식 베이스에 문제가 발생하여 답변할 수 없습니다."

    # (이전 대화의 맥락 오염을 방지하는 최종 강화된 지침)
    system_instruction = f"""
    당신은 한국어 존댓말로만 정중히 대답하는 '크리스찬메모리얼파크 AI 상담원'입니다. 당신의 임무는, 아래 제공되는 고도로 구조화된 '[공식 지식 베이스]'의 내용에만 근거하여 사용자의 질문에 가장 정확하고 도움이 되는 답변을 제공하는 것입니다.

    [답변 생성 핵심 원칙]
    1.  **오직 마지막 질문에만 집중 (Focus Solely on the Last Question):** 이것은 가장 중요한 규칙입니다. 당신의 답변은 반드시 사용자의 **마지막 질문 하나**에 대한 내용만 포함해야 합니다. 이전 대화 내용은 맥락을 이해하는 데에만 사용하고, 답변에 그 내용을 다시 언급하거나 요약해서는 절대 안 됩니다.

    2.  **철저한 근거 기반 답변:** 당신의 모든 답변은 반드시 '[공식 지식 베이스]'에서만 나와야 합니다. 당신의 외부 지식이나 추측은 절대 허용되지 않습니다.

    3.  **부가 정보 금지:** 사용자가 명시적으로 묻지 않은 내용은 절대 먼저 언급하지 마십시오. 답변은 사용자의 질문에 대한 직접적인 대답으로만 구성되어야 합니다.

    4.  **제한적인 정보 종합:** 사용자의 '마지막 질문'이 명백히 여러 정보(예: '계약금과 관리비')를 동시에 요구하는 경우에만 관련 정보를 종합하여 답변하십시오.

    5.  **간결한 형식 준수:** 답변은 1~3개의 문장, 50~300자 내외로 매우 간결하게 작성합니다. 어떠한 서식도 사용하지 마십시오.

    6.  **정보 부재 시 명확한 처리:** 지식 베이스 내에서 명확한 답변을 찾을 수 없다면, "{FALLBACK_MSG_NO_INFO}" 라고 일관되게 답변하십시오.

    [사고 과정 가이드]
    1.  전체 대화 기록을 읽고 맥락을 이해한다.
    2.  사용자의 **마지막 질문**의 핵심 의도를 파악한다.
    3.  오직 그 의도에 맞는 답변을 할 수 있는 가장 핵심적인 토픽을 지식 베이스에서 검색한다.
    4.  '답변 생성 핵심 원칙'에 따라, 다른 모든 부가 정보를 제외하고 마지막 질문에 대한 답변만으로 간결한 문장을 생성한다.
    ---
    [크리스찬메모리얼파크 공식 지식 베이스]
    {KNOWLEDGE_TEXTBOOK}
    ---
    """

    messages_to_send = history + [{"role": "user", "content": user_message}]
    
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_instruction}
            ] + messages_to_send,
        )
        ai_message = response.choices[0].message.content
        sanitized_text = re.sub(r"[\*#\`•]", "", ai_message).strip()
        return sanitized_text
        
    except Exception as e:
        print(f"🚨 ERROR: OpenAI API call failed for user message '{user_message}'. Details: {e}")
        return ERROR_MSG_AI_FAILED


# ===================================================================
#      Part 3: 모니터링 및 콜백 처리 로직 (대화 기억 기능 통합)
# ===================================================================

def send_to_jandi(user_id: str, user_query: str, bot_answer: str):
    """사용자 ID, 질문, 봇 답변을 JANDI 웹훅으로 전송합니다."""
    jandi_webhook_url = os.getenv("JANDI_WEBHOOK_URL")
    if not jandi_webhook_url:
        return

    headers = {
        "Accept": "application/vnd.tosslab.jandi-v2+json",
        "Content-Type": "application/json"
    }

    # JANDI 메시지에 user_id를 포함하여 디버깅 편의성 향상
    payload = {
        "body": f"💬 신규 챗봇 문의 (User: {user_id})",
        "connectColor": "#007AFF",
        "connectInfo": [
            {"title": "사용자 질문:", "description": user_query},
            {"title": "AI 답변:", "description": bot_answer}
        ]
    }

    try:
        resp = requests.post(jandi_webhook_url, json=payload, headers=headers, timeout=5)
        if resp.status_code != 200:
            print(f"⚠️ WARNING: JANDI notification failed. Status: {resp.status_code}, Body: {resp.text}")
    except requests.exceptions.RequestException as e:
        print(f"⚠️ WARNING: Failed to send JANDI notification: {e}")


def process_and_send_callback(user_id: str, user_message: str, callback_url: str):
    """백그라운드에서 AI 답변 생성, 로깅, DB/JANDI 전송, 콜백 전송을 모두 처리합니다."""
    print(f"INFO: Starting background processing for user_id: {user_id}")
    
    # 1. 이전 대화 기록 가져오기 (이제 SQLite에서 가져옵니다)
    history = get_conversation_history(user_id)
    
    # 2. AI 답변 생성 시 'history' 함께 전달
    ai_response_text = generate_ai_response_total_knowledge(user_message, history)

    final_text_for_user = ai_response_text
    if not final_text_for_user or not final_text_for_user.strip():
        print("🚨 CRITICAL: AI returned an empty or whitespace-only response. Using fallback message.")
        final_text_for_user = FALLBACK_MSG_EMPTY_RESPONSE

    # 3. 현재 대화를 DB에 저장 (이제 SQLite에 저장합니다)
    add_to_conversation_history(user_id, "user", user_message)
    add_to_conversation_history(user_id, "assistant", final_text_for_user)

    log_message = (
        f"{'='*50}\n"
        f"  [AI RESPONSE LOG for user_id: {user_id}]\n"
        f"  - User Query: {user_message}\n"
        f"  - Final Answer: {final_text_for_user}\n"
        f"{'='*50}"
    )
    print(log_message)

    # JANDI 알림
    send_to_jandi(user_id=user_id, user_query=user_message, bot_answer=final_text_for_user)

    # 조건부 퀵리플라이 (전화 버튼) 로직
    if FALLBACK_MSG_NO_INFO in final_text_for_user:
        final_response_data = {
            "version": "2.0",
            "template": {
                "outputs": [{"simpleText": {"text": final_text_for_user}}],
                "quickReplies": [
                    {"label": "관리사무실 전화", "action": "webLink", "webLinkUrl": "tel:0319571260"}
                ]
            }
        }
    else:
        final_response_data = {
            "version": "2.0",
            "template": {
                "outputs": [{"simpleText": {"text": final_text_for_user}}]
            }
        }

    # 강화된 콜백 전송 로직
    if not callback_url:
        print("🚨 ERROR: callback_url is empty. Cannot send reply to Kakao.")
        return

    try:
        headers = {"Content-Type": "application/json; charset=utf-8"}
        body = json.dumps(final_response_data, ensure_ascii=False).encode("utf-8")
        resp = requests.post(callback_url, data=body, headers=headers, timeout=10)
        
        if resp.status_code != 200:
            print(f"⚠️ WARNING: Kakao returned non-200. Status: {resp.status_code}, Body: {resp.text}")
        else:
            print("✅ INFO: Kakao callback POST completed successfully.")

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

    try:
        user_message = req['userRequest']['utterance']
        callback_url = req['userRequest'].get('callbackUrl')
        user_id = req['userRequest']['user']['id']
    except (KeyError, TypeError):
        return jsonify({"status": "error", "message": "Invalid request format"}), 400

    print(f"\n[INFO] New request received from user_id: {user_id}")
    print(f"[DEBUG] User Query: {user_message}")

    if callback_url:
        # 비동기 처리를 위해 user_id를 포함한 인자를 스레드에 전달
        thread = threading.Thread(target=process_and_send_callback, args=(user_id, user_message, callback_url))
        thread.start()
        return jsonify({"version": "2.0", "useCallback": True})
    else:
        # 동기식 처리 (카카오톡 테스트 콘솔 등)
        history = get_conversation_history(user_id)
        ai_response_text = generate_ai_response_total_knowledge(user_message, history)
        add_to_conversation_history(user_id, "user", user_message)
        add_to_conversation_history(user_id, "assistant", ai_response_text)
        
        print(f"[INFO] AI Response (Sync) for {user_id}: {ai_response_text}")
        send_to_jandi(user_id=user_id, user_query=user_message, bot_answer=ai_response_text)

        if FALLBACK_MSG_NO_INFO in ai_response_text:
             final_response_data = {
                "version": "2.0", "template": {"outputs": [{"simpleText": {"text": ai_response_text}}], "quickReplies": [{"label": "관리사무실 전화", "action": "webLink", "webLinkUrl": "tel:0319571260"}]}
            }
        else:
            final_response_data = {
                "version": "2.0", "template": {"outputs": [{"simpleText": {"text": ai_response_text}}]}
            }
        return jsonify(final_response_data)


# ===================================================================
#      서버 실행 (SQLite DB 초기화 로직 포함)
# ===================================================================

# Gunicorn과 같은 프로덕션 WSGI 서버로 실행될 때 이 부분이 먼저 호출됩니다.
init_db()
load_and_format_knowledge_base()

if __name__ == '__main__':
    # 로컬에서 직접 python app.py로 실행할 때를 위한 부분
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)