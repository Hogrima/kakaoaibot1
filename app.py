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

# [개선] 의도에 따른 세분화된 Fallback 메시지
FALLBACK_MSG_ESCALATION = "말씀주신 내용은 확인이 필요하여, 담당 직원에게 직접 문의하시면 가장 정확하고 신속하게 안내받으실 수 있습니다. 대표번호 031-957-1260으로 연락 부탁드립니다."
FALLBACK_MSG_OUT_OF_SCOPE = "저는 크리스찬메모리얼파크에 대한 안내를 도와드리는 AI 상담원입니다. 관련 내용으로 질문해주시면 정성껏 답변드리겠습니다."
FALLBACK_MSG_NO_INFO_IN_KB = "문의주신 내용에 대한 정보는 지식 베이스에서 찾을 수 없어 안내가 어렵습니다. 담당 직원에게 문의해주시면 감사하겠습니다."

# [추가] '전화 걸기' 퀵리플라이를 추가할 답변 목록
QUICK_REPLY_TRIGGERS = [FALLBACK_MSG_ESCALATION, FALLBACK_MSG_NO_INFO_IN_KB]

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
#       Part 2: AI 답변 생성 엔진
# ===================================================================

def generate_ai_response_total_knowledge(user_message: str, history: list) -> str:
    """AI 모델과 지식 베이스, 이전 대화 기록을 사용하여 답변을 생성합니다."""
    if not KNOWLEDGE_TEXTBOOK or ERROR_MSG_KNOWLEDGE_BASE in KNOWLEDGE_TEXTBOOK:
        return "죄송합니다. 현재 챗봇의 지식 베이스에 문제가 발생하여 답변할 수 없습니다."

    # [개선] 사용자의 의도를 먼저 파악하도록 강화된 시스템 지침
    system_instruction = f"""
    당신은 '크리스찬메모리얼파크 AI 상담원'입니다.

    [최상위 지시]
    가장 먼저 사용자의 마지막 질문 의도를 아래 4가지 유형으로 분류하고, 각 유형별 대응 원칙에 따라 답변을 생성하십시오.

    [질문 의도 유형 및 대응 원칙]
    1.  **정보 문의**: 지식 베이스에 있을 법한 내용(예: "봉안 자격이 어떻게 되나요?")을 질문하는 경우입니다.
        - **대응**: 아래의 '[답변 생성 핵심 원칙]'에 따라 지식 베이스에서 답변을 찾아 제공합니다.

    2.  **고객 서비스 및 개인 업무**: 입금, 예약 변경, 개인정보 확인 등 AI가 처리할 수 없는 개인적인 업무나 요청인 경우입니다. (예: "입금이 늦어서 죄송합니다", "예약 시간을 바꾸고 싶어요")
        - **대응**: 지식 베이스를 검색하지 말고, 즉시 다음 지정된 문장으로만 답변하십시오: "{FALLBACK_MSG_ESCALATION}"

    3.  **일반적 인사 또는 잡담**: "안녕하세요", "감사합니다" 와 같은 단순 인사나 대화 시도인 경우입니다.
        - **대응**: 지식 베이스를 검색하지 말고, "별 말씀을요. 더 궁금한 점이 있으시면 편하게 말씀해주세요." 와 같이 간단하고 정중한 인사로 답변합니다.

    4.  **범위 외 질문**: 크리스찬메모리얼파크와 관련 없는 질문인 경우입니다. (예: "오늘 날씨 어때요?")
        - **대응**: 지식 베이스를 검색하지 말고, 즉시 다음 지정된 문장으로만 답변하십시오: "{FALLBACK_MSG_OUT_OF_SCOPE}"

    [답변 생성 핵심 원칙 (오직 '정보 문의' 유형에만 적용)]
    1.  **오직 마지막 질문에만 집중**: 당신의 답변은 반드시 사용자의 **마지막 질문 하나**에 대한 내용만 포함해야 합니다. 이전 대화 내용은 맥락 파악에만 사용하고, 답변에 절대 다시 언급하거나 요약하지 마십시오.
    2.  **철저한 근거 기반 답변**: 당신의 모든 답변은 반드시 '[공식 지식 베이스]'에서만 나와야 합니다. 외부 지식이나 추측은 절대 허용되지 않습니다.
    3.  **부가 정보 금지**: 사용자가 명시적으로 묻지 않은 내용은 절대 먼저 언급하지 마십시오.
    4.  **간결한 형식 준수**: 답변은 1~3개의 문장, 50~300자 내외로 매우 간결하게 작성합니다. 서식을 사용하지 마십시오.
    5.  **정보 부재 시 명확한 처리**: 지식 베이스 내에서 명확한 답변을 찾을 수 없다면, 다음 지정된 문장으로만 답변하십시오: "{FALLBACK_MSG_NO_INFO_IN_KB}"

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

def create_kakao_response(text: str) -> dict:
    """[추가된 헬퍼 함수] 주어진 텍스트를 기반으로 카카오톡 응답 JSON 객체를 생성합니다."""
    response_template = {
        "outputs": [{"simpleText": {"text": text}}]
    }
    
    # 상수로 정의된 특정 메시지가 포함된 경우 '전화 걸기' 퀵리플라이 추가
    if any(trigger in text for trigger in QUICK_REPLY_TRIGGERS):
        response_template["quickReplies"] = [
            {
                "label": "관리사무실 전화연결",
                "action": "webLink",
                "webLinkUrl": "tel:031-957-1260"
            }
        ]
        
    return {"version": "2.0", "template": response_template}

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

    # [수정 완료] 헬퍼 함수를 사용하여 카카오톡 응답 JSON을 생성합니다.
    final_response_data = create_kakao_response(final_text_for_user)

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

        # [개선] 헬퍼 함수를 사용하여 카카오톡 응답 생성 (코드 간소화)
        final_response_data = create_kakao_response(ai_response_text)
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