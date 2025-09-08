# ===================================================================
#           KakaoTalk AI Chatbot - Asynchronous Callback Architecture
#
#   - Author: Gemini (as a world-class AI expert coder)
#   - Feature: Handles Kakao's 5-second timeout using the callback feature.
#              Performs multi-context retrieval and synthesis for complex queries.
#              Proactively suggests follow-up questions to enhance UX.
#
#   - Version: Production Ready
# ===================================================================

import os
import pandas as pd
import requests # 카카오 콜백 URL로 요청을 보내기 위한 라이브러리
import json
import threading # AI 처리와 같은 시간이 걸리는 작업을 백그라운드에서 실행하기 위한 라이브러리
from flask import Flask, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv

# --- 환경 설정 및 클라이언트 초기화 ---
load_dotenv()
app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- 시스템 동작을 제어하는 설정값들 ---
# 검색 시 상위 몇 개의 결과를 AI에게 전달할지 결정합니다.
MAX_CONTEXT_RESULTS = 3
# 검색 결과로 인정할 최소한의 키워드 매칭 점수입니다. 1점은 키워드가 하나라도 겹치면 된다는 의미입니다.
SCORE_THRESHOLD = 1


# --- 지식 베이스(knowledge.csv) 로딩 ---
try:
    knowledge_base = pd.read_csv('knowledge.csv')
    print("✅ Knowledge base loaded successfully.")
    print(f"Total entries: {len(knowledge_base)}")
except FileNotFoundError:
    print("🚨 FATAL ERROR: knowledge.csv file not found. The chatbot cannot function without it.")
    knowledge_base = pd.DataFrame()


# ===================================================================
#      Part 1: 정보 검색(Retrieval) 로직
# ===================================================================
def find_relevant_info_advanced(query: str) -> list[str]:
    """
    사용자의 질문과 관련된 상위 N개의 컨텍스트를 검색하는 고도화된 함수.
    
    Args:
        query: 사용자의 원본 질문 문자열.

    Returns:
        가장 관련성 높은 답변(answer)들의 리스트.
    """
    if knowledge_base.empty:
        return []

    # 사용자의 질문을 단어 집합으로 만들어 중복을 제거하고 검색 속도를 높입니다.
    query_keywords = set(query.split())
    
    # 각 문서(row)의 점수를 계산하여 리스트에 저장합니다.
    scored_results = []
    for index, row in knowledge_base.iterrows():
        question_keywords = set(str(row.get('question', '')).split())
        
        # 키워드 집합의 교집합(intersection)을 통해 겹치는 키워드의 수를 계산합니다.
        common_keywords = query_keywords.intersection(question_keywords)
        score = len(common_keywords)

        # 최소 점수 기준을 넘는 경우에만 결과 후보에 추가합니다.
        if score >= SCORE_THRESHOLD:
            scored_results.append((score, str(row.get('answer', ''))))

    # 계산된 점수가 높은 순으로 결과를 정렬합니다.
    scored_results.sort(key=lambda x: x[0], reverse=True)

    # 정렬된 결과 중 상위 N개의 'answer' 텍스트만 추출하여 최종 컨텍스트 리스트를 만듭니다.
    final_contexts = [answer for score, answer in scored_results[:MAX_CONTEXT_RESULTS]]
    
    # 디버깅 및 모니터링을 위해 어떤 컨텍스트가 선택되었는지 서버 로그에 기록합니다.
    print(f"Found {len(final_contexts)} relevant contexts for query: '{query}'")
    for i, context in enumerate(final_contexts):
        print(f"  - Context {i+1}: {context[:50]}...") # 답변의 앞 50자만 출력

    return final_contexts


# ===================================================================
#      Part 2: AI 답변 생성(Reasoning) 로직
# ===================================================================
def generate_ai_response_advanced(user_message: str, contexts: list[str]) -> str:
    """
    여러 개의 컨텍스트를 종합하여 하나의 완성된 답변을 생성하고, 관련 후속 질문까지 제안합니다.
    
    Args:
        user_message: 사용자의 원본 질문.
        contexts: find_relevant_info_advanced 함수가 찾아낸 답변들의 리스트.

    Returns:
        AI가 생성한 최종 답변 문자열 (후속 질문 제안 포함).
    """
    # 컨텍스트 리스트를 하나의 긴 문자열로 합칩니다. 각 컨텍스트는 명확히 구분되도록 합니다.
    context_str = "\n\n---\n\n".join(contexts)

    # 만약 검색된 컨텍스트가 전혀 없다면, 정보가 없다는 표준 메시지를 반환합니다.
    if not contexts:
        return "죄송하지만 문의하신 내용과 관련된 정보를 찾지 못했습니다. 조금 더 구체적으로 질문해주시거나, 고객센터로 문의해주시면 감사하겠습니다."

    # AI에게 고차원적인 임무를 부여하는 시스템 프롬프트입니다.
    system_instruction = f"""
    당신은 추모공원의 최상급 AI 안내원입니다. 당신의 임무는 사용자의 복잡한 질문에 대해, 아래에 제공되는 여러 개의 '참고 자료'를 종합하여 하나의 완벽하고 논리적인 답변을 생성하는 것입니다.

    [핵심 규칙]
    1.  **종합적 사고:** 사용자의 질문에는 여러 의도가 담겨있을 수 있습니다. '참고 자료' 목록을 모두 검토하여 질문의 모든 부분에 대해 답변해야 합니다.
    2.  **근거 기반 답변:** 답변은 반드시 '참고 자료'에 있는 내용에만 근거해야 합니다. 당신의 사전 지식이나 추측을 절대로 사용하지 마십시오.
    3.  **선별적 정보 제공:** '참고 자료'에 있더라도 사용자가 묻지 않은 내용은 굳이 언급할 필요가 없습니다. 질문의 의도에 집중하십시오.
    4.  **정보 부족 시 인정:** 만약 사용자가 물어본 내용에 대한 정보가 '참고 자료'에 없다면, 다른 내용에 대해서만 답변하고, "문의하신 OOO 정보는 현재 제가 참고할 수 있는 자료에 없어 안내가 어렵습니다." 와 같이 명확하고 솔직하게 전달하십시오.
    5.  **자연스러운 문장:** 여러 자료를 짜깁기한 느낌이 아니라, 처음부터 전문가가 작성한 것처럼 자연스럽고 유기적인 문장으로 답변을 재구성하십시오.

    ---
    [답변 후 행동 지침]
    1.  위의 규칙에 따라 답변 생성을 모두 마친 후, 문단을 나누는 선(---)을 추가하십시오.
    2.  그 다음, "더 궁금하신 점이 있으신가요?" 와 같은 마무리 인사를 하십시오.
    3.  마지막으로, 당신이 답변한 내용과 '참고 자료'를 바탕으로 사용자가 **다음에 가장 궁금해할 만한 관련 질문을 2~3개** 간결하게 추천해주십시오. 추천 질문은 사용자가 바로 복사-붙여넣기 해서 질문할 수 있는 완벽한 문장 형태여야 합니다.
    ---

    [참고 자료 묶음]
    {context_str}
    ---
    """

    try:
        response = client.chat.completions.create(
            model="gpt-5-mini", # 성능이 뛰어난 표준 모델을 명시합니다.
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_message}
            ],
            temperature=1, # 오류 로그에 따라 지원되는 기본값 1로 설정합니다.
            max_completion_tokens=1000, # 답변의 최대 길이를 제한합니다.
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"🚨 OpenAI API call failed: {e}")
        return "죄송합니다. AI 답변을 생성하는 중 시스템 오류가 발생했습니다. 잠시 후 다시 시도해주세요."


# ===================================================================
#      Part 3: 백그라운드 작업 및 콜백 전송 함수
# ===================================================================
def process_and_send_callback(user_message, callback_url):
    """
    (백그라운드에서 실행) AI 답변을 생성하고, 생성된 답변을 카카오 서버로 다시 전송합니다.
    """
    print("Starting background processing for callback...")
    # 1. 정보 검색
    contexts = find_relevant_info_advanced(user_message)
    # 2. AI 답변 생성
    ai_response_text = generate_ai_response_advanced(user_message, contexts)

    # 3. 카카오 서버로 보낼 최종 데이터 포맷(JSON)을 구성합니다.
    final_response_data = {
        "version": "2.0",
        "template": {
            "outputs": [
                {"simpleText": {"text": ai_response_text}}
            ]
        }
    }

    # 4. requests 라이브러리를 사용하여 카카오의 callbackUrl로 POST 요청을 보냅니다.
    headers = {'Content-Type': 'application/json'}
    try:
        # timeout을 설정하여 너무 오래 기다리지 않도록 합니다.
        response = requests.post(callback_url, data=json.dumps(final_response_data), headers=headers, timeout=10)
        response.raise_for_status()  # 응답 코드가 2xx가 아닐 경우 오류를 발생시킵니다.
        print("✅ Successfully sent the final response to Kakao via callback.")
    except requests.exceptions.RequestException as e:
        print(f"🚨 Failed to send callback to Kakao: {e}")


# ===================================================================
#      Part 4: 메인 서버 로직 (Flask Route)
# ===================================================================
@app.route('/callback', methods=['POST'])
def callback():
    # 카카오톡 서버로부터 받은 요청(request) 데이터를 JSON 형태로 파싱합니다.
    req = request.get_json()
    
    user_message = req['userRequest']['utterance']
    # 콜백 기능이 활성화된 경우에만 전달되는 callbackUrl을 추출합니다.
    callback_url = req['userRequest'].get('callbackUrl')

    print(f"\n--- New Request ---")
    print(f"User Query: {user_message}")
    
    # callbackUrl이 존재하는 경우 (AI 챗봇 콜백 기능이 활성화된 경우)
    if callback_url:
        print(f"Callback URL received. Processing in asynchronous mode.")
        
        # 시간이 오래 걸리는 AI 처리 및 콜백 전송 작업을 별도의 스레드(백그라운드)에서 실행하도록 예약합니다.
        # 이렇게 하면 메인 스레드는 즉시 다음 코드로 넘어갈 수 있습니다.
        thread = threading.Thread(target=process_and_send_callback, args=(user_message, callback_url))
        thread.start()
        
        # 카카오 서버에게는 "작업을 백그라운드에서 시작했으니 기다려달라"는 의미의 응답을 5초 내에 즉시 보냅니다.
        # 이 응답을 받은 카카오 서버는 사용자에게 채널 설정에 입력된 대기 메시지를 보여줍니다.
        initial_response = {
            "version": "2.0",
            "useCallback": True, # 이 값이 True이면 카카오가 후속 응답(콜백)을 기다립니다.
        }
        return jsonify(initial_response)
    
    # callbackUrl이 없는 경우 (콜백 기능이 비활성화되었거나 테스트 환경일 경우)
    else:
        print("No callback URL found. Processing in synchronous mode.")
        # 기존 방식대로 모든 처리가 끝날 때까지 기다렸다가 최종 답변을 한 번에 보냅니다.
        contexts = find_relevant_info_advanced(user_message)
        ai_response_text = generate_ai_response_advanced(user_message, contexts)
        return jsonify({
            "version": "2.0",
            "template": {
                "outputs": [
                    {"simpleText": {"text": ai_response_text}}
                ]
            }
        })


if __name__ == '__main__':
    # 개발 환경에서 테스트할 때 사용하는 서버 실행 코드입니다.
    # Render와 같은 실제 서버 환경에서는 gunicorn이 이 파일을 실행합니다.
    app.run(host='0.0.0.0', port=5000, debug=True)