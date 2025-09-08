# ===================================================================
#           KakaoTalk AI Chatbot - Advanced Architecture
#
#   - Author: Gemini (as a world-class AI expert coder)
#   - Architecture: Multi-Context Synthesis & Reasoning
# ===================================================================

import os
import pandas as pd
from flask import Flask, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv

# --- 환경 설정 및 클라이언트 초기화 ---
load_dotenv()
app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- (신규) 시스템 동작을 제어하는 설정값들 ---
# 검색 시 상위 몇 개의 결과를 AI에게 전달할지 결정합니다.
# 3개 정도가 정확도와 비용 사이의 가장 이상적인 균형을 제공합니다.
MAX_CONTEXT_RESULTS = 3
# 검색 결과로 인정할 최소한의 키워드 매칭 점수입니다.
SCORE_THRESHOLD = 1


# --- 지식 베이스 로딩 ---
try:
    knowledge_base = pd.read_csv('knowledge.csv')
    print("✅ Knowledge base loaded successfully.")
    print(f"Total entries: {len(knowledge_base)}")
except FileNotFoundError:
    print("🚨 FATAL ERROR: knowledge.csv file not found. The chatbot cannot function without it.")
    knowledge_base = pd.DataFrame()


# ===================================================================
#      Part 1: 정보 검색(Retrieval) 로직 고도화
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

    query_keywords = set(query.split())
    
    # 각 문서(row)의 점수를 계산하여 리스트에 저장합니다.
    scored_results = []
    for index, row in knowledge_base.iterrows():
        question_keywords = set(str(row.get('question', '')).split())
        
        # 키워드 집합의 교집합을 통해 매칭되는 키워드 수를 계산합니다.
        common_keywords = query_keywords.intersection(question_keywords)
        score = len(common_keywords)

        if score >= SCORE_THRESHOLD:
            scored_results.append((score, str(row.get('answer', ''))))

    # 점수가 높은 순으로 결과를 정렬합니다.
    scored_results.sort(key=lambda x: x[0], reverse=True)

    # 상위 N개의 결과에서 'answer' 텍스트만 추출하여 리스트로 반환합니다.
    # 예: [(3, "절차..."), (2, "가격...")] -> ["절차...", "가격..."]
    final_contexts = [answer for score, answer in scored_results[:MAX_CONTEXT_RESULTS]]
    
    # <<< CHANGED >>>
    # 디버깅을 위해 어떤 컨텍스트가 선택되었는지 로그를 남깁니다.
    print(f"Found {len(final_contexts)} relevant contexts for query: '{query}'")
    for i, context in enumerate(final_contexts):
        print(f"  - Context {i+1}: {context[:50]}...") # 답변의 앞 50자만 출력

    return final_contexts


# ===================================================================
#      Part 2: AI 답변 생성(Reasoning) 로직 지능화 - 최신 버전
# ===================================================================
def generate_ai_response_advanced(user_message: str, contexts: list[str]) -> str:
    """
    여러 개의 컨텍스트를 종합하여 하나의 완성된 답변을 생성하고,
    관련 후속 질문까지 제안하는 최종 고도화 함수.
    
    Args:
        user_message: 사용자의 원본 질문.
        contexts: find_relevant_info_advanced 함수가 찾아낸 답변들의 리스트.

    Returns:
        AI가 생성한 최종 답변 문자열 (후속 질문 제안 포함).
    """
    context_str = "\n\n---\n\n".join(contexts)

    if not contexts:
        print("No context found. Replying with a standard message.")
        return (
            "죄송하지만 문의하신 내용과 관련된 정보를 찾지 못했습니다. "
            "조금 더 구체적으로 질문해주시거나, 고객센터로 문의해주시면 감사하겠습니다."
        )

    # 시스템 프롬프트
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
    3.  마지막으로, 당신이 답변한 내용과 '참고 자료'를 바탕으로 사용자가 **다음에 가장 궁금해할 만한 관련 질문을 1~2개** 간결하게 추천해주십시오. 추천 질문은 사용자가 바로 복사-붙여넣기 해서 질문할 수 있는 완벽한 문장 형태여야 합니다.
    ---

    [참고 자료 묶음]
    {context_str}
    ---
    """

    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",   # <<< 최신 모델 적용
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_message}
            ],
            temperature=0.6,       # 후속 질문 추천을 위해 살짝 높임
            max_completion_tokens=1000,
        )
        ai_answer = response.choices[0].message.content
        print("✅ Successfully generated AI response with follow-up questions.")
        return ai_answer
    except Exception as e:
        print(f"🚨 OpenAI API call failed: {e}")
        return "죄송합니다. AI 답변을 생성하는 중 시스템 오류가 발생했습니다. 잠시 후 다시 시도해주세요."


# ===================================================================
#      Part 3: 메인 서버 로직 (통합)
# ===================================================================
@app.route('/callback', methods=['POST'])
def callback():
    req = request.get_json()
    user_message = req['userRequest']['utterance']
    print(f"\n--- New Request ---")
    print(f"User Query: {user_message}")

    # <<< CHANGED >>>
    # 고도화된 함수들을 호출합니다.
    contexts = find_relevant_info_advanced(user_message)
    ai_response_text = generate_ai_response_advanced(user_message, contexts)
    
    response = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": ai_response_text
                    }
                }
            ]
        }
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    