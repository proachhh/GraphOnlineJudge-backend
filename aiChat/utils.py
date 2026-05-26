from django.conf import settings
from sparkai.llm.llm import ChatSparkLLM
from sparkai.core.messages import ChatMessage
from openai import OpenAI


def get_spark_client():
    return ChatSparkLLM(
        spark_api_url=settings.SPARKAI_URL,
        spark_app_id=settings.SPARKAI_APP_ID,
        spark_api_key=settings.SPARKAI_API_KEY,
        spark_api_secret=settings.SPARKAI_API_SECRET,
        spark_llm_domain=settings.SPARKAI_DOMAIN,
        streaming=False,
    )


def get_deepseek_client():
    return OpenAI(
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
    )


def ask_spark(prompt: str) -> str:
    try:
        client = get_spark_client()
        messages = [ChatMessage(role="user", content=prompt)]
        response = client.generate([messages])
        return response.generations[0][0].text.strip()
    except Exception as e:
        return f"【错误】调用星火大模型失败：{str(e)}"


def ask_deepseek(prompt: str) -> str:
    try:
        client = get_deepseek_client()
        response = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}},
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"【错误】调用 DeepSeek 失败：{str(e)}"


def ask_ai(prompt: str, model: str = "deepseek") -> str:
    if model == "deepseek":
        return ask_deepseek(prompt)
    return ask_spark(prompt)
