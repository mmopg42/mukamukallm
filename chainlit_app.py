

# chainlit_app.py
# import chainlit as cl
# import httpx

# FASTAPI_SERVER_URL = ""  # FastAPI 서버 URL
import chainlit as cl
import httpx

FASTAPI_SERVER_URL = "https://kxxafcpgwrccxezh.tunnel-pt.elice.io/proxy/8000/generate"
# FASTAPI_SERVER_URL = "http://127.0.0.1:8000

@cl.on_message
async def main(message: str):
    """
    Chainlit 메시지를 처리하고 FastAPI 서버에 전달.
    """
    # 메시지가 문자열인지 확인하고 필요하면 변환
    if not isinstance(message, str):
        message = message.content  # chainlit.message.Message 객체에서 텍스트 추출


    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(FASTAPI_SERVER_URL, json={"message": message})
            print(f"Response status code: {response.status_code}")
            print(f"Response headers: {response.headers}")
            print(f"Response text: {response.text}")

        if response.status_code == 200:
            reply = response.json()["reply"]
            await cl.Message(content=reply).send()
        else:
            await cl.Message(content=f"FastAPI 서버 오류: {response.text}").send()
    except Exception as e:
        import traceback
        traceback.print_exc()
        await cl.Message(content=f"오류 발생: {str(e)}").send()