from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# 언리얼에서 보낼 데이터 구조 정의
class Message(BaseModel):
    text: str

@app.post("/test")
async def test_communication(msg: Message):
    print(f"언리얼에서 온 메시지: {msg.text}")
    # 언리얼로 다시 보낼 가짜 응답 데이터
    return {
        "status": "success",
        "received_text": msg.text,
        "emotion_score": 0.5
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)