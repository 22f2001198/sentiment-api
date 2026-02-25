import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

# ✅ FIXED: Safe OpenAI client for Railway
try:
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://aipipe.org/openai/v1")
    )
except:
    client = None  # Fallback for startup

class CommentRequest(BaseModel):
    comment: str

response_schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "sentiment_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                "rating": {"type": "integer", "minimum": 1, "maximum": 5}
            },
            "required": ["sentiment", "rating"],
            "additionalProperties": False
        }
    }
}

@app.post("/comment")
async def analyze_comment(request: CommentRequest):
    if not request.comment or not request.comment.strip():
        raise HTTPException(status_code=400, detail="Comment cannot be empty")
    
    if not client:
        raise HTTPException(status_code=500, detail="AI service unavailable")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Analyze sentiment. Return ONLY JSON."},
                {"role": "user", "content": f"Comment: {request.comment}"}
            ],
            response_format=response_schema,
            temperature=0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Analysis failed")

@app.get("/")
def root():
    return {
        "status": "Sentiment API ready!",
        "ai_ready": client is not None
    }

# Railway production startup
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
