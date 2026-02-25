import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

# Load API key (set with: export OPENAI_API_KEY="sk-...")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class CommentRequest(BaseModel):
    comment: str

# ✅ FIXED JSON SCHEMA - This is what OpenAI expects
response_schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "sentiment_analysis",  # ← REQUIRED FIELD!
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"]
                },
                "rating": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5
                }
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
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Analyze customer feedback sentiment. Return ONLY valid JSON."
                },
                {
                    "role": "user",
                    "content": f"Comment: {request.comment}"
                }
            ],
            response_format=response_schema,
            temperature=0
        )
        
        # Parse guaranteed JSON
        content = response.choices[0].message.content
        result = json.loads(content)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/")
def root():
    return {"message": "Sentiment API ready! POST to /comment"}
