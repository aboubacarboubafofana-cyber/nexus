from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

TEXT_MODEL = os.getenv("TEXT_MODEL", "openai/gpt-oss-120b")
VISION_MODELS = [
    os.getenv("VISION_MODEL", "qwen/qwen3.6-27b"),
    "meta-llama/llama-4-scout-17b-16e-instruct",
]


class ChatRequest(BaseModel):
    message: str
    history: list = []
    lang: str = "auto"
    image: str = ""


@app.get("/", response_class=HTMLResponse)
def root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/chat")
def chat(request: ChatRequest):
    try:
        if request.lang == "auto":
            lang_rule = "Réponds dans la langue utilisée par l'utilisateur."
        elif request.lang == "Bambara":
            lang_rule = (
                "Réponds en bambara simple (alphabet latin). "
                "Si tu n'es pas sûr d'un mot, ajoute une courte explication en français."
            )
        else:
            lang_rule = f"Réponds en {request.lang}."

        system = (
            "Tu es Fof-IA, un assistant IA intelligent créé par Aboubacar Boubafofana à Abidjan. "
            "Sois clair, utile et chaleureux. Tes réponses seront lues à voix haute : "
            "reste concis et évite les listes et les symboles inutiles. "
            + lang_rule
        )

        past = [
            {"role": m.get("role"), "content": str(m.get("content", ""))}
            for m in request.history[-10:]
            if isinstance(m, dict) and m.get("role") in ("user", "assistant")
        ]

        if request.image:
            user_msg = {
                "role": "user",
                "content": [
                    {"type": "text", "text": request.message},
                    {"type": "image_url", "image_url": {"url": request.image}},
                ],
            }
            messages = [{"role": "system", "content": system}, *past, user_msg]
            last_error = None
            for model in VISION_MODELS:
                try:
                    completion = client.chat.completions.create(model=model, messages=messages)
                    return {"response": completion.choices[0].message.content}
                except Exception as e:
                    last_error = e
            return {"response": f"Erreur de lecture d'image : {str(last_error)}"}

        completion = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[{"role": "system", "content": system}, *past,
                      {"role": "user", "content": request.message}],
        )
        return {"response": completion.choices[0].message.content}
    except Exception as e:
        return {"response": f"Erreur : {str(e)}"}
