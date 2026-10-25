from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, JSONResponse
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os
import io
import base64
import urllib.parse
import requests

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
VISION_MODELS = ["qwen/qwen3.8-27b"]

class ChatRequest(BaseModel):
    message: str
    history: list = []
    lang: str = "auto"
    image: str = ""

def generer_affiche_pro(prompt_utilisateur: str):
    try:
        # 1. L'IA traduit et améliore la demande pour faire une VRAIE affiche
        traduction = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": "Tu es un directeur artistique expert. Transforme la demande en un prompt d'image en anglais pour créer une affiche publicitaire professionnelle. Le prompt doit décrire une scène complète (ex: 'A professional advertising poster for a store, vibrant colors, high quality'). Ajoute les détails des produits. Ne réponds QUE par le prompt en anglais."},
                {"role": "user", "content": prompt_utilisateur}
            ]
        )
        prompt_en = traduction.choices[0].message.content.strip()
        
        # 2. On utilise Pollinations avec le modèle FLUX (le plus puissant)
        prompt_encode = urllib.parse.quote(prompt_en)
        url = f"https://image.pollinations.ai/prompt/{prompt_encode}?width=1024&height=1024&nologo=true&model=flux"
        
        response = requests.get(url, timeout=60)
        if response.status_code == 200:
            return base64.b64encode(response.content).decode("utf-8")
        return None
    except Exception as e:
        print(f"Erreur: {e}")
        return None

@app.get("/", response_class=HTMLResponse)
def root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/chat")
def chat(request: ChatRequest):
    try:
        mots_cles = ["crée", "génère", "dessine", "affiche", "image", "photo", "pub"]
        if any(mot in request.message.lower() for mot in mots_cles) and not request.image:
            img_b64 = generer_affiche_pro(request.message)
            if img_b64:
                return {
                    "response": "Voici l'affiche que j'ai générée pour toi :",
                    "image": "data:image/jpeg;base64," + img_b64
                }
        
        system = "Tu es Fof-IA, un assistant IA intelligent. Sois clair et chaleureux. Réponds en français."
        completion = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": request.message}],
        )
        return {"response": completion.choices[0].message.content}
    except Exception as e:
        return {"response": f"Erreur : {str(e)}"}
