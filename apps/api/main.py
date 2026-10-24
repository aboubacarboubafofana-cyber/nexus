import os
import base64
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, JSONResponse
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv 

load_dotenv()

LOGO_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
<defs>
<radialGradient id="bg" cx="25%" cy="0%" r="110%"><stop offset="0" stop-color="#4a1230"/><stop offset="0.6" stop-color="#1a0812"/><stop offset="1" stop-color="#0d0408"/></radialGradient>
<linearGradient id="tx" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#e6c27a"/><stop offset="1" stop-color="#d96a8c"/></linearGradient>
</defs>
<rect width="512" height="512" fill="url(#bg)"/>
<text x="256" y="292" text-anchor="middle" font-family="Georgia, 'Times New Roman', serif" font-style="italic" font-weight="bold" font-size="124" fill="url(#tx)">Fof-IA</text>
</svg>"""

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
HF_TOKEN = os.getenv("HF_TOKEN")

TEXT_MODEL = os.getenv("TEXT_MODEL", "openai/gpt-oss-120b")
VISION_MODELS = [
    os.getenv("VISION_MODEL", "qwen/qwen3.8-27b"),
]


class ChatRequest(BaseModel):
    message: str
    history: list = []
    lang: str = "auto"
    image: str = ""


# --- FONCTION POUR GÉNÉRER UNE IMAGE AVEC HUGGING FACE ---
def generer_image_hf(prompt: str):
    url = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {"inputs": prompt}
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            img_base64 = base64.b64encode(response.content).decode("utf-8")
            return img_base64
        else:
            return None
    except Exception:
        return None


@app.get("/", response_class=HTMLResponse)
def root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/logo.svg")
def logo():
    return Response(content=LOGO_SVG, media_type="image/svg+xml")


@app.get("/manifest.json")
def manifest():
    return JSONResponse({
        "name": "Fof-IA",
        "short_name": "Fof-IA",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#1a0812",
        "theme_color": "#2a0d1c",
        "icons": [
            {"src": "/logo.svg", "sizes": "any", "type": "image/svg+xml", "purpose": "any maskable"}
        ],
    })


@app.post("/api/chat")
def chat(request: ChatRequest):
    try:
        if request.lang == "auto":
            lang_rule = "Réponds dans la langue utilisée par l'utilisateur."
        elif request.lang == "Bambara":
            lang_rule = "Réponds en bambara simple (alphabet latin). Si tu n'es pas sûr d'un mot, ajoute une courte explication en français."
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

        # --- DÉTECTION DEMANDE D'IMAGE ---
        mots_cles_image = ["crée", "génère", "dessine", "affiche", "image", "photo"]
        if any(mot in request.message.lower() for mot in mots_cles_image) and not request.image:
            prompt_visuel_response = client.chat.completions.create(
                model=TEXT_MODEL,
                messages=[
                    {"role": "system", "content": "Tu es un expert en génération d'images. Transforme la demande en un prompt visuel très détaillé en anglais. Ne réponds QUE par le prompt."},
                    {"role": "user", "content": request.message}
                ]
            )
            prompt_visuel = prompt_visuel_response.choices[0].message.content
            img_b64 = generer_image_hf(prompt_visuel)
            
            if img_b64:
                return {
                    "response": "Voici l'image que j'ai générée pour toi :",
                    "image": f"data:image/png;base64,{img_b64}"
                }
            else:
                return {"response": "Je n'ai pas pu générer l'image, désolé."}

        # --- GESTION DE L'IMAGE ENVOYÉE ---
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

        # --- TEXTE NORMAL ---
        completion = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[{"role": "system", "content": system}, *past,
                      {"role": "user", "content": request.message}],
        )
        return {"response": completion.choices[0].message.content}
    except Exception as e:
        return {"response": f"Erreur : {str(e)}"}
