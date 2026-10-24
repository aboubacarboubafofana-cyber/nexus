from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, JSONResponse
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os
import io
import base64

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


@app.get("/logo.svg")
def logo():
    return Response(content=LOGO_SVG, media_type="image/svg+xml")
def generer_image_pollinations(prompt: str):
    import urllib.parse
    prompt_encode = urllib.parse.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{prompt_encode}?width=1024&height=1024&nologo=true&model=flux"
    try:
        import requests
        response = requests.get(url, timeout=60)
        if response.status_code == 200:
            return base64.b64encode(response.content).decode("utf-8")
    except Exception:
        pass
    return None

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


class ImageRequest(BaseModel):
    prompt: str


@app.post("/api/image")
def make_image(req: ImageRequest):
    try:
        # 1. On demande à Groq de traduire et d'améliorer le prompt en anglais
        prompt_response = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": "Tu es un expert en prompt d'image. Traduis la demande de l'utilisateur en un prompt très détaillé en anglais pour une IA générative. Ne réponds QUE par le prompt en anglais, sans aucune autre phrase."},
                {"role": "user", "content": req.prompt}
            ]
        )
        prompt_en = prompt_response.choices[0].message.content.strip()

        # 2. On génère l'image avec Pollinations en anglais
        import urllib.parse
        prompt_encode = urllib.parse.quote(prompt_en)
        url = f"https://image.pollinations.ai/prompt/{prompt_encode}?width=1024&height=1024&nologo=true&model=flux"
        
        import requests
        response = requests.get(url, timeout=60)
        
        if response.status_code == 200:
            b64 = base64.b64encode(response.content).decode("utf-8")
            return {"image": "data:image/jpeg;base64," + b64}
        else:
            return {"error": "Erreur de génération d'image."}
            
    except Exception as e:
        return {"error": str(e)}


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
