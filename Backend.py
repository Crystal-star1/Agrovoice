from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import torch, timm
import torchvision.transforms as T
from PIL import Image
import requests, tempfile, io, traceback

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

API_KEY = "YOUR_REAL_KEY"
LABELS = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]

KB = {
    "CBB": {"name": "Cassava Bacterial Blight",
        "keywords": ["water soaked","wet spot","gum","ooze","dark spot","wilt","dieback","sticky"],
        "explanation_points": [
            "Caused by bacteria (Xanthomonas) that spread mainly through rain splashes and contaminated tools.",
            "Shows as dark, water-soaked spots on leaves and sticky gum oozing from stems.",
            "Spreads fastest in wet, humid conditions, especially right after rainfall."],
        "action_points": [
            "No chemical spray currently cures this disease.",
            "Clean and disinfect farm tools between plants to avoid spreading it further.",
            "Use only certified, disease-free cuttings for your next planting."]},
    "CBSD": {"name": "Cassava Brown Streak Disease",
        "keywords": ["root rot","brown root","harvest","internal rot","root damage"],
        "explanation_points": [
            "A virus spread by whitefly insects feeding on the plant.",
            "Often shows little sign on leaves, making it easy to miss until harvest.",
            "Damages the inside of the root, causing brown, corky rot."],
        "action_points": [
            "No chemical treatment exists for this virus.",
            "Use certified CBSD-resistant cassava cuttings for your next planting.",
            "Check roots at harvest time even if leaves looked healthy."]},
    "CGM": {"name": "Cassava Green Mite",
        "keywords": ["mottle","speckle","curl","curling","curled","candle stick","shoot tip","small yellow spot"],
        "explanation_points": [
            "Caused by tiny mites, not a fungus or virus, feeding on young leaves.",
            "Leads to speckled, mottled spots and curling on new growth.",
            "Spreads quickly in dry, dusty conditions."],
        "action_points": [
            "Spray neem oil on affected leaves as a safe first option.",
            "Abamectin-based acaricide can be used if neem is unavailable.",
            "Encourage natural predator insects on your farm to help control mites long-term."]},
    "CMD": {"name": "Cassava Mosaic Disease",
        "keywords": ["mosaic","patchy","patchwork","patch work","blotchy","checkered","yellow green","distort","stunt"],
        "explanation_points": [
            "A virus spread by whitefly insects, one of the most damaging cassava diseases in Africa.",
            "Causes a yellow and green patchy, mosaic-like pattern on leaves.",
            "Leads to stunted growth and smaller harvests."],
        "action_points": [
            "No chemical cure exists for this virus.",
            "Remove and destroy infected plants to stop it spreading to healthy ones.",
            "Plant certified disease-resistant cassava cuttings next season."]},
    "Healthy": {"name": "a healthy plant",
        "keywords": ["fine","healthy","no problem","normal"],
        "explanation_points": [
            "No signs of disease were found in the photo or description.",
            "Leaf color and shape appear normal for cassava.",
            "Continued healthy growth is likely if conditions stay good."],
        "action_points": [
            "No treatment is needed right now.",
            "Keep checking your plants regularly to catch problems early.",
            "Maintain good spacing and drainage to prevent future disease."]}
}

cv_model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=5)
cv_model.load_state_dict(torch.load('/content/drive/MyDrive/cassava_checkpoints/epoch_10.pt', map_location='cpu'))
cv_model.eval()

transform = T.Compose([T.Resize((224,224)), T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

def cv_probs(image):
    img = transform(image.convert('RGB')).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(cv_model(img), dim=1).squeeze().tolist()
    return dict(zip(LABELS, probs))

def symptom_scores(transcript):
    t = transcript.lower()
    return {c: sum(1 for k in e["keywords"] if k in t)/max(len(e["keywords"]),1) for c, e in KB.items()}

def fuse(sym, cv, w_v=0.4, w_i=0.6):
    f = {c: w_v*sym.get(c,0) + w_i*cv.get(c,0) for c in LABELS}
    tot = sum(f.values()) or 1
    return {k: v/tot for k, v in f.items()}

def transcribe_sahara(path, lang):
    ext = path.split('.')[-1]
    mime_map = {"wav": "audio/wav", "m4a": "audio/x-m4a", "mp3": "audio/mpeg", "ogg": "audio/ogg"}
    mime = mime_map.get(ext, "audio/wav")
    with open(path, "rb") as f:
        r = requests.post("https://infer.voice.intron.io/file/v1/upload/sync",
            headers={"Authorization": f"Bearer {API_KEY}"},
            data={"audio_file_name": f"clip.{ext}", "use_language_asr_input": lang},
            files={"audio_file_blob": (f"clip.{ext}", f, mime)})
    if r.status_code != 200:
        raise Exception(f"Sahara STT error {r.status_code}: {r.text}")
    return r.json()["data"]["audio_transcript"]

def speak(text, lang):
    accent = "yoruba" if lang == "yo" else "pidgin"
    r = requests.post("https://infer.voice.intron.io/tts/v1/generate",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={"text": text, "voice_language": lang, "voice_accent": accent, "voice_gender": "female"})
    if r.status_code != 200:
        raise Exception(f"Sahara TTS error {r.status_code}: {r.text}")
    return r.json()["data"]["audio_path"]

@app.post("/diagnose")
async def diagnose(audio: UploadFile = File(...), image: UploadFile = File(...), lang: str = Form(...)):
    try:
        ext = audio.filename.split('.')[-1] if audio.filename and '.' in audio.filename else 'wav'
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(await audio.read())
            audio_path = tmp.name

        img = Image.open(io.BytesIO(await image.read()))

        transcript = transcribe_sahara(audio_path, lang)
        sym = symptom_scores(transcript)
        cv = cv_probs(img)
        fused = fuse(sym, cv)

        top = max(fused, key=fused.get)
        info = KB[top]
        conf = round(fused[top]*100)

        speech_text = (f"Your cassava plant looks like it has {info['name']} I am about {conf} percent confident "
                        + " ".join(info['explanation_points']) + " " + " ".join(info['action_points'])
                       ).replace(".", " ").replace(",", " ")
        audio_url = speak(speech_text, lang)

        return {
            "transcript": transcript,
            "fused_scores": fused,
            "top_diagnosis": info['name'],
            "confidence": conf,
            "explanation_points": info['explanation_points'],
            "action_points": info['action_points'],
            "tts_audio_url": audio_url
        }
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}
