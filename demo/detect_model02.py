import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "Hello-SimpleAI/chatgpt-detector-roberta"
MAX_LENGTH = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()


def predict_text(text):
    if not text or not text.strip():
        return "Empty input", "0.0000", {
            "Human-written": 0.0,
            "AI-generated": 0.0
        }

    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()

    human_prob = float(probs[0])
    ai_prob = float(probs[1])

    if ai_prob > human_prob:
        label = "AI-generated"
        confidence = ai_prob
    else:
        label = "Human-written"
        confidence = human_prob

    scores = {
        "Human-written": round(human_prob, 4),
        "AI-generated": round(ai_prob, 4)
    }

    return label, f"{confidence:.4f}", scores


demo = gr.Interface(
    fn=predict_text,
    inputs=gr.Textbox(lines=10, placeholder="Enter a piece of text here..."),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Confidence"),
        gr.Label(label="Class Probabilities")
    ],
    title="AI-Generated Text Detection Demo (Model 02)",
    description="This demo uses Hello-SimpleAI/chatgpt-detector-roberta."
)

demo.launch()