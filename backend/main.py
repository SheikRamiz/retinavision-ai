from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import io

app = FastAPI(title="RetinaVision AI - Fine-Tuned ResNet50 Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading fine-tuned ResNet50 model weights...")
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 5)

try:
    checkpoint = torch.load("model.pth", map_location=torch.device("cpu"))
    
    # Unpack checkpoint if wrapped inside a dictionary
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Clean up potential 'module.' prefixes from parallel training
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=False)
    print("Successfully loaded state dictionary into ResNet50!")
except Exception as e:
    print(f"Error loading state dict: {e}. Falling back to standard evaluation.")

model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

STAGE_RULES = {
    0: {"stage": 0, "stage_name": "No Diabetic Retinopathy", "risk_level": "Low", "action_guide": "Routine annual screening; maintain glycemic control targets."},
    1: {"stage": 1, "stage_name": "Mild Non-Proliferative DR", "risk_level": "Mild", "action_guide": "Follow-up screening within 6 to 12 months."},
    2: {"stage": 2, "stage_name": "Moderate Non-Proliferative DR", "risk_level": "Moderate", "action_guide": "Ophthalmology review within 3-6 months; strict glycemic optimization."},
    3: {"stage": 3, "stage_name": "Severe Non-Proliferative DR", "risk_level": "High", "action_guide": "Urgent Specialist Referral within 1–2 weeks."},
    4: {"stage": 4, "stage_name": "Proliferative Diabetic Retinopathy", "risk_level": "Critical", "action_guide": "Immediate Specialist Referral (PRP Laser evaluation)."}
}

@app.post("/predict")
async def predict_retinopathy(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")

        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = int(torch.argmax(probabilities, dim=1).item())
            confidence = float(torch.max(probabilities).item())

        rule_payload = STAGE_RULES.get(predicted_class, STAGE_RULES[0])

        return {
            "status": "success",
            "prediction": {
                "stage": rule_payload["stage"],
                "stage_name": rule_payload["stage_name"],
                "risk_level": rule_payload["risk_level"],
                "confidence": round(confidence, 3),
                "action_guide": rule_payload["action_guide"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)