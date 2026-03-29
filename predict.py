import torch
from torchvision import transforms
from PIL import Image
from model import get_model
import json

with open("classes.json") as f:
    classes = json.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = get_model(len(classes))
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    output = model(img)
    pred = output.argmax(1).item()

    label = classes[pred]
    plant=label
    if "leaf" in image_path.lower():
        type_ = "leaf"
    else:
        type_ = "plant"

    print("Type:", type_)
    print("Plant:", plant)

# Example
predict("test2.jpg")