import torch
from PIL import Image
from .model import SyntheticDetector
from .preprocess import transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SyntheticDetector()
model.load_state_dict(torch.load("models/synthetic_detector.pth", map_location=device))
model.to(device)
model.eval()


def predict(image: Image):

    image = image.convert("RGB")
    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(img)
        probabilities = torch.softmax(output, dim=1)

        pred = torch.argmax(probabilities, 1).item()
        confidence = probabilities[0][pred].item()

    return pred, confidence, probabilities.cpu().numpy()[0]