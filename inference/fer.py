import torch
from torchvision import datasets, models, transforms
from PIL import Image
import sys
sys.path.insert(0, '../')

from model import InceptionV3FER

class FER:
    def __init__(self, ckpt_path):
        self.model = InceptionV3FER.load_from_checkpoint(ckpt_path)
        self.data_transform = transforms.Compose([  transforms.Resize((326, 326)),
                                                    transforms.ToTensor() ])
        self.class_mapping = ['Afraid', 'Angry', 'Disgusted', 'Happy', 'Neutral', 'Sad', 'Surprised']
    
    def predict(self, img):
        img_t = self.data_transform(img)
        img_t = torch.unsqueeze(img_t, 0)

        self.model.eval()
        output = self.model(img_t)

        _, pred = torch.max(output, 1)

        return self.class_mapping[pred]