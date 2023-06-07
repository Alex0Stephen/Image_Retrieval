import torch
import torch.nn as nn
from torchvision import models, transforms


class FeatureExtractor:
    def __init__(self):
        vgg = models.vgg16(pretrained=True)
        self.model = nn.Sequential(*list(vgg.features.children())[:-1])
        self.model.eval()
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract(self, img):
        # print(self.model._modules.keys())
        img = self.transforms(img).unsqueeze(0)  # transform and add batch dimension
        with torch.no_grad():
            # feature = self.model(img).squeeze().numpy() # squeeze batch dimension and convert to numpy array
            feature = self.model(img)
            feature = torch.flatten(feature)
            # feature = feature / np.linalg.norm(feature)  # Normalize
            feature = feature.numpy()
        return feature
