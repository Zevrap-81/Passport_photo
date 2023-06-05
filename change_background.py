import cv2
from matplotlib import pyplot as plt
import torch
from U2net import U2NET
from torchvision import transforms
from PIL import Image
import numpy as np


def normalize(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn


class ChangeBackground:
    def __init__(self, model_path: str = "/passport_photo/model_ckpts/u2net_rmbg.pth"):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = U2NET(3, 1).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((320, 320)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def forward(self, image: np.ndarray) -> Image.Image:
        """Creates a background mask from the input image"""
        input_batch = (
            self.preprocess(image).unsqueeze(0).to(self.device)
        )  # create a batch

        pred = self.model(input_batch)[0]
        pred = normalize(pred).squeeze().cpu().detach().numpy()

        mask = Image.fromarray(pred * 255).convert("RGB")
        mask = mask.resize((image.size), resample=Image.BILINEAR)
        return mask

    def __call__(self, image: np.ndarray, background_image: Image.Image) -> Image.Image:
        image = Image.fromarray(image)
        background_image = Image.fromarray(background_image)
        background_image = background_image.resize((image.size), resample=Image.NEAREST)
        mask = self.forward(image)

        return Image.composite(image, background_image, mask.convert("L"))


if __name__ == "__main__":
    image = cv2.imread("photos/child.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # background_image = Image.new("RGBA", (30, 30), (255, 255, 255))
    # background_image.save("/passport_photo/photos/white_background.png")
    background_image = cv2.imread("photos/backgrounds/white.png")[:, :, ::-1]

    background_changer = ChangeBackground()
    image = background_changer(image, background_image)
    plt.imshow(image)
    plt.show()
