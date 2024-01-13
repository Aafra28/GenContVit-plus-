import os
import torch
import cv2
from torchvision import transforms, datasets
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    ShiftScaleRotate,
    CLAHE,
    RandomRotate90,
    Transpose,
    ShiftScaleRotate,
    HueSaturationValue,
    GaussNoise,
    Sharpen,
    Emboss,
    RandomBrightnessContrast,
    OneOf,
    Compose,
)
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from tqdm import tqdm
from torchvision.models.video import mc3_18

# Define the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the MC3-18 model
mc3_model = mc3_18(weights=None).to(device).eval()

# Data augmentation setup
def strong_aug(p=0.5):
    return Compose(
        [
            RandomRotate90(p=0.2),
            Transpose(p=0.2),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            OneOf(
                [
                    GaussNoise(),
                ],
                p=0.2,
            ),
            ShiftScaleRotate(p=0.2),
            OneOf(
                [
                    CLAHE(clip_limit=2),
                    Sharpen(),
                    Emboss(),
                    RandomBrightnessContrast(),
                ],
                p=0.2,
            ),
            HueSaturationValue(p=0.2),
        ],
        p=p,
    )

# Data loading and normalization
def load_and_preprocess_frames(video_file, num_frames=15):
    vr = VideoReader(video_file, ctx=cpu(0))
    step_size = max(1, len(vr) // num_frames)
    frames = vr.get_batch(list(range(0, len(vr), step_size))[:num_frames]).asnumpy()

    # Augment each frame individually
    aug = strong_aug(p=0.9)
    augmented_frames = [aug(Image.fromarray(frame)) for frame in frames]

    # Normalize frames and convert to tensors
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    input_tensors = [transform(np.array(frame)) for frame in augmented_frames]

    return input_tensors

# Perform spatiotemporal analysis
def spatiotemporal_analysis(frames_batch):
    # Assume frames_batch is a list of input tensors
    frames_batch = torch.stack(frames_batch).to(device)

    # Assuming mc3_model is the pre-trained MC3-18 model
    with torch.no_grad():
        # Perform spatiotemporal analysis using the MC3-18 model
        logits = mc3_model(frames_batch)

        # Convert logits to probabilities using softmax
        probs = torch.nn.functional.softmax(logits, dim=1)

        # Assuming the model is trained for binary classification
        # Use the class with the highest probability as the final prediction
        predictions = torch.argmax(probs, dim=1)

        # Print logits and probabilities
        print(logits)
        print(probs)

        # Return predictions
        return predictions.item()

# Example usage
video_file = 'path/to/your/video.mp4'
num_frames = 15
frames_batch = load_and_preprocess_frames(video_file, num_frames)
prediction = spatiotemporal_analysis(frames_batch)
print(f"Final Prediction: {prediction}")
