import torch
from weights_downloader import WeightsDownloader
from diffusers import ControlNetModel
from controlnet_aux import (
    AnylineDetector,
)

# Caches and URLs
SDXL_MODEL_CACHE = "./sdxl-cache"
REFINER_MODEL_CACHE = "./refiner-cache"
SAFETY_CACHE = "./safety-cache"
FEATURE_EXTRACTOR = "./feature-extractor"
SDXL_URL = "https://weights.replicate.delivery/default/RealVisXL/RealVisXL_V3.0.tar"
REFINER_URL = (
    "https://weights.replicate.delivery/default/sdxl/refiner-no-vae-no-encoder-1.0.tar"
)
SAFETY_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"

CONTROLNET_MODEL_CACHE = "./controlnet-cache"
CONTROLNET_URL = "https://weights.replicate.delivery/default/controlnet/sdxl-cn-canny-depth-softe-pose-qr.tar"


CONTROLNET_PREPROCESSOR_MODEL_CACHE = "./controlnet-preprocessor-cache"
CONTROLNET_PREPROCESSOR_URL = "https://weights.replicate.delivery/default/controlnet/cn-preprocess-leres-midas-pidi-hed-lineart-openpose.tar"

ANYLINE_PREPROCESSOR_MODEL_PATH = "./anyline-detector-weights"


print("Downloading Safety Checker weights...")
WeightsDownloader.download_if_not_exists(SAFETY_URL, SAFETY_CACHE)

print("Downloading SDXL Model weights...")
WeightsDownloader.download_if_not_exists(SDXL_URL, SDXL_MODEL_CACHE)

print("Downloading Refiner weights...")
WeightsDownloader.download_if_not_exists(REFINER_URL, REFINER_MODEL_CACHE)

print("Downloading ControlNet weights...")
WeightsDownloader.download_if_not_exists(CONTROLNET_URL, CONTROLNET_MODEL_CACHE)

print("Downloading ControlNet Preprocessor weights...")
WeightsDownloader.download_if_not_exists(
    CONTROLNET_PREPROCESSOR_URL, CONTROLNET_PREPROCESSOR_MODEL_CACHE
)

# Initialize anyline controlnet to download weights
print("Initializing Anyline controlnet...")
anyline_control_net = ControlNetModel.from_pretrained(
    "TheMistoAI/MistoLine",
    cache_dir=CONTROLNET_MODEL_CACHE,
    torch_dtype=torch.float16,
    variant="fp16",
)

print("Initializing Anyline ControlNet Preprocessor...")
anyline_control_net_preprocessor = AnylineDetector.from_pretrained(
    ANYLINE_PREPROCESSOR_MODEL_PATH,
    filename="MTEED.pth",
)

print("All weights downloaded.")
