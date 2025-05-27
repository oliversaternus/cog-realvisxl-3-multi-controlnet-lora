# RealVisXL V3.0 with multi ControlNet and custom Loras

Fork of https://github.com/fofr/cog-realvisxl-3-multi-controlnet-lora
with additional anyline controlnet and local weights loader script.

Before pushing model, load the weights by executing `cog run python download-weights-local.py`

Using anyline controlnet from https://huggingface.co/TheMistoAI/MistoLine

RealVis XL V3.0 with:

- img2img
- inpainting
- custom Replicate lora loading
- up to 3 simultaneous controlnets with different images
  - canny
  - midas depth
  - leres depth
  - soft edge hed
  - soft edge pidi
  - openpose
  - QR Monster (illusions)
  - lineart
  - lineart anime
  - lineart anyline
- img2img plus controlnet
- inpainting plus controlnet
- controlnet conditioning strengths
- controlnet start and end controls
- SDXL refiner
- Image resizing based on width/height, input image or a control image
- Disable safety checker via API
