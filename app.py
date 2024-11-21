import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from IPython.display import display

model_id = "timbrooks/instruct-pix2pix"

pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    safety_checker=None
)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

url = "https://images.alphacoders.com/436/436232.jpg"
image = download_image(url)

prompt = "Make the tiger look majestic, dressed in a golden robe with intricate embroidery, a velvet cape, and a jeweled crown. Place the tiger on a grand throne with soft, dramatic lighting for elegance."

images = pipe(
    prompt, 
    image=image, 
    num_inference_steps=10, 
    image_guidance_scale=1
).images

output_path = "generated_image.png"
images[0].save(output_path) 
display(images[0])      
