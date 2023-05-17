from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("mps")

# Recommended if your computer has < 64 GB of RAM
pipe.enable_attention_slicing()

# prompt = "a photo of an astronaut riding a horse on mars"
prompt = input("Enter a prompt: ")

# First-time "warmup" pass if PyTorch version is 1.13 (see explanation above)
_ = pipe(prompt, num_inference_steps=1)

# Results match those from the CPU device after the warmup pass.
filename = prompt.replace(" ", "_")
for i, image in enumerate(pipe(prompt).images):
    image.save(f"{filename}-{i}.png")
