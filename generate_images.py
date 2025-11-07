import json
from pathlib import Path
import sys
import torch
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).parent / "OmniGen2"))
from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel

# Config
PROMPTS_FILE = "samples/input/sample_prompts.jsonl"
OUTPUT_DIR = Path("samples/outputs")
NUM_IMAGES = 10

OUTPUT_DIR.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_path = "OmniGen2/OmniGen2"
weight_dtype = torch.float16

pipeline = OmniGen2Pipeline.from_pretrained(
    model_path,
    torch_dtype=weight_dtype,
    trust_remote_code=True,
)

# pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
#     model_path,
#     subfolder="transformer",
#     torch_dtype=weight_dtype,
# )

pipeline.to(device)

# Load prompts
prompts = []
with open(PROMPTS_FILE, "r") as f:
    for i, line in enumerate(f):
        if i >= NUM_IMAGES:
            break
        prompts.append(json.loads(line)["prompt"])
    
print(f"Loaded {len(prompts)} prompts.")

# Generate
for i, prompt in enumerate(prompts, start=1):
    print(f"[{i}/{len(prompts)}] Generating: {prompt}")
    
    image = pipeline(prompt)

    # save image
    out_path = OUTPUT_DIR / f"image_{i:03d}.png"
    image.save(out_path)
    print(f"Save to {out_path}")

print("\nDONE!")