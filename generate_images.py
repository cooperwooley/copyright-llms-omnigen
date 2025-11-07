import json
from pathlib import Path
import sys
import torch
from torchvision.utils import save_image
import gc

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

if not hasattr(pipeline.transformer, 'enable_teacache'):
    pipeline.transformer.enable_teacache = False

# pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
#     model_path,
#     subfolder="transformer",
#     torch_dtype=weight_dtype,
# )

# Enabled for potential 2X speedup
pipeline.enable_taylorseer = True

from omnigen2.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
scheduler = DPMSolverMultistepScheduler(
    algorithm_type="dpmsolver++",
    solver_type="midpoint",
    solver_order=2,
    prediction_type="flow_prediction",
)
pipeline.scheduler = scheduler

pipeline.to(device)

# Free up memory after loading
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

# Load prompts
prompts = []
with open(PROMPTS_FILE, "r") as f:
    for i, line in enumerate(f):
        if i >= NUM_IMAGES:
            break
        prompts.append(json.loads(line)["prompt"])
    
print(f"Loaded {len(prompts)} prompts.")

# Generate images one at a time
for i, prompt in enumerate(prompts, start=1):
    print(f"[{i}/{len(prompts)}] Generating: {prompt}")
    
    # Generate image
    image = pipeline(
        prompt,
        num_inference_steps=28,
        cfg_range=(0.0, 0.8),
        )

    # save image
    out_path = OUTPUT_DIR / f"image_{i:03d}.png"
    image.save(out_path)
    print(f"Save to {out_path}")

    # Clean up memory after each gen
    if torch.cuda.is_available():
            torch.cuda.empty_cache()
    gc.collect()

print("\nDONE!")