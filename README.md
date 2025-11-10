# Image Generator Using OmniGen2

This repository contains a script that generates images from a set of given prompts.

## How to Run

1. Currently, the script is only set up for a given `.jsonl` file. You can modify the file: `samples/input/sample_prompts.jsonl` or you can modify the script `generate_images.py` line `13` and redirect the variable `PROMPTS_FILE`.

2. Select the number of images you want to generate ($\leq$ length of your prompts file) by rewriting the variable `NUM_IMAGES` in line `15` of `generate_images.py`.

3. Optionally, modify the number of inference steps in the generation by modifying `num_inference_steps` on line `76` of `generate_images.py`. 50 is the default value, and 28 is what I found to be best, however, lower values will result in faster generation but lower quality.

4. Run the entire script (installs dependencies and OmniGen2 repository). Expect this to require about **12-15GB** of storage.
```bash
    chmod +x run.sh
    ./run.sh
```

## Notes 

* I will eventually add file arguments to streamline the script (i.e. `PROMPTS_FILE`, `NUM_IMAGES`, `num_inference_steps`)

* I will still work on optimizing image generation. There is a recommended package that could cut generation time in half at no cost to quality, but I was having conflicts with my `pip` version.

* This model recommends an **NVIDIA RTX 3090** or equivalent with **17GB of VRAM**. You will need about **15GB of storage**

## OmniGen2

* https://github.com/VectorSpaceLab/OmniGen2