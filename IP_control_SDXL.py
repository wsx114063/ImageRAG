import argparse
import os
from PIL import Image
import numpy as np
import openai
import torch
from diffusers import AutoPipelineForText2Image, DiffusionPipeline, ControlNetModel, AutoencoderKL
from transformers import CLIPVisionModelWithProjection

from utils import *
from retrieval import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="imageRAG pipeline")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--ip_scale", type=float, default=0.5)
    parser.add_argument("--data_lim", type=int, default=-1)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--out_name", type=str, default="out")
    parser.add_argument("--out_path", type=str, default="results")
    parser.add_argument("--embeddings_path", type=str, default="")
    parser.add_argument("--mode", type=str, default="sd_first", choices=['sd_first', 'generation'])
    parser.add_argument("--only_rephrase", action='store_true')
    parser.add_argument("--retrieval_method", type=str, default="CLIP", choices=['CLIP', 'SigLIP', 'MoE', 'gpt_rerank'])

    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)
    out_txt_file = os.path.join(args.out_path, args.out_name + ".txt")
    f = open(out_txt_file, "w")
    device = f"cuda:{args.device}" if int(args.device) >= 0 else "cuda"
    data_path = f"datasets/{args.dataset}"

    prompt_w_retreival = args.prompt

    retrieval_image_paths = [os.path.join(data_path, fname) for fname in os.listdir(data_path)]
    if args.data_lim != -1:
        retrieval_image_paths = retrieval_image_paths[:args.data_lim]

    embeddings_path = args.embeddings_path or f"datasets/embeddings/{args.dataset}"

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter",
        subfolder="models/image_encoder",
        torch_dtype=torch.float16,
        cache_dir=args.hf_cache_dir
    )
    '''
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0",
        torch_dtype=torch.float16
    ).to(device)

    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=torch.float16
    ).to(device)
    '''
    pipe_ip_control = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        # controlnet=controlnet,
        # controlnet_conditioning_scale = 0.4,
        # vae=vae,
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
        cache_dir=args.hf_cache_dir
    ).to(device)
    pipe_ip_control.load_ip_adapter("h94/IP-Adapter",
                            subfolder="sdxl_models",
                            weight_name=[
                                "ip-adapter-plus_sdxl_vit-h.safetensors",
                                "ip-adapter-plus_sdxl_vit-h.safetensors"
                            ],
                            cache_dir=args.hf_cache_dir)

    generator1 = torch.Generator(device="cuda").manual_seed(args.seed)

    cur_out_path = os.path.join(args.out_path, f"{args.out_name}_no_imageRAG.png")
    if not os.path.exists(cur_out_path):
        ip_image = Image.open("datasets/example_dataset/Common_Tern_0014_149194.jpg").convert("RGB")

        control_image = Image.open("datasets/example_dataset/Toyota_celica.png").convert("RGB")

        pipe_ip_control.set_ip_adapter_scale([args.ip_scale/2]*2)

        out_image = pipe_ip_control(
            prompt=args.prompt,
            ip_adapter_image=[ip_image, control_image],
            negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
            num_inference_steps=50,
            #num_inference_steps=28
            #guidance_scale=6.5,
            generator=generator1,
        ).images[0]

        cur_out_path = os.path.join(args.out_path, f"{args.out_name}.png")
        out_image.save(cur_out_path)