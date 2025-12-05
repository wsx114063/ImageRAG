import argparse
import os
from PIL import Image
import numpy as np
import openai
import torch
from diffusers import AutoPipelineForText2Image, DiffusionPipeline, ControlNetModel, AutoencoderKL
from transformers import CLIPVisionModelWithProjection

from utils import *
from retrieval import init_faiss_retrieval, search_bird_image_path, search_car_image_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="imageRAG pipeline")
    parser.add_argument("--openai_api_key", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--ip_scale", type=float, default=0.7)
    parser.add_argument("--data_lim", type=int, default=-1)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--out_name", type=str, default="out")
    parser.add_argument("--out_path", type=str, default="results")
    parser.add_argument("--embeddings_path", type=str, default="")
    parser.add_argument("--mode", type=str, default="sd_first", choices=['sd_first', 'generation'])
    parser.add_argument("--only_rephrase", action='store_true')
    parser.add_argument("--retrieval_method", type=str, default="CLIP", choices=['CLIP', 'SigLIP', 'MoE', 'gpt_rerank'])
    parser.add_argument("--bird_index_dir", type=str, default="datasets/bird/index", help="Bird FAISS index 目錄")
    parser.add_argument("--car_index_dir", type=str, default="datasets/car/index", help="Car FAISS index 目錄")

    args = parser.parse_args()
    
    openai.api_key = args.openai_api_key
    os.environ["OPENAI_API_KEY"] = openai.api_key
    client = openai.OpenAI()

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

    # 初始化 FAISS 檢索引擎 (只載入一次)
    init_faiss_retrieval(
        bird_index_dir=args.bird_index_dir,
        car_index_dir=args.car_index_dir,
        device=device
    )

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter",
        subfolder="models/image_encoder",
        torch_dtype=torch.float16,
        cache_dir=args.hf_cache_dir
    )
    
    pipe_ip_control = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
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
        # 使用 extract_keywords 解析關鍵字
        keywords = extract_keywords(args.prompt, client)

        # 解析 JSON 並存入變數
        bird_keyword = keywords.get("bird", "")
        car_keyword = keywords.get("car", "")
        print(f"🐦 Bird keyword: {bird_keyword}")
        print(f"🚗 Car keyword: {car_keyword}")
        
        # 使用 FAISS 做Retrieval
        bird_image_path = search_bird_image_path(bird_keyword, k=1, index_type="combined")
        car_image_path = search_car_image_path(car_keyword, k=1, index_type="combined")
        
        print(f"🐦 Bird 搜尋結果: {bird_image_path}")
        print(f"🚗 Car 搜尋結果: {car_image_path}")
        
        ip_image = Image.open(bird_image_path).convert("RGB")
        control_image = Image.open(car_image_path).convert("RGB")

        pipe_ip_control.set_ip_adapter_scale([0.35, 0.35])

        out_image = pipe_ip_control(
            prompt=args.prompt,
            ip_adapter_image=[ip_image, control_image],
            negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
            num_inference_steps=50,
            generator=generator1,
        ).images[0]

        cur_out_path = os.path.join(args.out_path, f"{args.out_name}.png")
        out_image.save(cur_out_path)