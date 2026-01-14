import torch
import transformers
import numpy as np

from decord import VideoReader, cpu, gpu
from PIL import Image
import argparse
import os
import pickle

import json
# This file is used to extract features from videos. In particular, it takes the video path and text data as input, uses clip model (or other models) 
# to extract frame-level features and compute text-frame similarity scores. The extracted feature embeddings and similarity scores are saved to the output_features folder.

def parse_argument():
    parser = argparse.ArgumentParser(description="Extract features from videos and text using a pre-trained model.")
    parser.add_argument("--dataset_name", type=str, default="longvideobench", help="Name of the dataset. ")
    parser.add_argument("--video_path", type=str, default= "/mnt/data/shuoxing/vllm_frame_select/hub/datasets--longvideobench--LongVideoBench/snapshots/60d1c89c1919a198b73be39c2babb213b29d6a5c/", help="Path to the input image.")
    parser.add_argument("--model_name", type=str, default="clip", help="Name of the pre-trained model to use.")
    parser.add_argument("--output_path", type=str, default = "./output_features", help="Path to save the extracted features.")
    return parser.parse_args()

def find_video_paths(dataset_name, video_base_path):
    if dataset_name == "longvideobench":
        return video_base_path
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

def main(args):

    # Prepare dataset paths
    video_base_path = find_video_paths(args.dataset_name, args.video_path)
    if args.dataset_name == "longvideobench":
        label_file = os.path.join(video_base_path, "lvb_val.json")
        video_path = os.path.join(video_base_path, "videos")
    else:
        raise ValueError(f"Dataset {args.dataset_name} not supported.")
    
    with open(label_file, 'r') as f:
        datas = json.load(f)

    # Load pre-trained model
    if args.model_name == 'clip': 
        from transformers import CLIPProcessor, CLIPModel
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        model.to('cuda' if torch.cuda.is_available() else 'cpu')
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    else:
        # TODO: Add support for other models
        raise NotImplementedError(f"Model {args.model_name} not supported yet.")
    
    # Start feature extraction
    # 1. Prepare output path
    os.makedirs(os.path.join(args.output_path, args.dataset_name), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, args.dataset_name, args.model_name), exist_ok=True)
    output_feature_path = os.path.join(args.output_path, args.dataset_name, args.model_name)

    # 2. Extract features
    scores = {}
    fn = {}
    embs = {}
    for data in datas:
        text = data['question']

        if args.dataset_name == "longvideobench":
            video_file = os.path.join(video_path, data['video_path'])
        else:
            raise ValueError(f"Dataset {args.dataset_name} not supported.")
        
        duration = data.get('duration', None)
        try:
            vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
            fps = vr.get_avg_fps()
            frame_nums = int(len(vr)/int(fps))
        except Exception as e:
            print(f"Error reading video {video_file}: {e}")
            continue

        score = []
        frame_num = []
        embedding = []
        if args.model_name == 'clip':
            input_text = processor(text=text, return_tensors="pt", padding=True, truncation = True).to('cuda' if torch.cuda.is_available() else 'cpu')
            text_features = model.get_text_features(**input_text)
            for i in range(frame_nums):
                frame = (vr[i * int(fps)]).asnumpy()
                image = Image.fromarray(frame)
                input_image = processor(images=image, return_tensors="pt", padding=True).to('cuda' if torch.cuda.is_available() else 'cpu')
                with torch.no_grad():
                    image_features = model.get_image_features(**input_image)
                clip_score = torch.cosine_similarity(text_features, image_features)
                embedding.append(image_features.cpu().numpy())
                score.append(clip_score.cpu().item())
                frame_num.append(i * int(fps))
        # score = score-min(score)
        scores[data['video_id']] = score
        fn[data['video_id']] = frame_num
        embs[data['video_id']] = embedding
    
    output_score_file = os.path.join(output_feature_path, "scores.json")
    output_emb_file = os.path.join(output_feature_path, "embeddings.json")
    output_frame_file = os.path.join(output_feature_path, "frame_nums.json")
    with open(output_score_file, 'w') as f:
        json.dump(scores, f)
    with open(output_emb_file, 'w') as f:
        json.dump(embs, f)
    with open(output_frame_file, 'w') as f:
        json.dump(fn, f)

if __name__ == "__main__":
    args = parse_argument()
    main(args)

