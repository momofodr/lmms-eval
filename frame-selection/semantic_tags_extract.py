# Runnning command: uv run ./frame-selection/semantic_tags_extract.py
import numpy as np
import os
import json
import argparse
import torch
from decord import VideoReader, cpu, gpu

from PIL import Image
import argparse
import os
from keybert import KeyBERT
import pickle

def parse_argument():
    argparser = argparse.ArgumentParser(description="Extract semantic tags from datasets.")
    argparser.add_argument("--dataset_name", type=str, default="longvideobench", help="Name of the dataset")
    argparser.add_argument("--video_path", type=str, default= "/mnt/data/shuoxing/vllm_frame_select/hub/datasets--longvideobench--LongVideoBench/snapshots/60d1c89c1919a198b73be39c2babb213b29d6a5c/", help="Path to the input image.")
    argparser.add_argument("--model_name", type=str, default="clip", help="Name of the pre-trained model to use.")
    argparser.add_argument("--output_path", type=str, default = "./output_features", help="Path to save the extracted features.")
    return argparser.parse_args()

def find_video_paths(dataset_name, video_base_path):
    if dataset_name == "longvideobench":
        return video_base_path
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

def format_tag_for_text(key_word_list):
    tag_texts = []
    for  words, _ in key_word_list:
        
        tag_texts.append(f"a photo containing information about {words} .")
    return tag_texts

def main(args):
    ################# First, read video and text data ##################
    video_base_path = find_video_paths(args.dataset_name, args.video_path)
    if args.dataset_name == "longvideobench":
        label_file = os.path.join(video_base_path, "lvb_val.json")
        video_path = os.path.join(video_base_path, "videos")
    else:
        raise ValueError(f"Dataset {args.dataset_name} not supported.")
    
    with open(label_file, 'r') as f:
        datas = json.load(f)
    
    #  Load pre-trained model
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

    
    scores = []
    kw_model = KeyBERT()
    print(f"There are {len(datas)} number of videos in the dataset.")
    for data in datas:
        text = data['question']
        keywords = kw_model.extract_keywords(
            text, 
            keyphrase_ngram_range=(1, 3),  # 1-3 word phrases
            stop_words='english',
            top_n=20
        )
        
        tag_texts = format_tag_for_text(keywords)
        # Validate tag_texts is a non-empty list of strings
        if len(tag_texts) == 0 or not all(isinstance(tag, str) for tag in tag_texts):
            print(f"No valid tags extracted from text: {text}. The length of scores is {len(scores)}")

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

        # first, load the text data from the dataset, then extract tags using existing models, then read frames and compute similarity and save the similarity data.
        #  In particular, the saved data should be a list of np arrays, which is of size (num_frames, num_tags)

        N = len(tag_texts)
        score = np.zeros((frame_nums, N)) 
        frame_num = []
        if args.model_name == 'clip':
            input_text = processor(text=tag_texts, return_tensors="pt", padding=True, truncation = True).to('cuda' if torch.cuda.is_available() else 'cpu')
            with torch.no_grad():
                text_features = model.get_text_features(**input_text)
            for i in range(frame_nums):
                frame = (vr[i * int(fps)]).asnumpy()
                image = Image.fromarray(frame)
                input_image = processor(images=image, return_tensors="pt", padding=True).to('cuda' if torch.cuda.is_available() else 'cpu')
                with torch.no_grad():
                    image_features = model.get_image_features(**input_image)
                similarities = torch.cosine_similarity(
                        image_features.unsqueeze(1),  # Shape: (1, 1, embedding_dim)
                        text_features.unsqueeze(0),   # Shape: (1, N, embedding_dim)
                        dim=2
                    )  # Shape: (1, N)
                
                score[i,:]= similarities.cpu().squeeze().numpy()

            score_min = score.min()
            score_max = score.max()
            # if score_max - score_min > 1e-8:
            #     score = (score - score_min) / (score_max - score_min)
            video_result = {'video_id': data['video_id'],
                'importance_scores': [key_word_score for phrase, key_word_score in keywords],  # KeyBERT scores
                'similarity_matrix': score  # CLIP similarity scores (frames, tags)
            }
            scores.append(video_result) 

    output_tags_score_file = os.path.join(output_feature_path, "tags_score_with_dict.pkl")
    print(f"Videos processing completed. Waiting to be written into output files {output_tags_score_file}.")
    with open(output_tags_score_file, 'wb') as f:
        pickle.dump(scores, f)

if __name__ == "__main__":
    args = parse_argument()
    main(args)