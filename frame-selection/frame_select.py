import json
import os
import argparse
import numpy as np
import pickle
import torch



def parse_argument():
    argparser = argparse.ArgumentParser(description = "Extract the frame numbers selected by submodular optimization.")
    argparser.add_argument("--dataset_name", type = str, default = "longvideobench", help = "Name of the dataset")
    argparser.add_argument("--text_frame_score_path", type=str, default = "./output_features/longvideobench/clip/scores.json", help="Path to the text-frame similarity scores.")
    argparser.add_argument("--embedding_path", type=str, default = "./output_features/longvideobench/clip/embeddings.json", help="Path to the frame embeddings.")
    argparser.add_argument("--frame_nums_path", type=str, default = "./output_features/longvideobench/clip/frame_nums.json", help="Path to the index of frames in each video.")
    argparser.add_argument("--pairwise_sim_score_path", type=str, default = "./output_features/longvideobench/clip/pairwise_similarities.json", help="Path to the pairwise similarity scores between frames.")
    argparser.add_argument("--topk_coef", type=float, default=1.0)
    argparser.add_argument("--div_coef", type=float, default=1.0)
    argparser.add_argument("--cov_coef", type = float, default=1.0)
    argparser.add_argument("--output_path", type=str, default = "./output_features/longvideobench/clip", help="Path to save the selected frame indices.")
    argparser.add_argument("--max_frame_nums", type=int, default=32, help="Maximum number of frames to select per video.")

    return argparser.parse_args()

def compute_pairwise_similarities(embedding_dict):
    num_videos = len(embedding_dict)
    sim_matrix_dict = {}
    for video_id in embedding_dict.keys():
        num_frames = len(embedding_dict[video_id])
        embeddings = embedding_dict[video_id]
        sim_matrix = np.zeros((num_frames, num_frames))
        for j in range(num_frames):
            for l in range(num_frames):
                sim_matrix[j, l] = np.dot(embeddings[j][0], embeddings[l][0]) / (np.linalg.norm(embeddings[j]) * np.linalg.norm(embeddings[l]) + 1e-10)
        sim_matrix_dict[video_id] = sim_matrix
    return sim_matrix_dict

def deltaf(max_sim_list, text_frame_scores, pairwise_similarities, candidate_ind, alpha, beta):
    gain = alpha * text_frame_scores[candidate_ind]
    diversity_gain = 0.0
    for i in range(len(max_sim_list)):
        diversity_gain += max(0, pairwise_similarities[i][candidate_ind] - max_sim_list[i])
    diversity_gain *= beta/ len(max_sim_list)
    gain += diversity_gain
    return gain

def greedy_submodular_selection(text_frame_scores, pairwise_similarities, frame_nums, alpha, beta, k):
    if len(text_frame_scores) <= k:
        return sorted(frame_nums)
    selected_indices = []
    max_sim_list = np.zeros(len(text_frame_scores))
    for i in range(k):
        best_gain = -float('inf')
        best_index = -1
        for j in range(len(text_frame_scores)):
            if j in selected_indices:
                continue
            gain = deltaf(max_sim_list, text_frame_scores, pairwise_similarities, j, alpha, beta)
            if gain > best_gain:
                best_gain = gain
                best_index = j
        if best_index == -1:
            # randomly select a frame if no positive gain exists
            remaining_indices = [idx for idx in range(len(text_frame_scores)) if idx not in selected_indices]
            best_index = np.random.choice(remaining_indices)
        selected_indices.append(best_index)
        max_sim_list = np.maximum(max_sim_list, pairwise_similarities[best_index])
    
    selected_frames = sorted([frame_nums[idx] for idx in selected_indices])
    return selected_frames

def is_videoid_match(text_frame_scores_dict, frame_nums_dict, pairwise_similarities_dict):
    if text_frame_scores_dict == frame_nums_dict and frame_nums_dict== pairwise_similarities_dict:
        return True
    else:
        return False

def main():
    args = parse_argument()

    # Step 1: Load data
    if os.path.exists(args.text_frame_score_path):
        with open(args.text_frame_score_path, 'r') as f:
            text_frame_scores_dict = json.load(f)
    else:
        raise FileNotFoundError(f"Text-frame score file not found at {args.text_frame_score_path}")
    
    if os.path.exists(args.embedding_path):
        with open(args.embedding_path, 'r') as f:
            frame_embeddings = json.load(f)
    else:
        raise FileNotFoundError(f"Frame embeddings file not found at {args.embedding_path}")
    
    if os.path.exists(args.frame_nums_path):
        with open(args.frame_nums_path, 'r') as f:
            frame_nums_dict= json.load(f)
    else:
        raise FileNotFoundError(f"Frame numbers file not found at {args.frame_nums_path}")


    # Step 2: check if the features and scores are extracted correctly
    if len(next(iter(text_frame_scores_dict))) == 0:
        raise ValueError("Text-frame similarity scores are empty...")
    print(f"Loaded {len(next(iter(text_frame_scores_dict)))} scores for first video")
    print(f"Sample scores: {next(iter(text_frame_scores_dict))[:3]}")

    if len(frame_embeddings) == 0:
        raise ValueError("Frame embeddings are empty. Please check the feature extraction step.")    
    else:
        print(f"The embedding shape is: {next(iter(frame_embeddings))[0].shape}")

    # Step 3: check if the pairwise similarity scores exists already, otherwise compute them
    if os.path.exists(args.pairwise_sim_score_path):
        with open(args.pairwise_sim_score_path, 'r') as f:
            pairwise_similarities_dict = json.load(f)
        # pairwise_similarities_dict is a list of numpy arrays
    else:
        print("Pairwise similarity scores not found, computing them now...")
        pairwise_similarities_dict = compute_pairwise_similarities(frame_embeddings)
        with open(args.pairwise_sim_score_path, 'w') as f:
            json.dump(pairwise_similarities_dict, f)
    
    # Step 4: perform frame selection using submodular optimization. 
    # First, check length for all needed lists. iterate over all videos, for each video, define the greedy algorithm (This can be changed to threshold greedy later)
    if len(text_frame_scores_dict) != len(frame_nums_dict) or len(text_frame_scores_dict) != len(pairwise_similarities_dict):
        raise ValueError("Length mismatch among text-frame scores, frame embeddings, frame numbers, and pairwise similarities.")
    
    if not is_videoid_match(text_frame_scores_dict, frame_nums_dict, pairwise_similarities_dict):
        raise ValueError("The video IDs in different input files are not the same.")
    selected_frame_nums_dict= {}


    for video_id in text_frame_scores_dict.keys():
        text_frame_scores =  text_frame_scores_dict[video_id]
        pairwise_similarities = pairwise_similarities_dict[video_id]
        frame_nums = frame_nums_dict[video_id]
        selected_frame_nums = greedy_submodular_selection(text_frame_scores, pairwise_similarities, frame_nums, args.topk_coef, args.div_coef, args.max_frame_nums)
        selected_frame_nums_dict[video_id]=(selected_frame_nums)
    
    # Step 5: save the selected frame indices
    output_frame_file = os.path.join(args.output_path, f"vfs_selected_frame_nums_{args.topk_coef}_{args.div_coef}_{args.cov_coef}.json")
    with open(output_frame_file, 'w') as f:
        json.dump(selected_frame_nums_dict, f)

if __name__ == "__main__":
    main()