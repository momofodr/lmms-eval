import json
import os
import argparse
import logging
import numpy as np



def parse_argument():
    argparser = argparse.ArgumentParser(description = "Extract the frame numbers selected by submodular optimization.")
    argparser.add_argument("--dataset_name", type = str, default = "longvideobench", help = "Name of the dataset")
    argparser.add_argument("--model_name", type=str, default="clip", help="Name of the feature extractor model.")
    argparser.add_argument("--topk_coef", type=float, default=1.0)
    argparser.add_argument("--div_coef", type=float, default=1.0)
    argparser.add_argument("--cov_coef", type = float, default=1.0)
    argparser.add_argument("--output_path", type=str, default = "./output_features", help="Base path to save the selected frame indices.")
    argparser.add_argument("--max_frame_nums", type=int, default=32, help="Maximum number of frames to select per video.")
    argparser.add_argument(
        "--refresh_pairwise_cache",
        action="store_true",
        help="Recompute pairwise similarities from the current embeddings instead of reusing a saved cache.",
    )

    return argparser.parse_args()


def setup_logger(output_dir):
    logger = logging.getLogger("frame_select")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(os.path.join(output_dir, "frame_select.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def compute_pairwise_similarities(embedding_dict):
    sim_matrix_dict = {}
    for video_key, embeddings in embedding_dict.items():
        num_frames = len(embeddings)
        sim_matrix = np.zeros((num_frames, num_frames))
        for j in range(num_frames):
            embedding_j = np.asarray(embeddings[j][0], dtype=float)
            for l in range(num_frames):
                embedding_l = np.asarray(embeddings[l][0], dtype=float)
                sim_matrix[j, l] = np.dot(embedding_j, embedding_l) / (
                    np.linalg.norm(embedding_j) * np.linalg.norm(embedding_l) + 1e-10
                )
        sim_matrix_dict[video_key] = sim_matrix
    return sim_matrix_dict


def load_json(path, description):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{description} file not found at {path}")
    with open(path, "r") as f:
        return json.load(f)


def serialize_pairwise_similarities(pairwise_similarities_dict):
    # JSON cannot store numpy arrays directly, so convert each matrix to nested Python lists.
    return {video_key: sim_matrix.tolist() for video_key, sim_matrix in pairwise_similarities_dict.items()}


def deserialize_pairwise_similarities(pairwise_similarities_dict):
    # Restore JSON-loaded lists back to numpy arrays for numerical operations during selection.
    return {video_key: np.asarray(sim_matrix) for video_key, sim_matrix in pairwise_similarities_dict.items()}

def deltaf(max_sim_list, text_frame_scores, pairwise_similarities, candidate_ind, alpha, beta, gamma, semantic_scores, semantic_vector):
    gain = alpha * text_frame_scores[candidate_ind]
    diversity_gain = 0.0
    for i in range(len(max_sim_list)):
        diversity_gain += max(0, pairwise_similarities[i][candidate_ind] - max_sim_list[i])
    diversity_gain *= beta/ len(max_sim_list)
    gain += diversity_gain
    gain += gamma * np.dot(semantic_vector, semantic_scores[ candidate_ind, :])
    return gain

def greedy_submodular_selection(text_frame_scores, pairwise_similarities, semantic_results, frame_nums, alpha, beta, gamma, k):
    if len(text_frame_scores) == 0:
        return []
    if len(text_frame_scores) <= k:
        return sorted(frame_nums)
    selected_indices = []
    max_sim_list = np.zeros(len(text_frame_scores))
    semantic_vector = np.array(semantic_results["importance_scores"])
    semantic_scores = np.array(semantic_results["similarity_matrix"])
    semantic_scores = np.clip(semantic_scores, 0.0, 1.0)
    for i in range(k):
        best_gain = -float('inf')
        best_index = -1
        for j in range(len(text_frame_scores)):
            if j in selected_indices:
                continue
            gain = deltaf(max_sim_list, text_frame_scores, pairwise_similarities, j, alpha, beta, gamma, semantic_scores, semantic_vector)
            if gain > best_gain:
                best_gain = gain
                best_index = j
        if best_index == -1:
            # randomly select a frame if no positive gain exists
            remaining_indices = [idx for idx in range(len(text_frame_scores)) if idx not in selected_indices]
            best_index = np.random.choice(remaining_indices)
        selected_indices.append(best_index)
        max_sim_list = np.maximum(max_sim_list, pairwise_similarities[best_index])
        semantic_vector = semantic_vector * (1-semantic_scores[ best_index, :])
    
    selected_frames = sorted([frame_nums[idx] for idx in selected_indices])
    
    return selected_frames

def is_video_key_match(frame_nums_dict, frame_embeddings, pairwise_similarities_dict):
    """Check that all video-level inputs share the same set of keys."""
    return set(frame_nums_dict.keys()) == set(frame_embeddings.keys()) == set(pairwise_similarities_dict.keys())


def validate_pairwise_shapes(frame_embeddings, pairwise_similarities_dict):
    for video_key, embeddings in frame_embeddings.items():
        num_frames = len(embeddings)
        if video_key not in pairwise_similarities_dict:
            raise ValueError(f"Missing pairwise similarity matrix for video {video_key}.")
        pairwise_matrix = pairwise_similarities_dict[video_key]
        if pairwise_matrix.shape != (num_frames, num_frames):
            raise ValueError(
                f"Pairwise similarity shape mismatch for video {video_key}: expected {(num_frames, num_frames)}, got {pairwise_matrix.shape}"
            )


def validate_semantic_alignment(question_id, video_key, frame_nums, semantic_results):
    if semantic_results.get("video_path") != video_key:
        raise ValueError(
            f"Semantic result video mismatch for question {question_id}: expected {video_key}, got {semantic_results.get('video_path')}"
        )
    semantic_frame_nums = semantic_results.get("frame_nums", [])
    if semantic_frame_nums != frame_nums:
        raise ValueError(
            f"Semantic frame indices mismatch for question {question_id}: expected {len(frame_nums)} frames, got {len(semantic_frame_nums)}"
        )

    similarity_matrix = np.asarray(semantic_results.get("similarity_matrix", []))
    importance_scores = np.asarray(semantic_results.get("importance_scores", []))
    if similarity_matrix.ndim != 2:
        raise ValueError(f"Semantic similarity matrix must be 2D for question {question_id}.")
    if similarity_matrix.shape[0] != len(frame_nums):
        raise ValueError(
            f"Semantic similarity matrix row mismatch for question {question_id}: expected {len(frame_nums)}, got {similarity_matrix.shape[0]}"
        )
    if similarity_matrix.shape[1] != len(importance_scores):
        raise ValueError(
            f"Semantic similarity matrix/tag weight mismatch for question {question_id}: expected {similarity_matrix.shape[1]} tag weights, got {len(importance_scores)}"
        )

def main():
    args = parse_argument()
    feature_output_path = os.path.join(args.output_path, args.dataset_name, args.model_name)
    os.makedirs(feature_output_path, exist_ok=True)
    logger = setup_logger(feature_output_path)
    text_frame_score_path = os.path.join(feature_output_path, "scores.json")
    embedding_path = os.path.join(feature_output_path, "video_embeddings.json")
    frame_nums_path = os.path.join(feature_output_path, "video_frame_nums.json")
    question_to_video_path = os.path.join(feature_output_path, "question_to_video.json")
    semantic_tags_score_path = os.path.join(feature_output_path, "tags_score_with_dict.json")
    pairwise_sim_score_path = os.path.join(feature_output_path, "pairwise_similarities.json")

    logger.info("Starting frame selection")
    logger.info("Args: %s", vars(args))
    logger.info("Feature directory: %s", feature_output_path)

    # Step 1: Load data
    text_frame_scores_dict = load_json(text_frame_score_path, "Text-frame score")
    frame_embeddings = load_json(embedding_path, "Frame embeddings")
    frame_nums_dict = load_json(frame_nums_path, "Frame numbers")
    question_to_video = load_json(question_to_video_path, "Question-to-video mapping")
    semantic_tags_score_dict = load_json(semantic_tags_score_path, "Semantic tags scores")
    semantic_results_by_id = {item["id"]: item for item in semantic_tags_score_dict}
    logger.info(
        "Loaded inputs: text_scores=%d videos_with_embeddings=%d videos_with_frames=%d question_to_video=%d semantic_results=%d",
        len(text_frame_scores_dict),
        len(frame_embeddings),
        len(frame_nums_dict),
        len(question_to_video),
        len(semantic_results_by_id),
    )


    # Step 2: check if the features and scores are extracted correctly
    if not text_frame_scores_dict:
        raise ValueError("Text-frame similarity scores are empty.")
    first_question_id = next(iter(text_frame_scores_dict))
    first_score_list = text_frame_scores_dict[first_question_id]
    if len(first_score_list) == 0:
        raise ValueError("Text-frame similarity scores are empty...")
    logger.info("Loaded %d scores for first question %s", len(first_score_list), first_question_id)
    logger.info("Sample scores: %s", first_score_list[:3])

    if len(frame_embeddings) == 0:
        raise ValueError("Frame embeddings are empty. Please check the feature extraction step.")    
    else:
        first_video_key = next(iter(frame_embeddings))
        first_embedding = np.asarray(frame_embeddings[first_video_key][0])
        logger.info("First embedding shape for video %s: %s", first_video_key, first_embedding.shape)

    # Step 3: check if the pairwise similarity scores exists already, otherwise compute them
    if os.path.exists(pairwise_sim_score_path) and not args.refresh_pairwise_cache:
        logger.info("Loading cached pairwise similarities from %s", pairwise_sim_score_path)
        with open(pairwise_sim_score_path, 'r') as f:
            pairwise_similarities_dict = deserialize_pairwise_similarities(json.load(f))
    else:
        logger.info("Computing pairwise similarity scores from current embeddings")
        pairwise_similarities_dict = compute_pairwise_similarities(frame_embeddings)
        with open(pairwise_sim_score_path, 'w') as f:
            json.dump(serialize_pairwise_similarities(pairwise_similarities_dict), f)
        logger.info("Saved pairwise similarities to %s", pairwise_sim_score_path)
    
    # Step 4: perform frame selection using submodular optimization.
    # Scores are keyed by question id, while embeddings/frame metadata are keyed by video path.
    if not is_video_key_match(frame_nums_dict, frame_embeddings, pairwise_similarities_dict):
        raise ValueError("The keys in frame embeddings, frame numbers, and pairwise similarities are not the same.")
    validate_pairwise_shapes(frame_embeddings, pairwise_similarities_dict)
    logger.info("Validated video-level embeddings, frame indices, and pairwise similarity shapes")

    missing_question_mappings = set(text_frame_scores_dict.keys()) - set(question_to_video.keys())
    if missing_question_mappings:
        raise ValueError(f"Missing question_to_video entries for {len(missing_question_mappings)} questions.")
    logger.info("Validated question-to-video mappings for %d questions", len(text_frame_scores_dict))

    selected_frame_nums_dict= {}

    for question_id, text_frame_scores in text_frame_scores_dict.items():
        video_key = question_to_video[question_id]
        if video_key not in frame_nums_dict or video_key not in pairwise_similarities_dict or question_id not in semantic_results_by_id:
            raise ValueError(f"Missing video-level data for question {question_id} and video {video_key}.")


        pairwise_similarities = pairwise_similarities_dict[video_key]
        frame_nums = frame_nums_dict[video_key]
        semantic_results = semantic_results_by_id[question_id]
        validate_semantic_alignment(question_id, video_key, frame_nums, semantic_results)
        if len(text_frame_scores) != len(frame_nums) or len(text_frame_scores) != len(pairwise_similarities):
            raise ValueError(
                f"Length mismatch for question {question_id}: scores={len(text_frame_scores)} frame_nums={len(frame_nums)} pairwise={len(pairwise_similarities)}"
            )
        logger.info(
            "Selecting frames for question %s on video %s: frames=%d tags=%d",
            question_id,
            video_key,
            len(frame_nums),
            len(semantic_results["importance_scores"]),
        )
        selected_frame_nums = greedy_submodular_selection(
            text_frame_scores,
            pairwise_similarities,
            semantic_results,
            frame_nums,
            args.topk_coef,
            args.div_coef,
            args.cov_coef,
            args.max_frame_nums,
        )
        selected_frame_nums_dict[question_id] = selected_frame_nums
        logger.info(
            "Selected %d frames for question %s: %s",
            len(selected_frame_nums),
            question_id,
            selected_frame_nums,
        )
    
    # Step 5: save the selected frame indices
    output_frame_file = os.path.join(
        feature_output_path,
        f"vfs_selected_frame_nums_{args.topk_coef}_{args.div_coef}_{args.cov_coef}.json",
    )
    with open(output_frame_file, 'w') as f:
        json.dump(selected_frame_nums_dict, f)
    logger.info("Saved selected frames for %d questions to %s", len(selected_frame_nums_dict), output_frame_file)

if __name__ == "__main__":
    main()
