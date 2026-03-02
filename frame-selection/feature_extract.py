import argparse
import json
import logging
import os

import numpy as np
import torch
from decord import VideoReader, cpu
from keybert import KeyBERT
from PIL import Image


# Define the CLI entry points for the combined feature and semantic-tag extractor.
def parse_argument():
    parser = argparse.ArgumentParser(
        description="Extract frame features, text-frame scores, and semantic-tag similarities from videos."
    )
    parser.add_argument("--dataset_name", type=str, default="longvideobench", help="Name of the dataset.")
    parser.add_argument(
        "--video_path",
        type=str,
        default="/mnt/data/shuoxing/vllm_frame_select/hub/datasets--longvideobench--LongVideoBench/snapshots/60d1c89c1919a198b73be39c2babb213b29d6a5c/",
        help="Dataset root path.",
    )
    parser.add_argument("--model_name", type=str, default="clip", help="Name of the pre-trained model to use.")
    parser.add_argument("--output_path", type=str, default="./output_features", help="Path to save extracted outputs.")
    parser.add_argument(
        "--skip_text_scores",
        action="store_true",
        help="Skip question-to-frame score extraction and only compute semantic tag outputs.",
    )
    parser.add_argument(
        "--skip_semantic_tags",
        action="store_true",  # Default is False, meaning we will extract semantic tags unless this flag is set
        help="Skip semantic tag extraction and only compute question-to-frame scores.",
    )
    parser.add_argument(
        "--semantic_top_n",
        type=int,
        default=20,
        help="Maximum number of semantic tags to extract per question.",
    )
    return parser.parse_args()


def find_video_paths(dataset_name, video_base_path):
    if dataset_name == "longvideobench":
        return video_base_path
    raise ValueError(f"Dataset {dataset_name} not supported.")


# Load a JSON file if it exists; otherwise keep a caller-provided empty/default structure.
def load_json_or_default(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Warning: failed to load {path}: {e}. Starting from empty data.")
        return default


def save_json_atomic(path, data):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f)
    os.replace(tmp_path, path)


# Configure both console logging and a persistent log file in the output directory.
def setup_logger(output_feature_path):
    logger = logging.getLogger("feature_extract")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(os.path.join(output_feature_path, "feature_extract.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def format_tag_for_text(keyword_list):
    return [f"a photo containing information about {words}." for words, _ in keyword_list]


# Semantic results are stored as a JSON list of records, one per question.
def load_semantic_results_or_default(path, default):
    records = load_json_or_default(path, default)
    if not isinstance(records, list):
        return default
    return records


def save_semantic_results_atomic(path, semantic_results_by_id):
    # Persist numpy matrices as plain lists so semantic results stay JSON-serializable.
    serializable_results = []
    for result in semantic_results_by_id.values():
        item = dict(result)
        item["similarity_matrix"] = item["similarity_matrix"].tolist()
        serializable_results.append(item)
    save_json_atomic(path, serializable_results)


def extract_video_embeddings(video_file, processor, model, device):
    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    fps = vr.get_avg_fps()
    if fps <= 0:
        raise ValueError(f"Invalid FPS {fps} for {video_file}")

    # Sample one frame per second to keep the feature set aligned with the original pipeline.
    duration_seconds = len(vr) / fps
    second_marks = range(int(duration_seconds))
    frame_nums = []
    embeddings = []

    # For each sampled frame, run the CLIP image encoder and keep both the frame index and embedding.
    for sec in second_marks:
        frame_idx = min(int(sec * fps), len(vr) - 1)
        image = Image.fromarray(vr[frame_idx].asnumpy())
        input_image = processor(images=image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**input_image)
        embeddings.append(image_features.cpu().numpy().tolist())
        frame_nums.append(frame_idx)

    return frame_nums, embeddings


def get_dataset_entries(dataset_name, video_base_path):
    if dataset_name == "longvideobench":
        label_file = os.path.join(video_base_path, "lvb_val.json")
        video_path = os.path.join(video_base_path, "videos")
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    # The label file provides the question text, question id, and video path for each example.
    with open(label_file, "r") as f:
        datas = json.load(f)

    return label_file, video_path, datas


# Load the vision-language model components used for both frame and semantic-tag scoring.
def load_model(model_name):
    if model_name != "clip":
        raise NotImplementedError(f"Model {model_name} not supported yet.")

    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return processor, model, device


def extract_text_frame_scores(text, embeddings, processor, model, device):
    # Encode the full question once, then compare it against every sampled frame embedding.
    input_text = processor(text=text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**input_text)

    scores = []
    for image_embedding in embeddings:
        image_features = torch.tensor(image_embedding, device=device)
        clip_score = torch.cosine_similarity(text_features, image_features)
        scores.append(clip_score.cpu().item())
    return scores


def extract_semantic_tag_result(text, frame_nums, embeddings, kw_model, processor, model, device, top_n):
    # KeyBERT expands the question into several focused phrases before CLIP scoring.
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
        top_n=top_n,
    )
    tag_texts = format_tag_for_text(keywords)
    if not tag_texts or not all(isinstance(tag, str) for tag in tag_texts):
        return None

    tag_phrases = [phrase for phrase, _ in keywords]
    importance_scores = [keyword_score for _, keyword_score in keywords]

    # Encode all semantic-tag prompts together so we can score each frame against every tag.
    input_text = processor(text=tag_texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**input_text)

    similarity_matrix = np.zeros((len(frame_nums), len(tag_texts)))
    # Build a frame-by-tag similarity matrix for downstream frame selection or inspection.
    for frame_idx, image_embedding in enumerate(embeddings):
        image_features = torch.tensor(image_embedding, device=device)
        similarities = torch.cosine_similarity(
            image_features.unsqueeze(1),
            text_features.unsqueeze(0),
            dim=2,
        )
        similarity_matrix[frame_idx, :] = similarities.cpu().squeeze().numpy()

    return {
        "tag_texts": tag_texts,
        "tag_phrases": tag_phrases,
        "importance_scores": importance_scores,
        "similarity_matrix": similarity_matrix,
    }


def run_extraction(args):
    # Resolve dataset metadata and load the shared CLIP model once for the entire run.
    video_base_path = find_video_paths(args.dataset_name, args.video_path)
    label_file, video_path, datas = get_dataset_entries(args.dataset_name, video_base_path)
    processor, model, device = load_model(args.model_name)

    # Create the standard output directory layout used by the frame-selection pipeline.
    os.makedirs(os.path.join(args.output_path, args.dataset_name), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, args.dataset_name, args.model_name), exist_ok=True)
    output_feature_path = os.path.join(args.output_path, args.dataset_name, args.model_name)
    logger = setup_logger(output_feature_path)

    # JSON files hold both per-question scores and the shared per-video frame metadata.
    output_score_file = os.path.join(output_feature_path, "scores.json")
    output_emb_file = os.path.join(output_feature_path, "video_embeddings.json")
    output_frame_file = os.path.join(output_feature_path, "video_frame_nums.json")
    output_q2v_file = os.path.join(output_feature_path, "question_to_video.json")
    output_tags_score_file = os.path.join(output_feature_path, "tags_score_with_dict.json")

    logger.info("Starting feature extraction")
    logger.info("Args: %s", vars(args))
    logger.info("Label file: %s", label_file)
    logger.info("Video directory: %s", video_path)
    logger.info("Output directory: %s", output_feature_path)

    # Reload previous outputs so interrupted runs can resume without recomputing everything.
    scores = load_json_or_default(output_score_file, {})
    frame_nums_by_video = load_json_or_default(output_frame_file, {})
    embeddings_by_video = load_json_or_default(output_emb_file, {})
    question_to_video = load_json_or_default(output_q2v_file, {})
    semantic_results = load_semantic_results_or_default(output_tags_score_file, [])

    # Reuse whichever cache already has a video's sampled frames and CLIP embeddings.
    video_cache = {
        video_key: (frame_nums_by_video[video_key], embeddings_by_video[video_key])
        for video_key in frame_nums_by_video.keys() & embeddings_by_video.keys()
    }

    # Index semantic results by question id so each question can resume independently.
    semantic_results_by_id = {}
    for item in semantic_results:
        if not isinstance(item, dict) or "id" not in item or "similarity_matrix" not in item:
            continue
        restored_item = dict(item)
        # Convert JSON lists back to numpy arrays for downstream numeric operations.
        restored_item["similarity_matrix"] = np.array(restored_item["similarity_matrix"])
        semantic_results_by_id[item["id"]] = restored_item

    kw_model = None
    if not args.skip_semantic_tags:
        # KeyBERT is only needed when the semantic-tag branch is enabled.
        kw_model = KeyBERT()
        logger.info("Loaded KeyBERT model")

    save_every = 20
    processed_since_save = 0
    logger.info(
        "Resume state loaded: scores=%d videos_with_frames=%d videos_with_embeddings=%d question_to_video=%d semantic_results=%d total_records=%d",
        len(scores),
        len(frame_nums_by_video),
        len(embeddings_by_video),
        len(question_to_video),
        len(semantic_results_by_id),
        len(datas),
    )

    for idx, data in enumerate(datas, start=1):
        # Each record corresponds to one question-video pair from the dataset annotations.
        text = data["question"]
        video_key = data["video_path"]
        video_file = os.path.join(video_path, video_key)
        question_id = data["id"]

        # Text scores and semantic tags can be resumed independently.
        need_text_score = not args.skip_text_scores and (
            question_id not in scores or question_id not in question_to_video
        )
        need_semantic = not args.skip_semantic_tags and question_id not in semantic_results_by_id
        if not need_text_score and not need_semantic:
            logger.info("Skipping existing question %s (%d/%d)", question_id, idx, len(datas))
            continue

        try:
            if video_key in video_cache:
                # Reuse previously computed embeddings when multiple questions point to the same video.
                frame_nums, embeddings = video_cache[video_key]
                logger.info("Using cached embeddings for video %s", video_key)
            else:
                logger.info(
                    "Extracting video embeddings for video %s (%d/%d) from %s",
                    video_key,
                    idx,
                    len(datas),
                    video_file,
                )
                frame_nums, embeddings = extract_video_embeddings(video_file, processor, model, device)
                video_cache[video_key] = (frame_nums, embeddings)
                logger.info("Extracted %d sampled frames for video %s", len(frame_nums), video_key)
        except Exception as e:
            logger.exception("Error reading video %s: %s", video_file, e)
            continue

        # Keep shared frame metadata in sync regardless of which output branch is being computed.
        frame_nums_by_video[video_key] = frame_nums
        embeddings_by_video[video_key] = embeddings

        if need_text_score:
            logger.info("Computing text-frame scores for question %s on video %s", question_id, video_key)
            scores[question_id] = extract_text_frame_scores(text, embeddings, processor, model, device)
            question_to_video[question_id] = video_key

        if need_semantic:
            logger.info("Computing semantic tag scores for question %s on video %s", question_id, video_key)
            semantic_result = extract_semantic_tag_result(
                text,
                frame_nums,
                embeddings,
                kw_model,
                processor,
                model,
                device,
                args.semantic_top_n,
            )
            if semantic_result is None:
                logger.warning("No valid semantic tags extracted for question %s: %s", question_id, text)
            else:
                # Store semantic outputs per question so repeated questions on one video stay distinct.
                semantic_result["id"] = question_id
                semantic_result["video_path"] = video_key
                semantic_result["frame_nums"] = frame_nums
                semantic_results_by_id[question_id] = semantic_result

        # Save periodically so a long extraction job can be resumed after interruption.
        processed_since_save += 1
        logger.info(
            "Finished question %s: text_scores=%s semantic_tags=%s",
            question_id,
            "yes" if need_text_score else "no",
            "yes" if need_semantic and question_id in semantic_results_by_id else "no",
        )

        if processed_since_save >= save_every:
            save_json_atomic(output_score_file, scores)
            save_json_atomic(output_emb_file, embeddings_by_video)
            save_json_atomic(output_frame_file, frame_nums_by_video)
            save_json_atomic(output_q2v_file, question_to_video)
            save_semantic_results_atomic(output_tags_score_file, semantic_results_by_id)
            logger.info(
                "Checkpoint saved after %d new questions. scores=%d videos=%d semantic_results=%d",
                processed_since_save,
                len(scores),
                len(embeddings_by_video),
                len(semantic_results_by_id),
            )
            processed_since_save = 0

    # Persist the final state for all outputs after the loop completes.
    save_json_atomic(output_score_file, scores)
    save_json_atomic(output_emb_file, embeddings_by_video)
    save_json_atomic(output_frame_file, frame_nums_by_video)
    save_json_atomic(output_q2v_file, question_to_video)
    save_semantic_results_atomic(output_tags_score_file, semantic_results_by_id)
    logger.info(
        "Run complete. Final counts: scores=%d videos=%d question_to_video=%d semantic_results=%d",
        len(scores),
        len(embeddings_by_video),
        len(question_to_video),
        len(semantic_results_by_id),
    )


def main(args):
    run_extraction(args)


if __name__ == "__main__":
    main(parse_argument())
