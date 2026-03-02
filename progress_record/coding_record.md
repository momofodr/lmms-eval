# Coding Record

---- command for summarizing using llm: Summarize what you did in a clean and clear manner and put it as the work of 03/02 in coding_record.m

## 02/27: Pass Frame Indices via YAML Config

The model needs pre-computed frame indices at inference time. Instead of
loading a file inside the model code, we pass the path through the YAML
task config and inject frame indices into each dataset row before inference.

### Changes

- **`lmms_eval/api/task.py`**: When `metadata` exists in the task config, pass it as a second argument to `process_docs` (backward-compatible — calls without metadata still work)
- **`longvideobench_val_i.yaml`**: Added `process_docs: !function utils.add_frame_idx_to_docs` and `metadata.frame_idx_path` pointing to the selected frames JSON
- **`longvideobench/utils.py`**: Added `add_frame_idx_to_docs(dataset, metadata)` — reads the JSON file from `metadata["frame_idx_path"]`, looks up each doc by its `id`, and attaches matching frame indices as `doc["frame_idx"]`

### Flow

```
YAML metadata.frame_idx_path
  -> task.py calls process_docs(dataset, metadata)
    -> add_frame_idx_to_docs loads JSON, adds doc["frame_idx"] per row
      -> llava_onevision.py reads doc["frame_idx"] at inference
```

## 03/01: Use Question-Level ID as Key Across Pipeline

LongVideoBench has multiple questions per video. Frame selection is
question-dependent (CLIP scores use the question text), so `video_id` caused
overwrites. Changed all files to use question-level `id` instead.

### Changes

- **`feature_extract.py`**: Key `video_id` → `id`; fixed embedding JSON serialization (`.tolist()`); added `torch.no_grad()` for text features
- **`frame_select.py`**: Renamed `is_videoid_match` → `is_key_match` with corrected set-based key comparison; renamed loop var to `doc_id`
- **`semantic_tags_extract.py`**: Result dict key `video_id` → `id`
- **`llava_onevision.py`**: Simplified `load_video_with_ind` to accept frame indices directly from `doc['frame_idx']` instead of loading a file internally

### End-to-End Flow

```
feature_extract.py (id) -> frame_select.py (id) -> selected_frames.json (id)
  -> YAML metadata -> process_docs (lookup by doc["id"]) -> doc["frame_idx"]
    -> llava_onevision.py load_video_with_ind
```

## 03/02: Merge Semantic Tag Extraction into Feature Extraction

Unified the frame-score pipeline and the semantic-tag pipeline so both can be
run from a single extractor. This reduces duplicated logic around video
loading, CLIP embedding extraction, caching, and resume behavior.

### Changes

- **`feature_extract.py`**: Merged the functionality from `semantic_tags_extract.py` into the main extractor
- **`feature_extract.py`**: Added CLI flags to control which branch runs: `--skip_text_scores`, `--skip_semantic_tags`, and `--semantic_top_n`
- **`feature_extract.py`**: Added KeyBERT-based semantic tag extraction and frame-by-tag CLIP similarity computation
- **`feature_extract.py`**: Changed semantic result storage from pickle to JSON (`tags_score_with_dict.json`) by serializing `similarity_matrix` with `.tolist()`
- **`feature_extract.py`**: Removed the redundant `tag_video_features.pkl` cache and reused the existing JSON frame metadata and embedding files instead
- **`feature_extract.py`**: Added explanatory comments throughout the file to document the purpose of each major section
- **`semantic_tags_extract.py`**: Simplified into a compatibility wrapper that calls `feature_extract.py` in semantic-only mode
- **`README.md`**: Updated documentation to reflect the merged workflow and the new JSON semantic output format

### Updated Flow

```
feature_extract.py
  -> sample video frames and extract CLIP image embeddings
  -> compute question-to-frame similarity scores (scores.json)
  -> optionally extract semantic tags with KeyBERT
  -> compute frame-by-tag similarity matrix (tags_score_with_dict.json)
  -> save shared frame metadata for resume and downstream frame selection

semantic_tags_extract.py
  -> calls feature_extract.py with text-score branch disabled
```
