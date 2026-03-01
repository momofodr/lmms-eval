# Coding Record

## 02/27: Pass Frame Indices via YAML Config

Pass pre-computed frame selection indices through the YAML task config.

### Changes

- **`lmms_eval/api/task.py`**: Pass `metadata` to `process_docs` when available (backward-compatible)
- **`longvideobench_val_i.yaml`**: Added `process_docs` and `metadata.frame_idx_path`
- **`longvideobench/utils.py`**: Added `add_frame_idx_to_docs()` — loads frame indices JSON, maps onto dataset rows

### Flow

```
YAML metadata.frame_idx_path -> process_docs -> doc['frame_idx'] -> load_video_with_ind
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
