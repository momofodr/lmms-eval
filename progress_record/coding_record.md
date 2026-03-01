# Coding Record

## 02/27: Pass Frame Indices via YAML Config

Pass pre-computed frame selection indices through the YAML task config.

### Changes

- **`lmms_eval/api/task.py`**: Pass `metadata` to `process_docs` when available (backward-compatible)
- **`longvideobench_val_i.yaml`**: Added `process_docs` and `metadata.frame_idx_path`
- **`longvideobench/utils.py`**: Added `add_frame_idx_to_docs()` — loads frame indices JSON, maps onto dataset rows

### Flow

```
YAML metadata.frame_idx_path -> process_docs -> doc['frame_idx'] -> load_video_index
```
