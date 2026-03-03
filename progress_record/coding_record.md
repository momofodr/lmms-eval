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
- **`frame_select.py`**: Updated input defaults to the new file names (`video_embeddings.json`, `video_frame_nums.json`)
- **`frame_select.py`**: Added `question_to_video.json` as an input so question-level CLIP scores can be matched to video-level frame embeddings and frame indices
- **`frame_select.py`**: Fixed the selection pipeline to join question ids to video paths before loading pairwise similarities and sampled frame numbers
- **`frame_select.py`**: Cleaned up pairwise similarity computation and validation checks for missing mappings or inconsistent lengths
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

frame_select.py
  -> load question-level scores from scores.json
  -> map each question id to its source video with question_to_video.json
  -> load video-level embeddings and frame indices
  -> compute/load pairwise similarities per video
  -> run submodular selection and save selected frames per question
```

## 03/03: Update Frame Selection for Semantic Coverage and New Extractor Outputs

Refined `frame_select.py` so it matches the merged extractor outputs and can
use semantic tag similarities inside the selection objective. The selection
pipeline now combines question-level relevance, frame diversity, and a
probabilistic semantic coverage term.

### Changes

- **`frame_select.py`**: Simplified the CLI to take `dataset_name` and `model_name`, then derive all input paths internally from the standard output directory layout
- **`frame_select.py`**: Switched input defaults to the current extractor outputs: `scores.json`, `video_embeddings.json`, `video_frame_nums.json`, `question_to_video.json`, and `tags_score_with_dict.json`
- **`frame_select.py`**: Joined question-level score vectors to video-level embeddings and frame indices through `question_to_video.json`
- **`frame_select.py`**: Added explicit JSON serialization/deserialization for pairwise similarity matrices, since numpy arrays cannot be written to JSON directly
- **`frame_select.py`**: Extended the greedy objective to include a semantic coverage term based on `importance_scores` and `similarity_matrix` from the semantic-tag output
- **`frame_select.py`**: Normalized semantic scores for coverage by clipping cosine similarities into `[0, 1]` before updating the residual semantic coverage state
- **`frame_select.py`**: Added validation checks for semantic result alignment, including matching `video_path`, frame indices, and matrix dimensions
- **`frame_select.py`**: Added validation for pairwise similarity cache shapes and a `--refresh_pairwise_cache` flag to recompute similarities from current embeddings when needed
- **`frame_select.py`**: Added explicit handling for zero-frame cases to avoid downstream divide-by-zero or shape errors
- **`frame_select.py`**: Added structured logging (`frame_select.log`) for arguments, cache behavior, validation status, and per-question selection results

### Updated Flow

```
frame_select.py
  -> derive model-specific input paths from output_features/<dataset>/<model>/
  -> load question-level CLIP scores and semantic-tag results
  -> map each question id to its source video with question_to_video.json
  -> load or recompute pairwise frame similarities per video
  -> validate semantic/frame alignment and cache consistency
  -> run greedy submodular selection with relevance + diversity + semantic coverage
  -> save selected frames per question and log the full run
```

### Addendum: Wire Selected Frames into LongVideoBench Evaluation

Connected the selected-frame JSON output to the actual `lmms-eval` inference
path so LongVideoBench can consume question-specific frame indices during
LLaVA-OneVision evaluation.

### Changes

- **`longvideobench_val_i.yaml`**: Replaced the hardcoded placeholder frame-index path with `metadata.frame_idx_path: "${FRAME_IDX_PATH}"`
- **`longvideobench/utils.py`**: Expanded `FRAME_IDX_PATH` from the environment, added validation for unresolved variables and missing files, and logged how many dataset IDs are missing from the selected-frame JSON before evaluation starts
- **`llava_onevision.py`**: Removed the unused `frame_ind_file` model argument after switching fully to `doc["frame_idx"]` injected by task preprocessing
- **`llava_onevision.py`**: Hardened `load_video_with_ind` by clamping out-of-range frame indices before `VideoReader.get_batch(...)` and falling back safely when no valid indices remain
- **`frame-selection/scripts/llava_onevision_longvideo_bench.sh`**: Updated the actual evaluation launcher to require `FRAME_IDX_PATH`, check that the file exists before startup, and enable selected-frame inference with `use_topk=True`
- **`examples/models/llava_onevision_lvbench.sh`**: Reverted an accidental launcher edit so only the intended frame-selection script carries the new selected-frame behavior

### Updated Eval Flow

```
frame_select.py
  -> save selected frame indices per question to JSON

llava_onevision_longvideo_bench.sh
  -> require FRAME_IDX_PATH and verify the file exists
  -> run lmms-eval with use_topk=True

longvideobench_val_i.yaml + utils.add_frame_idx_to_docs
  -> read FRAME_IDX_PATH from task metadata
  -> attach doc["frame_idx"] to each dataset example
  -> log any missing question IDs before evaluation

llava_onevision.py
  -> read doc["frame_idx"]
  -> load the selected frames (with index validation)
  -> run inference on the selected-frame subset
```
