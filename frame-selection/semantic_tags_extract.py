# Running command: uv run ./frame-selection/semantic_tags_extract.py
from feature_extract import parse_argument, run_extraction


def main():
    args = parse_argument()
    args.skip_text_scores = True
    args.skip_semantic_tags = False
    run_extraction(args)


if __name__ == "__main__":
    main()
