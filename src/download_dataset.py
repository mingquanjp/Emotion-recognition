import argparse
import json
from pathlib import Path

from datasets import load_dataset


DATASET_NAME = "abhilash88/fer2013-enhanced"
EXPECTED_SPLITS = ("train", "validation", "test")
IMAGE_COLUMNS = ("image", "pixels")
LABEL_COLUMNS = ("emotion", "emotion_name")


def find_first_existing(columns, candidates):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def summarize_dataset(dataset):
    summary = {
        "dataset_name": DATASET_NAME,
        "splits": {},
    }

    for split_name, split_data in dataset.items():
        columns = list(split_data.column_names)
        image_column = find_first_existing(columns, IMAGE_COLUMNS)
        label_column = find_first_existing(columns, LABEL_COLUMNS)

        summary["splits"][split_name] = {
            "num_rows": len(split_data),
            "columns": columns,
            "image_column": image_column,
            "label_column": label_column,
        }

    return summary


def validate_dataset(summary):
    missing_splits = [split for split in EXPECTED_SPLITS if split not in summary["splits"]]
    if missing_splits:
        raise ValueError(f"Missing expected split(s): {missing_splits}")

    for split_name, split_summary in summary["splits"].items():
        if split_summary["image_column"] is None:
            raise ValueError(f"Split '{split_name}' has no image column. Columns: {split_summary['columns']}")
        if split_summary["label_column"] is None:
            raise ValueError(f"Split '{split_name}' has no label column. Columns: {split_summary['columns']}")


def save_summary(summary, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Download and validate fer2013-enhanced from Hugging Face.")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument(
        "--summary-out",
        default="reports/dataset_load_summary.json",
        help="Where to save the dataset loading summary.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = load_dataset(DATASET_NAME, cache_dir=args.cache_dir)
    summary = summarize_dataset(dataset)
    validate_dataset(summary)

    output_path = Path(args.summary_out)
    save_summary(summary, output_path)

    print(f"Loaded dataset: {DATASET_NAME}")
    for split_name in EXPECTED_SPLITS:
        split_summary = summary["splits"][split_name]
        print(
            f"- {split_name}: {split_summary['num_rows']} rows, "
            f"image='{split_summary['image_column']}', label='{split_summary['label_column']}'"
        )
    print(f"Saved summary to: {output_path}")


if __name__ == "__main__":
    main()
