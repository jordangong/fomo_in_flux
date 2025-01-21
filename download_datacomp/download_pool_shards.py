# adapted from: https://github.com/mlfoundations/datacomp/blob/main/download_upstream.py

import argparse
import os
import re
import shutil
from pathlib import Path

import img2dataset
from huggingface_hub import snapshot_download

from scale_configs import available_scales

def cleanup_dir(path):
    assert isinstance(path, Path)
    if isinstance(path, Path):
        shutil.rmtree(path)
    else:
        path.rmtree()

CONSIDERED_PARQUETS = [
    'https://huggingface.co/datasets/mlfoundations/datacomp_small/resolve/main/006731584dd46fed36eafe8956742f7f.parquet',
    'https://huggingface.co/datasets/mlfoundations/datacomp_small/resolve/main/02ad5a8f481c6a0a71fec68556d0d18a.parquet',
    'https://huggingface.co/datasets/mlfoundations/datacomp_small/resolve/main/0239539d00c512673efa3efb7d9f4957.parquet',
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scale",
        type=str,
        required=False,
        choices=available_scales(simple_names=True)[1:] + ["datacomp_1b"],
        default="small",
        help="Competition scale.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Path to directory where the data (webdataset shards) will be stored.",
    )
    parser.add_argument(
        "--metadata_dir",
        type=Path,
        default=None,
        help="Path to directory where the metadata will be stored. If not set, infer from data_dir.",
    )
    parser.add_argument(
        "--download_npz",
        help="If true, also download npz files.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--skip_shards",
        help="If true, only download metadata.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--overwrite_metadata",
        help="If true, force re-download of the metadata files.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--skip_bbox_blurring",
        help="If true, skip bounding box blurring on images while downloading.",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--processes_count",
        type=int,
        required=False,
        default=2,
        help="Number of processes for download.",
    )
    parser.add_argument(
        "--thread_count",
        type=int,
        required=False,
        default=8,
        help="Number of threads for download.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        required=False,
        default=224,
        help="Size images need to be downloaded to.",
    )
    parser.add_argument(
        "--resize_mode",
        type=str,
        required=False,
        choices=["no", "border", "keep_ratio", "keep_ratio_largest", "center_crop"],
        default="no",
        help="Resizing mode used by img2dataset when downloading images.",
    )
    parser.add_argument(
        "--no_resize_only_if_bigger",
        help="If true, do not resize only if images are bigger than target size.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--encode_format",
        type=str,
        required=False,
        choices=["png", "jpg", "webp"],
        default="jpg",
        help="Images encoding format.",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        required=False,
        choices=["webdataset", "tfrecord", "parquet", "files"],
        default="webdataset",
        help="Output format used by img2dataset when downloading images.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        required=False,
        default=0,
        help="Number of time a download should be retried (default 2)",
    )
    parser.add_argument(
        "--enable_wandb",
        action="store_true",
        default=True,
        help="Whether to enable wandb logging (default True)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        required=False,
        default="datacomp",
        help="Name of W&B project used (default datacomp)",
    )

    args = parser.parse_args()

    hf_repo = (
        f"mlfoundations/datacomp_{args.scale}"
        if args.scale != "datacomp_1b"
        else "mlfoundations/datacomp_1b"
    )

    metadata_dir = args.metadata_dir
    m_exists = False
    if os.path.exists(metadata_dir):
        if len(os.listdir(metadata_dir)) > 0:
            assert len(os.listdir(metadata_dir)) == 3, 'there should be 3 parquets in the metadata directory, please delete the directory and rerun this script'
            m_exists = True
    else:
        os.makedirs(metadata_dir, exist_ok=True)

    if not m_exists:
        for url in CONSIDERED_PARQUETS:
            # Extract the filename from the URL
            file_name = url.split("/")[-1]
            # Construct the full path to save the file
            file_path = os.path.join(metadata_dir, file_name)
            # Use wget command to download the file
            os.system(f"wget {url} -O {file_path}")
    else:
        print('Metadata already exists here: {}'.format(metadata_dir))

    if not args.skip_shards:
        # Download images.
        shard_dir = args.data_dir / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading images to {shard_dir}")

        bbox_col = None if args.skip_bbox_blurring else "face_bboxes"

        ### updated a few args from the original script for faster downloading
        img2dataset.download(
            url_list=str(metadata_dir),
            output_folder=str(shard_dir),
            processes_count=args.processes_count,
            thread_count=args.thread_count,
            resize_mode=args.resize_mode,
            encode_format=args.encode_format,
            output_format=args.output_format,
            input_format="parquet",
            url_col="url",
            caption_col="text",
            number_sample_per_shard=10000,
            oom_shard_count=8,
            enable_wandb=args.enable_wandb,
            wandb_project=args.wandb_project,
        )
    else:
        print(f"Skipping image data download.")

    print("Done!")