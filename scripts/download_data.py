from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_DRIVE_URL = "https://drive.google.com/drive/folders/1IoZqSIesr5mvtALuoEVaXBEI4tkbx5PA?usp=sharing"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download project data from Google Drive.")
    parser.add_argument("--url", type=str, default=DEFAULT_DRIVE_URL)
    parser.add_argument("--output", type=str, default="data")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import gdown
    except ImportError as exc:
        raise ImportError(
            "Downloading Google Drive folders requires gdown. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    gdown.download_folder(
        url=args.url,
        output=str(output),
        quiet=args.quiet,
        use_cookies=False,
    )

    print(f"Downloaded data to: {output.resolve()}")


if __name__ == "__main__":
    main()
