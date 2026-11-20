
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "UrbanSound8K"
META_PATH = DATA_ROOT / "metadata" / "UrbanSound8K.csv"


def load_metadata(data_root=DATA_ROOT):
    data_root = Path(data_root)
    meta_path = data_root / "metadata" / "UrbanSound8K.csv"

    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {meta_path}"
        )

    meta = pd.read_csv(meta_path)

    required_cols = {
        "slice_file_name",
        "fold",
        "classID",
        "class",
    }

    missing_cols = required_cols - set(meta.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    meta["fold"] = meta["fold"].astype(int)
    meta["classID"] = meta["classID"].astype(int)

    meta["filepath"] = meta.apply(
        lambda row: str(
            data_root
            / "audio"
            / f"fold{row.fold}"
            / row.slice_file_name
        ),
        axis=1,
    )

    meta["file_exists"] = meta["filepath"].map(
        lambda path: Path(path).is_file()
    )

    print("Metadata rows:", len(meta))
    print("Classes:", meta["class"].nunique())
    print("Existing audio files:", int(meta["file_exists"].sum()))
    print("Missing audio files:", int((~meta["file_exists"]).sum()))

    if not meta["file_exists"].all():
        raise RuntimeError(
            "Some audio files are missing. Check the dataset."
        )

    if len(meta) != 8732:
        raise ValueError(
            f"Expected 8732 rows, found {len(meta)}."
        )

    return meta
