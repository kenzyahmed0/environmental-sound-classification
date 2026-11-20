
from pathlib import Path

from src.datasets.loader import load_metadata
from src.datasets.splits import create_splits


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def main():
    print("========== Validating UrbanSound8K ==========")

    metadata = load_metadata()

    print("\n========== Creating Official Fold Splits ==========")

    train_df, val_df, test_df = create_splits(metadata)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(
        PROCESSED_DIR / "train_metadata.csv",
        index=False
    )

    val_df.to_csv(
        PROCESSED_DIR / "validation_metadata.csv",
        index=False
    )

    test_df.to_csv(
        PROCESSED_DIR / "test_metadata.csv",
        index=False
    )

    print("\n========== Preprocessing Completed ==========")
    print("Train samples:", len(train_df))
    print("Validation samples:", len(val_df))
    print("Test samples:", len(test_df))
    print("Saved metadata splits to:", PROCESSED_DIR)


if __name__ == "__main__":
    main()
