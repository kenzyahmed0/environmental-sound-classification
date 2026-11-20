
from pathlib import Path
import tarfile
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_ROOT = DATA_DIR / "UrbanSound8K"
ARCHIVE_PATH = DATA_DIR / "UrbanSound8K.tar.gz"

DATA_URL = (
    "https://zenodo.org/records/1203745/files/"
    "UrbanSound8K.tar.gz?download=1"
)


def download_dataset():
    metadata_file = DATA_ROOT / "metadata" / "UrbanSound8K.csv"

    if metadata_file.exists():
        print("Dataset already downloaded and extracted.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not ARCHIVE_PATH.exists() or ARCHIVE_PATH.stat().st_size == 0:
        print("Downloading UrbanSound8K...")
        urllib.request.urlretrieve(DATA_URL, ARCHIVE_PATH)
        print("Download completed.")

    print(f"Archive size: {ARCHIVE_PATH.stat().st_size / 1e9:.2f} GB")
    print("Extracting dataset...")

    with tarfile.open(ARCHIVE_PATH, "r:gz") as archive:
        # Prevent extraction outside the project data folder.
        for member in archive.getmembers():
            target = (DATA_DIR / member.name).resolve()
            if not target.is_relative_to(DATA_DIR.resolve()):
                raise ValueError(f"Unsafe archive path: {member.name}")

        archive.extractall(DATA_DIR)

    if not metadata_file.exists():
        raise FileNotFoundError("UrbanSound8K metadata file was not found.")

    print("Dataset is ready:", DATA_ROOT)


if __name__ == "__main__":
    download_dataset()
