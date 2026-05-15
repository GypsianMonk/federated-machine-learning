"""
Download and verify MIT-BIH Arrhythmia Database records using WFDB.

This script is designed for PhD-grade reproducibility:
- Deterministic directory structure
- Restart-safe (skips existing records)
- Clear logging for thesis and experiments

Dataset:
MIT-BIH Arrhythmia Database (PhysioNet)
Sampling rate: 360 Hz
Records: 48 half-hour ECG recordings

Reference:
Moody, G. B., & Mark, R. G. (2001). The impact of the MIT-BIH Arrhythmia Database.
IEEE Engineering in Medicine and Biology Magazine.
"""

import os
import wfdb
from wfdb import processing
from typing import List

# -----------------------------
# Configuration
# -----------------------------

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
RAW_DIR = os.path.join(BASE_DIR, "raw", "mitbih")

# MIT-BIH record list (standard 48 records)
MITBIH_RECORDS: List[str] = [
    "100", "101", "102", "103", "104", "105", "106", "107",
    "108", "109", "111", "112", "113", "114", "115", "116",
    "117", "118", "119", "121", "122", "123", "124", "200",
    "201", "202", "203", "205", "207", "208", "209", "210",
    "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234"
]

# -----------------------------
# Utility Functions
# -----------------------------

def ensure_directory(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def record_exists(record: str) -> bool:
    """Check whether a record already exists locally."""
    record_path = os.path.join(RAW_DIR, record)
    return os.path.exists(record_path + ".dat") and os.path.exists(record_path + ".hea")


# -----------------------------
# Download Logic
# -----------------------------

def download_mitbih_records() -> None:
    """
    Download MIT-BIH Arrhythmia Database records using WFDB.
    Skips records that already exist.
    """
    ensure_directory(RAW_DIR)

    print("Starting MIT-BIH Arrhythmia Database download...")
    print(f"Target directory: {RAW_DIR}\n")

    for record in MITBIH_RECORDS:
        if record_exists(record):
            print(f"[SKIP] Record {record} already exists")
            continue

        try:
            print(f"[DOWNLOAD] Record {record}")
            wfdb.dl_database(
                "mitdb",
                RAW_DIR,
                records=[record],
                keep_subdirs=False,
            )

            # Basic integrity check
            signal, fields = wfdb.rdsamp(os.path.join(RAW_DIR, record))
            assert signal.shape[0] > 0, "Empty signal"

            print(f"[OK] Record {record} | Shape: {signal.shape}")

        except Exception as e:
            print(f"[ERROR] Record {record}: {e}")

    print("\nDownload process completed.")


# -----------------------------
# Entry Point
# -----------------------------

if __name__ == "__main__":
    download_mitbih_records()

