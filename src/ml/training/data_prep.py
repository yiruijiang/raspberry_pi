"""
data_prep.py — Dataset download and preparation utilities for YAMNet cry-detector training.

Responsibilities
----------------
1. Download and extract ESC-50 (class 38 = "crying_baby").
2. Validate that Donate-a-cry clips exist in the expected layout under ``data/cry/``.
3. Resample all clips to 16 kHz mono WAV (YAMNet requirement).
4. Return file lists and integer labels (0 = non_cry, 1 = cry) ready for embedding
   extraction in ``train_cry_detector.py``.

Directory layout produced / expected
-------------------------------------
::

    data/
      raw/
        ESC-50/                   # cloned from github.com/karolpiczak/ESC-50
          audio/                  # *.wav clips, already 44.1 kHz
          meta/esc50.csv          # label metadata
        baby_cry_detection/       # cloned from github.com/giulbia/baby_cry_detection
          data/
            baby_cry/             # *.wav infant cry clips
            noise/                # *.wav non-cry (ambient, white noise, speech)
      processed/
        cry/                      # 16 kHz mono WAV — positive class
        non_cry/                  # 16 kHz mono WAV — negative class

All paths below are relative to the repository root unless otherwise stated.

YAMNet audio requirements
--------------------------
- Mono (single channel)
- 16 000 Hz sample rate
- float32, values in [-1, 1]
- Duration: any length — YAMNet processes arbitrary-length waveforms.
  The training script takes the mean-pooled embedding, so even short clips
  (<1 s) work fine.

Usage
-----
As a library (called from ``train_cry_detector.py``)::

    from data_prep import prepare_dataset
    train_files, val_files, test_files = prepare_dataset(repo_root, force_resample=False)

As a standalone script::

    python src/ml/training/data_prep.py --repo-root /path/to/repo

Donate-a-cry sourcing instructions
------------------------------------
1. Clone the repository::

       git clone https://github.com/giulbia/baby_cry_detection.git data/raw/baby_cry_detection

2. The clone contains ``data/baby_cry/`` (~930 WAV clips) and ``data/noise/`` (~930 WAV clips).
3. No additional licence steps are needed — the corpus is MIT-licenced.
4. If you have additional labelled cry audio, place it in ``data/raw/extra_cry/`` and it
   will be picked up automatically as positive-class samples.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SR: int = 16_000          # Hz — YAMNet hard requirement
ESC50_GIT_URL: str = "https://github.com/karolpiczak/ESC-50.git"
DONATE_GIT_URL: str = "https://github.com/giulbia/baby_cry_detection.git"
ESC50_CRY_CLASS: int = 38        # ESC-50 label index for "crying_baby"

# Train / val / test split fractions (must sum to 1.0)
SPLIT_TRAIN: float = 0.70
SPLIT_VAL: float = 0.15
SPLIT_TEST: float = 0.15

# Minimum clips per class before issuing a warning
MIN_CLIPS_PER_CLASS: int = 500


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def prepare_dataset(
    repo_root: Path,
    force_resample: bool = False,
    seed: int = 42,
) -> Tuple[
    List[Tuple[str, int]],
    List[Tuple[str, int]],
    List[Tuple[str, int]],
]:
    """Download datasets if absent, resample to 16 kHz, and return train/val/test splits.

    Parameters
    ----------
    repo_root:
        Absolute path to the repository root (parent of ``src/``).
    force_resample:
        If ``True``, re-run resampling even if processed files already exist.
    seed:
        Random seed for reproducible splits.

    Returns
    -------
    tuple of (train, val, test)
        Each element is a list of ``(wav_path: str, label: int)`` tuples where
        ``label`` is ``1`` for cry and ``0`` for non-cry.

    Raises
    ------
    RuntimeError
        If no cry clips or no non-cry clips are found after dataset preparation.
    """
    raw_dir = repo_root / "data" / "raw"
    processed_dir = repo_root / "data" / "processed"

    # -- Step 1: Ensure datasets are present --------------------------------
    _ensure_esc50(raw_dir)
    _ensure_donate_a_cry(raw_dir)

    # -- Step 2: Resample to 16 kHz mono ------------------------------------
    cry_out = processed_dir / "cry"
    non_cry_out = processed_dir / "non_cry"
    cry_out.mkdir(parents=True, exist_ok=True)
    non_cry_out.mkdir(parents=True, exist_ok=True)

    cry_wavs = _collect_cry_wavs(raw_dir)
    non_cry_wavs = _collect_non_cry_wavs(raw_dir)

    if len(cry_wavs) < MIN_CLIPS_PER_CLASS:
        logger.warning(
            "Only %d cry clips found (target >= %d). "
            "Consider adding more data to data/raw/extra_cry/.",
            len(cry_wavs),
            MIN_CLIPS_PER_CLASS,
        )
    if len(non_cry_wavs) < MIN_CLIPS_PER_CLASS:
        logger.warning(
            "Only %d non-cry clips found (target >= %d). "
            "Consider adding more data to data/raw/extra_non_cry/.",
            len(non_cry_wavs),
            MIN_CLIPS_PER_CLASS,
        )

    logger.info(
        "Raw clips: %d cry, %d non-cry — resampling to %d Hz mono",
        len(cry_wavs),
        len(non_cry_wavs),
        TARGET_SR,
    )

    cry_processed = _resample_batch(cry_wavs, cry_out, force=force_resample)
    non_cry_processed = _resample_batch(non_cry_wavs, non_cry_out, force=force_resample)

    if not cry_processed:
        raise RuntimeError(
            "No cry clips found after dataset preparation. "
            "Check data/raw/ESC-50 and data/raw/baby_cry_detection are present."
        )
    if not non_cry_processed:
        raise RuntimeError(
            "No non-cry clips found after dataset preparation. "
            "Check data/raw/baby_cry_detection/data/noise/ is present."
        )

    logger.info(
        "Processed: %d cry, %d non-cry clips at %d Hz mono",
        len(cry_processed),
        len(non_cry_processed),
        TARGET_SR,
    )

    # -- Step 3: Stratified train/val/test split ----------------------------
    rng = random.Random(seed)

    def _split(paths: List[str], label: int) -> Tuple[
        List[Tuple[str, int]],
        List[Tuple[str, int]],
        List[Tuple[str, int]],
    ]:
        shuffled = list(paths)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * SPLIT_TRAIN)
        n_val = int(n * SPLIT_VAL)
        train = [(p, label) for p in shuffled[:n_train]]
        val = [(p, label) for p in shuffled[n_train:n_train + n_val]]
        test = [(p, label) for p in shuffled[n_train + n_val:]]
        return train, val, test

    cry_train, cry_val, cry_test = _split(cry_processed, label=1)
    non_cry_train, non_cry_val, non_cry_test = _split(non_cry_processed, label=0)

    train = cry_train + non_cry_train
    val = cry_val + non_cry_val
    test = cry_test + non_cry_test

    # Shuffle within each split so classes are interleaved during training.
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(train),
        len(val),
        len(test),
    )
    return train, val, test


# ---------------------------------------------------------------------------
# Dataset download helpers
# ---------------------------------------------------------------------------


def _ensure_esc50(raw_dir: Path) -> None:
    """Clone ESC-50 into ``raw_dir/ESC-50`` if not already present.

    Parameters
    ----------
    raw_dir:
        Parent directory for raw dataset downloads.
    """
    dest = raw_dir / "ESC-50"
    if (dest / "meta" / "esc50.csv").exists():
        logger.info("ESC-50 already present at %s", dest)
        return

    logger.info("Cloning ESC-50 from %s ...", ESC50_GIT_URL)
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "clone", "--depth=1", ESC50_GIT_URL, str(dest)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to clone ESC-50:\n{result.stderr}\n"
            f"Manually clone with: git clone {ESC50_GIT_URL} {dest}"
        )
    logger.info("ESC-50 cloned successfully")


def _ensure_donate_a_cry(raw_dir: Path) -> None:
    """Clone Donate-a-cry into ``raw_dir/baby_cry_detection`` if not already present.

    Parameters
    ----------
    raw_dir:
        Parent directory for raw dataset downloads.
    """
    dest = raw_dir / "baby_cry_detection"
    cry_data = dest / "data" / "baby_cry"
    if cry_data.exists() and any(cry_data.rglob("*.wav")):
        logger.info("Donate-a-cry already present at %s", dest)
        return

    logger.info("Cloning Donate-a-cry from %s ...", DONATE_GIT_URL)
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "clone", "--depth=1", DONATE_GIT_URL, str(dest)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to clone Donate-a-cry:\n{result.stderr}\n\n"
            "Donate-a-cry sourcing instructions:\n"
            "  1. Manually clone:\n"
            f"     git clone {DONATE_GIT_URL} {dest}\n"
            "  2. The repo contains data/baby_cry/ (~930 WAV clips)\n"
            "     and data/noise/ (~930 non-cry clips).\n"
            "  3. Place any additional cry WAVs in data/raw/extra_cry/\n"
            "     and non-cry WAVs in data/raw/extra_non_cry/.\n"
        )
    logger.info("Donate-a-cry cloned successfully")


# ---------------------------------------------------------------------------
# File collection helpers
# ---------------------------------------------------------------------------


def _collect_cry_wavs(raw_dir: Path) -> List[str]:
    """Collect all raw cry WAV paths from ESC-50 class 38 and Donate-a-cry.

    Parameters
    ----------
    raw_dir:
        Root raw data directory.

    Returns
    -------
    list of str
        Absolute paths to raw WAV files for the positive (cry) class.
    """
    paths: List[str] = []

    # ESC-50 class 38 ("crying_baby")
    esc50_meta = raw_dir / "ESC-50" / "meta" / "esc50.csv"
    esc50_audio = raw_dir / "ESC-50" / "audio"
    if esc50_meta.exists():
        with esc50_meta.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row["target"]) == ESC50_CRY_CLASS:
                    wav = esc50_audio / row["filename"]
                    if wav.exists():
                        paths.append(str(wav))
        logger.debug("ESC-50 class 38: %d files", len(paths))
    else:
        logger.warning("ESC-50 metadata not found at %s", esc50_meta)

    # Donate-a-cry corpus
    donate_cry = raw_dir / "baby_cry_detection" / "data" / "baby_cry"
    if donate_cry.exists():
        donate_paths = list(donate_cry.rglob("*.wav"))
        logger.debug("Donate-a-cry: %d files", len(donate_paths))
        paths.extend(str(p) for p in donate_paths)
    else:
        logger.warning(
            "Donate-a-cry baby_cry directory not found at %s. "
            "See TASK-201 for sourcing instructions.",
            donate_cry,
        )

    # Optional extra cry clips dropped in by the user
    extra_cry = raw_dir / "extra_cry"
    if extra_cry.exists():
        extra = list(extra_cry.rglob("*.wav"))
        logger.debug("Extra cry clips: %d files", len(extra))
        paths.extend(str(p) for p in extra)

    return paths


def _collect_non_cry_wavs(raw_dir: Path) -> List[str]:
    """Collect non-cry WAV paths from ESC-50 (excluding class 38) and Donate-a-cry noise.

    We cap ESC-50 non-cry samples to 5 per class to avoid over-representing any
    single environment sound category and keep the dataset balanced.

    Parameters
    ----------
    raw_dir:
        Root raw data directory.

    Returns
    -------
    list of str
        Absolute paths to raw WAV files for the negative (non-cry) class.
    """
    paths: List[str] = []

    # ESC-50 — all classes except 38, capped to balance with cry count
    esc50_meta = raw_dir / "ESC-50" / "meta" / "esc50.csv"
    esc50_audio = raw_dir / "ESC-50" / "audio"
    if esc50_meta.exists():
        # Collect by class so we can apply a per-class cap
        class_files: dict[int, List[str]] = {}
        with esc50_meta.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cls = int(row["target"])
                if cls == ESC50_CRY_CLASS:
                    continue
                wav = esc50_audio / row["filename"]
                if wav.exists():
                    class_files.setdefault(cls, []).append(str(wav))
        # Take at most 5 clips per class (ESC-50 has 40 clips/class = 49 * 5 = 245 clips)
        for cls_files in class_files.values():
            paths.extend(cls_files[:5])
        logger.debug("ESC-50 non-cry (capped): %d files", len(paths))

    # Donate-a-cry noise
    donate_noise = raw_dir / "baby_cry_detection" / "data" / "noise"
    if donate_noise.exists():
        noise_paths = list(donate_noise.rglob("*.wav"))
        logger.debug("Donate-a-cry noise: %d files", len(noise_paths))
        paths.extend(str(p) for p in noise_paths)
    else:
        logger.warning(
            "Donate-a-cry noise directory not found at %s",
            donate_noise,
        )

    # Optional extra non-cry clips
    extra_non_cry = raw_dir / "extra_non_cry"
    if extra_non_cry.exists():
        extra = list(extra_non_cry.rglob("*.wav"))
        logger.debug("Extra non-cry clips: %d files", len(extra))
        paths.extend(str(p) for p in extra)

    return paths


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def _resample_batch(
    src_paths: List[str],
    out_dir: Path,
    force: bool = False,
) -> List[str]:
    """Resample a list of WAV files to 16 kHz mono and write to ``out_dir``.

    Uses ``soundfile`` for reading and ``librosa`` for resampling.  Both are
    available in the training requirements.

    Parameters
    ----------
    src_paths:
        Input WAV file paths (any sample rate, any channel count).
    out_dir:
        Directory to write resampled output files.  Created if absent.
    force:
        If ``True``, overwrite existing output files.

    Returns
    -------
    list of str
        Absolute paths to the resampled output WAV files.

    Notes
    -----
    File names are derived from the source path stem to avoid collisions across
    datasets a 6-hex content-hash suffix is appended when stems clash.
    """
    import hashlib

    import librosa
    import soundfile as sf

    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths: List[str] = []
    stem_seen: set[str] = set()

    for src in src_paths:
        src_path = Path(src)
        stem = src_path.stem
        if stem in stem_seen:
            # Disambiguate with a short hash of the full source path
            suffix = hashlib.md5(src.encode()).hexdigest()[:6]
            stem = f"{stem}_{suffix}"
        stem_seen.add(stem)

        out_path = out_dir / f"{stem}.wav"

        if out_path.exists() and not force:
            output_paths.append(str(out_path))
            continue

        try:
            # librosa.load always returns float32 in [-1, 1], resamples if needed
            audio, _ = librosa.load(src, sr=TARGET_SR, mono=True, dtype="float32")
            sf.write(str(out_path), audio, TARGET_SR, subtype="PCM_16")
            output_paths.append(str(out_path))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to resample %s: %s — skipping", src, exc)

    return output_paths


# ---------------------------------------------------------------------------
# Audio loading utility (used by training script)
# ---------------------------------------------------------------------------


def load_wav_16k(path: str) -> "numpy.ndarray":  # noqa: F821
    """Load a 16 kHz mono WAV file as a float32 numpy array.

    Assumes the file was produced by :func:`_resample_batch` and is already
    at 16 kHz mono.  Uses ``soundfile`` directly (no resampling) for speed.

    Parameters
    ----------
    path:
        Path to a 16 kHz mono WAV file.

    Returns
    -------
    numpy.ndarray
        1-D float32 array, values in ``[-1, 1]``.

    Raises
    ------
    AssertionError
        If the file is not at 16 kHz or is not mono.
    """
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    assert sr == TARGET_SR, (
        f"Expected {TARGET_SR} Hz, got {sr} Hz for {path}. "
        "Run data_prep with force_resample=True."
    )
    if audio.ndim == 2:
        # Unexpected stereo — mix down to mono
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Standalone entry point for dataset preparation.

    Returns
    -------
    int
        Exit code: 0 on success, 1 on error.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Prepare ESC-50 and Donate-a-cry datasets for YAMNet fine-tuning"
    )
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parent.parent.parent.parent),
        help="Path to the repository root (default: auto-detected from script location)",
    )
    parser.add_argument(
        "--force-resample",
        action="store_true",
        help="Re-run resampling even if processed files already exist",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    logger.info("Repository root: %s", repo_root)

    try:
        train, val, test = prepare_dataset(
            repo_root=repo_root,
            force_resample=args.force_resample,
        )
    except Exception as exc:
        logger.error("Dataset preparation failed: %s", exc)
        return 1

    print(f"\nDataset ready:")
    print(f"  Train: {len(train)} clips")
    print(f"  Val:   {len(val)} clips")
    print(f"  Test:  {len(test)} clips")

    # Report class balance in each split
    for split_name, split in [("Train", train), ("Val", val), ("Test", test)]:
        n_cry = sum(1 for _, lbl in split if lbl == 1)
        n_non = sum(1 for _, lbl in split if lbl == 0)
        print(f"  {split_name}: {n_cry} cry / {n_non} non-cry")

    return 0


if __name__ == "__main__":
    sys.exit(main())
