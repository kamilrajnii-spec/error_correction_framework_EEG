"""Train the Phase 2 DAE using real EEGdenoiseNet artifact pairs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eeg_denoising.data_loading.load_eegdenoisenet import load_eegdenoisenet  # noqa: E402
from eeg_denoising.models.dae import (  # noqa: E402
    ConvDAE,
    load_dae_checkpoint,
    write_model_summary,
)
from eeg_denoising.evaluation.metrics import rmse, rrmse, snr, snr_gain  # noqa: E402
from eeg_denoising.pipeline.hybrid_pipeline import apply_dae_to_epochs  # noqa: E402
from eeg_denoising.preprocessing.artifact_mixing import (  # noqa: E402
    _as_2d_epochs,
    create_artifact_pairs,
)
from eeg_denoising.training.eeg_dataset import (  # noqa: E402
    EEGPairDataset,
    normalize_pairs,
)
from eeg_denoising.training.train_dae import TrainingConfig, train_dae_model  # noqa: E402
from eeg_denoising.wavelet.dwt_denoising import denoise_epochs_dwt  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "results" / "phase2"
CHECKPOINT_PATH = OUTPUT_DIR / "dae_best_model.pt"
LOSS_CSV_PATH = OUTPUT_DIR / "training_loss.csv"
LOSS_PLOT_PATH = OUTPUT_DIR / "training_loss_curve.png"
SUMMARY_PATH = OUTPUT_DIR / "model_summary.txt"
MANIFEST_PATH = OUTPUT_DIR / "training_manifest.csv"
HELDOUT_EVALUATION_PATH = OUTPUT_DIR / "heldout_evaluation_table.csv"


def main() -> int:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    splits, manifest = build_grouped_pair_splits(args)
    normalized_splits = normalize_split_pairs(splits)

    train_loader = DataLoader(
        EEGPairDataset(
            normalized_splits["train"]["dae_input"],
            normalized_splits["train"]["clean"],
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        EEGPairDataset(
            normalized_splits["validation"]["dae_input"],
            normalized_splits["validation"]["clean"],
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = ConvDAE()
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        spectral_weight=args.spectral_weight,
        device=args.device,
        checkpoint_path=CHECKPOINT_PATH,
    )
    history = train_dae_model(model, train_loader, validation_loader, config)

    pd.DataFrame(history).to_csv(LOSS_CSV_PATH, index=False)
    pd.DataFrame(manifest).to_csv(MANIFEST_PATH, index=False)
    save_loss_plot(history)

    best_model = load_dae_checkpoint(CHECKPOINT_PATH, device=args.device)
    heldout_rows = evaluate_heldout_test_set(best_model, splits["test"], args.device)
    pd.DataFrame(heldout_rows).to_csv(HELDOUT_EVALUATION_PATH, index=False)

    write_model_summary(
        model,
        SUMMARY_PATH,
        extra_lines=[
            "Training input: DWT wavelet output",
            "Training target: clean EEG",
            f"Training pairs used: {len(splits['train']['clean'])}",
            f"Validation pairs used: {len(splits['validation']['clean'])}",
            f"Held-out test pairs used: {len(splits['test']['clean'])}",
            f"Epochs requested: {args.epochs}",
            f"Batch size: {args.batch_size}",
            f"Best checkpoint: {CHECKPOINT_PATH.relative_to(PROJECT_ROOT)}",
        ],
    )

    print(f"Created {CHECKPOINT_PATH}")
    print(f"Created {LOSS_CSV_PATH}")
    print(f"Created {LOSS_PLOT_PATH}")
    print(f"Created {MANIFEST_PATH}")
    print(f"Created {HELDOUT_EVALUATION_PATH}")
    print(f"Created {SUMMARY_PATH}")

    return 0


def build_grouped_pair_splits(
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, object]]]:
    """Create split-safe Wavelet-to-clean DAE pairs.

    The clean EEG epoch IDs are split first. Artifact pairs are then created
    inside each split so the same clean epoch cannot appear in both training
    and validation/test data.
    """
    eegdenoisenet = load_eegdenoisenet(PROJECT_ROOT / "data" / "eegdenoisenet")

    clean = _as_2d_epochs(eegdenoisenet["clean_eeg"])[: args.max_clean_epochs]
    eog = _as_2d_epochs(eegdenoisenet["eog"])
    emg = _as_2d_epochs(eegdenoisenet["emg"])

    split_indices = split_clean_epoch_indices(
        n_epochs=clean.shape[0],
        seed=args.seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )

    splits = {}
    manifest = []
    for split_name, clean_indices in split_indices.items():
        split_data, split_manifest = create_split_pairs(
            split_name=split_name,
            clean=clean,
            eog=eog,
            emg=emg,
            clean_indices=clean_indices,
            snr_levels=args.snr_levels,
        )
        splits[split_name] = split_data
        manifest.extend(split_manifest)

    return splits, manifest


def split_clean_epoch_indices(
    n_epochs: int,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, np.ndarray]:
    """Split clean epoch IDs before artifact mixing to avoid leakage."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_epochs)

    train_end = int(train_fraction * n_epochs)
    validation_end = train_end + int(validation_fraction * n_epochs)

    return {
        "train": indices[:train_end],
        "validation": indices[train_end:validation_end],
        "test": indices[validation_end:],
    }


def create_split_pairs(
    split_name: str,
    clean: np.ndarray,
    eog: np.ndarray,
    emg: np.ndarray,
    clean_indices: np.ndarray,
    snr_levels: list[float],
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    """Create raw noisy, DWT input, and clean target arrays for one split."""
    split_clean = clean[clean_indices]
    eog_indices = clean_indices % eog.shape[0]
    emg_indices = clean_indices % emg.shape[0]
    split_eog = eog[eog_indices]
    split_emg = emg[emg_indices]

    pairs = create_artifact_pairs(split_clean, split_eog, split_emg, snr_levels_db=snr_levels)

    raw_noisy_arrays = []
    dae_input_arrays = []
    clean_arrays = []
    manifest = []

    for pair in pairs:
        dae_input = denoise_epochs_dwt(pair.noisy)
        raw_noisy_arrays.append(pair.noisy)
        dae_input_arrays.append(dae_input)
        clean_arrays.append(pair.clean)

        for local_id, clean_epoch_id in enumerate(clean_indices):
            manifest.append(
                {
                    "split": split_name,
                    "clean_epoch_id": int(clean_epoch_id),
                    "artifact_epoch_id": artifact_epoch_id(
                        pair.artifact_type,
                        int(eog_indices[local_id]),
                        int(emg_indices[local_id]),
                    ),
                    "artifact_type": pair.artifact_type,
                    "target_snr_db": pair.target_snr_db,
                    "dae_input": "dwt_wavelet_output",
                    "target": "clean_eeg",
                }
            )

    return (
        {
            "raw_noisy": np.vstack(raw_noisy_arrays),
            "dae_input": np.vstack(dae_input_arrays),
            "clean": np.vstack(clean_arrays),
        },
        manifest,
    )


def artifact_epoch_id(
    artifact_type: str,
    eog_index: int,
    emg_index: int,
) -> str:
    """Return a readable artifact ID for the manifest."""
    if artifact_type == "blink":
        return f"eog:{eog_index}"
    if artifact_type == "muscle":
        return f"emg:{emg_index}"

    return f"eog:{eog_index};emg:{emg_index}"


def normalize_split_pairs(
    splits: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    """Normalize DAE input and clean target inside each split."""
    normalized = {}
    for split_name, split_data in splits.items():
        dae_input, clean, _, _ = normalize_pairs(
            split_data["dae_input"],
            split_data["clean"],
        )
        normalized[split_name] = {
            "dae_input": dae_input,
            "clean": clean,
        }

    return normalized


def evaluate_heldout_test_set(
    model: ConvDAE,
    test_split: dict[str, np.ndarray],
    device: str,
) -> list[dict[str, object]]:
    """Evaluate noisy, wavelet-only, and hybrid outputs on held-out pairs."""
    clean = test_split["clean"]
    raw_noisy = test_split["raw_noisy"]
    wavelet = test_split["dae_input"]
    hybrid = apply_dae_to_epochs(wavelet, model, device=device)

    return [
        evaluation_row("noisy_input", clean, raw_noisy, raw_noisy),
        evaluation_row("wavelet_only", clean, raw_noisy, wavelet),
        evaluation_row("hybrid_wavelet_dae", clean, raw_noisy, hybrid),
    ]


def evaluation_row(
    method: str,
    clean: np.ndarray,
    noisy_reference: np.ndarray,
    output: np.ndarray,
) -> dict[str, object]:
    return {
        "split": "test",
        "method": method,
        "snr_db": round(snr(clean, output), 6),
        "snr_gain_db": round(snr_gain(clean, noisy_reference, output), 6),
        "rmse": round(rmse(clean, output), 6),
        "rrmse": round(rrmse(clean, output), 6),
        "n_pairs": clean.shape[0],
    }


def save_loss_plot(history: list[dict[str, float]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(history)

    plt.figure(figsize=(8, 4))
    plt.plot(frame["epoch"], frame["train_loss"], marker="o", label="train")
    plt.plot(frame["epoch"], frame["validation_loss"], marker="o", label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Phase 2 DAE training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=150)
    plt.close()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--spectral-weight", type=float, default=0.10)
    parser.add_argument("--max-clean-epochs", type=int, default=600)
    parser.add_argument("--snr-levels", type=float, nargs="+", default=[-5.0, 0.0, 5.0])
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
