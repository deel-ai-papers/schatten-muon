#!/usr/bin/env python3
"""
Plot optimal p_star values across all layers, components, and training steps.
Optimized for speed using multiprocessing and PyTorch memory mapping.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Configure matplotlib for LaTeX-like appearance without requiring LaTeX
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif", "Times New Roman", "serif"]
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

# Configuration
DATA_DIR = Path("data")
STEPS = [1] + list(range(20, 440, 20))

# Layer names from the actual data files (Vision Transformer blocks)
LAYER_COMPONENTS = [
    "blocks_0_attn_qkv",
    "blocks_3_attn_qkv",
    "blocks_6_attn_qkv",
    "blocks_9_attn_qkv",
]
GLOBAL_COMPONENTS = ["patch_embed_proj", "head"]
ALL_COMPONENTS = GLOBAL_COMPONENTS + LAYER_COMPONENTS


def _process_single_file(file_info):
    """
    Worker function to process a single checkpoint file.
    Must be at the top level so the multiprocessing module can pickle it.
    """
    filepath, component, step = file_info
    try:
        # mmap=True is the secret sauce here. It maps the file to virtual memory
        # instead of physically loading hundreds of megabytes of tensors into RAM.
        checkpoint = torch.load(
            filepath, map_location="cpu", weights_only=False, mmap=True
        )

        if "p_star" in checkpoint:
            return (component, step, float(checkpoint["p_star"]), None)
        else:
            return (component, step, None, "p_star key missing")
    except Exception as e:
        return (component, step, None, str(e))


def load_pstar_data():
    """Load p_star values from all checkpoint files in parallel."""
    data = defaultdict(dict)

    # Build list of all files to load
    files_to_load = []
    for step in STEPS:
        for component in ALL_COMPONENTS:
            filename = f"step_{step:03d}_{component}_weight.pth"
            filepath = DATA_DIR / filename
            if filepath.exists():
                files_to_load.append((filepath, component, step))

    total_files = len(files_to_load)
    if total_files == 0:
        return data

    # Use up to 8 workers, or however many cores you have (whichever is lower)
    # This prevents thrashing your CPU/Disk while still being massively faster
    n_workers = min(8, multiprocessing.cpu_count())

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all jobs to the pool
        future_to_info = {
            executor.submit(_process_single_file, info): info for info in files_to_load
        }

        with tqdm(
            total=total_files, desc=f"Loading with {n_workers} workers", unit="file"
        ) as pbar:
            for future in as_completed(future_to_info):
                component, step, p_star_val, error = future.result()

                if error:
                    tqdm.write(f"Error loading {component} (Step {step}): {error}")
                elif p_star_val is not None:
                    data[component][step] = p_star_val

                pbar.update(1)

    return data


def get_pstar_values(data, component):
    """Get p_star values for a specific component across steps."""
    if component not in data or len(data[component]) == 0:
        return None

    values = [data[component].get(step, np.nan) for step in STEPS]
    return values


def create_mega_plot(data):
    """Create a comprehensive visualization of all p_star values."""

    # Count valid components
    valid_global = [c for c in GLOBAL_COMPONENTS if c in data and len(data[c]) > 0]
    valid_layers = [c for c in LAYER_COMPONENTS if c in data and len(data[c]) > 0]

    # Create figure with 2 subplots: one for global components, one for layer components
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot global components (conv1, fc)
    if valid_global:
        ax = axes[0]
        for component in valid_global:
            values = get_pstar_values(data, component)
            if values is not None:
                # Filter out NaN values for cleaner plotting
                steps_valid = [
                    s for i, s in enumerate(STEPS) if not np.isnan(values[i])
                ]
                values_valid = [v for v in values if not np.isnan(v)]

                if steps_valid:
                    ax.plot(
                        steps_valid,
                        values_valid,
                        marker="o",
                        label=component,
                        linewidth=2,
                        markersize=8,
                    )

        ax.set_xlabel("Training Step", fontsize=11.88)
        ax.set_ylabel(r"$p^*$", fontsize=13.2)
        ax.set_title("Patch Embedding and Classification Layer", fontsize=14)
        ax.legend(fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(STEPS[::2])  # Show every other step for readability
    else:
        axes[0].text(
            0.5,
            0.5,
            "No global component data available",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )

    # Plot layer components (layer1-4)
    if valid_layers:
        ax = axes[1]
        for component in valid_layers:
            values = get_pstar_values(data, component)
            if values is not None:
                # Filter out NaN values for cleaner plotting
                steps_valid = [
                    s for i, s in enumerate(STEPS) if not np.isnan(values[i])
                ]
                values_valid = [v for v in values if not np.isnan(v)]

                if steps_valid:
                    ax.plot(
                        steps_valid,
                        values_valid,
                        marker="o",
                        label=component,
                        linewidth=2,
                        markersize=6,
                    )

        ax.set_xlabel("Training Step", fontsize=11.88)
        ax.set_ylabel(r"$p^*$", fontsize=13.2)
        ax.set_title("Attention Blocks", fontsize=14)
        ax.legend(fontsize=11.7, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(STEPS[::2])  # Show every other step for readability
    else:
        axes[1].text(
            0.5,
            0.5,
            "No layer component data available",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )

    plt.tight_layout()
    return fig


def print_summary_stats(data):
    """Print summary statistics of p_star values."""
    print("\n" + "=" * 80)
    print("P_STAR SUMMARY STATISTICS")
    print("=" * 80)

    for component in ALL_COMPONENTS:
        if component not in data or len(data[component]) == 0:
            continue

        all_values = [v for v in data[component].values() if not np.isnan(v)]

        if all_values:
            all_values = np.array(all_values)
            print(f"\n{component.upper()}:")
            print(f"  Mean: {np.mean(all_values):.4f}")
            print(f"  Std:  {np.std(all_values):.4f}")
            print(f"  Min:  {np.min(all_values):.4f}")
            print(f"  Max:  {np.max(all_values):.4f}")
            print(f"  Range: {np.max(all_values) - np.min(all_values):.4f}")


def main():
    """Main execution function."""
    print("Loading p_star data from checkpoints...")
    data = load_pstar_data()

    if not data:
        print(
            "No data loaded! Check that the data directory contains checkpoint files."
        )
        return

    print(f"Loaded data for {len(data)} components")

    # Print summary statistics
    print_summary_stats(data)

    # Create visualization
    print("\nCreating visualization...")
    fig = create_mega_plot(data)

    # Save figure
    output_path = "pstar_analysis.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")

    # Also save as PDF for better quality
    output_pdf = "pstar_analysis.pdf"
    fig.savefig(output_pdf, bbox_inches="tight")
    print(f"Plot saved to: {output_pdf}")

    print("\nDone!")


if __name__ == "__main__":
    # This guard is absolutely required for multiprocessing to work on some OS's
    main()
