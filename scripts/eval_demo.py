from pathlib import Path
import sys
import pandas as pd
from synth.copula import CopulaGenerator
from eval.fidelity import basic_fidelity_report
from eval.utility import utility_transfer_report

def main():
    if len(sys.argv) < 3:
        print("Usage: python -m scripts.eval_demo data/<real.csv> <target_col> [n_synth]")
        sys.exit(1)

    real_path = Path(sys.argv[1])
    target = sys.argv[2]
    n_synth = int(sys.argv[3]) if len(sys.argv) > 3 else 5000

    real = pd.read_csv(real_path)
    print(f"Loaded real: {real_path} shape={real.shape}")

    gen = CopulaGenerator().fit(real)
    synth = gen.sample(n_synth)
    print(f"Generated synth: shape={synth.shape}")

    # Fidelity
    fid = basic_fidelity_report(real, synth)
    print("\n=== Fidelity (headline) ===")
    print(fid["headline"])
    print("\nWorst corr deltas (top 10):")
    for row in fid["corr_worst"]:
        print(row)

    # Utility
    if target in real.columns:
        util = utility_transfer_report(real, synth, target_col=target)
        print("\n=== Utility Transfer ===")
        for k, v in util.items():
            print(f"{k}: {v}")
    else:
        print(f"\n[skip utility] target '{target}' not found in real columns.")

if __name__ == "__main__":
    main()