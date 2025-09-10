import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from synth.copula import CopulaGenerator

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/smoke_copula.py data/<yourfile>.csv [N]")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

    df = pd.read_csv(csv_path)
    print(f"Loaded real data: {csv_path} shape={df.shape}")

    gen = CopulaGenerator().fit(df)
    synth = gen.sample(n)

    out = csv_path.parent / "synthetic_sample.csv"
    synth.to_csv(out, index=False)
    print(f"Generated synthetic data: {synth.shape} → wrote {out}")
    print("Preview:")
    print(synth.head(5).to_string(index=False))

if __name__ == "__main__":
    main()