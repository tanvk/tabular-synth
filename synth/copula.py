from __future__ import annotations
import pandas as pd

try:
    # SDV 1.x modern API
    from sdv.single_table import GaussianCopulaSynthesizer
    from sdv.metadata import SingleTableMetadata
    _USE_SINGLE_TABLE = True
except Exception:  # fallback for older SDV
    from sdv.tabular import GaussianCopula  # type: ignore
    _USE_SINGLE_TABLE = False


class CopulaGenerator:
    """Thin wrapper around SDV's Gaussian Copula for single-table synthesis."""

    def __init__(self, enforce_min_max: bool = True, enforce_uniqueness: bool = False):
        self.enforce_min_max = enforce_min_max
        self.enforce_uniqueness = enforce_uniqueness
        self._model = None
        self._columns: list[str] = []

    def fit(self, df: pd.DataFrame) -> "CopulaGenerator":
        self._columns = df.columns.tolist()

        if _USE_SINGLE_TABLE:
            # Infer metadata automatically
            md = SingleTableMetadata()
            md.detect_from_dataframe(df)
            synth = GaussianCopulaSynthesizer(
                metadata=md,
                enforce_min_max_values=self.enforce_min_max,
                enforce_rounding=True,
            )
            synth.fit(df)
            self._model = synth
        else:
            # Older API
            model = GaussianCopula(
                enforce_min_max_values=self.enforce_min_max,
                enforce_rounding=True,
            )
            model.fit(df)
            self._model = model
        return self

    def sample(self, n: int) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit(df) first.")
        if _USE_SINGLE_TABLE:
            out = self._model.sample(num_rows=n)
        else:
            out = self._model.sample(n)
        # Keep original column order when possible
        return out[self._columns] if set(self._columns).issubset(out.columns) else out

    def sample_like(self, df_like: pd.DataFrame) -> pd.DataFrame:
        """Convenience: sample the same number of rows as df_like."""
        return self.sample(len(df_like))