# synth/ctgan.py
from __future__ import annotations
import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

class CTGANGenerator:
    def __init__(self, epochs: int = 300, batch_size: int = 500, embedding_dim: int = 128):
        self.epochs = epochs
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self._model = None
        self._cols = []

    def fit(self, df: pd.DataFrame) -> "CTGANGenerator":
        self._cols = df.columns.tolist()
        md = SingleTableMetadata()
        md.detect_from_dataframe(df)
        ctgan = CTGANSynthesizer(
            metadata=md,
            epochs=self.epochs,
            batch_size=self.batch_size,
            embedding_dim=self.embedding_dim,
            enforce_rounding=True,
        )
        ctgan.fit(df)
        self._model = ctgan
        return self

    def sample(self, n: int) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        out = self._model.sample(num_rows=n)
        return out[self._cols] if set(self._cols).issubset(out.columns) else out