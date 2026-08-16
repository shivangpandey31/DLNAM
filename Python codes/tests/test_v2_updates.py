"""Small regression tests for the compatibility additions in V2_UPDATE_NOTES.md."""

import pandas as pd
import torch
import sys

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DLNAM import (
    DataProcessor,
    LayerSpec,
    ModelConfig,
    SurfaceTermSpec,
    TrainConfig,
    Trainer,
)


def test_grouped_lags_do_not_cross_groups():
    df = pd.DataFrame({
        "id": ["A"] * 5 + ["B"] * 5,
        "tas": [1, 2, 3, 4, 5, 101, 102, 103, 104, 105],
        "event": [0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    })
    cfg = ModelConfig(
        terms={
            "tas": SurfaceTermSpec(
                lag_max=2,
                layers=(LayerSpec(8),),
                num_subnets=1,
            )
        },
        link="logit",
        encoding_configs=[{
            "name": "person",
            "col": "id",
            "num_categories": 2,
            "order": [],
            "hidden_layers": [],
            "encoding_type": "embedding",
            "embedding_dim": 1,
        }],
    )
    trainer = Trainer(
        cfg,
        TrainConfig(epochs=0, n_ensemble=1, loss="bernoulli"),
        device=torch.device("cpu"),
    )
    prepared = DataProcessor(cfg).prepare(
        df,
        trainer.ensemble,
        target_col="event",
        input_type="grouped",
        groupby_col="id",
    )
    assert prepared.raw["tas"].tolist() == [3, 4, 5, 103, 104, 105]
    assert prepared.inputs["person"].tolist() == [0, 0, 0, 1, 1, 1]
    assert prepared.category_orders["person"] == ["A", "B"]
