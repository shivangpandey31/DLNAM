import pandas as pd
import torch

from DLNAM import DataProcessor, LayerSpec, ModelConfig, SurfaceTermSpec, TrainConfig, Trainer


def _strata_cfg(n=2):
    return {
        "name": "person",
        "col": "id",
        "num_categories": n,
        "order": [],
        "encoding_type": "embedding",
        "embedding_dim": 1,
        "hidden_layers": [],
    }


def test_grouped_lags_and_neural_strata_are_separate():
    df = pd.DataFrame({
        "id": ["A"] * 5 + ["B"] * 5,
        "date": pd.date_range("2020-01-01", periods=5).tolist() * 2,
        "tas": [1, 2, 3, 4, 5, 101, 102, 103, 104, 105],
        "event": [0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    })
    cfg = ModelConfig(
        terms={"tas": SurfaceTermSpec(lag_max=2, layers=(LayerSpec(8),), num_subnets=1)},
        strata_config=_strata_cfg(2),
        link="logit",
    )
    trainer = Trainer(
        cfg,
        TrainConfig(epochs=0, n_ensemble=1, loss="bernoulli", show_progress=False),
        device=torch.device("cpu"),
    )
    prepared = DataProcessor(cfg).prepare(
        df,
        trainer.ensemble,
        target_col="event",
        input_type="grouped",
        groupby_col="id",
        time_col="date",
    )
    assert prepared.raw["tas"].tolist() == [3, 4, 5, 103, 104, 105]
    assert prepared.inputs["person"].tolist() == [0, 0, 0, 1, 1, 1]
    assert prepared.category_orders["person"] == ["A", "B"]
    assert cfg.terms["person"].role == "strata"


def test_strata_term_is_sum_to_zero_centered():
    cfg = ModelConfig(terms={}, strata_config=_strata_cfg(3), link="logit")
    trainer = Trainer(
        cfg,
        TrainConfig(epochs=0, n_ensemble=1, loss="bernoulli", show_progress=False),
        device=torch.device("cpu"),
    )
    term = trainer.ensemble[0].term("person")
    with torch.no_grad():
        term.emb.weight[:, 0] = torch.tensor([1.0, 2.0, 6.0])
    out = term(torch.tensor([0, 1, 2])).reshape(-1)
    assert torch.allclose(out.mean(), torch.tensor(0.0), atol=1e-6)


def test_random_minibatch_training_runs_with_neural_strata():
    rows = []
    for ident, offset in [("A", 0), ("B", 10), ("C", 20)]:
        for day in range(8):
            rows.append({"id": ident, "tas": offset + day, "event": float(day == 7)})
    df = pd.DataFrame(rows)
    cfg = ModelConfig(
        terms={"tas": SurfaceTermSpec(lag_max=2, layers=(LayerSpec(4),), num_subnets=1)},
        strata_config=_strata_cfg(3),
        link="logit",
    )
    trainer = Trainer(
        cfg,
        TrainConfig(
            epochs=1,
            n_ensemble=1,
            loss="bernoulli",
            batch_fraction=0.25,
            schedule="none",
            show_progress=False,
            strata_weight_decay=0.0,
        ),
        device=torch.device("cpu"),
    )
    prepared = DataProcessor(cfg).prepare(
        df, trainer.ensemble, target_col="event", input_type="grouped", groupby_col="id"
    )
    trainer.fit(prepared.inputs, prepared.y)
    assert trainer.fit_summary["n_samples"] == prepared.n_samples
