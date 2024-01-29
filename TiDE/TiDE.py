import torch
from darts.models import TiDEModel
from Helper import DataPreprocessorTiDE
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.progress.progress_bar import ProgressBar
from darts import TimeSeries


def build_tide_1(
        dataset: DataPreprocessorTiDE,
        input_chunk_length,
        output_chunk_length,
        epochs=200,
        batch_size=32,
        use_future_covs: bool = True,
        cov_names=None,
        n_extend: int = 24):
    """
    Builds, compiles, and fits our Multivariate_TiDE baseline model.
    """
    # Model configuration
    # *************************************************************************
    cov_names = [] if cov_names is None else cov_names

    optimizer_kwargs = {
        "lr": 1e-3,
    }

    # PyTorch Lightning Trainer arguments
    pl_trainer_kwargs = {
        "gradient_clip_val": 1,
        "max_epochs": epochs,
        "accelerator": "auto",
        "callbacks": [EarlyStopping(monitor="val_loss", patience=50, min_delta=0.001, verbose=True)],
    }

    # learning rate scheduler
    lr_scheduler_cls = torch.optim.lr_scheduler.ExponentialLR
    lr_scheduler_kwargs = {
        "gamma": 0.999,
    }

    common_model_args = {
        "input_chunk_length": input_chunk_length,  # lookback window
        "output_chunk_length": output_chunk_length,  # forecast/lookahead window
        "optimizer_kwargs": optimizer_kwargs,
        "pl_trainer_kwargs": pl_trainer_kwargs,
        "lr_scheduler_cls": lr_scheduler_cls,
        "lr_scheduler_kwargs": lr_scheduler_kwargs,
        "likelihood": None,  # use a likelihood for probabilistic forecasts
        "save_checkpoints": False,  # checkpoint to retrieve the best performing model state,
        "force_reset": True,
        "random_state": 42,
        "batch_size": batch_size,
        "n_epochs": epochs,
    }

    # Create the model
    # *************************************************************************
    model_tide = TiDEModel(**common_model_args, use_reversible_instance_norm=False)

    train_target_series, train_past_covs = dataset.train_series()
    test_target_series, test_past_covs = dataset.test_series()
    future_covs = dataset.future_series(cov_names, n_extend) if use_future_covs else None

    model_tide.fit(
        series=train_target_series,
        past_covariates=train_past_covs,
        future_covariates=future_covs,
        val_series=test_target_series,
        val_past_covariates=test_past_covs,
        verbose=True)

    return model_tide
