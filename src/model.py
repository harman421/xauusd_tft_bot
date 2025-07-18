# src/model.py (FINAL VERSION for PyTorch Lightning v2+)

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import pandas as pd

import config

def create_tft_dataset(data: pd.DataFrame):
    # This function is correct and needs no changes.
    training_cutoff = data["time_idx"].max() - config.PREDICTION_LENGTH
    time_varying_unknown_reals = [
        "Close", "Open", "High", "Low", "Volume", "hma", "hma_slope", "macd",
        "adx", "macd_signal", "rsi", "hma_dist_pct", "vwap_dist_pct",
        "atr_norm", "bb_width_norm", "bb_lower", "bb_upper", "vwap", "volume_zscore",
    ]
    dataset = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="Close",
        group_ids=["group"],
        min_encoder_length=config.ENCODER_LENGTH,
        max_encoder_length=config.ENCODER_LENGTH,
        min_prediction_length=config.PREDICTION_LENGTH,
        max_prediction_length=config.PREDICTION_LENGTH,
        static_categoricals=["group"],
        time_varying_known_categoricals=["hour", "day_of_week", "month"],
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=GroupNormalizer(
            groups=["group"], transformation="softplus"
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    return dataset


# --- ### THE FIX IS IN THIS FUNCTION ### ---
def train_tft_model(dataset: TimeSeriesDataSet, data: pd.DataFrame):
    """
    Initializes and trains the TFT model using PyTorch Lightning v2 syntax.
    """
    validation = TimeSeriesDataSet.from_dataset(dataset, data, predict=True, stop_randomization=True)

    train_dataloader = dataset.to_dataloader(train=True, batch_size=128, num_workers=2)
    val_dataloader = validation.to_dataloader(train=False, batch_size=128, num_workers=2)

    pl.seed_everything(42)
    
    # Define callbacks for Early Stopping and saving the best model
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min")
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")

    # The Trainer initialization is slightly different in v2
    trainer = pl.Trainer(
        max_epochs=25,
        accelerator="auto", # 'auto' is the modern way to select GPU/CPU
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback, checkpoint_callback] # Pass callbacks here
    )

    tft = TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=0.01,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=32,
        output_size=7,
        loss=QuantileLoss(),
        # No need for log_interval and reduce_on_plateau_patience, handled by callbacks
    )
    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

    # The trainer.fit() call is the same
    trainer.fit(
        tft, # The model is now correctly recognized as a LightningModule
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    return tft, trainer

# The save/load functions need a small tweak for the new checkpointing
def save_model(trainer, path):
    """Saves the best model from the trainer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # The best_model_path is now directly on the checkpoint_callback
    best_model_path = trainer.checkpoint_callback.best_model_path
    # No need to re-save, we can just copy the checkpoint file
    import shutil
    shutil.copy(best_model_path, path)
    print(f"Best model saved to {path}")

def load_model(path):
    """Loads a pre-trained model."""
    print(f"Loading model from {path}")
    return TemporalFusionTransformer.load_from_checkpoint(path)