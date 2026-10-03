from dataclasses import dataclass, asdict


@dataclass
class ExperimentConfig:
    seed: int = 42
    data_root: str = "data"
    image_size: int = 224
    val_ratio: float = 0.20
    batch_size: int = 32
    num_workers: int = 4

    warmup_epochs: int = 3
    finetune_epochs: int = 10
    warmup_lr: float = 1e-3
    finetune_lr: float = 1e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 4

    scratch_epochs: int = 15
    scratch_lr: float = 3e-4

    output_root: str = "outputs"
    use_amp: bool = True

    def to_dict(self):
        return asdict(self)
