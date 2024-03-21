from lit import LitModel, LitModelCfg
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary, StochasticWeightAveraging, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import lightning.pytorch as pl

import torch
torch.set_float32_matmul_precision('medium')

from datamodule import AFHQDataModule, AFHQDataset



class TrainConfig(LitModelCfg):

    ###### GENERAL TRAINING ##########

    seed: int = 42
    accelerator: str = "gpu"
    num_nodes: int = 1
    devices: int = 1
    strategy: Optional[str] = "auto"
    max_epochs : int = 1000
    gradient_clipping : float = 0.
    debug_single : bool = False
    accumulate_grad_batches : int = 1
    swa : bool = False

    ####### DATAMODULE CONFIG ##########
    data_folder : str = 'afhq/train'
    batch_size : int = 32
    num_workers : int = 4
    debug_data : bool = False

    im_size : int = 256
    normalize : bool = True

    ######## LOGGING ##########
    progress_bar: bool = True
    log_every_n_steps: int = 10

    checkpoint : bool = False
    checkpoint_dir: Optional[str] = None
    checkpoint_every_n_min: int = 10  # minutes

    wandb: bool = False
    wandb_project: str = "ch_test"
    wandb_entity: Optional[str] = "loluwot"
    wandb_run_id: str = None
    wandb_run_name: str = None

    ##### DEBUG ######
    debug_single : bool = False
    debug_num : int = 1

    class Config(pydantic_cli.DefaultConfig):
        extra = "forbid"
        CLI_BOOL_PREFIX = ("--enable_", "--disable_")


def train(cfg):
    pl.seed_everything(cfg.seed, workers=True)
    
    if cfg.checkpoint_dir is None:  # set to some random unique folder
        rand_dir = pathlib.Path(__file__).parents[0] / "checkpoints" / str(uuid.uuid4())
        assert not rand_dir.exists()
        cfg.checkpoint_dir = str(rand_dir)
    ckpt_dir = pathlib.Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = ckpt_dir.resolve()

    base_dataset = AFHQDataset(
        cfg.data_folder,
        im_size = cfg.im_size,
        use_normalize = cfg.normalize
    )

    datamodule = ChallengeDataModule(
        base_dataset, 
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
                                        
    model = LitModel(config=dict(cfg))
    
    callbacks = [ModelSummary(max_depth=2)]

    if cfg.wandb:
        project = cfg.wandb_project# + ("_debug" if cfg.debug else "")
        logger = WandbLogger(
            name=cfg.wandb_run_name,
            project=project,
            entity=cfg.wandb_entity,
            log_model=False,
            save_dir=ckpt_dir,
            config=dict(cfg),
            id=cfg.wandb_run_id,
            resume="allow",
        )
        # logger.watch(model)
        callbacks.append(LearningRateMonitor())
    else:
        logger = False

    if cfg.checkpoint:
        ckpt_callbacks = [
            ModelCheckpoint(
                dirpath=ckpt_dir,
                monitor='val_loss',
                filename='checkpoint-{val_loss:02f}',
                save_top_k=2,
                save_last=True,
                verbose=True,
                # train_time_interval=datetime.timedelta(minutes=cfg.checkpoint_every_n_min),
                every_n_epochs=1
            ),
        ]
        callbacks.extend(ckpt_callbacks)

    if cfg.swa:
        callbacks.append(StochasticWeightAveraging(swa_lrs=1e-2))

    # callbacks.append(EarlyStopping(monitor="train/loss", check_finite=True))

    trainer = pl.Trainer(
        accelerator=cfg.accelerator,
        num_nodes=cfg.num_nodes,
        devices=cfg.devices,
        strategy=cfg.strategy,
        callbacks=callbacks,
        enable_checkpointing=cfg.checkpoint,
        logger=logger,
        max_epochs=cfg.max_epochs,
        log_every_n_steps=cfg.log_every_n_steps,
        enable_progress_bar=cfg.progress_bar,
        num_sanity_val_steps=0,
        use_distributed_sampler=False,  # already handled in DataModule!
        gradient_clip_val=cfg.gradient_clipping,
        limit_train_batches = cfg.debug_num if cfg.debug_single else None,
        limit_val_batches = 0,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
    )

    ckpt_path = ckpt_dir / "last.ckpt"
    ckpt_path = str(ckpt_path) if ckpt_path.exists() else None
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    return 0