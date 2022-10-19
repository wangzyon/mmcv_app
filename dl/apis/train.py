from ..utils import find_latest_checkpoint
from ..datasets import build_dataloader, build_dataset, build_sampler
from mmcv.runner import (Fp16OptimizerHook, build_optimizer, build_runner)
from ..models import build_separator
from mmcv.runner.hooks.evaluation import EvalHook

__all__ = ["train_model"]


def train_model(cfg, logger, device, meta=None):
    # 是否进行验证
    validate = cfg.data.get('val', None) is not None
    for workflow, _ in cfg.workflow:
        if workflow == 'val':
            validate = True

    # build data loaders
    train_samples_per_gpu = cfg.data.train.pop('samples_per_gpu', 1)
    train_dataset = build_dataset(cfg.data.train)
    train_data_loader = build_dataloader(
        dataset=train_dataset,
        sampler_cfg=cfg.data.get('sampler', None),
        batch_sampler_cfg=cfg.data.get('batch_sampler', None),
        samples_per_gpu=train_samples_per_gpu,
        num_workers=cfg.data.workers_per_gpu,
        shuffle=cfg.data.get('shuffle', False),
        drop_last=cfg.data.get('drop_last', False))
    if validate:
        """
        val阶段，sampler、batch_sampler、shuffle均可使val_dataloader生成数据乱序，与dataset顺序不一致;
        若采用val_data_loader进行evaluation，其forward_test产生results和dataset顺序不一致，无法进行metric计算；
        因此，需要执行evaluation时，应使用方案之一：
        - 将sampler、batch_sampler置为None,shuffle置为False，避免乱序；
        - 在evaluation中对data_info也进行相应顺序调整，使其与results顺序一致；
        """

        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        val_dataset = build_dataset(cfg.data.val)
        val_data_loader = build_dataloader(
            dataset=val_dataset,
            sampler_cfg=None,    # cfg.data.get('sampler', None),
            batch_sampler_cfg=None,    #cfg.data.get('batch_sampler', None),
            samples_per_gpu=val_samples_per_gpu,
            num_workers=cfg.data.workers_per_gpu,
            shuffle=False,
            drop_last=cfg.data.get('drop_last', False))

    data_loader_dict = {"train": train_data_loader, "val": val_data_loader}
    data_loaders = [data_loader_dict[mode] for mode, _ in cfg.workflow]

    # build model
    if cfg.task_name.startswith("Separation"):
        model = build_separator(cfg.model)
        model.classes = train_dataset.classes
    model.to(device)
    model.device = device

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = build_runner(
        cfg.runner, default_args=dict(model=model, optimizer=optimizer, work_dir=cfg.work_dir, logger=logger, meta=meta))

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(**cfg.optimizer_config, **fp16_cfg, distributed=False)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))

    # register eval hooks
    if validate:
        eval_cfg = cfg.get('evaluation', {})
        runner.register_hook(EvalHook(val_data_loader, **eval_cfg), priority='LOW')

    # resume or load from
    resume_from = None
    if cfg.resume_from is None and cfg.get('auto_resume'):
        resume_from = find_latest_checkpoint(cfg.work_dir)
    if resume_from is not None:
        cfg.resume_from = resume_from

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    # start
    runner.run(data_loaders, cfg.workflow)


