from functools import partial
from dl.utils.collate import collate
from mmcv.utils import Registry, build_from_cfg
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler

DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')
SAMPLER = Registry('sampler')


def build_dataset(cfg, default_args=None):
    dataset = build_from_cfg(cfg, DATASETS, default_args)
    return dataset


def build_sampler(cfg, **kwargs):
    sampler = build_from_cfg({**cfg, **kwargs}, SAMPLER)
    return sampler


def build_dataloader(dataset,
                     sampler_cfg,
                     batch_sampler_cfg,
                     samples_per_gpu,
                     num_workers,
                     shuffle=False,
                     drop_last=False,
                     pin_memory=False,
                     **kwargs):
    """Build PyTorch DataLoader."""

    # set collate_fn
    if hasattr(dataset, 'collate_fn'):
        collate_fn = partial(collate, samples_per_gpu=samples_per_gpu, collate_fn=dataset.collate_fn)
    else:
        collate_fn = partial(collate, samples_per_gpu=samples_per_gpu)

    if sampler_cfg is None:
        sampler = None
    else:
        sampler = build_sampler(sampler_cfg, data_source=dataset, shuffle=shuffle)

    if batch_sampler_cfg is None:
        batch_sampler = None
    else:
        sampler = None
        batch_sampler = build_sampler(
            batch_sampler_cfg,
            sampler=SequentialSampler(dataset),
            batch_size=samples_per_gpu,
            drop_last=drop_last,
            shuffle=shuffle)

    if sampler is None:
        if batch_sampler is None:
            data_loader = DataLoader(
                dataset,
                batch_size=samples_per_gpu,
                num_workers=num_workers,
                drop_last=drop_last,
                shuffle=shuffle,
                collate_fn=collate_fn,
                pin_memory=pin_memory,
                **kwargs)
        else:
            data_loader = DataLoader(
                dataset,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=collate_fn,
                pin_memory=pin_memory,
                **kwargs)
    else:
        data_loader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=samples_per_gpu,
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            **kwargs)
    return data_loader
