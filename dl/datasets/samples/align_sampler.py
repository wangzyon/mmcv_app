import numpy as np
from torch.utils.data import Sampler, BatchSampler
from ..builder import SAMPLER

__all__ = ["AlignSampler"]


@SAMPLER.register_module()
class AlignSampler():

    def __init__(self, sampler, batch_size, drop_last, shuffle=True, **kwargs):
        self.sampler = sampler
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size    # type: ignore
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        batches = []
        sample_indices = np.arange(len(self.sampler))
        pulse_nums = np.array([data_info["total_pulse_num"] for data_info in self.sampler.data_source.data_infos])
        sample_indices_sort = sample_indices[np.argsort(pulse_nums)]

        for start in range(0, len(self.sampler), self.batch_size):
            if (start + self.batch_size > len(self.sampler)):
                batches.append(sample_indices_sort[start:])
            else:
                batches.append(sample_indices_sort[start:start + self.batch_size])

        if self.shuffle:
            np.random.shuffle(batches)
        return iter(batches)
