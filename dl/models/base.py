# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import torch
from mmcv.runner import BaseModule, auto_fp16
import torch.distributed as dist


class BaseModel(BaseModule, metaclass=ABCMeta):
    """Base class for signal separation."""

    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.fp16_enabled = False

    @abstractmethod
    def extract_feat(self, imgs):
        """Extract features from signal."""
        pass

    @abstractmethod
    def forward_train(self, dtoas, tags, **kwargs):
        pass

    @abstractmethod
    def forward_test(self, dtoas, **kwargs):
        pass

    def convert_tensor_to_model_deivce(self, x):
        return x.to(self.device) if isinstance(x, torch.Tensor) else x

    def __call__(self, *args, **kwargs):
        """conovert data to model device"""
        args = [self.convert_tensor_to_model_deivce(x) for x in args]
        kwargs = {k: self.convert_tensor_to_model_deivce(v) for k, v in kwargs.items()}
        return super()._call_impl(*args, **kwargs)

    @auto_fp16(apply_to=('dtoas',))
    def forward(self, dtoas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.
        """
        if return_loss:
            return self.forward_train(dtoas, **kwargs)
        else:
            return self.forward_test(dtoas, **kwargs)

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)
        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()
        return loss, log_vars

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """

        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data['data_metas']))

        return outputs

    def val_step(self, data, optimizer=None):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data['data_metas']))

        return outputs
