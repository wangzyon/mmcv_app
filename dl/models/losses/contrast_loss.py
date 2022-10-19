import torch
import torch.nn as nn
from .cross_entropy_loss import CrossEntropyLoss, LOSSES


@LOSSES.register_module()
class BatchContrastLoss(CrossEntropyLoss):
    """batch内进行对比学习"""

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 ignore_index=None,
                 loss_weight=1.0,
                 avg_non_ignore=False,
                 T=0.1):
        """
        对比学习损失

        Args:
            T (float, optional): softmax temperature. Defaults to 0.1.
        """
        super().__init__(use_sigmoid, use_mask, reduction, class_weight, ignore_index, loss_weight, avg_non_ignore)
        self.T = T

    def caculate_logits_and_labels(self, x, input_tags, device):
        """构建对比学习正负样本和标签"""
        # L2标准化
        x = nn.functional.normalize(x, dim=-1)

        # 当前batch样本信息
        batch_size, num_plus, dim = x.shape
        max_mix_num = input_tags.max() + 1
        batch_num_plus = batch_size * num_plus

        # 计算子信号特征向量（子信号所有脉冲特征向量的平均）
        tags = torch.zeros(batch_size, max_mix_num, num_plus, device=device)
        tags[torch.arange(batch_size).repeat_interleave(num_plus).view(batch_size, -1), input_tags, torch.arange(num_plus)] = 1
        tags_weight = torch.nn.functional.normalize(tags, p=1, dim=-1)
        valid_flags = tags.sum(dim=-1, keepdim=True).bool()
        centres = torch.einsum('bcn,bnd->bcd', [tags_weight, x]).view(batch_size * max_mix_num, dim)
        centres = centres[valid_flags.view(-1)]

        # 计算脉冲标签矩阵
        help_mask = torch.eye(batch_size, device=device)
        help_mask = help_mask.repeat_interleave(max_mix_num, dim=0).repeat_interleave(num_plus, dim=1)
        tags = tags.repeat(1, batch_size, 1).permute(1, 0, 2).reshape(batch_size * max_mix_num, -1)
        tags = (tags * help_mask)[valid_flags.view(-1)]

        # 获取query
        indices = (torch.where(tags.view(-1) == 1)[0] / batch_num_plus).long()
        query = centres[indices]

        # 获取正样本pos, [batch_num_plus, dim]
        indices = torch.where(tags.view(-1) == 1)[0] % batch_num_plus
        pos = x.view(-1, dim)[indices]

        # 获取负样本batch_neg, [batch_num_plus, num_plus, dim]
        batch_neg = pos.reshape(batch_size, num_plus, dim).repeat_interleave(num_plus, dim=0)
        indices = (torch.where(tags.view(-1) == 1)[0] / batch_num_plus).long().reshape(batch_size, -1)
        help_mask = indices - indices.min(dim=1)[0][:, None]
        help_mask = (help_mask[:, None, :] == help_mask[:, :, None]) * (-2) + 1
        help_mask = help_mask.view(batch_num_plus, num_plus)
        batch_neg = torch.einsum("bnd,bn->bnd", [batch_neg, help_mask])

        # 计算余弦相似度
        # l_pos，[batch_num_pulse, 1]
        l_pos = torch.einsum('bd,bd->b', [query, pos]).unsqueeze(-1)
        # l_batch_neg，[batch_num_pulse, num_pulse]
        l_batch_neg = torch.einsum('bd,bcd->bc', [query, batch_neg])
        # logit，[batch_num_pulse, num_pulse+1]
        logits = torch.cat([l_pos, l_batch_neg], dim=1)
        logits /= self.T    # 余弦相似度[-1,1]，需要对区间进行拉伸

        # 设置分类标签，pos总在第0位置，因此标签值全0
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)
        return logits, labels

    def forward(self, tags, x, weight=None, avg_factor=None, reduction_override=None, ignore_index=None, **kwargs):
        logits, labels = self.caculate_logits_and_labels(x, tags, x.device)
        loss = super().forward(logits, labels, weight, avg_factor, reduction_override, ignore_index, **kwargs)
        return loss


@LOSSES.register_module()
class MocoContrastLoss(CrossEntropyLoss):
    """Moco全局对比学习"""

    def __init__(self,
                 encoder_q,
                 encoder_k,
                 dim,
                 K=65536,
                 T=0.1,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 ignore_index=None,
                 loss_weight=1.0,
                 avg_non_ignore=False):
        """
        对比学习损失

        Args:
            T (float, optional): softmax temperature. Defaults to 0.1.
        """
        super().__init__(use_sigmoid, use_mask, reduction, class_weight, ignore_index, loss_weight, avg_non_ignore)
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k
        self.T = T
        self.K = K

        # initialize and set k do not update by gradient
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(K, dim))
        self.queue = nn.functional.normalize(self.queue, dim=-1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        # replace the keys at ptr (dequeue and enqueue)
        if (ptr + batch_size) > self.K:
            self.queue[ptr:self.K, :] = keys[:self.K - ptr]
            self.queue[:(batch_size + ptr - self.K), :] = keys[self.K - ptr:]
            ptr = batch_size + ptr - self.K
        else:
            self.queue[ptr:ptr + batch_size, :] = keys
            ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def caculate_logits_and_labels(self, feat_q, feat_k, input_tags, device):
        """构建对比学习正负样本和标签"""
        # L2标准化
        feat_q = nn.functional.normalize(feat_q, dim=-1)
        with torch.no_grad():
            feat_k = nn.functional.normalize(feat_k, dim=-1)

        # 当前batch样本信息
        batch_size, num_plus, dim = feat_q.shape
        max_mix_num = input_tags.max() + 1
        batch_num_plus = batch_size * num_plus

        # 计算子信号特征向量（子信号所有脉冲特征向量的平均）
        # centres_q，[centres_num,dim], encoder_q编码后子信号特征向量
        # centres_k，[centres_num,dim], encoder_k编码后子信号特征向量
        tags = torch.zeros(batch_size, max_mix_num, num_plus, device=device)
        tags[torch.arange(batch_size).repeat_interleave(num_plus).view(batch_size, -1), input_tags, torch.arange(num_plus)] = 1
        tags_weight = torch.nn.functional.normalize(tags, p=1, dim=-1)
        valid_flags = tags.sum(dim=-1, keepdim=True).bool()
        centres_q = torch.einsum('bcn,bnd->bcd', [tags_weight, feat_q]).view(batch_size * max_mix_num, dim)
        centres_q = centres_q[valid_flags.view(-1)]
        with torch.no_grad():
            centres_k = torch.einsum('bcn,bnd->bcd', [tags_weight, feat_k]).view(batch_size * max_mix_num, dim)
            centres_k = centres_k[valid_flags.view(-1)]

        # 计算脉冲标签矩阵[]
        help_mask = torch.eye(batch_size, device=device)
        help_mask = help_mask.repeat_interleave(max_mix_num, dim=0).repeat_interleave(num_plus, dim=1)
        tags = tags.repeat(1, batch_size, 1).permute(1, 0, 2).reshape(batch_size * max_mix_num, -1)
        tags = (tags * help_mask)[valid_flags.view(-1)]

        # 获取query
        indices = (torch.where(tags.view(-1) == 1)[0] / batch_num_plus).long()
        query = centres_q[indices]

        # 获取正样本pos, [batch_num_plus, dim]
        indices = torch.where(tags.view(-1) == 1)[0] % batch_num_plus
        pos = feat_k.view(-1, dim)[indices]

        # 获取负样本batch_neg, [batch_num_plus, num_plus, dim]
        batch_neg = pos.reshape(batch_size, num_plus, dim).repeat_interleave(num_plus, dim=0)
        indices = (torch.where(tags.view(-1) == 1)[0] / batch_num_plus).long().reshape(batch_size, -1)
        help_mask = indices - indices.min(dim=1)[0][:, None]
        help_mask = (help_mask[:, None, :] == help_mask[:, :, None]) * (-2) + 1
        help_mask = help_mask.view(batch_num_plus, num_plus)
        batch_neg = torch.einsum("bnd,bn->bnd", [batch_neg, help_mask])

        # 获取queue_neg, [K, dim]
        queue_neg = self.queue.clone().detach()

        # 计算余弦相似度
        # l_pos，[batch_num_pulse, 1]
        l_pos = torch.einsum('bd,bd->b', [query, pos]).unsqueeze(-1)
        # l_batch_neg，[batch_num_pulse, num_pulse]
        l_batch_neg = torch.einsum('bd,bcd->bc', [query, batch_neg])
        # l_queue_neg [batch_num_pulse, k]
        l_queue_neg = torch.einsum('bd,kd->bk', [query, queue_neg])

        # logit，[batch_num_pulse, num_pulse+1+k]
        logits = torch.cat([l_pos, l_batch_neg, l_queue_neg], dim=1)
        logits /= self.T    # 余弦相似度[-1,1]，需要对区间进行拉伸

        # 设置分类标签，pos总在第0位置，因此标签值全0
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)

        # dequeue and enqueue
        self._dequeue_and_enqueue(centres_k)
        return logits, labels

    def forward(self, tags, feat_q, feat_k, weight=None, avg_factor=None, reduction_override=None, ignore_index=None, **kwargs):
        logits, labels = self.caculate_logits_and_labels(feat_q, feat_k, tags, feat_q.device)
        loss = super().forward(logits, labels, weight, avg_factor, reduction_override, ignore_index, **kwargs)
        return loss