from typing import overload
from ..builder import SEPARATORS, build_backbone, build_head, build_loss, build_cluster
import torch
from torch import nn
from ..base import BaseModel


@SEPARATORS.register_module()
class SeparatorI(BaseModel):
    """信号调制模式分选，属于分选的第一阶段，即混叠信号包含多种不同调制模式、不同调制参数的子信号，将不同调制模式的信号完全分离开"""

    def __init__(self, backbone, head, loss, train_cfg=None, test_cfg=None, init_cfg=None):
        """_summary_

        Args:
            backbone (dict): 对比特征提取骨干
            head (dict): 对比头
            loss (dict): 对比损失
            train_cfg (dict, optional): 训练参数配置. Defaults to None.
            test_cfg (dict, optional): 测试参数配置. Defaults to None.
            init_cfg (dict, optional): 权重初始化配置. Defaults to None.
        """
        super().__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        self.head = build_head(head)
        self.loss = build_loss(loss)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, x):
        """提取信号各脉冲的特征"""
        x = self.backbone(x)
        x = self.head(x)
        return x

    def caculate_loss(self, x, tags):
        """分选损失"""
        loss = self.loss(x, tags)
        return loss

    def forward_train(self, dtoas, modes, **kwargs):
        """信号分选训练"""
        loss = self.caculate_loss(self.extract_feat(dtoas), modes)
        losses = {"loss": loss}
        return losses

    def forward_test(self, dtoas, **kwargs):
        """信号分选推理"""
        out_features = self.extract_feat(dtoas)
        pre = torch.argmax(pre.reshape(-1, out_features.shape[-1]), dim=1).reshape(-1)
        return {'pre': pre}


# @SEPARATORS.register_module()
# class SeparatorIIV1(BaseModel):
#     """信号实例分选，属于分选的第二阶段，即混叠信号包含1~n个同种调制模式、不同调制参数的子信号，将1~n信号实例完全分离开"""

#     def __init__(self,
#                  contrast_backbone,
#                  contrast_head,
#                  contrast_loss,
#                  classify_backbone,
#                  classify_head,
#                  classify_loss,
#                  cluster,
#                  train_cfg=None,
#                  test_cfg=None,
#                  init_cfg=None):
#         """_summary_

#         Args:
#             contrast_backbone (dict): 对比特征提取骨干
#             contrast_head (dict): 对比头
#             contrast_loss (dict): 对比损失
#             classify_backbone (dict): 分类特征提取骨干
#             classify_head (dict): 分类头
#             classify_loss (dict): 分类损失
#             cluster (dict): _description_
#             train_cfg (dict, optional): 训练参数配置. Defaults to None.
#             test_cfg (dict, optional): 测试参数配置. Defaults to None.
#             init_cfg (dict, optional): 权重初始化配置. Defaults to None.
#         """
#         super().__init__(init_cfg)
#         self.contrast_backbone = build_backbone(contrast_backbone)
#         self.contrast_head = build_head(contrast_head)
#         self.contrast_loss = build_loss(contrast_loss)

#         self.classify_backbone = build_backbone(classify_backbone)
#         self.classify_head = build_head(classify_head)
#         self.classify_loss = build_loss(classify_loss)

#         self.cluster = build_cluster(cluster)
#         self.train_cfg = train_cfg
#         self.test_cfg = test_cfg

#     def extract_feat(self, x):
#         pass

#     def extract_classify_feat(self, x):
#         """提取分类特征"""
#         x = self.classify_backbone(x)
#         x = self.classify_head(x)
#         return x

#     def extract_contrast_feat(self, x):
#         """提取信号序列标注特征"""
#         x = self.contrast_backbone(x)
#         x = self.contrast_head(x)
#         return x

#     def caculate_classify_loss(self, x, tags):
#         """
#         计算分类损失：

#         类别 0->信号由单一信号组成，不需要分选；
#         类别 1->信号由多个子信号混叠而成，需要分选；
#         Args:
#             x (tensor): [batch_size, dim] 分类特征；
#             tags (tensor): [batch_size, n] 信号脉冲标签；

#         Returns:
#             loss (tesnor)
#         """
#         need_separations = (tags.max(dim=1)[0] != 0).long()    # 脉冲标签若大于0,表示存在多个子信号
#         loss = self.classify_loss(x, need_separations)
#         return loss

#     def caculate_contrast_loss(self, x, tags):
#         """对比学习损失"""
#         loss = self.contrast_loss(x, tags)
#         return loss

#     def forward_train(self, dtoas, tags, **kwargs):
#         """信号分选训练"""
#         classify_loss = self.caculate_classify_loss(self.extract_classify_feat(dtoas), tags)
#         contrast_loss = self.caculate_contrast_loss(self.extract_contrast_feat(dtoas), tags)
#         losses = {"contrast_loss": contrast_loss, "classify_loss": classify_loss}
#         return losses

#     def forward_test(self, dtoas, **kwargs):
#         """信号分选推理"""
#         need_separations = self.test_classify(dtoas)
#         signal_features = self.test_contrast(dtoas)
#         assert len(signal_features) == len(need_separations)
#         results = self.cluster(signal_features, need_separations)

#         if kwargs.get('tags') is not None:
#             for signal_feature, result, tags in zip(signal_features, results, kwargs.get('tags')):
#                 signal_cosine = self.caculate_signal_cosine(signal_feature, tags)
#                 result['signal_cosine'] = signal_cosine
#         return results

#     def test_classify(self, dtoas):
#         """信号分类推理"""
#         features = self.extract_classify_feat(dtoas)
#         need_separations = torch.argmax(features.reshape(-1, features.shape[-1]), dim=1).reshape(-1)
#         return need_separations

#     def test_contrast(self, dtoas):
#         """对比学习推理"""
#         features = self.extract_contrast_feat(dtoas)
#         features = nn.functional.normalize(features, dim=-1)
#         return features

#     def caculate_signal_cosine(self, features: torch.Tensor, tags: torch.Tensor):
#         """
#         计算信号脉冲类间平均余弦相似度

#         1.信号特征features.shape->[n,dim]，每个脉冲特征[1,dim]，计算脉冲两两间余弦相似度;
#         2.根据脉冲标签，统计类间和类内相似度，获取相似度矩阵signal_cosine，signal.shape->[n,n]；
#         3.若信号含有4个子信号，即4类脉冲，则signal_cosine.shape->[4,4],signal_cosine[1,2]表示类别1脉冲和类别2脉冲特征平均余弦相似度；

#         """
#         features = torch.nn.functional.normalize(features, dim=1)
#         signal_num = len(tags.unique())
#         weight = tags.repeat(len(tags)) + signal_num * tags.repeat_interleave(len(tags))
#         weight = weight[None, :].repeat(signal_num * signal_num, 1)
#         weight = (weight == torch.arange(signal_num * signal_num, device=features.device)[:, None]).float()
#         weight = torch.nn.functional.normalize(weight, p=1, dim=1)
#         cosine = torch.einsum('id,jd->ij', [features, features])
#         signal_cosine = torch.einsum('n,kn->k', [cosine.view(-1), weight]).view(signal_num, signal_num)
#         return signal_cosine


@SEPARATORS.register_module()
class SeparatorII(BaseModel):
    """信号实例分选，属于分选的第二阶段，即混叠信号包含1~n个同种调制模式、不同调制参数的子信号，将1~n子信号实例完全分离开"""

    def __init__(self,
                 contrast_backbone,
                 contrast_head,
                 contrast_loss,
                 classify_backbone,
                 classify_head,
                 classify_loss,
                 cluster,
                 moco_moment=0.999,
                 use_moco=False,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        """_summary_

        Args:
            contrast_backbone (dict): 对比特征提取骨干
            contrast_head (dict): 对比头
            contrast_loss (dict): 对比损失
                * use_moco=False，损失函数仅支持BatchContrastLoss，
                * use_moco=True，损失函数仅支持MocoContrastLoss
            classify_backbone (dict): 分类特征提取骨干
            classify_head (dict): 分类头
            classify_loss (dict): 分类损失
            cluster (dict): 聚类器
            use_moco (bool): 使用moco对比学习算法，相比于batch对比学习，负样本更丰富，准确性理论更佳，具体视场景而定；
                同时显存占用增加，训练更耗时(耗时增加约25%)，需要针对根据提升程度酌情使用；
            moco_moment (float): 模型参数更新动量因子，use_moco=True时有效；
            train_cfg (dict, optional): 训练参数配置. Defaults to None.
            test_cfg (dict, optional): 测试参数配置. Defaults to None.
            init_cfg (dict, optional): 权重初始化配置. Defaults to None.
        """
        super().__init__(init_cfg)

        # moco对比学习相关
        self.use_moco = use_moco
        self.moco_moment = moco_moment

        if self.use_moco:
            assert contrast_loss.type == "MocoContrastLoss", "moco mode(use_moco=True) only support MocoContrastLoss"
            self.contrast_encoder_q = nn.Sequential(build_backbone(contrast_backbone), build_head(contrast_head))
            self.contrast_encoder_k = nn.Sequential(build_backbone(contrast_backbone), build_head(contrast_head))
            self.contrast_loss = build_loss(contrast_loss, encoder_q=self.contrast_encoder_q, encoder_k=self.contrast_encoder_k)
        else:
            assert contrast_loss.type == "BatchContrastLoss", "batch mode(use_moco=False) only support BatchContrastLoss"
            self.contrast_encoder = nn.Sequential(build_backbone(contrast_backbone), build_head(contrast_head))
            self.contrast_loss = build_loss(contrast_loss)

        self.classify_backbone = build_backbone(classify_backbone)
        self.classify_head = build_head(classify_head)
        self.classify_loss = build_loss(classify_loss)

        self.cluster = build_cluster(cluster)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, x):
        pass

    def extract_classify_feat(self, x):
        """提取分类特征"""
        x = self.classify_backbone(x)
        x = self.classify_head(x)
        return x

    def caculate_classify_loss(self, x, tags):
        """
        计算分类损失：
        
        类别 0->信号由单一信号组成，不需要分选；
        类别 1->信号由多个子信号混叠而成，需要分选；
        Args:
            x (tensor): [batch_size, dim] 分类特征； 
            tags (tensor): [batch_size, n] 信号脉冲标签；

        Returns:
            loss (tesnor)
        """
        need_separations = (tags.max(dim=1)[0] != 0).long()    # 脉冲标签若大于0,表示存在多个子信号
        loss = self.classify_loss(x, need_separations)
        return loss

    def test_classify(self, dtoas):
        """信号分类推理"""
        features = self.extract_classify_feat(dtoas)
        need_separations = torch.argmax(features.reshape(-1, features.shape[-1]), dim=1).reshape(-1)
        return need_separations

    def extract_contrast_feat(self, x):
        """提取信号序列标注特征"""
        x = self.contrast_encoder(x)
        return x

    def extract_contrast_feat_q(self, x):
        """提取来自编码器q的信号序列标注特征"""
        x = self.contrast_encoder_q(x)
        return x

    @torch.no_grad()
    def extract_contrast_feat_k(self, x, update_params=True):
        """
        提取来自编码器k的信号序列标注特征
        
        update_first：Momentum update of the key encoder
        """
        if update_params:
            for param_q, param_k in zip(self.contrast_encoder_q.parameters(), self.contrast_encoder_k.parameters()):
                param_k.data = param_k.data * self.moco_moment + param_q.data * (1. - self.moco_moment)
        x = self.contrast_encoder_k(x)
        return x

    def caculate_contrast_loss(self, tags, *args):
        """对比学习损失"""
        loss = self.contrast_loss(tags, *args)
        return loss

    def test_contrast(self, dtoas):
        """对比学习推理"""
        if self.use_moco:
            features = self.extract_contrast_feat_k(dtoas, update_params=False)
        else:
            features = self.extract_contrast_feat(dtoas)
        features = nn.functional.normalize(features, dim=-1)
        return features

    def forward_train(self, dtoas, tags, **kwargs):
        """信号分选训练"""
        classify_loss = self.caculate_classify_loss(self.extract_classify_feat(dtoas), tags)
        if self.use_moco:
            contrast_loss = self.caculate_contrast_loss(tags, self.extract_contrast_feat_q(dtoas),
                                                        self.extract_contrast_feat_k(dtoas))
        else:
            contrast_loss = self.caculate_contrast_loss(tags, self.extract_contrast_feat(dtoas))
        losses = {"contrast_loss": contrast_loss, "classify_loss": classify_loss}
        return losses

    def forward_test(self, dtoas, **kwargs):
        """信号分选推理"""
        need_separations = self.test_classify(dtoas)
        signal_features = self.test_contrast(dtoas)
        assert len(signal_features) == len(need_separations)
        results = self.cluster(signal_features, need_separations)

        if kwargs.get('tags') is not None:
            for signal_feature, result, tags in zip(signal_features, results, kwargs.get('tags')):
                signal_cosine = self.caculate_signal_cosine(signal_feature, tags)
                result['signal_cosine'] = signal_cosine
        return results

    def caculate_signal_cosine(self, features: torch.Tensor, tags: torch.Tensor):
        """
        计算信号脉冲类间平均余弦相似度
        
        1.信号特征features.shape->[n,dim]，每个脉冲特征[1,dim]，计算脉冲两两间余弦相似度;
        2.根据脉冲标签，统计类间和类内相似度，获取相似度矩阵signal_cosine，signal.shape->[n,n]；
        3.若信号含有4个子信号，即4类脉冲，则signal_cosine.shape->[4,4],signal_cosine[1,2]表示类别1脉冲和类别2脉冲特征平均余弦相似度；
        
        """
        features = torch.nn.functional.normalize(features, dim=1)
        signal_num = len(tags.unique())
        weight = tags.repeat(len(tags)) + signal_num * tags.repeat_interleave(len(tags))
        weight = weight[None, :].repeat(signal_num * signal_num, 1)
        weight = (weight == torch.arange(signal_num * signal_num, device=features.device)[:, None]).float()
        weight = torch.nn.functional.normalize(weight, p=1, dim=1)
        cosine = torch.einsum('id,jd->ij', [features, features])
        signal_cosine = torch.einsum('n,kn->k', [cosine.view(-1), weight]).view(signal_num, signal_num)
        return signal_cosine
