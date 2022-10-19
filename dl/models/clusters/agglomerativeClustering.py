from ..builder import CLUSTERS
from sklearn.cluster import AgglomerativeClustering
from torch import nn
from sklearn.metrics import silhouette_score
import numpy as np
from functools import partial


@CLUSTERS.register_module()
class AgglomerativeCluster(nn.Module):
    """基于numpy分层聚类封装"""

    def __init__(
        self,
        affinity="cosine",
        linkage="average",
        max_cluster_num=4,
    ):
        super().__init__()
        self.max_cluster_num = max_cluster_num
        self.cluster_factory = partial(AgglomerativeClustering, affinity=affinity, linkage=linkage)

    def tensor_to_array(self, x):
        """tesnor to numpy array"""
        return x.detach().cpu().numpy()

    def fit_predict(self, pulse_features, need_separation):
        """
        1.分类器预测need_separation为0（不需要分选），分类器预测need_separation为1（需要分选）
        2.采用对特征向量聚类预测方式进行分选，聚类中心依次选择[2,max_n_clusters],基于轮廓系数确定最佳聚类器；
        3.输出聚类预测结果和聚类中心数
        """
        if need_separation:
            max_silhouette_score = float("-inf")
            for n_clusters in range(2, self.max_cluster_num + 1):
                cluster = self.cluster_factory(n_clusters=n_clusters)
                cluster.fit(pulse_features)
                score = silhouette_score(pulse_features, cluster.labels_)    # 轮廓系数
                if score > max_silhouette_score:
                    max_silhouette_score = score
                    best_n_cluster = n_clusters
                    best_cluster_labels = cluster.labels_
            return dict(labels=best_cluster_labels, n_cluster=best_n_cluster, need_separation=1)
        else:
            return dict(labels=np.zeros(len(pulse_features)), n_cluster=1, need_separation=0)

    def forward(self, signal_features, need_separations):
        """返回期望是list[dict,dict]格式"""
        assert len(signal_features) == len(need_separations)
        signal_features = self.tensor_to_array(signal_features)
        need_separations = self.tensor_to_array(need_separations)
        results = list(map(self.fit_predict, signal_features, need_separations))
        return results