#----------------------------------------------------------------------------------------------------------------------------
# 关键信息屏蔽，仅做配置文件示例
data_root = "....."



#----------------------------------------------------------------------------------------------------------------------------
# LOAD DATASET
dataset_type = '...'
classes = ('....', '....', '....')

train_pipeline = [
    dict(type='LoadSignalFromFile', nrows=801, classes=classes),
    dict(type='ToTensor', keys=['dtoas', 'tags']),
    dict(type='Collect', keys=['dtoas', 'tags'], meta_keys=['signal_num', 'total_pulse_num', 'filename', 'duration', 'metas'])
]
val_pipeline = train_pipeline

test_pipeline = [
    dict(type='LoadMyDataFromFile', nrows=801, classes=classes),
    dict(type='ToTensor', keys=['dtoas']),
    dict(type='Collect', keys=['dtoas'], meta_keys=['total_pulse_num', 'filename'])
]

data = dict(
    workers_per_gpu=2,
    shuffle=True,
    drop_last=True,
    batch_sampler=dict(type='AlignSampler'),
    train=dict(
        type=dataset_type,
        data_dir=f"{data_root}/train",
        pipeline=train_pipeline,
        describe_file=f"{data_root}/train.json",
        samples_per_gpu=20,
        classes=classes,
    ),
    val=dict(
        type=dataset_type,
        data_dir=f"{data_root}/val",
        pipeline=val_pipeline,
        describe_file=f"{data_root}/val.json",
        samples_per_gpu=1,    # 序列不等长，evaluation，设置1跳过dataloader中collect_fn的对齐操作
        classes=classes,
    ))

#----------------------------------------------------------------------------------------------------------------------------
# MODEL
model = dict(
    type='SeparatorII',
    contrast_backbone=dict(
        type='BilstmEncoder', element_num=3002, embedding_dim=128, hidden_dim=128, num_layers=2, only_last_hidden=False),
    contrast_head=dict(
        type='TransformerHead',
        input_dim=256,
        norm_shape=[256],
        ffn_input_dim=256,
        ffn_hidden_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.5,
        use_bias=False),
    # contrast_loss=dict(type='MocoContrastLoss', dim=256, T=0.07, K=500),
    # use_moco=True,
    # moco_moment=0.99,
    contrast_loss=dict(type='BatchContrastLoss', T=0.1),
    use_moco=False,
    classify_backbone=dict(
        type='BilstmEncoder', element_num=3002, embedding_dim=128, hidden_dim=128, num_layers=1, only_last_hidden=True),
    classify_head=dict(type='DoubleLinearHead', input_dim=256, hidden_dim=128, output_dim=2, dropout=0.5),
    classify_loss=dict(type='CrossEntropyLoss'),
    cluster=dict(type='AgglomerativeCluster', max_cluster_num=4))

#----------------------------------------------------------------------------------------------------------------------------
# EVALUATION
evaluation = dict(
    start=40,
    interval=5,
    metric=['n_cluster_accuracy', 'need_separation_accuracy', 'avg_rand_score', 'signal_cosine'],
    save_best='avg_rand_score',
    greater_keys=['n_cluster_accuracy', 'need_separation_accuracy', 'avg_rand_score'])

#----------------------------------------------------------------------------------------------------------------------------
# SAVE AND LOAD
task_name = "SeparationII_Sta_mask"
checkpoint_config = dict(interval=1)
load_from = None
resume_from = None
work_dir = f"..../{task_name}"
#----------------------------------------------------------------------------------------------------------------------------
# LOG
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
    # dict(type='TensorboardLoggerHook')
    ])

#----------------------------------------------------------------------------------------------------------------------------
# CUSTOM HOOKS
# None

#----------------------------------------------------------------------------------------------------------------------------
# TRAIN POLICY
optimizer = dict(type='Adam', lr=0.001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[30, 250], warmup='linear', warmup_iters=200, warmup_ratio=0.05)
fp16 = dict(loss_scale=512.)
workflow = [('train', 1)]
runner = dict(type='EpochBasedRunner', max_epochs=200)
