default_scope = 'mmdet'
model = dict(
    backbone=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        arch='P5',
        channel_attention=True,
        deepen_factor=0.167,
        expand_ratio=0.5,
        init_cfg=None,
        norm_cfg=dict(type='BN'),
        norm_eval=True,
        type='CSPNeXt',
        widen_factor=0.375),
    bbox_head=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        anchor_generator=dict(
            offset=0, strides=[
                8,
                16,
                32,
            ], type='MlvlPointGenerator'),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        feat_channels=96,
        in_channels=96,
        loss_bbox=dict(loss_weight=2.0, type='GIoULoss'),
        loss_cls=dict(
            beta=2.0,
            loss_weight=2.0,
            type='QualityFocalLoss',
            use_sigmoid=True),
        loss_mask=dict(
            eps=5e-06, loss_weight=4.0, reduction='mean', type='DiceLoss'),
        mask_loss_stride=1,
        norm_cfg=dict(type='BN'),
        num_classes=1,
        pred_kernel_size=1,
        share_conv=True,
        stacked_convs=2,
        test_cfg=dict(
            mask_thr_binary=0.5,
            max_per_img=100,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.6, type='nms'),
            nms_pre=1000,
            score_thr=0.05),
        train_cfg=dict(
            allowed_border=-1,
            assigner=dict(topk=13, type='DynamicSoftLabelAssigner'),
            debug=False,
            pos_weight=-1),
        type='RTMDetInsSepBNHead'),
    data_preprocessor=dict(
        batch_augments=None,
        bgr_to_rgb=False,
        mean=[
            103.53,
            116.28,
            123.675,
        ],
        pad_mask=True,
        pad_size_divisor=32,
        std=[
            57.375,
            57.12,
            58.395,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        expand_ratio=0.5,
        in_channels=[
            96,
            192,
            384,
        ],
        norm_cfg=dict(type='BN'),
        num_csp_blocks=1,
        out_channels=96,
        type='CSPNeXtPAFPN'),
    test_cfg=dict(
        mask_thr_binary=0.5,
        max_per_img=100,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.6, type='nms'),
        nms_pre=1000,
        score_thr=0.05),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(topk=13, type='DynamicSoftLabelAssigner'),
        debug=False,
        pos_weight=-1),
    type='RTMDet')
