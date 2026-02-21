from UniDxMD.models.xmuda_arch import Net2DSeg, Net3DSeg, JointPrototypeLearning_add_relation
from UniDxMD.models.metric import SegIoU


def build_model_2d(cfg, codebook):
    model = Net2DSeg(num_classes=cfg.MODEL_2D.NUM_CLASSES,
                     backbone_2d=cfg.MODEL_2D.TYPE,
                     backbone_2d_kwargs=cfg.MODEL_2D[cfg.MODEL_2D.TYPE],
                     dual_head=cfg.MODEL_2D.DUAL_HEAD,
                     codebook=codebook,
                     sigma = cfg.MODEL_2D.sigma
                     )
    train_metric = SegIoU(cfg.MODEL_2D.NUM_CLASSES, name='seg_iou_2d')
    return model, train_metric


def build_model_3d(cfg, codebook):
    model = Net3DSeg(num_classes=cfg.MODEL_3D.NUM_CLASSES,
                     backbone_3d=cfg.MODEL_3D.TYPE,
                     backbone_3d_kwargs=cfg.MODEL_3D[cfg.MODEL_3D.TYPE],
                     dual_head=cfg.MODEL_3D.DUAL_HEAD,
                     codebook = codebook,
                     sigma = cfg.MODEL_3D.sigma,
                     use_color = cfg.MODEL_3D.USE_COLOR
                     )
    train_metric = SegIoU(cfg.MODEL_3D.NUM_CLASSES, name='seg_iou_3d')
    return model, train_metric

def build_model_JointPrototypeLearning_add_relation(cfg, codebook_dim):
    assert cfg.MODEL_2D.NUM_CLASSES == cfg.MODEL_3D.NUM_CLASSES
    model = JointPrototypeLearning_add_relation(num_classes=cfg.MODEL_3D.NUM_CLASSES, feature_dim=codebook_dim)
    train_metric = SegIoU(cfg.MODEL_3D.NUM_CLASSES, name='seg_iou_fus')
    return model, train_metric
