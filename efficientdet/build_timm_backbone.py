import timm
from detectron2.modeling import BACKBONE_REGISTRY, Backbone
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool

@BACKBONE_REGISTRY.register()
def build_timm_backbone(cfg, input_shape):
    model_name = cfg.MODEL.TIMM.NAME
    pretrained = getattr(cfg.MODEL.TIMM, "PRETRAINED", True)
    out_indices = tuple(getattr(cfg.MODEL.TIMM, "OUT_INDICES", (1, 2, 3, 4)))
    timm_model = timm.create_model(model_name, pretrained=pretrained, features_only=True, out_indices=out_indices)

    class _TimmBottomUp(Backbone):
        def __init__(self, timm_backbone):
            super().__init__()
            self.body = timm_backbone
            info = self.body.feature_info
            channels = info.channels()
            reductions = info.reduction()
            self._feature_names = [f"res{2 + i}" for i in range(len(channels))]
            self._out_feature_channels = {n: channels[i] for i, n in enumerate(self._feature_names)}
            self._out_feature_strides = {n: reductions[i] for i, n in enumerate(self._feature_names)}

        def forward(self, x):
            feats = self.body(x)
            out = {}
            for i, name in enumerate(self._feature_names):
                if i < len(feats):
                    out[name] = feats[i]
            return out

        def output_shape(self):
            return {
                name: ShapeSpec(channels=self._out_feature_channels[name], stride=self._out_feature_strides[name])
                for name in self._feature_names
            }

    bottom_up = _TimmBottomUp(timm_model)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = getattr(cfg.MODEL.FPN, "OUT_CHANNELS", 256)
    fpn = FPN(bottom_up=bottom_up, in_features=in_features, out_channels=out_channels, top_block=LastLevelMaxPool(), fuse_type="sum")
    return fpn

if __name__ == "__main__":
    print("module build_timm_backbone loaded (not intended to be executed standalone).")
