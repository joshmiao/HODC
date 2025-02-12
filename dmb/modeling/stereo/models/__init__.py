from . import HODC_PSMNet


_META_ARCHITECTURES = {
    "HODC": HODC_PSMNet.HODC_PSMNet,
   
}


def build_stereo_model(cfg):
    meta_arch = _META_ARCHITECTURES[cfg.model.meta_architecture]
    return meta_arch(cfg)
