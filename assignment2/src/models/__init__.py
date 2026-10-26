from .unet import UNet


def build_model(cfg):
    """Returns (model, output_type). output_type is 'metric' (depth in m) or 'relative' (inverse depth)."""
    if cfg["model"] == "unet":
        return UNet(), "metric"
    if cfg["model"] == "midas_small":
        from .midas import MiDaSSmall
        return MiDaSSmall(freeze_encoder=cfg.get("freeze_encoder", False)), "relative"
    raise ValueError(f"unknown model {cfg['model']}")