"""Model exports for the AF3 baseline and adapted audio backbones."""

from importlib import import_module

_EXPORTS = {
    "SpatialAF3GenerativeModel": (".spatial_generative", "SpatialAF3GenerativeModel"),
    "SpatialAFNextGenerativeModel": (".spatial_afnext_generative", "SpatialAFNextGenerativeModel"),
    "AFNextTemporalConcatFusion": (".spatial_afnext_generative", "AFNextTemporalConcatFusion"),
    "FrozenAFNextGenerativeBackbone": (".afnext_generative_backbone", "FrozenAFNextGenerativeBackbone"),
    "SpatialQwen3OmniGenerativeModel": (".spatial_qwen3_omni_generative", "SpatialQwen3OmniGenerativeModel"),
    "Qwen3OmniTemporalConcatFusion": (".spatial_qwen3_omni_generative", "Qwen3OmniTemporalConcatFusion"),
    "FrozenQwen3OmniGenerativeBackbone": (".qwen3_omni_generative_backbone", "FrozenQwen3OmniGenerativeBackbone"),
    "FrozenMiDashengLMGenerativeBackbone": (".midashenglm_generative_backbone", "FrozenMiDashengLMGenerativeBackbone"),
    "SpatialAudioProjectorSubsample": (".midashenglm_spatial_projector", "SpatialAudioProjectorSubsample"),
    "SpatialMiDashengLMGenerativeBackbone": (".midashenglm_spatial_generative_backbone", "SpatialMiDashengLMGenerativeBackbone"),
    "SpatialMiDashengLMTrainingModel": (".midashenglm_spatial_training", "SpatialMiDashengLMTrainingModel"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
