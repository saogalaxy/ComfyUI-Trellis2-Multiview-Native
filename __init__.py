"""ComfyUI-Trellis2-Multiview-Native: native-style multi-view conditioning for Trellis2."""

from .trellis2_multiview import comfy_entrypoint as _entry

comfy_entrypoint = _entry

__all__ = ["comfy_entrypoint"]
