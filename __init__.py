from .krea2_regional import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .ideogram_bridge import (BRIDGE_NODE_CLASS_MAPPINGS,
                              BRIDGE_NODE_DISPLAY_NAME_MAPPINGS)
from .krea2_builder import (BUILDER_NODE_CLASS_MAPPINGS,
                            BUILDER_NODE_DISPLAY_NAME_MAPPINGS)

try:  # registers HTTP routes for the builder's LoRA info panel
    from . import server_routes  # noqa: F401
except Exception as _e:  # never block node loading on route registration
    import logging
    logging.warning("[Krea2Regional] server routes unavailable: %s", _e)

NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **BRIDGE_NODE_CLASS_MAPPINGS,
                       **BUILDER_NODE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS,
                              **BRIDGE_NODE_DISPLAY_NAME_MAPPINGS,
                              **BUILDER_NODE_DISPLAY_NAME_MAPPINGS}

WEB_DIRECTORY = "./web/js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS",
           "WEB_DIRECTORY"]
