import importlib

if "bpy" in locals():
    from . import core, operators, scene
    from .core import blend as core_blend
    from .core import infill as core_infill
    from .scene import bake as scene_bake
    from .scene import images as scene_images
    from .scene import progress_overlay as scene_progress_overlay

    importlib.reload(core_infill)
    importlib.reload(core_blend)
    importlib.reload(core)
    importlib.reload(scene_images)
    importlib.reload(scene_progress_overlay)
    importlib.reload(scene_bake)
    importlib.reload(scene)
    importlib.reload(operators)

bl_info = {
    "name": "Mask Luminance",
    "author": "Genies",
    "version": (0, 1, 0),
    "blender": (5, 1, 0),
    "location": "Image Editor > Sidebar > Mask Luminance",
    "description": "Layered mask-channel luminance correction for texture albedos",
    "category": "Paint",
}

try:
    import bpy
except ImportError:
    bpy = None  # type: ignore[misc, assignment]

if bpy is not None:
    from . import core, operators, scene

    classes = operators.CLASSES

    def register():
        try:
            unregister()
        except Exception:
            pass

        for cls in classes:
            bpy.utils.register_class(cls)
        bpy.types.Scene.mask_luminance = bpy.props.PointerProperty(type=operators.MASKLUM_PG_settings)
        bpy.types.WindowManager.mask_luminance_bake_progress = bpy.props.PointerProperty(
            type=operators.MASKLUM_PG_BakeProgress
        )

    def unregister():
        del bpy.types.WindowManager.mask_luminance_bake_progress
        del bpy.types.Scene.mask_luminance
        for cls in reversed(classes):
            bpy.utils.unregister_class(cls)
