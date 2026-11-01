"""Load a retained mesh with an optional UV-identical material through trimesh."""
import numpy as np
import trimesh


def load_paint_base(geometry_path, material_path=None):
    scene = trimesh.load(geometry_path, force="scene", process=False)
    if len(scene.geometry) != 1:
        raise ValueError("Painting requires one UV-mapped base primitive")
    mesh = next(iter(scene.geometry.values()))
    if material_path is not None:
        material_scene = trimesh.load(material_path, force="scene", process=False)
        if len(material_scene.geometry) != 1:
            raise ValueError("Painting material requires one UV-mapped primitive")
        source = next(iter(material_scene.geometry.values()))
        if not (np.array_equal(mesh.vertices, source.vertices)
                and np.array_equal(mesh.faces, source.faces)
                and np.allclose(mesh.visual.uv, source.visual.uv, rtol=0, atol=1e-7)):
            raise ValueError("Material source must match the retained positions and indices exactly, and UVs within float32 round-trip precision")
        mesh.visual.material = source.visual.material
    if mesh.visual.material.baseColorTexture is None:
        raise ValueError("Painting base must retain a colour atlas for uncovered texels")
    return mesh
