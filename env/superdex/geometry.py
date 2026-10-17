"""Small deterministic geometry helpers used by SuperDex observations."""
from __future__ import annotations

import numpy as np


def surface_points(mesh, count: int = 200) -> np.ndarray:
    """Return deterministic area-weighted samples from a triangle mesh."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    area = np.asarray(mesh.area_faces, dtype=np.float64)
    rng = np.random.default_rng(0)
    face_ids = rng.choice(len(faces), size=count, p=area / area.sum())
    uv = rng.random((count, 2))
    fold = uv.sum(axis=1) > 1.0
    uv[fold] = 1.0 - uv[fold]
    chosen = triangles[face_ids]
    return (
        chosen[:, 0]
        + uv[:, :1] * (chosen[:, 1] - chosen[:, 0])
        + uv[:, 1:] * (chosen[:, 2] - chosen[:, 0])
    ).astype(np.float32)


def nearest_vectors(points: np.ndarray, surface: np.ndarray) -> np.ndarray:
    """Return the vector from each point to its nearest surface sample."""
    delta = surface[None, :, :] - points[:, None, :]
    nearest = np.argmin(np.einsum("ijk,ijk->ij", delta, delta), axis=1)
    return delta[np.arange(points.shape[0]), nearest]
