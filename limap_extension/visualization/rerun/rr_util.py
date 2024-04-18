from typing import TYPE_CHECKING, Optional, Union
from warnings import warn

import rerun as rr
import torch
import numpy as np


def prep_data(arr: Optional[Union[torch.Tensor, 'np.ndarray']]) -> 'np.ndarray':
    if arr is None:
        return
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    if (len(arr.shape) > 1) and (arr.shape[0] in [2, 3]) and (arr.shape[1] > 3):
        warn("Detected array given to rerun that is likely in the wrong shape. Transposing.")
        arr = arr.T
    return arr


def log_def_obj_edges(name: str, verts: Union[torch.Tensor, np.ndarray],
                      edges: Union[torch.Tensor, np.ndarray], colors: Optional[Union[torch.Tensor,
                                                                                     np.ndarray]],
                      radii: Optional[Union[float, 'np.ndarray']]) -> None:
    verts = prep_data(verts)
    edges = prep_data(edges)
    colors = prep_data(colors)

    # Use fancy indexing here to get the start and end vertices for each edge
    pt_start_idxs = edges[:, 0]
    pt_end_idxs = edges[:, 1]
    pts_start = verts[pt_start_idxs, :]
    pts_end = verts[pt_end_idxs, :]
    edge_strips = np.stack((pts_start, pts_end), axis=0)

    if radii is not None:
        radii = radii * 0.5

    rr.log(f"{name}/edges", rr.LineStrips3D(edge_strips, radii=radii, colors=colors))


def log_def_obj_config(name: str,
                       verts: torch.Tensor,
                       edges: torch.Tensor,
                       colors: Optional['np.ndarray'] = None,
                       radii: Optional[Union[float, 'np.ndarray']] = None) -> None:
    verts = prep_data(verts)
    edges = prep_data(edges)
    colors = prep_data(colors)
    log_def_obj_edges(name, verts, edges, colors, radii)
    rr.log(f"{name}/vertices", rr.Points3D(verts, radii=radii, colors=colors))
