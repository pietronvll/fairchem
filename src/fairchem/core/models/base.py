"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import Optional
import logging

import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph

from fairchem.core.common.utils import (
    compute_neighbors,
    get_pbc_distances,
    radius_graph_pbc,
    radius_graph_pbc_func,
    segment_coo_patch,
    segment_csr_patch,
)


class BaseModel(nn.Module):
    def __init__(self, num_atoms=None, bond_feat_dim=None, num_targets=None) -> None:
        super().__init__()
        self.num_atoms = num_atoms
        self.bond_feat_dim = bond_feat_dim
        self.num_targets = num_targets

    def forward(self, data):
        raise NotImplementedError

    def generate_graph(
        self,
        data,
        cutoff=None,
        max_neighbors=None,
        use_pbc=None,
        otf_graph=None,
        enforce_max_neighbors_strictly=None,
    ):
        cutoff = cutoff or self.cutoff
        max_neighbors = max_neighbors or self.max_neighbors
        use_pbc = use_pbc or self.use_pbc
        otf_graph = otf_graph or self.otf_graph

        if enforce_max_neighbors_strictly is not None:
            pass
        elif hasattr(self, "enforce_max_neighbors_strictly"):
            # Not all models will have this attribute
            enforce_max_neighbors_strictly = self.enforce_max_neighbors_strictly
        else:
            # Default to old behavior
            enforce_max_neighbors_strictly = True

        if not otf_graph:
            try:
                edge_index = data.edge_index

                if use_pbc:
                    cell_offsets = data.cell_offsets
                    neighbors = data.neighbors

            except AttributeError:
                logging.warning(
                    "Turning otf_graph=True as required attributes not present in data object"
                )
                otf_graph = True

        if use_pbc:
            if otf_graph:
                edge_index, cell_offsets, neighbors = radius_graph_pbc(
                    data,
                    cutoff,
                    max_neighbors,
                    enforce_max_neighbors_strictly,
                )

            out = get_pbc_distances(
                data.pos,
                edge_index,
                data.cell,
                cell_offsets,
                neighbors,
                return_offsets=True,
                return_distance_vec=True,
            )

            edge_index = out["edge_index"]
            edge_dist = out["distances"]
            cell_offset_distances = out["offsets"]
            distance_vec = out["distance_vec"]
        else:
            if otf_graph:
                edge_index = radius_graph(
                    data.pos,
                    r=cutoff,
                    batch=data.batch,
                    max_num_neighbors=max_neighbors,
                )

            j, i = edge_index
            distance_vec = data.pos[j] - data.pos[i]

            edge_dist = distance_vec.norm(dim=-1)
            cell_offsets = torch.zeros(edge_index.shape[1], 3, device=data.pos.device)
            cell_offset_distances = torch.zeros_like(
                cell_offsets, device=data.pos.device
            )
            neighbors = compute_neighbors(data, edge_index)

        return (
            edge_index,
            edge_dist,
            distance_vec,
            cell_offsets,
            cell_offset_distances,
            neighbors,
        )

    def generate_graph_func(
        self,
        atom_pos,
        natoms,
        cell,
        batch_ids,
        data_pbc,
        cutoff=None,
        max_neighbors=None,
        use_pbc=None,
        otf_graph=None,
        enforce_max_neighbors_strictly=None,
        edge_index: Optional[torch.Tensor] = None,
        cell_offsets: Optional[torch.Tensor] = None,
        neighbors: Optional[torch.Tensor] = None
    ):
        cutoff = cutoff or self.cutoff
        max_neighbors = max_neighbors or self.max_neighbors
        use_pbc = use_pbc or self.use_pbc
        otf_graph = otf_graph or self.otf_graph

        if enforce_max_neighbors_strictly is not None:
            pass
        elif hasattr(self, "enforce_max_neighbors_strictly"):
            # Not all models will have this attribute
            enforce_max_neighbors_strictly = self.enforce_max_neighbors_strictly
        else:
            # Default to old behavior
            enforce_max_neighbors_strictly = True

        if not otf_graph:
            need_otf_graph = (
                edge_index is not None or (
                    (not use_pbc) and (
                        cell_offsets is not None or neighbors is not None
                    )
                )
            )
            if need_otf_graph:
                logging.warning(
                    "Turning otf_graph=True as required attributes not present in data object"
                )
                otf_graph = True

        if use_pbc:
            if otf_graph:
                edge_index, cell_offsets, neighbors = radius_graph_pbc_func(
                    atom_pos,
                    natoms,
                    cell,
                    cutoff,
                    max_neighbors,
                    data_pbc=data_pbc,
                    enforce_max_neighbors_strictly=enforce_max_neighbors_strictly,
                )

            out = get_pbc_distances(
                atom_pos,
                edge_index,
                cell,
                cell_offsets,
                neighbors,
                return_offsets=True,
                return_distance_vec=True,
            )

            edge_index = out["edge_index"]
            edge_dist = out["distances"]
            cell_offset_distances = out["offsets"]
            distance_vec = out["distance_vec"]
        else:
            if otf_graph:
                edge_index = radius_graph(
                    atom_pos,
                    r=cutoff,
                    batch=batch_ids,
                    max_num_neighbors=max_neighbors,
                )

            j, i = edge_index
            distance_vec = atom_pos[j] - atom_pos[i]

            edge_dist = distance_vec.norm(dim=-1)
            cell_offsets = torch.zeros(edge_index.shape[1], 3, device=atom_pos.device)
            cell_offset_distances = torch.zeros_like(
                cell_offsets, device=atom_pos.device
            )

            # Get number of neighbors
            # segment_coo assumes sorted index
            ones = edge_index[1].new_ones(1).expand_as(edge_index[1])
            num_neighbors = segment_coo_patch(
                ones, edge_index[1], dim_size=natoms.sum()
            )

            # Get number of neighbors per image
            image_indptr = torch.zeros(
                natoms.shape[0] + 1, device=atom_pos.device, dtype=torch.long
            )
            image_indptr[1:] = torch.cumsum(natoms, dim=0)
            neighbors = segment_csr_patch(num_neighbors, image_indptr)

        return (
            edge_index,
            edge_dist,
            distance_vec,
            cell_offsets,
            cell_offset_distances,
            neighbors,
        )

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.jit.ignore
    def no_weight_decay(self) -> list:
        """Returns a list of parameters with no weight decay."""
        no_wd_list = []
        for name, _ in self.named_parameters():
            if "embedding" in name or "frequencies" in name or "bias" in name:
                no_wd_list.append(name)
        return no_wd_list
