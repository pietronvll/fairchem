"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import torch
from torch_geometric.nn import SchNet
from torch_scatter import scatter

from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad
from fairchem.core.models.base import BaseModel


@registry.register_model("schnet")
class SchNetWrap(SchNet, BaseModel):
    r"""Wrapper around the continuous-filter convolutional neural network SchNet from the
    `"SchNet: A Continuous-filter Convolutional Neural Network for Modeling
    Quantum Interactions" <https://arxiv.org/abs/1706.08566>`_. Each layer uses interaction
    block of the form:

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \odot
        h_{\mathbf{\Theta}} ( \exp(-\gamma(\mathbf{e}_{j,i} - \mathbf{\mu}))),

    Args:
        num_atoms (int): Unused argument
        bond_feat_dim (int): Unused argument
        num_targets (int): Number of targets to predict.
        use_pbc (bool, optional): If set to :obj:`True`, account for periodic boundary conditions.
            (default: :obj:`True`)
        regress_forces (bool, optional): If set to :obj:`True`, predict forces by differentiating
            energy with respect to positions.
            (default: :obj:`True`)
        otf_graph (bool, optional): If set to :obj:`True`, compute graph edges on the fly.
            (default: :obj:`False`)
        hidden_channels (int, optional): Number of hidden channels.
            (default: :obj:`128`)
        num_filters (int, optional): Number of filters to use.
            (default: :obj:`128`)
        num_interactions (int, optional): Number of interaction blocks
            (default: :obj:`6`)
        num_gaussians (int, optional): The number of gaussians :math:`\mu`.
            (default: :obj:`50`)
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        readout (string, optional): Whether to apply :obj:`"add"` or
            :obj:`"mean"` global aggregation. (default: :obj:`"add"`)
    """

    def __init__(
        self,
        num_atoms: int,  # not used
        bond_feat_dim: int,  # not used
        num_targets: int,
        use_pbc: bool = True,
        regress_forces: bool = True,
        otf_graph: bool = False,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        readout: str = "add",
    ) -> None:
        self.num_targets = num_targets
        self.regress_forces = regress_forces
        self.use_pbc = use_pbc
        self.cutoff = cutoff
        self.otf_graph = otf_graph
        self.max_neighbors = 50
        self.reduce = readout
        super().__init__(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            readout=readout,
        )

    @conditional_grad(torch.enable_grad())
    def _forward(self, data):
        z = data.atomic_numbers.long()
        pos = data.pos
        batch = data.batch

        (
            edge_index,
            edge_weight,
            distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(data)

        if self.use_pbc:
            assert z.dim() == 1
            assert z.dtype == torch.long

            edge_attr = self.distance_expansion(edge_weight)

            h = self.embedding(z)
            for interaction in self.interactions:
                h = h + interaction(h, edge_index, edge_weight, edge_attr)

            h = self.lin1(h)
            h = self.act(h)
            h = self.lin2(h)

            batch = torch.zeros_like(z) if batch is None else batch
            energy = scatter(h, batch, dim=0, reduce=self.reduce)
        else:
            energy = super().forward(z, pos, batch)
        return energy

    def forward(self, data):
        if self.regress_forces:
            data.pos.requires_grad_(True)
        energy = self._forward(data)
        outputs = {"energy": energy}

        if self.regress_forces:
            forces = -1 * (
                torch.autograd.grad(
                    energy,
                    data.pos,
                    grad_outputs=torch.ones_like(energy),
                    create_graph=True,
                )[0]
            )
            outputs["forces"] = forces

        return outputs

    def descriptors(
        self,
        atom_pos,
        natoms,
        cell,
        batch_ids,
        data_pbc,
        atomic_numbers,
        interaction_block,
        edge_index=None,
        cell_offsets=None,
        neighbors=None,
        edge_weight=None
    ):
        """
        Forward pass for the SchNet model to get the embedded representations of the input data

        """
        # Get the atomic numbers of the input data
        z = atomic_numbers.long()
        # Get the edge index, edge weight and other attributes of the input data
        if edge_index is None or edge_weight is None:
            (
                edge_index,
                edge_weight,
                _,
                _,
                _,
                _,
            ) = self.generate_graph_func(
                atom_pos, natoms, cell, batch_ids, data_pbc,
                edge_index=edge_index, cell_offsets=cell_offsets, neighbors=neighbors
            )

        assert z.dim() == 1 and z.dtype == torch.long

        edge_attr = self.distance_expansion(edge_weight)

        # Get the embedded representations of the input data
        h = self.embedding(z)
        for interaction in self.interactions[:interaction_block]:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        return h

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
