import abc
from pathlib import Path
from collections import defaultdict, OrderedDict
from typing import Optional, Union, List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.spatial
import random
import pickle
from copy import deepcopy


def sparse_dropout(sp_mat: torch.Tensor, p: float, fill_value: float = 0.0) -> torch.Tensor:
    r"""Dropout function for sparse matrix. This function will return a new sparse matrix with the same shape as the input sparse matrix, but with some elements dropped out.
    
    Args:
        ``sp_mat`` (``torch.Tensor``): The sparse matrix with format ``torch.sparse_coo_tensor``.
        ``p`` (``float``): Probability of an element to be dropped. 
        ``fill_value`` (``float``): The fill value for dropped elements. Defaults to ``0.0``.
    """
    device = sp_mat.device
    sp_mat = sp_mat.coalesce()
    assert 0 <= p <= 1
    if p == 0:
        return sp_mat
    p = torch.ones(sp_mat._nnz(), device=device) * p
    keep_mask = torch.bernoulli(1 - p).to(device)
    fill_values = torch.logical_not(keep_mask) * fill_value
    new_sp_mat = torch.sparse_coo_tensor(
        sp_mat._indices(),
        sp_mat._values() * keep_mask + fill_values,
        size=sp_mat.size(),
        device=sp_mat.device,
        dtype=sp_mat.dtype,
    )
    return new_sp_mat



class BaseHypergraph:
    r"""The ``BaseHypergraph`` class is the base class for all hypergraph structures.

    Args:
        ``num_v`` (``int``): The number of vertices in the hypergraph.
        ``e_list_v2e`` (``Union[List[int], List[List[int]]]``, optional): A list of hyperedges describes how the vertices point to the hyperedges. Defaults to ``None``.
        ``e_list_e2v`` (``Union[List[int], List[List[int]]]``, optional): A list of hyperedges describes how the hyperedges point to the vertices. Defaults to ``None``.
        ``w_list_v2e`` (``Union[List[float], List[List[float]]]``, optional): The weights are attached to the connections from vertices to hyperedges, which has the same shape
            as ``e_list_v2e``. If set to ``None``, the value ``1`` is used for all connections. Defaults to ``None``.
        ``w_list_e2v`` (``Union[List[float], List[List[float]]]``, optional): The weights are attached to the connections from the hyperedges to the vertices, which has the
            same shape to ``e_list_e2v``. If set to ``None``, the value ``1`` is used for all connections. Defaults to ``None``.
        ``e_weight`` (``Union[float, List[float]]``, optional): A list of weights for hyperedges. If set to ``None``, the value ``1`` is used for all hyperedges. Defaults to ``None``.
        ``v_weight`` (``Union[float, List[float]]``, optional): Weights for vertices. If set to ``None``, the value ``1`` is used for all vertices. Defaults to ``None``.
        ``device`` (``torch.device``, optional): The deivce to store the hypergraph. Defaults to ``torch.device('cpu')``.
    """

    def __init__(
        self,
        num_v: int,
        e_list_v2e: Optional[Union[List[int], List[List[int]]]] = None,
        e_list_e2v: Optional[Union[List[int], List[List[int]]]] = None,
        w_list_v2e: Optional[Union[List[float], List[List[float]]]] = None,
        w_list_e2v: Optional[Union[List[float], List[List[float]]]] = None,
        e_weight: Optional[Union[float, List[float]]] = None,
        v_weight: Optional[List[float]] = None,
        device: torch.device = torch.device("cpu"),
    ):
        assert isinstance(num_v, int) and num_v > 0, "num_v should be a positive integer"
        self.clear()
        self._num_v = num_v
        self.device = device

    @abc.abstractmethod
    def __repr__(self) -> str:
        r"""Print the hypergraph information.
        """

    @property
    @abc.abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        r"""Get the state dict of the hypergraph.
        """

    @abc.abstractmethod
    def save(self, file_path: Union[str, Path]):
        r"""Save the DHG's hypergraph structure to a file.

        Args:
            ``file_path`` (``str``): The file_path to store the DHG's hypergraph structure.
        """

    @abc.abstractstaticmethod
    def load(file_path: Union[str, Path]):
        r"""Load the DHG's hypergraph structure from a file.

        Args:
            ``file_path`` (``str``): The file path to load the DHG's hypergraph structure.
        """

    @abc.abstractstaticmethod
    def from_state_dict(state_dict: dict):
        r"""Load the DHG's hypergraph structure from the state dict.

        Args:
            ``state_dict`` (``dict``): The state dict to load the DHG's hypergraph.
        """

    def clear(self):
        r"""Remove all hyperedges and caches from the hypergraph.
        """
        self._clear_raw()
        self._clear_cache()

    def _clear_raw(self):
        self._v_weight = None
        self._raw_groups = {}

    def _clear_cache(self, group_name: Optional[str] = None):
        self.cache = {}
        if group_name is None:
            self.group_cache = defaultdict(dict)
        else:
            self.group_cache.pop(group_name, None)

    @abc.abstractmethod
    def clone(self) -> "BaseHypergraph":
        r"""Return a copy of this type of hypergraph.
        """

    def to(self, device: torch.device):
        r"""Move the hypergraph to the specified device.

        Args:
            ``device`` (``torch.device``): The device to store the hypergraph.
        """
        self.device = device
        for v in self.vars_for_DL:
            if v in self.cache and self.cache[v] is not None:
                self.cache[v] = self.cache[v].to(device)
            for name in self.group_names:
                if v in self.group_cache[name] and self.group_cache[name][v] is not None:
                    self.group_cache[name][v] = self.group_cache[name][v].to(device)
        return self

    # utils
    def _hyperedge_code(self, src_v_set: List[int], dst_v_set: List[int]) -> Tuple:
        r"""Generate the hyperedge code.

        Args:
            ``src_v_set`` (``List[int]``): The source vertex set.
            ``dst_v_set`` (``List[int]``): The destination vertex set.
        """
        return tuple([src_v_set, dst_v_set])

    def _merge_hyperedges(self, e1: dict, e2: dict, op: str = "mean"):
        assert op in ["mean", "sum", "max",], "Hyperedge merge operation must be one of ['mean', 'sum', 'max']"
        _func = {
            "mean": lambda x, y: (x + y) / 2,
            "sum": lambda x, y: x + y,
            "max": lambda x, y: max(x, y),
        }
        _e = {}
        if "w_v2e" in e1 and "w_v2e" in e2:
            for _idx in range(len(e1["w_v2e"])):
                _e["w_v2e"] = _func[op](e1["w_v2e"][_idx], e2["w_v2e"][_idx])
        if "w_e2v" in e1 and "w_e2v" in e2:
            for _idx in range(len(e1["w_e2v"])):
                _e["w_e2v"] = _func[op](e1["w_e2v"][_idx], e2["w_e2v"][_idx])
        _e["w_e"] = _func[op](e1["w_e"], e2["w_e"])
        return _e

    @staticmethod
    def _format_e_list(e_list: Union[List[int], List[List[int]]]) -> List[List[int]]:
        r"""Format the hyperedge list.

        Args:
            ``e_list`` (``List[int]`` or ``List[List[int]]``): The hyperedge list.
        """
        if type(e_list[0]) in (int, float):
            return [tuple(sorted(e_list))]
        elif type(e_list) == tuple:
            e_list = list(e_list)
        elif type(e_list) == list:
            pass
        else:
            raise TypeError("e_list must be List[int] or List[List[int]].")
        for _idx in range(len(e_list)):
            e_list[_idx] = tuple(sorted(e_list[_idx]))
        return e_list

    @staticmethod
    def _format_e_list_and_w_on_them(
        e_list: Union[List[int], List[List[int]]], w_list: Optional[Union[List[int], List[List[int]]]] = None,
    ):
        r"""Format ``e_list`` and ``w_list``.

        Args:
            ``e_list`` (Union[List[int], List[List[int]]]): Hyperedge list.
            ``w_list`` (Optional[Union[List[int], List[List[int]]]]): Weights on connections. Defaults to ``None``.
        """
        bad_connection_msg = (
            "The weight on connections between vertices and hyperedges must have the same size as the hyperedges."
        )
        if isinstance(e_list, tuple):
            e_list = list(e_list)
        if w_list is not None and isinstance(w_list, tuple):
            w_list = list(w_list)
        if isinstance(e_list[0], int) and w_list is None:
            w_list = [1] * len(e_list)
            e_list, w_list = [e_list], [w_list]
        elif isinstance(e_list[0], int) and w_list is not None:
            assert len(e_list) == len(w_list), bad_connection_msg
            e_list, w_list = [e_list], [w_list]
        elif isinstance(e_list[0], list) and w_list is None:
            w_list = [[1] * len(e) for e in e_list]
        assert len(e_list) == len(w_list), bad_connection_msg
        # TODO: this step can be speeded up
        for idx in range(len(e_list)):
            assert len(e_list[idx]) == len(w_list[idx]), bad_connection_msg
            cur_e, cur_w = np.array(e_list[idx]), np.array(w_list[idx])
            sorted_idx = np.argsort(cur_e)
            e_list[idx] = tuple(cur_e[sorted_idx].tolist())
            w_list[idx] = cur_w[sorted_idx].tolist()
        return e_list, w_list

    def _fetch_H_of_group(self, direction: str, group_name: str):
        r"""Fetch the H matrix of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Args:
            ``direction`` (``str``): The direction of hyperedges can be either ``'v2e'`` or ``'e2v'``.
            ``group_name`` (``str``): The name of the group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        assert direction in ["v2e", "e2v"], "direction must be one of ['v2e', 'e2v']"
        if direction == "v2e":
            select_idx = 0
        else:
            select_idx = 1
        num_e = len(self._raw_groups[group_name])
        e_idx, v_idx = [], []
        for _e_idx, e in enumerate(self._raw_groups[group_name].keys()):
            sub_e = e[select_idx]
            v_idx.extend(sub_e)
            e_idx.extend([_e_idx] * len(sub_e))
        H = torch.sparse_coo_tensor(
            torch.tensor([v_idx, e_idx], dtype=torch.long),
            torch.ones(len(v_idx)),
            torch.Size([self.num_v, num_e]),
            device=self.device,
        ).coalesce().to(self.device)
        return H

    def _fetch_R_of_group(self, direction: str, group_name: str):
        r"""Fetch the R matrix of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Args:
            ``direction`` (``str``): The direction of hyperedges can be either ``'v2e'`` or ``'e2v'``.
            ``group_name`` (``str``): The name of the group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        assert direction in ["v2e", "e2v"], "direction must be one of ['v2e', 'e2v']"
        if direction == "v2e":
            select_idx = 0
        else:
            select_idx = 1
        num_e = len(self._raw_groups[group_name])
        e_idx, v_idx, w_list = [], [], []
        for _e_idx, e in enumerate(self._raw_groups[group_name].keys()):
            sub_e = e[select_idx]
            v_idx.extend(sub_e)
            e_idx.extend([_e_idx] * len(sub_e))
            w_list.extend(self._raw_groups[group_name][e][f"w_{direction}"])
        R = torch.sparse_coo_tensor(
            torch.vstack([v_idx, e_idx]), torch.tensor(w_list), torch.Size([self.num_v, num_e]), device=self.device,
        ).coalesce().to(self.device)
        return R

    def _fetch_W_of_group(self, group_name: str):
        r"""Fetch the W matrix of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Args:
            ``group_name`` (``str``): The name of the group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        w_list = [content["w_e"] for content in self._raw_groups[group_name].values()]
        W = torch.tensor(w_list, device=self.device).view((-1, 1)).to(self.device)
        return W

    # some structure modification functions
    def add_hyperedges(
        self,
        e_list_v2e: Union[List[int], List[List[int]]],
        e_list_e2v: Union[List[int], List[List[int]]],
        w_list_v2e: Optional[Union[List[float], List[List[float]]]] = None,
        w_list_e2v: Optional[Union[List[float], List[List[float]]]] = None,
        e_weight: Optional[Union[float, List[float]]] = None,
        merge_op: str = "mean",
        group_name: str = "main",
    ):
        r"""Add hyperedges to the hypergraph. If the ``group_name`` is not specified, the hyperedges will be added to the default ``main`` hyperedge group.

        Args:
            ``num_v`` (``int``): The number of vertices in the hypergraph.
            ``e_list_v2e`` (``Union[List[int], List[List[int]]]``): A list of hyperedges describes how the vertices point to the hyperedges.
            ``e_list_e2v`` (``Union[List[int], List[List[int]]]``): A list of hyperedges describes how the hyperedges point to the vertices.
            ``w_list_v2e`` (``Union[List[float], List[List[float]]]``, optional): The weights are attached to the connections from vertices to hyperedges, which has the same shape
                as ``e_list_v2e``. If set to ``None``, the value ``1`` is used for all connections. Defaults to ``None``.
            ``w_list_e2v`` (``Union[List[float], List[List[float]]]``, optional): The weights are attached to the connections from the hyperedges to the vertices, which has the
                same shape to ``e_list_e2v``. If set to ``None``, the value ``1`` is used for all connections. Defaults to ``None``.
            ``e_weight`` (``Union[float, List[float]]``, optional): A list of weights for hyperedges. If set to ``None``, the value ``1`` is used for all hyperedges. Defaults to ``None``.
            ``merge_op`` (``str``): The merge operation for the conflicting hyperedges. The possible values are ``mean``, ``sum``, ``max``, and ``min``. Defaults to ``mean``.
            ``group_name`` (``str``, optional): The target hyperedge group to add these hyperedges. Defaults to the ``main`` hyperedge group.
        """
        e_list_v2e, w_list_v2e = self._format_e_list_and_w_on_them(e_list_v2e, w_list_v2e)
        e_list_e2v, w_list_e2v = self._format_e_list_and_w_on_them(e_list_e2v, w_list_e2v)
        if e_weight is None:
            e_weight = [1.0] * len(e_list_v2e)
        assert len(e_list_v2e) == len(e_weight), "The number of hyperedges and the number of weights are not equal."
        assert len(e_list_v2e) == len(e_list_e2v), "Hyperedges of 'v2e' and 'e2v' must have the same size."
        for _idx in range(len(e_list_v2e)):
            self._add_hyperedge(
                self._hyperedge_code(e_list_v2e[_idx], e_list_e2v[_idx]),
                {"w_v2e": w_list_v2e[_idx], "w_e2v": w_list_e2v[_idx], "w_e": e_weight[_idx],},
                merge_op,
                group_name,
            )
        self._clear_cache(group_name)

    def _add_hyperedge(
        self, hyperedge_code: Tuple[List[int], List[int]], content: Dict[str, Any], merge_op: str, group_name: str,
    ):
        r"""Add a hyperedge to the specified hyperedge group.

        Args:
            ``hyperedge_code`` (``Tuple[List[int], List[int]]``): The hyperedge code.
            ``content`` (``Dict[str, Any]``): The content of the hyperedge.
            ``merge_op`` (``str``): The merge operation for the conflicting hyperedges.
            ``group_name`` (``str``): The target hyperedge group to add this hyperedge.
        """
        if group_name not in self.group_names:
            self._raw_groups[group_name] = {}
            self._raw_groups[group_name][hyperedge_code] = content
        else:
            if hyperedge_code not in self._raw_groups[group_name]:
                self._raw_groups[group_name][hyperedge_code] = content
            else:
                self._raw_groups[group_name][hyperedge_code] = self._merge_hyperedges(
                    self._raw_groups[group_name][hyperedge_code], content, merge_op
                )

    def remove_hyperedges(
        self,
        e_list_v2e: Union[List[int], List[List[int]]],
        e_list_e2v: Union[List[int], List[List[int]]],
        group_name: Optional[str] = None,
    ):
        r"""Remove the specified hyperedges from the hypergraph.

        Args:
            ``e_list_v2e`` (``Union[List[int], List[List[int]]]``): A list of hyperedges describes how the vertices point to the hyperedges.
            ``e_list_e2v`` (``Union[List[int], List[List[int]]]``): A list of hyperedges describes how the hyperedges point to the vertices.
            ``group_name`` (``str``, optional): Remove these hyperedges from the specified hyperedge group. If not specified, the function will
                remove those hyperedges from all hyperedge groups. Defaults to the ``None``.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        assert len(e_list_v2e) == len(e_list_e2v), "Hyperedges of 'v2e' and 'e2v' must have the same size."
        e_list_v2e = self._format_e_list(e_list_v2e)
        e_list_e2v = self._format_e_list(e_list_e2v)
        if group_name is None:
            for _idx in range(len(e_list_v2e)):
                e_code = self._hyperedge_code(e_list_v2e[_idx], e_list_e2v[_idx])
                for name in self.group_names:
                    self._raw_groups[name].pop(e_code, None)
        else:
            for _idx in range(len(e_list_v2e)):
                e_code = self._hyperedge_code(e_list_v2e[_idx], e_list_e2v[_idx])
                self._raw_groups[group_name].pop(e_code, None)
        self._clear_cache(group_name)

    @abc.abstractmethod
    def drop_hyperedges(self, drop_rate: float, ord="uniform"):
        r"""Randomly drop hyperedges from the hypergraph. This function will return a new hypergraph with non-dropped hyperedges.

        Args:
            ``drop_rate`` (``float``): The drop rate of hyperedges.
            ``ord`` (``str``): The order of dropping edges. Currently, only ``'uniform'`` is supported. Defaults to ``uniform``.
        """

    @abc.abstractmethod
    def drop_hyperedges_of_group(self, group_name: str, drop_rate: float, ord="uniform"):
        r"""Randomly drop hyperedges from the specified hyperedge group. This function will return a new hypergraph with non-dropped hyperedges.

        Args:
            ``group_name`` (``str``): The name of the hyperedge group.
            ``drop_rate`` (``float``): The drop rate of hyperedges.
            ``ord`` (``str``): The order of dropping edges. Currently, only ``'uniform'`` is supported. Defaults to ``uniform``.
        """

    # properties for the hypergraph
    @property
    def v(self) -> List[int]:
        r"""Return the list of vertices.
        """
        if self.cache.get("v") is None:
            self.cache["v"] = list(range(self.num_v))
        return self.cache["v"]

    @property
    def v_weight(self) -> List[float]:
        r"""Return the vertex weights of the hypergraph.
        """
        if self._v_weight is None:
            self._v_weight = [1.0] * self.num_v
        return self._v_weight

    @v_weight.setter
    def v_weight(self, v_weight: List[float]):
        r"""Set the vertex weights of the hypergraph.
        """
        assert len(v_weight) == self.num_v, "The length of vertex weights must be equal to the number of vertices."
        self._v_weight = v_weight
        self._clear_cache()

    @property
    @abc.abstractmethod
    def e(self) -> Tuple[List[List[int]], List[float]]:
        r"""Return all hyperedges and weights in the hypergraph.
        """

    @abc.abstractmethod
    def e_of_group(self, group_name: str) -> Tuple[List[List[int]], List[float]]:
        r"""Return all hyperedges and weights in the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """

    @property
    def num_v(self) -> int:
        r"""Return the number of vertices in the hypergraph.
        """
        return self._num_v

    @property
    def num_e(self) -> int:
        r"""Return the number of hyperedges in the hypergraph.
        """
        _num_e = 0
        for name in self.group_names:
            _num_e += len(self._raw_groups[name])
        return _num_e

    def num_e_of_group(self, group_name: str) -> int:
        r"""Return the number of hyperedges in the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        return len(self._raw_groups[group_name])

    @property
    def num_groups(self) -> int:
        r"""Return the number of hyperedge groups in the hypergraph.
        """
        return len(self._raw_groups)

    @property
    def group_names(self) -> List[str]:
        r"""Return the names of hyperedge groups in the hypergraph.
        """
        return list(self._raw_groups.keys())

    # properties for deep learning
    @property
    @abc.abstractmethod
    def vars_for_DL(self) -> List[str]:
        r"""Return a name list of available variables for deep learning in this type of hypergraph.
        """

    @property
    def W_v(self) -> torch.Tensor:
        r"""Return the vertex weight matrix of the hypergraph.
        """
        if self.cache["W_v"] is None:
            self.cache["W_v"] = torch.tensor(self.v_weight, dtype=torch.float, device=self.device).view(-1, 1).to(self.device)
        return self.cache["W_v"]

    @property
    def W_e(self) -> torch.Tensor:
        r"""Return the hyperedge weight matrix of the hypergraph.
        """
        if self.cache["W_e"] is None:
            _tmp = [self.W_e_of_group(name) for name in self.group_names]
            self.cache["W_e"] = torch.cat(_tmp, dim=0)
        return self.cache["W_e"]

    def W_e_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the hyperedge weight matrix of the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name]["W_e"] is None:
            self.group_cache[group_name]["W_e"] = self._fetch_W_of_group(group_name)
        return self.group_cache[group_name]["W_e"]

    @property
    @abc.abstractmethod
    def H(self) -> torch.Tensor:
        r"""Return the hypergraph incidence matrix.
        """

    @property
    @abc.abstractmethod
    def H_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the hypergraph incidence matrix in the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """

    @property
    def H_v2e(self) -> torch.Tensor:
        r"""Return the hypergraph incidence matrix with ``sparse matrix`` format.
        """
        if self.cache.get("H_v2e") is None:
            _tmp = [self.H_v2e_of_group(name) for name in self.group_names]
            self.cache["H_v2e"] = torch.cat(_tmp, dim=1)
        return self.cache["H_v2e"]

    def H_v2e_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the hypergraph incidence matrix with ``sparse matrix`` format in the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("H_v2e") is None:
            self.group_cache[group_name]["H_v2e"] = self._fetch_H_of_group("v2e", group_name)
        return self.group_cache[group_name]["H_v2e"]

    @property
    def H_e2v(self) -> torch.Tensor:
        r"""Return the hypergraph incidence matrix with ``sparse matrix`` format.
        """
        if self.cache.get("H_e2v") is None:
            _tmp = [self.H_e2v_of_group(name) for name in self.group_names]
            self.cache["H_e2v"] = torch.cat(_tmp, dim=1)
        return self.cache["H_e2v"]

    def H_e2v_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the hypergraph incidence matrix with ``sparse matrix`` format in the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("H_e2v") is None:
            self.group_cache[group_name]["H_e2v"] = self._fetch_H_of_group("e2v", group_name)
        return self.group_cache[group_name]["H_e2v"]

    @property
    def R_v2e(self) -> torch.Tensor:
        r"""Return the weight matrix of connections (vertices point to hyperedges) with ``sparse matrix`` format.
        """
        if self.cache.get("R_v2e") is None:
            _tmp = [self.R_v2e_of_group(name) for name in self.group_names]
            self.cache["R_v2e"] = torch.cat(_tmp, dim=1)
        return self.cache["R_v2e"]

    def R_v2e_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the weight matrix of connections (vertices point to hyperedges) with ``sparse matrix`` format in the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("R_v2e") is None:
            self.group_cache[group_name]["R_v2e"] = self._fetch_R_of_group("v2e", group_name)
        return self.group_cache[group_name]["R_v2e"]

    @property
    def R_e2v(self) -> torch.Tensor:
        r"""Return the weight matrix of connections (hyperedges point to vertices) with ``sparse matrix`` format.
        """
        if self.cache.get("R_e2v") is None:
            _tmp = [self.R_e2v_of_group(name) for name in self.group_names]
            self.cache["R_e2v"] = torch.cat(_tmp, dim=1)
        return self.cache["R_e2v"]

    def R_e2v_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the weight matrix of connections (hyperedges point to vertices) with ``sparse matrix`` format in the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("R_e2v") is None:
            self.group_cache[group_name]["R_e2v"] = self._fetch_R_of_group("e2v", group_name)
        return self.group_cache[group_name]["R_e2v"]

    # spectral-based smoothing
    def smoothing(self, X: torch.Tensor, L: torch.Tensor, lamb: float) -> torch.Tensor:
        r"""Spectral-based smoothing.

        .. math::
            X_{smoothed} = X + \lambda \mathcal{L} X

        Args:
            ``X`` (``torch.Tensor``): The vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``L`` (``torch.Tensor``): The Laplacian matrix with ``torch.sparse_coo_tensor`` format. Size :math:`(|\mathcal{V}|, |\mathcal{V}|)`.
            ``lamb`` (``float``): :math:`\lambda`, the strength of smoothing.
        """
        return X + lamb * torch.sparse.mm(L, X)

    # message passing functions
    @abc.abstractmethod
    def v2e_aggregation(
        self, X: torch.Tensor, aggr: str = "mean", v2e_weight: Optional[torch.Tensor] = None,
    ):
        r"""Message aggretation step of ``vertices to hyperedges``.

        Args:
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyepredges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
        """

    @abc.abstractmethod
    def v2e_aggregation_of_group(
        self, group_name: str, X: torch.Tensor, aggr: str = "mean", v2e_weight: Optional[torch.Tensor] = None,
    ):
        r"""Message aggregation step of ``vertices to hyperedges`` in specified hyperedge group.

        Args:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyepredges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
        """

    @abc.abstractmethod
    def v2e_update(self, X: torch.Tensor, e_weight: Optional[torch.Tensor] = None):
        r"""Message update step of ``vertices to hyperedges``.

        Args:
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
        """

    @abc.abstractmethod
    def v2e_update_of_group(self, group_name: str, X: torch.Tensor, e_weight: Optional[torch.Tensor] = None):
        r"""Message update step of ``vertices to hyperedges`` in specified hyperedge group.

        Args:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
        """

    @abc.abstractmethod
    def v2e(
        self,
        X: torch.Tensor,
        aggr: str = "mean",
        v2e_weight: Optional[torch.Tensor] = None,
        e_weight: Optional[torch.Tensor] = None,
    ):
        r"""Message passing of ``vertices to hyperedges``. The combination of ``v2e_aggregation`` and ``v2e_update``.

        Args:
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyepredges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
        """

    @abc.abstractmethod
    def v2e_of_group(
        self,
        group_name: str,
        X: torch.Tensor,
        aggr: str = "mean",
        v2e_weight: Optional[torch.Tensor] = None,
        e_weight: Optional[torch.Tensor] = None,
    ):
        r"""Message passing of ``vertices to hyperedges`` in specified hyperedge group. The combination of ``e2v_aggregation_of_group`` and ``e2v_update_of_group``.

        Args:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyepredges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
        """

    @abc.abstractmethod
    def e2v_aggregation(
        self, X: torch.Tensor, aggr: str = "mean", e2v_weight: Optional[torch.Tensor] = None,
    ):
        r"""Message aggregation step of ``hyperedges to vertices``.

        Args:
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
        """

    @abc.abstractmethod
    def e2v_aggregation_of_group(
        self, group_name: str, X: torch.Tensor, aggr: str = "mean", e2v_weight: Optional[torch.Tensor] = None,
    ):
        r"""Message aggregation step of ``hyperedges to vertices`` in specified hyperedge group.

        Args:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
        """

    @abc.abstractmethod
    def e2v_update(self, X: torch.Tensor, v_weight: Optional[torch.Tensor] = None):
        r"""Message update step of ``hyperedges to vertices``.

        Args:
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``v_weight`` (``torch.Tensor``, optional): The vertex weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
        """

    @abc.abstractmethod
    def e2v_update_of_group(self, group_name: str, X: torch.Tensor, v_weight: Optional[torch.Tensor] = None):
        r"""Message update step of ``hyperedges to vertices`` in specified hyperedge group.

        Args:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``v_weight`` (``torch.Tensor``, optional): The vertex weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
        """

    @abc.abstractmethod
    def e2v(
        self,
        X: torch.Tensor,
        aggr: str = "mean",
        e2v_weight: Optional[torch.Tensor] = None,
        v_weight: Optional[torch.Tensor] = None,
    ):
        r"""Message passing of ``hyperedges to vertices``. The combination of ``e2v_aggregation`` and ``e2v_update``.

        Args:
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``v_weight`` (``torch.Tensor``, optional): The vertex weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
        """

    @abc.abstractmethod
    def e2v_of_group(
        self,
        group_name: str,
        X: torch.Tensor,
        aggr: str = "mean",
        e2v_weight: Optional[torch.Tensor] = None,
        v_weight: Optional[torch.Tensor] = None,
    ):
        r"""Message passing of ``hyperedges to vertices`` in specified hyperedge group. The combination of ``e2v_aggregation_of_group`` and ``e2v_update_of_group``.

        Args:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``v_weight`` (``torch.Tensor``, optional): The vertex weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
        """

    @abc.abstractmethod
    def v2v(
        self,
        X: torch.Tensor,
        aggr: str = "mean",
        v2e_aggr: Optional[str] = None,
        v2e_weight: Optional[torch.Tensor] = None,
        e_weight: Optional[torch.Tensor] = None,
        e2v_aggr: Optional[str] = None,
        e2v_weight: Optional[torch.Tensor] = None,
        v_weight: Optional[torch.Tensor] = None,
    ):
        r"""Message passing of ``vertices to vertices``. The combination of ``v2e`` and ``e2v``.

        Args:
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``. If specified, this ``aggr`` will be used to both ``v2e`` and ``e2v``.
            ``v2e_aggr`` (``str``, optional): The aggregation method for hyperedges to vertices. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``. If specified, it will override the ``aggr`` in ``e2v``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyepredges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e2v_aggr`` (``str``, optional): The aggregation method for vertices to hyperedges. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``. If specified, it will override the ``aggr`` in ``v2e``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``v_weight`` (``torch.Tensor``, optional): The vertex weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
        """

    @abc.abstractmethod
    def v2v_of_group(
        self,
        group_name: str,
        X: torch.Tensor,
        aggr: str = "mean",
        v2e_aggr: Optional[str] = None,
        v2e_weight: Optional[torch.Tensor] = None,
        e_weight: Optional[torch.Tensor] = None,
        e2v_aggr: Optional[str] = None,
        e2v_weight: Optional[torch.Tensor] = None,
        v_weight: Optional[torch.Tensor] = None,
    ):
        r"""Message passing of ``vertices to vertices`` in specified hyperedge group. The combination of ``v2e_of_group`` and ``e2v_of_group``.

        Args:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``. If specified, this ``aggr`` will be used to both ``v2e_of_group`` and ``e2v_of_group``.
            ``v2e_aggr`` (``str``, optional): The aggregation method for hyperedges to vertices. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``. If specified, it will override the ``aggr`` in ``e2v_of_group``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyepredges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e2v_aggr`` (``str``, optional): The aggregation method for vertices to hyperedges. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``. If specified, it will override the ``aggr`` in ``v2e_of_group``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``v_weight`` (``torch.Tensor``, optional): The vertex weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
        """

class Hypergraph(BaseHypergraph):
    r"""The ``Hypergraph`` class is developed for hypergraph structures.

    Args:
        ``num_v`` (``int``): The number of vertices in the hypergraph.
        ``e_list`` (``Union[List[int], List[List[int]]]``, optional): A list of hyperedges describes how the vertices point to the hyperedges. Defaults to ``None``.
        ``e_weight`` (``Union[float, List[float]]``, optional): A list of weights for hyperedges. If set to ``None``, the value ``1`` is used for all hyperedges. Defaults to ``None``.
        ``v_weight`` (``Union[List[float]]``, optional): A list of weights for vertices. If set to ``None``, the value ``1`` is used for all vertices. Defaults to ``None``.
        ``merge_op`` (``str``): The operation to merge those conflicting hyperedges in the same hyperedge group, which can be ``'mean'``, ``'sum'`` or ``'max'``. Defaults to ``'mean'``.
        ``device`` (``torch.device``, optional): The deivce to store the hypergraph. Defaults to ``torch.device('cpu')``.
    """

    def __init__(
        self,
        num_v: int,
        e_list: Optional[Union[List[int], List[List[int]]]] = None,
        e_weight: Optional[Union[float, List[float]]] = None,
        v_weight: Optional[List[float]] = None,
        merge_op: str = "mean",
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(num_v, device=device)
        # init vertex weight
        if v_weight is None:
            self._v_weight = [1.0] * self.num_v
        else:
            assert len(v_weight) == self.num_v, "The length of vertex weight is not equal to the number of vertices."
            self._v_weight = v_weight
        # init hyperedges
        if e_list is not None:
            self.add_hyperedges(e_list, e_weight, merge_op=merge_op)

    def __repr__(self) -> str:
        r"""Print the hypergraph information.
        """
        return f"Hypergraph(num_v={self.num_v}, num_e={self.num_e})"

    @property
    def state_dict(self) -> Dict[str, Any]:
        r"""Get the state dict of the hypergraph.
        """
        return {"num_v": self.num_v, "raw_groups": self._raw_groups}

    def save(self, file_path: Union[str, Path]):
        r"""Save the DHG's hypergraph structure a file.

        Args:
            ``file_path`` (``Union[str, Path]``): The file path to store the DHG's hypergraph structure.
        """
        file_path = Path(file_path)
        assert file_path.parent.exists(), "The directory does not exist."
        data = {
            "class": "Hypergraph",
            "state_dict": self.state_dict,
        }
        with open(file_path, "wb") as fp:
            pickle.dump(data, fp)

    @staticmethod
    def load(file_path: Union[str, Path]):
        r"""Load the DHG's hypergraph structure from a file.

        Args:
            ``file_path`` (``Union[str, Path]``): The file path to load the DHG's hypergraph structure.
        """
        file_path = Path(file_path)
        assert file_path.exists(), "The file does not exist."
        with open(file_path, "rb") as fp:
            data = pickle.load(fp)
        assert data["class"] == "Hypergraph", "The file is not a DHG's hypergraph file."
        return Hypergraph.from_state_dict(data["state_dict"])

    def clear(self):
        r"""Clear all hyperedges and caches from the hypergraph.
        """
        return super().clear()

    def clone(self) -> "Hypergraph":
        r"""Return a copy of the hypergraph.
        """
        hg = Hypergraph(self.num_v, device=self.device)
        hg._raw_groups = deepcopy(self._raw_groups)
        hg.cache = deepcopy(self.cache)
        hg.group_cache = deepcopy(self.group_cache)
        return hg

    def to(self, device: torch.device):
        r"""Move the hypergraph to the specified device.

        Args:
            ``device`` (``torch.device``): The target device.
        """
        return super().to(device)

    # =====================================================================================
    # some construction functions
    @staticmethod
    def from_state_dict(state_dict: dict):
        r"""Load the hypergraph from the state dict.

        Args:
            ``state_dict`` (``dict``): The state dict to load the hypergraph.
        """
        _hg = Hypergraph(state_dict["num_v"])
        _hg._raw_groups = deepcopy(state_dict["raw_groups"])
        return _hg

    # =====================================================================================
    # some structure modification functions
    def add_hyperedges(
        self,
        e_list: Union[List[int], List[List[int]]],
        e_weight: Optional[Union[float, List[float]]] = None,
        merge_op: str = "mean",
        group_name: str = "main",
    ):
        r"""Add hyperedges to the hypergraph. If the ``group_name`` is not specified, the hyperedges will be added to the default ``main`` hyperedge group.

        Args:
            ``num_v`` (``int``): The number of vertices in the hypergraph.
            ``e_list`` (``Union[List[int], List[List[int]]]``): A list of hyperedges describes how the vertices point to the hyperedges.
            ``e_weight`` (``Union[float, List[float]]``, optional): A list of weights for hyperedges. If set to ``None``, the value ``1`` is used for all hyperedges. Defaults to ``None``.
            ``merge_op`` (``str``): The merge operation for the conflicting hyperedges. The possible values are ``"mean"``, ``"sum"``, and ``"max"``. Defaults to ``"mean"``.
            ``group_name`` (``str``, optional): The target hyperedge group to add these hyperedges. Defaults to the ``main`` hyperedge group.
        """
        e_list = self._format_e_list(e_list)
        if e_weight is None:
            e_weight = [1.0] * len(e_list)
        elif type(e_weight) in (int, float):
            e_weight = [e_weight]
        elif type(e_weight) is list:
            pass
        else:
            raise TypeError(f"The type of e_weight should be float or list, but got {type(e_weight)}")
        assert len(e_list) == len(e_weight), "The number of hyperedges and the number of weights are not equal."

        for _idx in range(len(e_list)):
            self._add_hyperedge(
                self._hyperedge_code(e_list[_idx], e_list[_idx]), {"w_e": float(e_weight[_idx])}, merge_op, group_name,
            )
        self._clear_cache(group_name)

    def add_hyperedges_from_feature_kNN(self, feature: torch.Tensor, k: int, group_name: str = "main"):
        r"""Add hyperedges from the feature matrix by k-NN. Each hyperedge is constructed by the central vertex and its :math:`k`-Nearest Neighbor vertices.

        Args:
            ``features`` (``torch.Tensor``): The feature matrix.
            ``k`` (``int``): The number of nearest neighbors.
            ``group_name`` (``str``, optional): The target hyperedge group to add these hyperedges. Defaults to the ``main`` hyperedge group.
        """
        assert (
            feature.shape[0] == self.num_v
        ), "The number of vertices in the feature matrix is not equal to the number of vertices in the hypergraph."
        e_list = Hypergraph._e_list_from_feature_kNN(feature, k)
        self.add_hyperedges(e_list, group_name=group_name)

    def add_hyperedges_from_graph(self, graph: "Graph", group_name: str = "main"):
        r"""Add hyperedges from edges in the graph. Each edge in the graph is treated as a hyperedge.

        Args:
            ``graph`` (``Graph``): The graph to join the hypergraph.
            ``group_name`` (``str``, optional): The target hyperedge group to add these hyperedges. Defaults to the ``main`` hyperedge group.
        """
        assert self.num_v == graph.num_v, "The number of vertices in the hypergraph and the graph are not equal."
        e_list, e_weight = graph.e
        self.add_hyperedges(e_list, e_weight=e_weight, group_name=group_name)

    def remove_hyperedges(
        self, e_list: Union[List[int], List[List[int]]], group_name: Optional[str] = None,
    ):
        r"""Remove the specified hyperedges from the hypergraph.

        Args:
            ``e_list`` (``Union[List[int], List[List[int]]]``): A list of hyperedges describes how the vertices point to the hyperedges.
            ``group_name`` (``str``, optional): Remove these hyperedges from the specified hyperedge group. If not specified, the function will
                remove those hyperedges from all hyperedge groups. Defaults to the ``None``.
        """
        assert (
            group_name is None or group_name in self.group_names
        ), "The specified group_name is not in existing hyperedge groups."
        e_list = self._format_e_list(e_list)
        if group_name is None:
            for _idx in range(len(e_list)):
                e_code = self._hyperedge_code(e_list[_idx], e_list[_idx])
                for name in self.group_names:
                    self._raw_groups[name].pop(e_code, None)
        else:
            for _idx in range(len(e_list)):
                e_code = self._hyperedge_code(e_list[_idx], e_list[_idx])
                self._raw_groups[group_name].pop(e_code, None)
        self._clear_cache(group_name)

    def remove_group(self, group_name: str):
        r"""Remove the specified hyperedge group from the hypergraph.

        Args:
            ``group_name`` (``str``): The name of the hyperedge group to remove.
        """
        self._raw_groups.pop(group_name, None)
        self._clear_cache(group_name)

    def drop_hyperedges(self, drop_rate: float, ord="uniform"):
        r"""Randomly drop hyperedges from the hypergraph. This function will return a new hypergraph with non-dropped hyperedges.

        Args:
            ``drop_rate`` (``float``): The drop rate of hyperedges.
            ``ord`` (``str``): The order of dropping edges. Currently, only ``'uniform'`` is supported. Defaults to ``uniform``.
        """
        if ord == "uniform":
            _raw_groups = {}
            for name in self.group_names:
                _raw_groups[name] = {k: v for k, v in self._raw_groups[name].items() if random.random() > drop_rate}
            state_dict = {
                "num_v": self.num_v,
                "raw_groups": _raw_groups,
            }
            _hg = Hypergraph.from_state_dict(state_dict)
            _hg = _hg.to(self.device)
        else:
            raise ValueError(f"Unkonwn drop order: {ord}.")
        return _hg

    def drop_hyperedges_of_group(self, group_name: str, drop_rate: float, ord="uniform"):
        r"""Randomly drop hyperedges from the specified hyperedge group. This function will return a new hypergraph with non-dropped hyperedges.

        Args:
            ``group_name`` (``str``): The name of the hyperedge group.
            ``drop_rate`` (``float``): The drop rate of hyperedges.
            ``ord`` (``str``): The order of dropping edges. Currently, only ``'uniform'`` is supported. Defaults to ``uniform``.
        """
        if ord == "uniform":
            _raw_groups = {}
            for name in self.group_names:
                if name == group_name:
                    _raw_groups[name] = {
                        k: v for k, v in self._raw_groups[name].items() if random.random() > drop_rate
                    }
                else:
                    _raw_groups[name] = self._raw_groups[name]
            state_dict = {
                "num_v": self.num_v,
                "raw_groups": _raw_groups,
            }
            _hg = Hypergraph.from_state_dict(state_dict)
            _hg = _hg.to(self.device)
        else:
            raise ValueError(f"Unkonwn drop order: {ord}.")
        return _hg

    # =====================================================================================
    # properties for representation
    @property
    def v(self) -> List[int]:
        r"""Return the list of vertices.
        """
        return super().v
    
    @property
    def v_weight(self) -> List[float]:
        r"""Return the list of vertex weights.
        """
        return self._v_weight

    @property
    def e(self) -> Tuple[List[List[int]], List[float]]:
        r"""Return all hyperedges and weights in the hypergraph.
        """
        if self.cache.get("e", None) is None:
            e_list, e_weight = [], []
            for name in self.group_names:
                _e = self.e_of_group(name)
                e_list.extend(_e[0])
                e_weight.extend(_e[1])
            self.cache["e"] = (e_list, e_weight)
        return self.cache["e"]

    def e_of_group(self, group_name: str) -> Tuple[List[List[int]], List[float]]:
        r"""Return all hyperedges and weights of the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("e", None) is None:
            e_list = [e_code[0] for e_code in self._raw_groups[group_name].keys()]
            e_weight = [e_content["w_e"] for e_content in self._raw_groups[group_name].values()]
            self.group_cache[group_name]["e"] = (e_list, e_weight)
        return self.group_cache[group_name]["e"]

    @property
    def num_v(self) -> int:
        r"""Return the number of vertices in the hypergraph.
        """
        return super().num_v

    @property
    def num_e(self) -> int:
        r"""Return the number of hyperedges in the hypergraph.
        """
        return super().num_e

    def num_e_of_group(self, group_name: str) -> int:
        r"""Return the number of hyperedges of the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        return super().num_e_of_group(group_name)

    @property
    def deg_v(self) -> List[int]:
        r"""Return the degree list of each vertex.
        """
        return self.D_v._values().cpu().view(-1).numpy().tolist()

    def deg_v_of_group(self, group_name: str) -> List[int]:
        r"""Return the degree list of each vertex of the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        return self.D_v_of_group(group_name)._values().cpu().view(-1).numpy().tolist()

    @property
    def deg_e(self) -> List[int]:
        r"""Return the degree list of each hyperedge.
        """
        return self.D_e._values().cpu().view(-1).numpy().tolist()

    def deg_e_of_group(self, group_name: str) -> List[int]:
        r"""Return the degree list of each hyperedge of the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        return self.D_e_of_group(group_name)._values().cpu().view(-1).numpy().tolist()

    def nbr_e(self, v_idx: int) -> List[int]:
        r"""Return the neighbor hyperedge list of the specified vertex.

        Args:
            ``v_idx`` (``int``): The index of the vertex.
        """
        return self.N_e(v_idx).cpu().numpy().tolist()

    def nbr_e_of_group(self, v_idx: int, group_name: str) -> List[int]:
        r"""Return the neighbor hyperedge list of the specified vertex of the specified hyperedge group.

        Args:
            ``v_idx`` (``int``): The index of the vertex.
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        return self.N_e_of_group(v_idx, group_name).cpu().numpy().tolist()

    def nbr_v(self, e_idx: int) -> List[int]:
        r"""Return the neighbor vertex list of the specified hyperedge.

        Args:
            ``e_idx`` (``int``): The index of the hyperedge.
        """
        return self.N_v(e_idx).cpu().numpy().tolist()

    def nbr_v_of_group(self, e_idx: int, group_name: str) -> List[int]:
        r"""Return the neighbor vertex list of the specified hyperedge of the specified hyperedge group.

        Args:
            ``e_idx`` (``int``): The index of the hyperedge.
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        return self.N_v_of_group(e_idx, group_name).cpu().numpy().tolist()

    @property
    def num_groups(self) -> int:
        r"""Return the number of hyperedge groups in the hypergraph.
        """
        return super().num_groups

    @property
    def group_names(self) -> List[str]:
        r"""Return the names of all hyperedge groups in the hypergraph.
        """
        return super().group_names

    # =====================================================================================
    # properties for deep learning
    @property
    def vars_for_DL(self) -> List[str]:
        r"""Return a name list of available variables for deep learning in the hypergraph including

        Sparse Matrices:

        .. math::
            \mathbf{H}, \mathbf{H}^\top, \mathcal{L}_{sym}, \mathcal{L}_{rw} \mathcal{L}_{HGNN},

        Sparse Diagnal Matrices:

        .. math::
            \mathbf{W}_v, \mathbf{W}_e, \mathbf{D}_v, \mathbf{D}_v^{-1}, \mathbf{D}_v^{-\frac{1}{2}}, \mathbf{D}_e, \mathbf{D}_e^{-1},

        Vectors:

        .. math::
            \overrightarrow{v2e}_{src}, \overrightarrow{v2e}_{dst}, \overrightarrow{v2e}_{weight},\\
            \overrightarrow{e2v}_{src}, \overrightarrow{e2v}_{dst}, \overrightarrow{e2v}_{weight}

        """
        return [
            "H",
            "H_T",
            "L_sym",
            "L_rw",
            "L_HGNN",
            "W_v",
            "W_e",
            "D_v",
            "D_v_neg_1",
            "D_v_neg_1_2",
            "D_e",
            "D_e_neg_1",
            "v2e_src",
            "v2e_dst",
            "v2e_weight" "e2v_src",
            "e2v_dst",
            "e2v_weight",
        ]

    @property
    def v2e_src(self) -> torch.Tensor:
        r"""Return the source vertex index vector :math:`\overrightarrow{v2e}_{src}` of the connections (vertices point to hyperedges) in the hypergraph.
        """
        return self.H_T._indices()[1].clone()

    def v2e_src_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the source vertex index vector :math:`\overrightarrow{v2e}_{src}` of the connections (vertices point to hyperedges) in the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        return self.H_T_of_group(group_name)._indices()[1].clone()

    @property
    def v2e_dst(self) -> torch.Tensor:
        r"""Return the destination hyperedge index vector :math:`\overrightarrow{v2e}_{dst}` of the connections (vertices point to hyperedges) in the hypergraph.
        """
        return self.H_T._indices()[0].clone()

    def v2e_dst_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the destination hyperedge index vector :math:`\overrightarrow{v2e}_{dst}` of the connections (vertices point to hyperedges) in the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        return self.H_T_of_group(group_name)._indices()[0].clone()

    @property
    def v2e_weight(self) -> torch.Tensor:
        r"""Return the weight vector :math:`\overrightarrow{v2e}_{weight}` of the connections (vertices point to hyperedges) in the hypergraph.
        """
        return self.H_T._values().clone()

    def v2e_weight_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the weight vector :math:`\overrightarrow{v2e}_{weight}` of the connections (vertices point to hyperedges) in the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        return self.H_T_of_group(group_name)._values().clone()

    @property
    def e2v_src(self) -> torch.Tensor:
        r"""Return the source hyperedge index vector :math:`\overrightarrow{e2v}_{src}` of the connections (hyperedges point to vertices) in the hypergraph.
        """
        return self.H._indices()[1].clone()

    def e2v_src_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the source hyperedge index vector :math:`\overrightarrow{e2v}_{src}` of the connections (hyperedges point to vertices) in the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        return self.H_of_group(group_name)._indices()[1].clone()

    @property
    def e2v_dst(self) -> torch.Tensor:
        r"""Return the destination vertex index vector :math:`\overrightarrow{e2v}_{dst}` of the connections (hyperedges point to vertices) in the hypergraph.
        """
        return self.H._indices()[0].clone()

    def e2v_dst_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the destination vertex index vector :math:`\overrightarrow{e2v}_{dst}` of the connections (hyperedges point to vertices) in the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        return self.H_of_group(group_name)._indices()[0].clone()

    @property
    def e2v_weight(self) -> torch.Tensor:
        r"""Return the weight vector :math:`\overrightarrow{e2v}_{weight}` of the connections (hyperedges point to vertices) in the hypergraph.
        """
        return self.H._values().clone()

    def e2v_weight_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the weight vector :math:`\overrightarrow{e2v}_{weight}` of the connections (hyperedges point to vertices) in the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        return self.H_of_group(group_name)._values().clone()

    @property
    def H(self) -> torch.Tensor:
        r"""Return the hypergraph incidence matrix :math:`\mathbf{H}` with ``torch.sparse_coo_tensor`` format.
        """
        if self.cache.get("H") is None:
            self.cache["H"] = self.H_v2e.to(self.device)
        return self.cache["H"]

    def H_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the hypergraph incidence matrix :math:`\mathbf{H}` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("H") is None:
            self.group_cache[group_name]["H"] = self.H_v2e_of_group(group_name)
        return self.group_cache[group_name]["H"]

    @property
    def H_T(self) -> torch.Tensor:
        r"""Return the transpose of the hypergraph incidence matrix :math:`\mathbf{H}^\top` with ``torch.sparse_coo_tensor`` format.
        """
        if self.cache.get("H_T") is None:
            self.cache["H_T"] = self.H.t().to(self.device)
        return self.cache["H_T"]

    def H_T_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the transpose of the hypergraph incidence matrix :math:`\mathbf{H}^\top` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("H_T") is None:
            self.group_cache[group_name]["H_T"] = self.H_of_group(group_name).t()
        return self.group_cache[group_name]["H_T"]
    
    @property
    def W_v(self) -> torch.Tensor:
        r"""Return the weight matrix :math:`\mathbf{W}_v` of vertices with ``torch.sparse_coo_tensor`` format.
        """
        if self.cache.get("W_v") is None:
            _tmp = torch.Tensor(self.v_weight)
            _num_v = _tmp.size(0)
            self.cache["W_v"] = torch.sparse_coo_tensor(
                torch.arange(0, _num_v, device=self.device).view(1, -1).repeat(2, 1),
                _tmp,
                torch.Size([_num_v, _num_v]),
                device=self.device,
            ).coalesce()
        return self.cache["W_v"]

    @property
    def W_e(self) -> torch.Tensor:
        r"""Return the weight matrix :math:`\mathbf{W}_e` of hyperedges with ``torch.sparse_coo_tensor`` format.
        """
        if self.cache.get("W_e") is None:
            _tmp = [self.W_e_of_group(name)._values().clone() for name in self.group_names]
            _tmp = torch.cat(_tmp, dim=0).view(-1)
            _num_e = _tmp.size(0)
            self.cache["W_e"] = torch.sparse_coo_tensor(
                torch.arange(0, _num_e, device=self.device).view(1, -1).repeat(2, 1),
                _tmp,
                torch.Size([_num_e, _num_e]),
                device=self.device,
            ).coalesce()
        return self.cache["W_e"]

    def W_e_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the weight matrix :math:`\mathbf{W}_e` of hyperedges of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("W_e") is None:
            _tmp = self._fetch_W_of_group(group_name).view(-1)
            _num_e = _tmp.size(0)
            self.group_cache[group_name]["W_e"] = torch.sparse_coo_tensor(
                torch.arange(0, _num_e, device=self.device).view(1, -1).repeat(2, 1),
                _tmp,
                torch.Size([_num_e, _num_e]),
                device=self.device,
            ).coalesce()
        return self.group_cache[group_name]["W_e"]

    @property
    def D_v(self) -> torch.Tensor:
        r"""Return the vertex degree matrix :math:`\mathbf{D}_v` with ``torch.sparse_coo_tensor`` format.
        """
        if self.cache.get("D_v") is None:
            _tmp = [self.D_v_of_group(name)._values().clone() for name in self.group_names]
            _tmp = torch.vstack(_tmp).sum(dim=0).view(-1)
            self.cache["D_v"] = torch.sparse_coo_tensor(
                torch.arange(0, self.num_v, device=self.device).view(1, -1).repeat(2, 1),
                _tmp,
                torch.Size([self.num_v, self.num_v]),
                device=self.device,
            ).coalesce()
        return self.cache["D_v"]

    def D_v_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the vertex degree matrix :math:`\mathbf{D}_v` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("D_v") is None:
            H = self.H_of_group(group_name).clone()
            w_e = self.W_e_of_group(group_name)._values().clone()
            val = w_e[H._indices()[1]] * H._values()
            H_ = torch.sparse_coo_tensor(H._indices(), val, size=H.shape, device=self.device).coalesce()
            _tmp = torch.sparse.sum(H_, dim=1).to_dense().clone().view(-1)
            _num_v = _tmp.size(0)
            self.group_cache[group_name]["D_v"] = torch.sparse_coo_tensor(
                torch.arange(0, _num_v, device=self.device).view(1, -1).repeat(2, 1),
                _tmp,
                torch.Size([_num_v, _num_v]),
                device=self.device,
            ).coalesce()
        return self.group_cache[group_name]["D_v"]

    @property
    def D_v_neg_1(self) -> torch.Tensor:
        r"""Return the vertex degree matrix :math:`\mathbf{D}_v^{-1}` with ``torch.sparse_coo_tensor`` format.
        """
        if self.cache.get("D_v_neg_1") is None:
            _mat = self.D_v.clone()
            _val = _mat._values() ** -1
            _val[torch.isinf(_val)] = 0
            self.cache["D_v_neg_1"] = torch.sparse_coo_tensor(
                _mat._indices(), _val, _mat.size(), device=self.device
            ).coalesce()
        return self.cache["D_v_neg_1"]

    def D_v_neg_1_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the vertex degree matrix :math:`\mathbf{D}_v^{-1}` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("D_v_neg_1") is None:
            _mat = self.D_v_of_group(group_name).clone()
            _val = _mat._values() ** -1
            _val[torch.isinf(_val)] = 0
            self.group_cache[group_name]["D_v_neg_1"] = torch.sparse_coo_tensor(
                _mat._indices(), _val, _mat.size(), device=self.device
            ).coalesce()
        return self.group_cache[group_name]["D_v_neg_1"]

    @property
    def D_v_neg_1_2(self) -> torch.Tensor:
        r"""Return the vertex degree matrix :math:`\mathbf{D}_v^{-\frac{1}{2}}` with ``torch.sparse_coo_tensor`` format.
        """
        if self.cache.get("D_v_neg_1_2") is None:
            _mat = self.D_v.clone()
            _val = _mat._values() ** -0.5
            _val[torch.isinf(_val)] = 0
            self.cache["D_v_neg_1_2"] = torch.sparse_coo_tensor(
                _mat._indices(), _val, _mat.size(), device=self.device
            ).coalesce()
        return self.cache["D_v_neg_1_2"]

    def D_v_neg_1_2_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the vertex degree matrix :math:`\mathbf{D}_v^{-\frac{1}{2}}` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("D_v_neg_1_2") is None:
            _mat = self.D_v_of_group(group_name).clone()
            _val = _mat._values() ** -0.5
            _val[torch.isinf(_val)] = 0
            self.group_cache[group_name]["D_v_neg_1_2"] = torch.sparse_coo_tensor(
                _mat._indices(), _val, _mat.size(), device=self.device
            ).coalesce()
        return self.group_cache[group_name]["D_v_neg_1_2"]

    @property
    def D_e(self) -> torch.Tensor:
        r"""Return the hyperedge degree matrix :math:`\mathbf{D}_e` with ``torch.sparse_coo_tensor`` format.
        """
        if self.cache.get("D_e") is None:
            _tmp = [self.D_e_of_group(name)._values().clone() for name in self.group_names]
            _tmp = torch.cat(_tmp, dim=0).view(-1)
            _num_e = _tmp.size(0)
            self.cache["D_e"] = torch.sparse_coo_tensor(
                torch.arange(0, _num_e, device=self.device).view(1, -1).repeat(2, 1),
                _tmp,
                torch.Size([_num_e, _num_e]),
                device=self.device,
            ).coalesce()
        return self.cache["D_e"]

    def D_e_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the hyperedge degree matrix :math:`\mathbf{D}_e` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("D_e") is None:
            _tmp = torch.sparse.sum(self.H_T_of_group(group_name), dim=1).to_dense().clone().view(-1)
            _num_e = _tmp.size(0)
            self.group_cache[group_name]["D_e"] = torch.sparse_coo_tensor(
                torch.arange(0, _num_e, device=self.device).view(1, -1).repeat(2, 1),
                _tmp,
                torch.Size([_num_e, _num_e]),
                device=self.device,
            ).coalesce()
        return self.group_cache[group_name]["D_e"]

    @property
    def D_e_neg_1(self) -> torch.Tensor:
        r"""Return the hyperedge degree matrix :math:`\mathbf{D}_e^{-1}` with ``torch.sparse_coo_tensor`` format.
        """
        if self.cache.get("D_e_neg_1") is None:
            _mat = self.D_e.clone()
            _val = _mat._values() ** -1
            _val[torch.isinf(_val)] = 0
            self.cache["D_e_neg_1"] = torch.sparse_coo_tensor(
                _mat._indices(), _val, _mat.size(), device=self.device
            ).coalesce()
        return self.cache["D_e_neg_1"]

    def D_e_neg_1_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the hyperedge degree matrix :math:`\mathbf{D}_e^{-1}` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("D_e_neg_1") is None:
            _mat = self.D_e_of_group(group_name).clone()
            _val = _mat._values() ** -1
            _val[torch.isinf(_val)] = 0
            self.group_cache[group_name]["D_e_neg_1"] = torch.sparse_coo_tensor(
                _mat._indices(), _val, _mat.size(), device=self.device
            ).coalesce()
        return self.group_cache[group_name]["D_e_neg_1"]

    def N_e(self, v_idx: int) -> torch.Tensor:
        r"""Return the neighbor hyperedges of the specified vertex with ``torch.Tensor`` format.

        .. note::
            The ``v_idx`` must be in the range of [0, :attr:`num_v`).

        Args:
            ``v_idx`` (``int``): The index of the vertex.
        """
        assert v_idx < self.num_v
        _tmp, e_bias = [], 0
        for name in self.group_names:
            _tmp.append(self.N_e_of_group(v_idx, name) + e_bias)
            e_bias += self.num_e_of_group(name)
        return torch.cat(_tmp, dim=0)

    def N_e_of_group(self, v_idx: int, group_name: str) -> torch.Tensor:
        r"""Return the neighbor hyperedges of the specified vertex of the specified hyperedge group with ``torch.Tensor`` format.

        .. note::
            The ``v_idx`` must be in the range of [0, :attr:`num_v`).

        Args:
            ``v_idx`` (``int``): The index of the vertex.
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        assert v_idx < self.num_v
        e_indices = self.H_of_group(group_name)[v_idx]._indices()[0]
        return e_indices.clone()

    def N_v(self, e_idx: int) -> torch.Tensor:
        r"""Return the neighbor vertices of the specified hyperedge with ``torch.Tensor`` format.

        .. note::
            The ``e_idx`` must be in the range of [0, :attr:`num_e`).

        Args:
            ``e_idx`` (``int``): The index of the hyperedge.
        """
        assert e_idx < self.num_e
        for name in self.group_names:
            if e_idx < self.num_e_of_group(name):
                return self.N_v_of_group(e_idx, name)
            else:
                e_idx -= self.num_e_of_group(name)

    def N_v_of_group(self, e_idx: int, group_name: str) -> torch.Tensor:
        r"""Return the neighbor vertices of the specified hyperedge of the specified hyperedge group with ``torch.Tensor`` format.

        .. note::
            The ``e_idx`` must be in the range of [0, :func:`num_e_of_group`).

        Args:
            ``e_idx`` (``int``): The index of the hyperedge.
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        assert e_idx < self.num_e_of_group(group_name)
        v_indices = self.H_T_of_group(group_name)[e_idx]._indices()[0]
        return v_indices.clone()

    # =====================================================================================
    # spectral-based convolution/smoothing
    def smoothing(self, X: torch.Tensor, L: torch.Tensor, lamb: float) -> torch.Tensor:
        return super().smoothing(X, L, lamb)

    @property
    def L_sym(self) -> torch.Tensor:
        r"""Return the symmetric Laplacian matrix :math:`\mathcal{L}_{sym}` of the hypergraph with ``torch.sparse_coo_tensor`` format.

        .. math::
            \mathcal{L}_{sym} = \mathbf{I} - \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}}
        """
        if self.cache.get("L_sym") is None:
            L_HGNN = self.L_HGNN.clone()
            self.cache["L_sym"] = torch.sparse_coo_tensor(
                torch.hstack([torch.arange(0, self.num_v, device=self.device).view(1, -1).repeat(2, 1), L_HGNN._indices(),]),
                torch.hstack([torch.ones(self.num_v, device=self.device), -L_HGNN._values()]),
                torch.Size([self.num_v, self.num_v]),
                device=self.device,
            ).coalesce()
        return self.cache["L_sym"]

    def L_sym_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the symmetric Laplacian matrix :math:`\mathcal{L}_{sym}` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        .. math::
            \mathcal{L}_{sym} = \mathbf{I} - \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}}

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("L_sym") is None:
            L_HGNN = self.L_HGNN_of_group(group_name).clone()
            self.group_cache[group_name]["L_sym"] = torch.sparse_coo_tensor(
                torch.hstack([torch.arange(0, self.num_v, device=self.device).view(1, -1).repeat(2, 1), L_HGNN._indices(),]),
                torch.hstack([torch.ones(self.num_v, device=self.device), -L_HGNN._values()]),
                torch.Size([self.num_v, self.num_v]),
                device=self.device,
            ).coalesce()
        return self.group_cache[group_name]["L_sym"]

    @property
    def L_rw(self) -> torch.Tensor:
        r"""Return the random walk Laplacian matrix :math:`\mathcal{L}_{rw}` of the hypergraph with ``torch.sparse_coo_tensor`` format.

        .. math::
            \mathcal{L}_{rw} = \mathbf{I} - \mathbf{D}_v^{-1} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top
        """
        if self.cache.get("L_rw") is None:
            _tmp = self.D_v_neg_1.mm(self.H).mm(self.W_e).mm(self.D_e_neg_1).mm(self.H_T)
            self.cache["L_rw"] = (
                torch.sparse_coo_tensor(
                    torch.hstack([torch.arange(0, self.num_v, device=self.device).view(1, -1).repeat(2, 1), _tmp._indices(),]),
                    torch.hstack([torch.ones(self.num_v, device=self.device), -_tmp._values()]),
                    torch.Size([self.num_v, self.num_v]),
                    device=self.device,
                )
                .coalesce()
                .clone()
            )
        return self.cache["L_rw"]

    def L_rw_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the random walk Laplacian matrix :math:`\mathcal{L}_{rw}` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        .. math::
            \mathcal{L}_{rw} = \mathbf{I} - \mathbf{D}_v^{-1} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("L_rw") is None:
            _tmp = (
                self.D_v_neg_1_of_group(group_name)
                .mm(self.H_of_group(group_name))
                .mm(self.W_e_of_group(group_name),)
                .mm(self.D_e_neg_1_of_group(group_name),)
                .mm(self.H_T_of_group(group_name),)
            )
            self.group_cache[group_name]["L_rw"] = (
                torch.sparse_coo_tensor(
                    torch.hstack([torch.arange(0, self.num_v, device=self.device).view(1, -1).repeat(2, 1), _tmp._indices(),]),
                    torch.hstack([torch.ones(self.num_v, device=self.device), -_tmp._values()]),
                    torch.Size([self.num_v, self.num_v]),
                    device=self.device,
                )
                .coalesce()
                .clone()
            )
        return self.group_cache[group_name]["L_rw"]

    ## HGNN Laplacian smoothing
    @property
    def L_HGNN(self) -> torch.Tensor:
        r"""Return the HGNN Laplacian matrix :math:`\mathcal{L}_{HGNN}` of the hypergraph with ``torch.sparse_coo_tensor`` format.

        .. math::
            \mathcal{L}_{HGNN} = \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}}
        """
        if self.cache.get("L_HGNN") is None:
            '''
            print(self.D_v_neg_1_2.shape, self.D_v_neg_1_2.dtype)
            print(self.H)
            print(self.W_e)
            print(self.D_e_neg_1)
            print(self.H_T)
            '''
            #_tmp = self.D_v_neg_1_2.mm(self.H).mm(self.W_e).mm(self.D_e_neg_1).mm(self.H_T,).mm(self.D_v_neg_1_2)
            tmp1 = self.D_v_neg_1_2.mm(self.H).mm(self.W_e).mm(self.D_e_neg_1)
            #print(tmp1)

            tmp1_dense = tmp1.to_dense()
            H_T_dense = self.H_T.to_dense()
            tmp4 = tmp1_dense.mm(H_T_dense)
            del tmp1_dense, H_T_dense

            D_v_dense = self.D_v_neg_1_2.to_dense()  # 转为密集对角矩阵
            _tmp = tmp4.mm(D_v_dense)  # 密集×密集乘法，CUDA支持
            del D_v_dense

            self.cache["L_HGNN"] = _tmp.to_sparse_coo().coalesce()
        return self.cache["L_HGNN"]

    def L_HGNN_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the HGNN Laplacian matrix :math:`\mathcal{L}_{HGNN}` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        .. math::
            \mathcal{L}_{HGNN} = \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}}

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("L_HGNN") is None:
            _tmp = (
                self.D_v_neg_1_2_of_group(group_name)
                .mm(self.H_of_group(group_name))
                .mm(self.W_e_of_group(group_name))
                .mm(self.D_e_neg_1_of_group(group_name),)
                .mm(self.H_T_of_group(group_name),)
                .mm(self.D_v_neg_1_2_of_group(group_name),)
            )
            self.group_cache[group_name]["L_HGNN"] = _tmp.coalesce()
        return self.group_cache[group_name]["L_HGNN"]

    def smoothing_with_HGNN(self, X: torch.Tensor, drop_rate: float = 0.0) -> torch.Tensor:
        r"""Return the smoothed feature matrix with the HGNN Laplacian matrix :math:`\mathcal{L}_{HGNN}`.

            .. math::
                \mathbf{X} = \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}} \mathbf{X}

        Args:
            ``X`` (``torch.Tensor``): The feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
    """
        if self.device != X.device:
            X = X.to(self.device)
        if drop_rate > 0.0:
            L_HGNN = sparse_dropout(self.L_HGNN, drop_rate)
        else:
            L_HGNN = self.L_HGNN
        return L_HGNN.mm(X)

    def smoothing_with_HGNN_of_group(self, group_name: str, X: torch.Tensor, drop_rate: float = 0.0) -> torch.Tensor:
        r"""Return the smoothed feature matrix with the HGNN Laplacian matrix :math:`\mathcal{L}_{HGNN}`.

            .. math::
                \mathbf{X} = \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}} \mathbf{X}

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
            ``X`` (``torch.Tensor``): The feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.device != X.device:
            X = X.to(self.device)
        if drop_rate > 0.0:
            L_HGNN = sparse_dropout(self.L_HGNN_of_group(group_name), drop_rate)
        else:
            L_HGNN = self.L_HGNN_of_group(group_name)
        return L_HGNN.mm(X)

    # =====================================================================================
    # spatial-based convolution/message-passing
    ## general message passing functions
    def v2e_aggregation(
        self, X: torch.Tensor, aggr: str = "mean", v2e_weight: Optional[torch.Tensor] = None, drop_rate: float = 0.0
    ):
        r"""Message aggretation step of ``vertices to hyperedges``.

        Args:
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyepredges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        assert aggr in ["mean", "sum", "softmax_then_sum"]
        if self.device != X.device:
            self.to(X.device)
        if v2e_weight is None:
            if drop_rate > 0.0:
                P = sparse_dropout(self.H_T, drop_rate)
            else:
                P = self.H_T
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                X = torch.sparse.mm(self.D_e_neg_1, X)
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method {aggr}.")
        else:
            # init message path
            assert (
                v2e_weight.shape[0] == self.v2e_weight.shape[0]
            ), "The size of v2e_weight must be equal to the size of self.v2e_weight."
            P = torch.sparse_coo_tensor(self.H_T._indices(), v2e_weight, self.H_T.shape, device=self.device)
            if drop_rate > 0.0:
                P = sparse_dropout(P, drop_rate)
            # message passing
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                D_e_neg_1 = torch.sparse.sum(P, dim=1).to_dense().view(-1, 1)
                D_e_neg_1[torch.isinf(D_e_neg_1)] = 0
                X = D_e_neg_1 * X
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method {aggr}.")
        return X

    def v2e_aggregation_of_group(
        self,
        group_name: str,
        X: torch.Tensor,
        aggr: str = "mean",
        v2e_weight: Optional[torch.Tensor] = None,
        drop_rate: float = 0.0,
    ):
        r"""Message aggregation step of ``vertices to hyperedges`` in specified hyperedge group.

        Args:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyepredges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        assert aggr in ["mean", "sum", "softmax_then_sum"]
        if self.device != X.device:
            self.to(X.device)
        if v2e_weight is None:
            if drop_rate > 0.0:
                P = sparse_dropout(self.H_T_of_group(group_name), drop_rate)
            else:
                P = self.H_T_of_group(group_name)
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                X = torch.sparse.mm(self.D_e_neg_1_of_group(group_name), X)
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method {aggr}.")
        else:
            # init message path
            assert (
                v2e_weight.shape[0] == self.v2e_weight_of_group(group_name).shape[0]
            ), f"The size of v2e_weight must be equal to the size of self.v2e_weight_of_group('{group_name}')."
            P = torch.sparse_coo_tensor(
                self.H_T_of_group(group_name)._indices(),
                v2e_weight,
                self.H_T_of_group(group_name).shape,
                device=self.device,
            )
            if drop_rate > 0.0:
                P = sparse_dropout(P, drop_rate)
            # message passing
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                D_e_neg_1 = torch.sparse.sum(P, dim=1).to_dense().view(-1, 1)
                D_e_neg_1[torch.isinf(D_e_neg_1)] = 0
                X = D_e_neg_1 * X
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method {aggr}.")
        return X

    def v2e_update(self, X: torch.Tensor, e_weight: Optional[torch.Tensor] = None):
        r"""Message update step of ``vertices to hyperedges``.

        Args:
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
        """
        if self.device != X.device:
            self.to(X.device)
        if e_weight is None:
            X = torch.sparse.mm(self.W_e, X)
        else:
            e_weight = e_weight.view(-1, 1)
            assert e_weight.shape[0] == self.num_e, "The size of e_weight must be equal to the size of self.num_e."
            X = e_weight * X
        return X

    def v2e_update_of_group(self, group_name: str, X: torch.Tensor, e_weight: Optional[torch.Tensor] = None):
        r"""Message update step of ``vertices to hyperedges`` in specified hyperedge group.

        Args:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.device != X.device:
            self.to(X.device)
        if e_weight is None:
            X = torch.sparse.mm(self.W_e_of_group(group_name), X)
        else:
            e_weight = e_weight.view(-1, 1)
            assert e_weight.shape[0] == self.num_e_of_group(
                group_name
            ), f"The size of e_weight must be equal to the size of self.num_e_of_group('{group_name}')."
            X = e_weight * X
        return X

    def v2e(
        self,
        X: torch.Tensor,
        aggr: str = "mean",
        v2e_weight: Optional[torch.Tensor] = None,
        e_weight: Optional[torch.Tensor] = None,
        drop_rate: float = 0.0,
    ):
        r"""Message passing of ``vertices to hyperedges``. The combination of ``v2e_aggregation`` and ``v2e_update``.

        Args:
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyepredges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        X = self.v2e_aggregation(X, aggr, v2e_weight, drop_rate=drop_rate)
        X = self.v2e_update(X, e_weight)
        return X

    def v2e_of_group(
        self,
        group_name: str,
        X: torch.Tensor,
        aggr: str = "mean",
        v2e_weight: Optional[torch.Tensor] = None,
        e_weight: Optional[torch.Tensor] = None,
        drop_rate: float = 0.0,
    ):
        r"""Message passing of ``vertices to hyperedges`` in specified hyperedge group. The combination of ``e2v_aggregation_of_group`` and ``e2v_update_of_group``.

        Args:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyepredges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        X = self.v2e_aggregation_of_group(group_name, X, aggr, v2e_weight, drop_rate=drop_rate)
        X = self.v2e_update_of_group(group_name, X, e_weight)
        return X

    def e2v_aggregation(
        self, X: torch.Tensor, aggr: str = "mean", e2v_weight: Optional[torch.Tensor] = None, drop_rate: float = 0.0
    ):
        r"""Message aggregation step of ``hyperedges to vertices``.

        Args:
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        assert aggr in ["mean", "sum", "softmax_then_sum"]
        if self.device != X.device:
            self.to(X.device)
        if e2v_weight is None:
            if drop_rate > 0.0:
                P = sparse_dropout(self.H, drop_rate)
            else:
                P = self.H
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                X = torch.sparse.mm(self.D_v_neg_1, X)
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method: {aggr}")
        else:
            # init message path
            assert (
                e2v_weight.shape[0] == self.e2v_weight.shape[0]
            ), "The size of e2v_weight must be equal to the size of self.e2v_weight."
            P = torch.sparse_coo_tensor(self.H._indices(), e2v_weight, self.H.shape, device=self.device)
            if drop_rate > 0.0:
                P = sparse_dropout(P, drop_rate)
            # message passing
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                D_v_neg_1 = torch.sparse.sum(P, dim=1).to_dense().view(-1, 1)
                D_v_neg_1[torch.isinf(D_v_neg_1)] = 0
                X = D_v_neg_1 * X
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method: {aggr}")
        return X

    def e2v_aggregation_of_group(
        self,
        group_name: str,
        X: torch.Tensor,
        aggr: str = "mean",
        e2v_weight: Optional[torch.Tensor] = None,
        drop_rate: float = 0.0,
    ):
        r"""Message aggregation step of ``hyperedges to vertices`` in specified hyperedge group.

        Args:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        assert aggr in ["mean", "sum", "softmax_then_sum"]
        if self.device != X.device:
            self.to(X.device)
        if e2v_weight is None:
            if drop_rate > 0.0:
                P = sparse_dropout(self.H_of_group(group_name), drop_rate)
            else:
                P = self.H_of_group(group_name)
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                X = torch.sparse.mm(self.D_v_neg_1_of_group(group_name), X)
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method: {aggr}")
        else:
            # init message path
            assert (
                e2v_weight.shape[0] == self.e2v_weight_of_group(group_name).shape[0]
            ), f"The size of e2v_weight must be equal to the size of self.e2v_weight_of_group('{group_name}')."
            P = torch.sparse_coo_tensor(
                self.H_of_group(group_name)._indices(),
                e2v_weight,
                self.H_of_group(group_name).shape,
                device=self.device,
            )
            if drop_rate > 0.0:
                P = sparse_dropout(P, drop_rate)
            # message passing
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                D_v_neg_1 = torch.sparse.sum(P, dim=1).to_dense().view(-1, 1)
                D_v_neg_1[torch.isinf(D_v_neg_1)] = 0
                X = D_v_neg_1 * X
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method: {aggr}")
        return X

    def e2v_update(self, X: torch.Tensor):
        r"""Message update step of ``hyperedges to vertices``.

        Args:
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
        """
        if self.device != X.device:
            self.to(X.device)
        return X

    def e2v_update_of_group(self, group_name: str, X: torch.Tensor):
        r"""Message update step of ``hyperedges to vertices`` in specified hyperedge group.

        Args:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.device != X.device:
            self.to(X.device)
        return X

    def e2v(
        self, X: torch.Tensor, aggr: str = "mean", e2v_weight: Optional[torch.Tensor] = None, drop_rate: float = 0.0,
    ):
        r"""Message passing of ``hyperedges to vertices``. The combination of ``e2v_aggregation`` and ``e2v_update``.

        Args:
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        X = self.e2v_aggregation(X, aggr, e2v_weight, drop_rate=drop_rate)
        X = self.e2v_update(X)
        return X

    def e2v_of_group(
        self,
        group_name: str,
        X: torch.Tensor,
        aggr: str = "mean",
        e2v_weight: Optional[torch.Tensor] = None,
        drop_rate: float = 0.0,
    ):
        r"""Message passing of ``hyperedges to vertices`` in specified hyperedge group. The combination of ``e2v_aggregation_of_group`` and ``e2v_update_of_group``.

        Args:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        X = self.e2v_aggregation_of_group(group_name, X, aggr, e2v_weight, drop_rate=drop_rate)
        X = self.e2v_update_of_group(group_name, X)
        return X

    def v2v(
        self,
        X: torch.Tensor,
        aggr: str = "mean",
        drop_rate: float = 0.0,
        v2e_aggr: Optional[str] = None,
        v2e_weight: Optional[torch.Tensor] = None,
        v2e_drop_rate: Optional[float] = None,
        e_weight: Optional[torch.Tensor] = None,
        e2v_aggr: Optional[str] = None,
        e2v_weight: Optional[torch.Tensor] = None,
        e2v_drop_rate: Optional[float] = None,
    ):
        r"""Message passing of ``vertices to vertices``. The combination of ``v2e`` and ``e2v``.

        Args:
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``. If specified, this ``aggr`` will be used to both ``v2e`` and ``e2v``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
            ``v2e_aggr`` (``str``, optional): The aggregation method for hyperedges to vertices. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``. If specified, it will override the ``aggr`` in ``e2v``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyepredges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``v2e_drop_rate`` (``float``, optional): Dropout rate for hyperedges to vertices. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. If specified, it will override the ``drop_rate`` in ``e2v``. Default: ``None``.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e2v_aggr`` (``str``, optional): The aggregation method for vertices to hyperedges. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``. If specified, it will override the ``aggr`` in ``v2e``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e2v_drop_rate`` (``float``, optional): Dropout rate for vertices to hyperedges. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. If specified, it will override the ``drop_rate`` in ``v2e``. Default: ``None``.
        """
        if v2e_aggr is None:
            v2e_aggr = aggr
        if e2v_aggr is None:
            e2v_aggr = aggr
        if v2e_drop_rate is None:
            v2e_drop_rate = drop_rate
        if e2v_drop_rate is None:
            e2v_drop_rate = drop_rate
        X = self.v2e(X, v2e_aggr, v2e_weight, e_weight, drop_rate=v2e_drop_rate)
        X = self.e2v(X, e2v_aggr, e2v_weight, drop_rate=e2v_drop_rate)
        return X

    def v2v_of_group(
        self,
        group_name: str,
        X: torch.Tensor,
        aggr: str = "mean",
        drop_rate: float = 0.0,
        v2e_aggr: Optional[str] = None,
        v2e_weight: Optional[torch.Tensor] = None,
        v2e_drop_rate: Optional[float] = None,
        e_weight: Optional[torch.Tensor] = None,
        e2v_aggr: Optional[str] = None,
        e2v_weight: Optional[torch.Tensor] = None,
        e2v_drop_rate: Optional[float] = None,
    ):
        r"""Message passing of ``vertices to vertices`` in specified hyperedge group. The combination of ``v2e_of_group`` and ``e2v_of_group``.

        Args:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``. If specified, this ``aggr`` will be used to both ``v2e_of_group`` and ``e2v_of_group``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
            ``v2e_aggr`` (``str``, optional): The aggregation method for hyperedges to vertices. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``. If specified, it will override the ``aggr`` in ``e2v_of_group``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyepredges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``v2e_drop_rate`` (``float``, optional): Dropout rate for hyperedges to vertices. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. If specified, it will override the ``drop_rate`` in ``e2v_of_group``. Default: ``None``.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e2v_aggr`` (``str``, optional): The aggregation method for vertices to hyperedges. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``. If specified, it will override the ``aggr`` in ``v2e_of_group``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e2v_drop_rate`` (``float``, optional): Dropout rate for vertices to hyperedges. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. If specified, it will override the ``drop_rate`` in ``v2e_of_group``. Default: ``None``.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if v2e_aggr is None:
            v2e_aggr = aggr
        if e2v_aggr is None:
            e2v_aggr = aggr
        if v2e_drop_rate is None:
            v2e_drop_rate = drop_rate
        if e2v_drop_rate is None:
            e2v_drop_rate = drop_rate
        X = self.v2e_of_group(group_name, X, v2e_aggr, v2e_weight, e_weight, drop_rate=v2e_drop_rate)
        X = self.e2v_of_group(group_name, X, e2v_aggr, e2v_weight, drop_rate=e2v_drop_rate)
        return X


class HGNNConv(nn.Module):
    r"""The HGNN convolution layer proposed in `Hypergraph Neural Networks <https://arxiv.org/pdf/1809.09401>`_ paper (AAAI 2019).
    Matrix Format:

    .. math::
        \mathbf{X}^{\prime} = \sigma \left( \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} 
        \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}} \mathbf{X} \mathbf{\Theta} \right).

    where :math:`\mathbf{X}` is the input vertex feature matrix, :math:`\mathbf{H}` is the hypergraph incidence matrix, 
    :math:`\mathbf{W}_e` is a diagonal hyperedge weight matrix, :math:`\mathbf{D}_v` is a diagonal vertex degree matrix, 
    :math:`\mathbf{D}_e` is a diagonal hyperedge degree matrix, :math:`\mathbf{\Theta}` is the learnable parameters.

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``out_channels`` (int): :math:`C_{out}` is the number of output channels.
        ``bias`` (``bool``): If set to ``False``, the layer will not learn the bias parameter. Defaults to ``True``.
        ``use_bn`` (``bool``): If set to ``True``, the layer will use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): If set to a positive number, the layer will use dropout. Defaults to ``0.5``.
        ``is_last`` (``bool``): If set to ``True``, the layer will not apply the final activation and dropout functions. Defaults to ``False``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, X: torch.Tensor, hg: Hypergraph) -> torch.Tensor:
        r"""The forward function.

        Args:
            X (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            hg (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        X = self.theta(X)
        X = hg.smoothing_with_HGNN(X)
        if not self.is_last:
            X = self.act(X)
            if self.bn is not None:
                X = self.bn(X)
            X = self.drop(X)
        return X

#############################################################################################
class NegativeHyperGraphConvDHG(nn.Module):
    """基于DHG的负聚合超图卷积 - 修正版本"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, X, hg: Hypergraph):
        # 获取关联矩阵 H (稀疏格式)
        H = hg.H
        
        # 获取度向量并确保是张量
        if hasattr(hg, 'deg_v'):
            deg_v = torch.tensor(hg.deg_v, device=X.device, dtype=X.dtype)
        else:
            deg_v = H.sum(dim=1)
            if isinstance(deg_v, torch.Tensor) and deg_v.is_sparse:
                deg_v = deg_v.to_dense()
        
        if hasattr(hg, 'deg_e'):
            deg_e = torch.tensor(hg.deg_e, device=X.device, dtype=X.dtype)
        else:
            deg_e = H.sum(dim=0)
            if isinstance(deg_e, torch.Tensor) and deg_e.is_sparse:
                deg_e = deg_e.to_dense()
        
        # 避免除零
        deg_v_inv = 1.0 / torch.clamp(deg_v, min=1e-8)
        deg_e_inv = 1.0 / torch.clamp(deg_e, min=1e-8)
        
        # 超边特征聚合: E = D_e^{-1} H^T X
        H_T = H.t()
        if H_T.is_sparse:
            H_T_X = torch.sparse.mm(H_T, X)
        else:
            H_T_X = torch.matmul(H_T, X)
        
        E = deg_e_inv.view(-1, 1) * H_T_X
        
        # 负向消息传递: A_negative = H E
        if H.is_sparse:
            A_negative = torch.sparse.mm(H, E)
        else:
            A_negative = torch.matmul(H, E)
        
        # 排斥更新: X' = X - D_v^{-1} A_negative
        X_update = X - deg_v_inv.view(-1, 1) * A_negative
        
        # 线性变换
        output = torch.matmul(X_update, self.weight)
        return output

class CosineRepulsionHyperGraphConvDHG(nn.Module):
    """基于DHG的余弦排斥超图卷积 - 修正版本"""
    def __init__(self, in_channels, out_channels, temperature=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temperature = temperature
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, X, hg: Hypergraph):
        # 获取关联矩阵 H
        H = hg.H
        
        # 特征归一化
        X_norm = F.normalize(X, p=2, dim=1)
        
        # 计算余弦相似度
        similarity = torch.matmul(X_norm, X_norm.T)
        
        # 构建超边连接掩码: M = (H H^T) > 0
        if H.is_sparse:
            # 对于稀疏矩阵，使用矩阵乘法得到邻接矩阵
            hyperedge_adj = torch.sparse.mm(H, H.t())
            # 转换为稠密并创建掩码
            if hyperedge_adj.is_sparse:
                hyperedge_adj_dense = hyperedge_adj.to_dense()
            else:
                hyperedge_adj_dense = hyperedge_adj
            mask = (hyperedge_adj_dense > 0).float()
        else:
            hyperedge_adj = torch.matmul(H, H.T)
            mask = (hyperedge_adj > 0).float()
        
        # 计算排斥权重
        masked_similarity = similarity * mask
        repulsion_weights = F.softmax(masked_similarity / self.temperature, dim=1)
        
        # 排斥更新
        repulsion_update = torch.matmul(repulsion_weights, X)
        X_update = X - repulsion_update
        
        # 线性变换
        output = torch.matmul(X_update, self.weight)
        return output

class FeatureDispersionHyperGraphConvDHG(nn.Module):
    """基于DHG的特征分散超图卷积 - 修正版本"""
    def __init__(self, in_channels, out_channels, alpha=0.5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = alpha
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, X, hg: Hypergraph):
        # 获取关联矩阵 H
        H = hg.H
        
        # 获取度向量并确保是张量
        if hasattr(hg, 'deg_v'):
            deg_v = torch.tensor(hg.deg_v, device=X.device, dtype=X.dtype)
        else:
            deg_v = H.sum(dim=1)
            if isinstance(deg_v, torch.Tensor) and deg_v.is_sparse:
                deg_v = deg_v.to_dense()
        
        if hasattr(hg, 'deg_e'):
            deg_e = torch.tensor(hg.deg_e, device=X.device, dtype=X.dtype)
        else:
            deg_e = H.sum(dim=0)
            if isinstance(deg_e, torch.Tensor) and deg_e.is_sparse:
                deg_e = deg_e.to_dense()
        
        # 避免除零
        deg_e_inv = 1.0 / torch.clamp(deg_e, min=1e-8)
        
        # 计算超边中心: C = D_e^{-1} H^T X
        H_T = H.t()
        if H_T.is_sparse:
            H_T_X = torch.sparse.mm(H_T, X)
        else:
            H_T_X = torch.matmul(H_T, X)
        
        C = deg_e_inv.view(-1, 1) * H_T_X
        
        # 计算分散方向: Δ = D_v X - H C
        D_v_X = deg_v.view(-1, 1) * X
        
        if H.is_sparse:
            H_C = torch.sparse.mm(H, C)
        else:
            H_C = torch.matmul(H, C)
        
        dispersion = D_v_X - H_C
        
        # 分散更新
        X_update = X + self.alpha * dispersion
        
        # 线性变换
        output = torch.matmul(X_update, self.weight)
        return output