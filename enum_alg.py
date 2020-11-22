from typing import List, Dict, Set
from functools import reduce


class Evidence:
    def __init__(self, idx: int, value: int):
        self.idx = idx
        self.value = value

    def __eq__(self, other):
        return self.idx == other.idx and self.value == other.value

    def __hash__(self):
        return hash(str(self.idx) + " " + str(self.value))


class Condition:
    def __init__(self, for_value: int, evidences: Set[Evidence]):
        self.for_value = for_value
        self.evidences = evidences

    def __eq__(self, obj):
        return isinstance(obj, Condition) \
               and obj.for_value == self.for_value \
               and obj.evidences == self.evidences

    def __str__(self):
        return str(self.for_value) + str(reduce(lambda a, b: a + b, [], list(map(lambda a: str(a), self.evidences))))

    def __hash__(self):
        return hash(str(self))

    def is_part_of(self, condition) -> bool:
        if self.for_value != condition.for_value:
            return False
        return self.evidences.issubset(condition.evidences)


class Node:
    idx: int
    dif_values: int
    parent_ids: List[int]
    probabilities: Dict[Condition, float]

    def __init__(self, idx: int, dif_values: int, parent_ids, probabilities: Dict[Condition, float]):
        self.idx = idx
        self.dif_values = dif_values
        self.parent_ids = parent_ids
        self.probabilities = probabilities

    def __eq__(self, other):
        return self.idx == other.idx

    def remove_cond_ind(self, condition: Condition) -> Condition:
        new_evidences = set(filter(lambda x: x.idx in self.parent_ids, condition.evidences))
        return Condition(condition.for_value, new_evidences)

    def get_prob(self, condition: Condition) -> float:
        if condition in self.probabilities:
            return self.probabilities[condition]

        min_condition = self.remove_cond_ind(condition)
        P_sum: float = 0

        for fix_condition in self.probabilities.keys():
            if min_condition.is_part_of(fix_condition):
                P_sum += self.probabilities[fix_condition]

        return P_sum


class Net:
    node_dict: Dict[int, Node]
    nodes: List[Node]

    def __init__(self, node_dict):
        self.node_dict = node_dict
        self.nodes = [node_dict[k] for k in node_dict.keys()]

    def node_from_idx(self, idx: int) -> Node:
        return self.node_dict[idx]


def enum_all(nodes: List[Node], evidences: Set[Evidence]) -> float:
    if len(nodes) == 0:
        return 1.0

    new_nodes = nodes.copy()
    Y = new_nodes.pop(0)

    for iv in evidences:
        if iv.idx == Y.idx:
            return Y.get_prob(Condition(iv.value, evidences)) * enum_all(new_nodes, evidences)

    P_sum = 0.0
    for yi in range(0, Y.dif_values):
        P_sum += Y.get_prob(Condition(yi, evidences)) \
                    * enum_all(new_nodes, set().union(evidences, {Evidence(Y.idx, yi)}))
    return P_sum


def normalize(Q: List[float]) -> List[float]:
    norm_const = 1 / reduce(lambda a, b: a + b, Q)
    return list(map(lambda a: norm_const * a, Q))


def enum_ask(queried_idx: int, evidences: Set[Evidence], net: Net) -> List[float]:
    Q = []
    queried_node = net.node_from_idx(queried_idx)

    for value in range(0, queried_node.dif_values):
        new_evidences = set().union(evidences, {Evidence(queried_node.idx, value)})
        Q.append(enum_all(net.nodes, new_evidences))

    return normalize(Q)