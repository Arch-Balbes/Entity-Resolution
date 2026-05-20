from __future__ import annotations

import pandas as pd


class UnionFind:
    def __init__(self, values: list[str]):
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in values}
        self.size = {value: 1 for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str, max_component_size: int | None = None) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return True
        if max_component_size is not None and self.size[left_root] + self.size[right_root] > max_component_size:
            return False
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return True


def build_components(
    profile_ids: list[str],
    edges: pd.DataFrame,
    max_component_size: int | None = None,
) -> tuple[dict[str, list[str]], pd.DataFrame, pd.DataFrame]:
    union_find = UnionFind(profile_ids)
    accepted_edges = []
    skipped_edges = []

    cols = ["left_profile_id", "right_profile_id"]
    if "score" in edges.columns:
        cols.append("score")
    edge_iter = edges
    if "score" in edges.columns:
        edge_iter = edges.sort_values("score", ascending=False)

    for row in edge_iter[cols].itertuples(index=False):
        score = getattr(row, "score", None) if len(cols) > 2 else None
        accepted = union_find.union(row.left_profile_id, row.right_profile_id, max_component_size)
        edge_record = {
            "left_profile_id": row.left_profile_id,
            "right_profile_id": row.right_profile_id,
            "score": score,
        }
        if accepted:
            accepted_edges.append(edge_record)
        else:
            skipped_edges.append(edge_record)

    components: dict[str, list[str]] = {}
    for profile_id in profile_ids:
        components.setdefault(union_find.find(profile_id), []).append(profile_id)
    return components, pd.DataFrame(accepted_edges), pd.DataFrame(skipped_edges)
