"""Tests for UnionFind data structure."""

from corp_entity_db.store import UnionFind


class TestUnionFind:
    def test_single_elements(self):
        uf = UnionFind([1, 2, 3])
        assert uf.find(1) == 1
        assert uf.find(2) == 2
        assert uf.find(3) == 3

    def test_basic_union(self):
        uf = UnionFind([1, 2, 3])
        uf.union(1, 2)
        assert uf.find(1) == uf.find(2)
        assert uf.find(3) != uf.find(1)

    def test_path_compression(self):
        uf = UnionFind([1, 2, 3, 4])
        uf.union(1, 2)
        uf.union(2, 3)
        uf.union(3, 4)
        root = uf.find(4)
        # After path compression, parent should point directly to root
        assert uf.parent[4] == root

    def test_groups(self):
        uf = UnionFind([1, 2, 3, 4])
        uf.union(1, 2)
        uf.union(3, 4)
        groups = uf.groups()
        assert len(groups) == 2
        group_sizes = sorted(len(v) for v in groups.values())
        assert group_sizes == [2, 2]

    def test_union_same_element(self):
        uf = UnionFind([1, 2])
        uf.union(1, 1)
        assert uf.find(1) == 1
        assert len(uf.groups()) == 2
