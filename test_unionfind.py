import pytest
from pythonchecker import UnionFind


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def uf5():
    """Fresh UnionFind over 5 nodes."""
    return UnionFind(5)


@pytest.fixture
def uf10():
    """Fresh UnionFind over 10 nodes."""
    return UnionFind(10)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_component_count_equals_n(self):
        assert UnionFind(7).components == 7

    def test_every_node_is_its_own_root(self, uf5):
        assert all(uf5.find(i) == i for i in range(5))

    def test_no_nodes_connected_initially(self, uf5):
        for i in range(5):
            for j in range(5):
                if i != j:
                    assert not uf5.connected(i, j)

    @pytest.mark.parametrize("n", [1, 2, 100, 10_000])
    def test_various_sizes(self, n):
        uf = UnionFind(n)
        assert uf.components == n


# ---------------------------------------------------------------------------
# find / path compression
# ---------------------------------------------------------------------------

class TestFind:
    def test_find_returns_self_for_isolated_node(self, uf5):
        assert uf5.find(0) == 0

    def test_find_is_idempotent(self, uf5):
        uf5.union(0, 1)
        root = uf5.find(0)
        assert uf5.find(0) == root

    def test_path_compression_flattens_tree(self, uf10):
        # Build a chain 0-1-2-3-4 so the tree is deep before compression
        for i in range(4):
            uf10.parent[i + 1] = i   # manually create chain without rank logic
        root = uf10.find(4)
        # After find, every node on the path should point directly to root
        for i in range(5):
            assert uf10.parent[i] == root


# ---------------------------------------------------------------------------
# union
# ---------------------------------------------------------------------------

class TestUnion:
    def test_union_connects_two_nodes(self, uf5):
        uf5.union(0, 1)
        assert uf5.connected(0, 1)

    def test_union_returns_true_on_new_merge(self, uf5):
        assert uf5.union(0, 1) is True

    def test_union_returns_false_when_already_connected(self, uf5):
        uf5.union(0, 1)
        assert uf5.union(0, 1) is False

    def test_union_same_node_is_no_op(self, uf5):
        assert uf5.union(2, 2) is False
        assert uf5.components == 5

    def test_union_decrements_component_count(self, uf5):
        uf5.union(0, 1)
        assert uf5.components == 4
        uf5.union(2, 3)
        assert uf5.components == 3

    def test_union_redundant_does_not_change_component_count(self, uf5):
        uf5.union(0, 1)
        before = uf5.components
        uf5.union(0, 1)
        assert uf5.components == before

    def test_union_all_nodes_into_one_component(self, uf5):
        for i in range(4):
            uf5.union(i, i + 1)
        assert uf5.components == 1

    @pytest.mark.parametrize("a,b", [(0, 4), (1, 3), (2, 4)])
    def test_union_various_pairs(self, uf5, a, b):
        uf5.union(a, b)
        assert uf5.connected(a, b)


# ---------------------------------------------------------------------------
# connected / transitivity
# ---------------------------------------------------------------------------

class TestConnected:
    def test_node_is_connected_to_itself(self, uf5):
        assert uf5.connected(3, 3)

    def test_transitivity(self, uf5):
        uf5.union(0, 1)
        uf5.union(1, 2)
        assert uf5.connected(0, 2)

    def test_no_cross_component_connectivity(self, uf5):
        uf5.union(0, 1)
        uf5.union(2, 3)
        assert not uf5.connected(0, 2)
        assert not uf5.connected(1, 3)

    def test_connected_is_symmetric(self, uf5):
        uf5.union(0, 4)
        assert uf5.connected(0, 4) == uf5.connected(4, 0)


# ---------------------------------------------------------------------------
# Larger / stress scenarios
# ---------------------------------------------------------------------------

class TestStress:
    def test_full_graph_single_component(self):
        n = 1000
        uf = UnionFind(n)
        for i in range(n - 1):
            uf.union(i, i + 1)
        assert uf.components == 1
        assert uf.connected(0, n - 1)

    def test_two_large_independent_components(self):
        n = 1000
        uf = UnionFind(n)
        for i in range(0, n // 2 - 1):
            uf.union(i, i + 1)
        for i in range(n // 2, n - 1):
            uf.union(i, i + 1)
        assert uf.components == 2
        assert not uf.connected(0, n - 1)

    def test_merging_two_large_components(self):
        n = 1000
        uf = UnionFind(n)
        for i in range(0, n // 2 - 1):
            uf.union(i, i + 1)
        for i in range(n // 2, n - 1):
            uf.union(i, i + 1)
        uf.union(0, n - 1)
        assert uf.components == 1
