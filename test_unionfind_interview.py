import unittest
from pythonchecker import UnionFind


class TestInit(unittest.TestCase):

    def test_component_count(self):
        uf = UnionFind(5)
        self.assertEqual(uf.components, 5)

    def test_each_node_is_own_root(self):
        uf = UnionFind(5)
        for i in range(5):
            self.assertEqual(uf.find(i), i)


class TestUnion(unittest.TestCase):

    def setUp(self):
        self.uf = UnionFind(5)

    def test_union_connects_nodes(self):
        self.uf.union(0, 1)
        self.assertTrue(self.uf.connected(0, 1))

    def test_union_returns_true_on_new_merge(self):
        self.assertTrue(self.uf.union(0, 1))

    def test_union_returns_false_if_already_connected(self):
        self.uf.union(0, 1)
        self.assertFalse(self.uf.union(0, 1))

    def test_union_same_node_is_no_op(self):
        self.assertFalse(self.uf.union(2, 2))
        self.assertEqual(self.uf.components, 5)

    def test_union_decrements_components(self):
        self.uf.union(0, 1)
        self.assertEqual(self.uf.components, 4)

    def test_union_all_nodes_gives_one_component(self):
        for i in range(4):
            self.uf.union(i, i + 1)
        self.assertEqual(self.uf.components, 1)


class TestConnected(unittest.TestCase):

    def setUp(self):
        self.uf = UnionFind(5)

    def test_node_connected_to_itself(self):
        self.assertTrue(self.uf.connected(0, 0))

    def test_unconnected_nodes(self):
        self.assertFalse(self.uf.connected(0, 1))

    def test_transitivity(self):
        self.uf.union(0, 1)
        self.uf.union(1, 2)
        self.assertTrue(self.uf.connected(0, 2))

    def test_no_cross_component_connection(self):
        self.uf.union(0, 1)
        self.uf.union(2, 3)
        self.assertFalse(self.uf.connected(0, 2))

    def test_symmetry(self):
        self.uf.union(0, 4)
        self.assertEqual(self.uf.connected(0, 4), self.uf.connected(4, 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
