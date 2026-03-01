class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        # union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        self.components -= 1
        return True

    def connected(self, x, y):
        return self.find(x) == self.find(y)


if __name__ == "__main__":
    uf = UnionFind(5)

    # basic union
    uf.union(0, 1)
    assert uf.connected(0, 1), "0 and 1 should be connected"
    assert not uf.connected(0, 2), "0 and 2 should not be connected"

    # transitivity
    uf.union(1, 2)
    assert uf.connected(0, 2), "0 and 2 should be connected transitively"

    # component count
    assert uf.components == 3, f"Expected 3 components, got {uf.components}"

    print("All tests passed!")
