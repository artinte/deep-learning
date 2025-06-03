import unittest


class GraphAdjMat:
    def __init__(self, vertices, edges):
        # Vertex list, elements represent "vertex value", index represents "vertex index".
        self.vertices = []
        # Adjacency matrix, row and column indices correspond to "vertex index".
        self.adj_mat = []

        for val in vertices:
            self.add_vertex(val)

        for e in edges:
            self.add_edge(e[0], e[1])

    def size(self):
        return len(self.vertices)

    def add_vertex(self, val):
        n = self.size()
        self.vertices.append(val)
        # Add a worw to the adjacency matrix
        new_row = [0] * n
        self.adj_mat.append(new_row)
        # Add a column to the adjacency matrix
        for row in self.adj_mat:
            row.append(0)

    def remove_vertex(self, index):
        if index >= self.size():
            raise IndexError()
        self.vertices.pop(index)
        self.adj_mat.pop(index)
        for row in self.adj_mat:
            row.pop(index)

    def add_edge(self, i, j):
        if i < 0 or j < 0 or i >= self.size() or j >= self.size() or i == j:
            raise IndexError()

        self.adj_mat[i][j] = 1
        self.adj_mat[j][i] = 1

    def remove_edge(self, i, j):
        if i < 0 or j < 0 or i >= self.size() or j >= self.size() or i == j:
            raise IndexError()
        self.adj_mat[i][j] = 0
        self.adj_mat[j][i] = 0

    def print(self):
        print("Vertex list =", self.vertices)
        print("Adjacency matrix =")
        print_matrix(self.adj_mat)


class TestGraphAdjMat(unittest.TestCase):
    def test_add_vertex(self):
        g = GraphAdjMat([], [])
        g.add_vertex("A")
        g.add_vertex("B")
        self.assertEqual(g.vertices, ["A", "B"])
        self.assertEqual(g.adj_mat, [[0, 0], [0, 0]])

    def test_add_edge(self):
        g = GraphAdjMat(["A", "B", "C"], [])
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        self.assertEqual(g.adj_mat[0][1], 1)
        self.assertEqual(g.adj_mat[1][0], 1)
        self.assertEqual(g.adj_mat[1][2], 1)
        self.assertEqual(g.adj_mat[2][1], 1)
    
    def test_remove_edge(self):
        g = GraphAdjMat(["A", "B"], [(0, 1)])
        g.remove_edge(0, 1)
        self.assertEqual(g.adj_mat[0][1], 0)
        self.assertEqual(g.adj_mat[1][0], 0)

    def test_remove_vertex(self):
        g = GraphAdjMat(["A", "B", "C"], [(0, 1), (1, 2)])
        g.remove_vertex(1)  # remove "B"
        self.assertEqual(g.vertices, ["A", "C"])
        self.assertEqual(len(g.adj_mat), 2)
        self.assertEqual(len(g.adj_mat[0]), 2)

unittest.main()
