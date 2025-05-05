import torch


def naive_vector_dot(vector_a, vector_b):
    assert len(vector_a.shape) == 1
    assert len(vector_b.shape) == 1
    assert vector_a.shape[0] == vector_b.shape[0]
    z = 0.0
    for i in range(vector_a.shape[0]):
        z += vector_a[i] * vector_b[i]
    return z


a = torch.tensor([1, 2, 3, 4, 5])
b = torch.tensor([1, 2, 3, 4, 5])
assert naive_vector_dot(a, b) == 55
assert naive_vector_dot(a, b) == torch.matmul(a, b)


def naive_matrix_vector_dot(matrix_a, vector_b):
    assert len(matrix_a.shape) == 2
    assert len(vector_b.shape) == 1
    assert matrix_a.shape[1] == vector_b.shape[0]
    result = torch.zeros(matrix_a.shape[0])
    for i in range(matrix_a.shape[0]):
        for j in range(matrix_a.shape[1]):
            result[i] += matrix_a[i, j] * vector_b[j]
    return result


c = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
d = torch.tensor([1, 2, 3, 4, 5])
assert (naive_matrix_vector_dot(c, d) == torch.tensor([55, 130])).all()
assert (naive_matrix_vector_dot(c, d) == torch.matmul(c, d)).all()


def naive_matrix_dot(matrix_a, matrix_b):
    assert len(matrix_a.shape) == 2
    assert len(matrix_b.shape) == 2
    assert matrix_a.shape[1] == matrix_b.shape[0]
    result = torch.zeros((matrix_a.shape[0], matrix_b.shape[1]))
    for i in range(matrix_a.shape[0]):
        for j in range(matrix_b.shape[1]):
            row_x = matrix_a[i, :]
            column_y = matrix_b[:, j]
            result[i, j] = naive_vector_dot(row_x, column_y)
    return result


e = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
f = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
assert (naive_matrix_dot(e, f) == torch.tensor([[95, 110], [220, 260]])).all()
assert (naive_matrix_dot(e, f) == torch.matmul(e, f)).all()
