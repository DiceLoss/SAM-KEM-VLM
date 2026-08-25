import numpy as np

from sam_kem.operators import CallableOperator, relative_adjoint_error
from sam_kem.reconstruction import bowsher_kernel, identity_kernel, kem, mlem, osem


def poisson_nll(measured, expected):
  expectation = np.maximum(np.asarray(expected, dtype=float), 1e-12)
  return float(np.sum(expectation - measured * np.log(expectation)))


def matrix_operator(matrix, image_shape, projection_shape):
  return CallableOperator(
    image_shape=image_shape,
    projection_shape=projection_shape,
    operator_id="test_matrix",
    forward_fn=lambda x: (matrix @ x.ravel()).reshape(projection_shape),
    adjoint_fn=lambda y: (matrix.T @ y.ravel()).reshape(image_shape),
  )


def test_adjoint_and_identity_kem_matches_mlem():
  matrix = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
  ], dtype=float)
  operator = matrix_operator(matrix, (2, 2), (2, 2))
  assert relative_adjoint_error(operator) < 1e-12
  measured = np.array([[8, 12], [9, 11]], dtype=float)
  em = mlem(measured, operator, iterations=5)[5]
  kernel_em = kem(measured, operator, identity_kernel((2, 2)), iterations=5)[5]
  np.testing.assert_allclose(em, kernel_em, rtol=1e-6, atol=1e-6)


def test_mlem_does_not_unit_normalize():
  matrix = np.eye(4)
  operator = matrix_operator(matrix, (2, 2), (2, 2))
  measured = np.array([[2, 4], [6, 8]], dtype=float)
  result = mlem(measured, operator, iterations=1)[1]
  np.testing.assert_allclose(result, measured)
  assert result.max() == 8


def test_mlem_poisson_objective_is_monotone_for_matched_model():
  matrix = np.array([
    [1.0, 0.2, 0.0, 0.0],
    [0.0, 1.0, 0.3, 0.0],
    [0.0, 0.0, 1.0, 0.4],
    [0.5, 0.0, 0.0, 1.0],
  ])
  operator = matrix_operator(matrix, (2, 2), (2, 2))
  measured = np.array([[7.0, 11.0], [5.0, 9.0]])
  iterates = mlem(
    measured, operator, iterations=12, save_iterations=range(13),
  )
  objectives = [poisson_nll(measured, operator.forward(iterates[i])) for i in range(13)]
  assert np.all(np.diff(objectives) <= 1e-7)


def test_projection_gain_is_used_in_mlem_scale():
  operator = matrix_operator(np.eye(4), (2, 2), (2, 2))
  measured = np.array([[20.0, 40.0], [60.0, 80.0]])
  gain = np.full((2, 2), 10.0)
  result = mlem(measured, operator, iterations=1, multiplicative=gain)[1]
  np.testing.assert_allclose(result, measured / gain)


def test_one_subset_osem_matches_mlem():
  matrix = np.array([
    [1.0, 0.0, 0.2, 0.0],
    [0.0, 1.0, 0.0, 0.3],
    [0.4, 0.0, 1.0, 0.0],
    [0.0, 0.5, 0.0, 1.0],
  ])
  operator = matrix_operator(matrix, (2, 2), (4, 1))
  measured = np.array([[8.0], [12.0], [9.0], [11.0]])
  em = mlem(measured, operator, iterations=6)[6]
  ordered = osem(measured, operator, iterations=6, subsets=1)[6]
  np.testing.assert_allclose(em, ordered, rtol=1e-6, atol=1e-6)


def test_bowsher_kem_kernel_is_nonnegative_and_row_normalized():
  feature = np.arange(16, dtype=float).reshape(4, 4)
  kernel = bowsher_kernel(feature, neighbours=3, radius=1)
  assert np.all(kernel.data >= 0)
  np.testing.assert_allclose(np.asarray(kernel.sum(axis=1)).ravel(), 1.0)
