from types import SimpleNamespace
import unittest

import numpy as np

import lattice_site_v9 as lattice


def shell_args(**overrides):
    values = {
        "energy_density": 0.0,
        "energy_width": 1.0,
        "Dynamics": "Ising",
        "mu": 0.0,
        "beta": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def shell_haar_args(**overrides):
    values = {
        "Dynamics": "shell-haar",
        "energy_density": None,
        "energy_width": 0.5,
        "mu": 0.0,
        "beta": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TestEnergyShell(unittest.TestCase):
  def test_energy_shell_labels_are_zero_centered_and_contiguous(self):
    labels = lattice._energy_shell_labels(
        np.array([-1.49, -0.5, -0.49, 0.0, 0.49, 0.5, 1.49]),
        nsites=1,
        energy_width=1.0,
    )

    np.testing.assert_array_equal(labels, [-1, 0, 0, 0, 0, 1, 1])


  def test_shell_haar_is_unitary_and_commutes_with_shell_projectors(self):
    hamiltonian = np.diag([-1.2, -0.9, 0.1, 0.3, 1.1, 1.4])
    unitary, labels, dimensions = lattice.shell_haar_unitary(
        hamiltonian, nsites=1, energy_width=1.0, rng=np.random.default_rng(4)
    )
    energy_basis_unitary = unitary
    different_shells = labels[:, None] != labels[None, :]

    np.testing.assert_allclose(unitary.conj().T @ unitary, np.eye(6), atol=1e-12)
    np.testing.assert_allclose(energy_basis_unitary[different_shells], 0.0, atol=1e-12)
    for shell in np.unique(labels):
      projector = np.diag(labels == shell)
      np.testing.assert_allclose(unitary @ projector, projector @ unitary, atol=1e-12)
    self.assertEqual(sum(size for _, size in dimensions), 6)
    self.assertGreater(len(dimensions), 1)


  def test_shell_haar_seed_controls_independent_block_draws(self):
    hamiltonian = np.diag([-1.2, -0.9, 0.1, 0.3, 1.1, 1.4])
    first, first_labels, _ = lattice.shell_haar_unitary(
        hamiltonian, nsites=1, energy_width=1.0, rng=np.random.default_rng(4)
    )
    repeat, repeat_labels, _ = lattice.shell_haar_unitary(
        hamiltonian, nsites=1, energy_width=1.0, rng=np.random.default_rng(4)
    )
    different, different_labels, _ = lattice.shell_haar_unitary(
        hamiltonian, nsites=1, energy_width=1.0, rng=np.random.default_rng(5)
    )

    np.testing.assert_array_equal(first_labels, repeat_labels)
    np.testing.assert_array_equal(first_labels, different_labels)
    np.testing.assert_allclose(first, repeat)
    self.assertFalse(np.allclose(first, different))


  def test_shell_haar_validation(self):
    cases = [
        (shell_haar_args(energy_width=None), "requires --energy_width"),
        (shell_haar_args(energy_width=0.0), "energy_width > 0"),
        (shell_haar_args(energy_density=0.0, mu=0.2), "cannot be combined"),
    ]
    for args, message in cases:
      with self.subTest(args=args):
        with self.assertRaisesRegex(ValueError, message):
          lattice._validate_energy_shell_args(args)
    self.assertFalse(
        lattice._validate_energy_shell_args(shell_haar_args(energy_density=0.0))
    )


  def test_beta_validation_is_restricted_to_ising_and_shell_haar(self):
    with self.assertRaisesRegex(ValueError, "only for Ising and shell-haar"):
      lattice._validate_energy_shell_args(shell_args(Dynamics="GOE", beta=1.0))
    with self.assertRaisesRegex(ValueError, "only for Ising and shell-haar"):
      lattice._validate_energy_shell_args(shell_args(Dynamics="Kicked", beta=1.0))


  def test_shell_haar_beta_weights_use_shell_centers_and_dimensions(self):
    labels = np.array([-2, -1, -1, 0, 0, 0, 1, 1, 2])
    vectors = np.eye(len(labels), dtype=complex)
    metadata = {"shell_labels": labels, "hamiltonian_vectors": vectors}
    beta = 0.7
    rho, energies, dimensions, factors, partition = lattice._shell_haar_thermal_density(
        metadata, nsites=1, energy_width=1.0, beta=beta
    )
    expected_factors = np.exp(-beta * np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    expected_partition = np.dot([1, 2, 3, 2, 1], expected_factors)
    expected_weights = np.repeat(expected_factors, [1, 2, 3, 2, 1]) / expected_partition

    np.testing.assert_allclose(energies, [-2, -1, 0, 1, 2])
    np.testing.assert_array_equal(dimensions, [1, 2, 3, 2, 1])
    np.testing.assert_allclose(factors, expected_factors)
    self.assertAlmostEqual(partition, expected_partition)
    np.testing.assert_allclose(np.diag(rho), expected_weights)
    self.assertAlmostEqual(np.trace(rho), 1.0)


  def test_shell_haar_beta_zero_is_uniform(self):
    labels = np.array([-1, -1, 0, 1])
    metadata = {
        "shell_labels": labels,
        "hamiltonian_vectors": np.eye(4, dtype=complex),
    }
    rho, _, _, _, _ = lattice._shell_haar_thermal_density(
        metadata, nsites=2, energy_width=0.5, beta=0.0
    )
    np.testing.assert_allclose(rho, np.eye(4) / 4)


  def test_shell_haar_build_dynamics_returns_unitary_matrix(self):
    args = SimpleNamespace(
        Dynamics="shell-haar", energy_width=0.5, tau=1.0, Periodic="N",
        hx=0.4, hz=0.7, g=0.1, disorder="Z",
    )
    xops = [lattice.local_op(lattice.X, 1, i, 2) for i in range(2)]
    zops = [lattice.local_op(lattice.Z, 1, i, 2) for i in range(2)]
    dynamics, unitary, metadata = lattice.build_dynamics(
        args, dim=4, rng=np.random.default_rng(2),
        xops=xops, yops=[], zops=zops,
        return_metadata=True,
    )

    self.assertTrue(unitary)
    np.testing.assert_allclose(dynamics.conj().T @ dynamics, np.eye(4), atol=1e-12)
    self.assertEqual(len(metadata["hamiltonian_energies"]), 4)
    self.assertEqual(sum(size for _, size in metadata["shell_dimensions"]), 4)


  def test_analysis_shell_uses_hamiltonian_energies_not_unitary_phases(self):
    args = shell_haar_args(energy_density=0.5, energy_width=0.5)
    args.ortho_obs = "N"
    identity = np.eye(4, dtype=complex)
    analysis_rho, selected_count, target, lower, upper = lattice._microcanonical_projector(
        np.array([0.0, 1.0, 2.0, 3.0]),
        identity,
        nsites=2,
        energy_density=0.5,
        energy_width=0.5,
    )
    _, weights, shell_count = lattice.run_realization(
        args,
        np.diag(np.exp(1j * np.array([0.0, 0.2, 0.4, 0.6]))),
        True,
        identity / 4,
        [identity],
        identity,
        np.array([0.0]),
        site_start=1,
        analysis_rho=analysis_rho,
        analysis_shell_count=selected_count,
        return_shell_count=True,
    )

    self.assertEqual((target, lower, upper), (1.0, 0.5, 1.5))
    self.assertEqual(selected_count, 1)
    self.assertEqual(shell_count, 1)
    np.testing.assert_allclose(weights[0, 0], [0.0, 1.0, 0.0, 0.0])


  def test_microcanonical_weights_scale_density_and_include_boundaries(self):
    weights, target, lower, upper = lattice._microcanonical_weights(
        np.array([-1.0, 1.0, 2.0, 3.0]),
        nsites=2,
        energy_density=0.75,
        energy_width=0.5,
    )

    self.assertAlmostEqual(target, 1.5)
    self.assertAlmostEqual(lower, 1.0)
    self.assertAlmostEqual(upper, 2.0)
    np.testing.assert_allclose(weights, [0.0, 0.5, 0.5, 0.0])
    self.assertAlmostEqual(weights.sum(), 1.0)


  def test_microcanonical_weights_empty_shell_reports_nearest_energies(self):
    with self.assertRaisesRegex(ValueError, "Energy shell is empty.*nearest energies"):
        lattice._microcanonical_weights(
            np.array([-1.0, 1.0, 2.0]),
            nsites=2,
            energy_density=2.0,
            energy_width=0.1,
            realization=7,
        )


  def test_energy_shell_argument_validation(self):
    cases = [
        ({"energy_width": None}, "must be supplied together"),
        ({"energy_density": None}, "must be supplied together"),
        ({"energy_width": 0.0}, "energy_width > 0"),
        ({"Dynamics": "Kicked"}, "only for -D Ising"),
        ({"mu": 0.2}, "cannot be combined"),
    ]
    for overrides, message in cases:
      with self.subTest(overrides=overrides):
        with self.assertRaisesRegex(ValueError, message):
          lattice._validate_energy_shell_args(shell_args(**overrides))


  def test_shell_run_uses_only_selected_eigenstates(self):
    args = shell_args(energy_density=0.75, energy_width=0.5, ortho_obs="N")
    identity = np.eye(4, dtype=complex)
    correlators, weights, selected_count = lattice.run_realization(
        args,
        np.diag([0.0, 1.0, 2.0, 3.0]),
        False,
        None,
        [identity],
        identity,
        np.array([0.0]),
        site_start=1,
        nsites=2,
        realization=0,
        return_shell_count=True,
    )

    self.assertEqual(correlators.shape, (1, 1, 4))
    self.assertEqual(selected_count, 2)
    np.testing.assert_allclose(weights[0, 0], [0.0, 0.5, 0.5, 0.0])
    self.assertAlmostEqual(np.sum(weights[0, 0]), 1.0)


  def test_run_without_shell_preserves_full_space_ensemble(self):
    args = shell_args(energy_density=None, energy_width=None, ortho_obs="N")
    identity = np.eye(2, dtype=complex)
    rho = identity / 2
    _, weights, selected_count = lattice.run_realization(
        args,
        np.diag([0.0, 1.0]),
        False,
        rho,
        [identity],
        identity,
        np.array([0.0]),
        site_start=1,
        return_shell_count=True,
    )

    self.assertIsNone(selected_count)
    np.testing.assert_allclose(weights[0, 0], [0.5, 0.5])


  def test_ising_beta_uses_exact_hamiltonian_energies(self):
    args = shell_args(
        beta=0.7, energy_density=None, energy_width=None, ortho_obs="N"
    )
    identity = np.eye(3, dtype=complex)
    _, weights = lattice.run_realization(
        args,
        np.diag([-1.0, 0.0, 2.0]),
        False,
        identity / 3,
        [identity],
        identity,
        np.array([0.0]),
        site_start=1,
    )
    factors = np.exp(-0.7 * np.array([-1.0, 0.0, 2.0]))
    np.testing.assert_allclose(weights[0, 0], factors / factors.sum())


if __name__ == "__main__":
  unittest.main()
