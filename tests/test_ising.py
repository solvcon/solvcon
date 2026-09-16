# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import unittest

import numpy as np

import solvcon as sc

# FIXME: The number and solution in this file have not been validated.


def build_hamiltonian(L, J, hx, periodic=True):
    # Bit i of the state index is spin i, 0 for up and 1 for down.  The
    # coupling counts agreeing minus disagreeing bonds on the diagonal; the
    # field connects each state to the L states one flip away.
    n = 1 << L
    H = np.zeros((n, n), dtype='float64')
    nbond = L if periodic else L - 1
    for s in range(n):
        for i in range(nbond):
            j = (i + 1) % L
            H[s, s] += J * (1 - 2 * (((s >> i) & 1) ^ ((s >> j) & 1)))
        for i in range(L):
            H[s ^ (1 << i), s] -= hx
    return H


def exact_energy(L, J, hx):
    # Jordan-Wigner ground energy of the periodic ring; valid for even L.
    k = np.pi * (2 * np.arange(L, dtype='float64') + 1) / L
    return -np.sum(np.sqrt(J**2 + hx**2 - 2 * J * hx * np.cos(k)))


def observables(L, psi):
    states = np.arange(1 << L, dtype='int64')

    def sz(i):
        return 1 - 2 * ((states >> i) & 1)

    mx = sum(psi @ psi[states ^ (1 << i)] for i in range(L)) / L
    corr = [np.mean([psi @ (sz(i) * sz((i + r) % L) * psi)
                     for i in range(L)]) for r in range(L)]
    mz = [psi @ (sz(i) * psi) for i in range(L)]
    return mx, corr, mz


@unittest.skipIf(sc.EigenSystem is None,
                 "sc.EigenSystem is not built (no vendor LAPACK)")
class TestIsingChainTC(unittest.TestCase):
    """Solve the four-spin transverse-field Ising ring by dense
    diagonalization, as the application page does, and pin the result to
    the closed-form energy of the periodic chain.
    """

    L, J, HX = 4, 1.0, 0.3
    # Closed form for L = 4, J = 1, hx = 0.3: -2 sqrt(2.18 + 2 sqrt(1.0081)).
    E0 = -4.092961599426859

    def _solve(self, H):
        solver = sc.EigenSystem(sc.SimpleArray(H), do_vl=False)
        solver.run()
        wr = np.array(solver.wr, dtype='float64')
        wi = np.array(solver.wi, dtype='float64')
        np.testing.assert_allclose(wi, 0.0, atol=1e-12)
        order = np.argsort(wr)
        psi = np.array(solver.vr, dtype='float64')[:, order[0]]
        return wr[order], psi / np.linalg.norm(psi)

    def test_hamiltonian_structure(self):
        H = build_hamiltonian(self.L, self.J, self.HX)
        self.assertEqual(H.shape, (16, 16))
        np.testing.assert_array_equal(H, H.T)
        self.assertLessEqual(np.count_nonzero(H, axis=1).max(), self.L + 1)

        # The diagonal is the classical bond energy: 4J for the two uniform
        # patterns, -4J for the two alternating ones, and 0 elsewhere.
        diag = np.zeros(16, dtype='float64')
        diag[[0b0000, 0b1111]] = 4 * self.J
        diag[[0b0101, 0b1010]] = -4 * self.J
        np.testing.assert_array_equal(np.diag(H), diag)

        # Off the diagonal, -hx sits exactly at single-bit distance.
        index = np.arange(16, dtype='int64')
        s, t = np.meshgrid(index, index, indexing='ij')
        single_flip = np.isin(s ^ t, [1, 2, 4, 8])
        np.testing.assert_array_equal(H[single_flip], -self.HX)
        np.testing.assert_array_equal(H[~single_flip & (s != t)], 0.0)

    def test_ground_energy_matches_closed_form(self):
        H = build_hamiltonian(self.L, self.J, self.HX)
        energies, psi = self._solve(H)
        np.testing.assert_allclose(energies[0], self.E0,
                                   rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(exact_energy(self.L, self.J, self.HX),
                                   self.E0, rtol=1e-14)
        # The eigen-equation is a self-check that needs no reference solver.
        np.testing.assert_allclose(H @ psi, energies[0] * psi, atol=1e-12)

    def test_spectrum_matches_numpy(self):
        H = build_hamiltonian(self.L, self.J, self.HX)
        energies, _ = self._solve(H)
        np.testing.assert_allclose(energies, np.linalg.eigvalsh(H),
                                   rtol=1e-10, atol=1e-12)
        # The spectrum is symmetric about zero, with a near-degenerate pair
        # at the bottom left over from the two-fold ordered ground state.
        np.testing.assert_allclose(energies, -energies[::-1], atol=1e-12)
        np.testing.assert_allclose(energies[1] - energies[0],
                                   0.004900297644744, rtol=1e-9)

    def test_closed_form_across_field_and_size(self):
        for L in (4, 6, 8):
            for hx in (0.0, 0.3, 1.0, 2.0):
                with self.subTest(L=L, hx=hx):
                    energies, _ = self._solve(build_hamiltonian(L, 1.0, hx))
                    np.testing.assert_allclose(
                        energies[0], exact_energy(L, 1.0, hx),
                        rtol=1e-10, atol=1e-12)

    def test_boundary_and_sign_conventions(self):
        # Open ends change the energy, so the golden value pins the ring.
        energies, _ = self._solve(
            build_hamiltonian(self.L, self.J, self.HX, periodic=False))
        np.testing.assert_allclose(energies[0], -3.143326796402813,
                                   rtol=1e-10)

        # Flipping every other spin maps J to -J on an even ring, so the
        # golden value cannot see the sign of J there; an odd ring can.
        flipped, _ = self._solve(build_hamiltonian(4, -self.J, self.HX))
        np.testing.assert_allclose(flipped[0], self.E0, rtol=1e-10)
        anti, _ = self._solve(build_hamiltonian(5, self.J, self.HX))
        ferro, _ = self._solve(build_hamiltonian(5, -self.J, self.HX))
        np.testing.assert_allclose(anti[0], -3.71250543187188, rtol=1e-10)
        np.testing.assert_allclose(ferro[0], -5.113788665585293, rtol=1e-10)

    def test_ground_state_observables(self):
        _, psi = self._solve(build_hamiltonian(self.L, self.J, self.HX))
        # The two alternating patterns dominate with equal weight.
        np.testing.assert_allclose(np.abs(psi[[0b0101, 0b1010]]),
                                   0.6984, atol=5e-5)

        mx, corr, mz = observables(self.L, psi)
        np.testing.assert_allclose(mx, 0.159733387137, rtol=1e-9)
        np.testing.assert_allclose(
            corr, [1.0, -0.975320383716, 0.974571404258, -0.975320383716],
            rtol=1e-9)
        # The global spin flip commutes with H, so no site favors a
        # direction along z; the order lives in the correlation.
        np.testing.assert_allclose(mz, 0.0, atol=1e-10)

    def test_kramers_wannier_duality(self):
        # Exchanging coupling and field maps the transverse magnetization at
        # hx onto the nearest-neighbor correlation at 1/hx.
        for hx in (0.3, 0.5):
            with self.subTest(hx=hx):
                _, psi = self._solve(build_hamiltonian(self.L, self.J, hx))
                mx, _, _ = observables(self.L, psi)
                _, psi = self._solve(
                    build_hamiltonian(self.L, self.J, 1.0 / hx))
                _, corr, _ = observables(self.L, psi)
                np.testing.assert_allclose(mx, -corr[1], rtol=1e-10)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
