# Transverse-Field Ising Chain

:::{admonition} Work in progress
:class: attention

This page documents a problem solver that is still being developed. Its text
may not be accurate. The text, code, and numbers may change.
:::

The Ising model is the simplest model of a magnet: a lattice of spins that
each point up or down, with every pair of neighbors coupled so that the bond
between them prefers the two spins either aligned or opposed. An Ising
problem asks what the spins do collectively under that coupling. This page
solves the smallest version that still counts as physics, a ring of four
quantum spins in a transverse magnetic field. It writes the energy of the
system as a matrix, finds the lowest eigenvalue and its eigenvector with
solvcon's `EigenSystem`, and checks the result against the exact solution to
machine precision. The eigenvector then answers the physical questions: how
far the spins lean along the field, and how the direction of one spin depends
on its neighbors.

## Ising Model

Magnetism comes from electrons that carry a tiny intrinsic magnetic moment,
called spin. In the model proposed by Lenz and worked out by Ising in 1925
[^ising1925][^brush1967], each atom of the material is reduced to a single
spin $s_i$ that points either up or down, written $s_i = +1$ or $s_i = -1$,
and only neighboring spins interact. The energy of one configuration of a
chain of $L$ spins is

```{math}
:label: e:ising:classical

E(s_0, \ldots, s_{L-1}) = J \sum_{i=0}^{L-1} s_i s_{i+1} ,
\qquad s_L \equiv s_0 ,
```

where the last identity closes the chain into a ring so that every spin has
two neighbors. The coupling $J$ sets what a bond prefers. With $J < 0$ a
bond lowers the energy when its two spins agree, so the spins line up and the
material is a ferromagnet, like iron. With $J > 0$ a bond lowers the energy
when its spins disagree, so they alternate, and the material is an
antiferromagnet. The sign in front of $J$ is chosen so that the
antiferromagnet solved below is the $J > 0$ case; most textbooks write $-J$
instead and take $J > 0$ as the ferromagnet. Every spin tries to satisfy all
of its bonds at once, and the behavior of the material is the collective
result.

At a finite temperature the spins are jostled by heat. At low temperature
they settle into the ordered pattern and a ferromagnet is magnetized. Heat it
past its Curie temperature and the alignment is scrambled, the magnetization
vanishes, and the magnet loses its strength [^kittel]. That abrupt change of
character is a phase transition. Ising showed that the one-dimensional chain
has no such transition at any temperature above zero; Onsager later solved
the two-dimensional model exactly and showed that it does [^onsager1944].
The model has since become the standard test bed for phase transitions, and
its variants describe alloys, lattice gases, and neural networks as well as
magnets.

## Transverse Field

At the atomic scale a spin is not simply up or down. It is a two-state
quantum system, and quantum mechanics lets it sit in a superposition of both
[^feynman3]. The spin along $z$ is represented by the Pauli matrix
$\sigma^z$, whose two eigenstates are up and down. A magnetic field applied
along $x$, transverse to the spin axis, acts through $\sigma^x$, which turns
up into down and back:

$$
\sigma^z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix} , \qquad
\sigma^x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix} .
$$

A transverse field of strength $h_x$ therefore makes every spin tunnel
between up and down. These quantum fluctuations play a role much like
temperature, fighting the ordering tendency of the coupling, with one crucial
difference: they act even at absolute zero, where there is no heat at all.

The transverse-field Ising chain sets the coupling and the field against each
other. On the ring of $L$ spins its energy operator, the Hamiltonian, is

```{math}
:label: e:ising:hamiltonian

\hat{H}
  = J \sum_{i=0}^{L-1} \hat{\sigma}^z_i \hat{\sigma}^z_{i+1}
  - h_x \sum_{i=0}^{L-1} \hat{\sigma}^x_i ,
\qquad \hat{\sigma}_L \equiv \hat{\sigma}_0 ,
```

where $\hat{\sigma}^a_i$ acts as $\sigma^a$ on spin $i$ and leaves every
other spin alone. Turn the field off and the ground states, the
lowest-energy states the chain settles into, are the perfectly ordered
patterns of Eq. {eq}`e:ising:classical`. Turn it up and the fluctuations
eventually destroy the order. In the limit of a long chain the change happens
sharply at $h_x = J$: a phase transition at zero temperature, driven by a
parameter of the model rather than by heat, which is why this chain is the
textbook example of a quantum phase transition [^sachdev]. In one dimension
the model is exactly solvable [^katsura1962][^pfeuty1970], which makes it a
trusted benchmark for a numerical method.

{numref}`f:ising:ring` shows the case solved on this page: $L = 4$ spins on
a ring, antiferromagnetic coupling $J = 1$, and a weak field $h_x = 0.3$,
deep in the ordered phase.

```{eval-rst}
.. pstake:: schematic/ising_ring.tex
   :align: center
   :name: f:ising:ring
   :width: 55%

   Four spins on a ring.  Each spin is measured along :math:`z` and is drawn
   in the alternating pattern the antiferromagnetic coupling :math:`J`
   favors.  The transverse field :math:`h_x` is applied along :math:`x`, at
   a right angle to the spin axis, and makes each spin flip.
```

## Eigenvalue Problem

One spin has two states, so $L$ spins have $2^L$ joint states, and the state
of the chain is a vector with $2^L$ entries. Label the joint states by the
bit patterns $\lvert s \rangle = \lvert s_{L-1} \cdots s_1 s_0 \rangle$, with
bit $s_i = 0$ for spin $i$ up and $s_i = 1$ for down; the pattern read as an
integer is the index of the state. The ket $\lvert s \rangle$ is the basis
column vector of length $2^L$ with a single 1 at position $s$, the bra
$\langle s' \vert$ is its transpose, and $\langle s' \vert \hat{H} \vert s
\rangle$ is the matrix entry in row $s'$ and column $s$. Applied to a basis
state, $\hat{\sigma}^z_i$ multiplies it by $1 - 2 s_i$, and
$\hat{\sigma}^x_i$ turns it into the state with bit $i$ flipped, so the
matrix element of Eq. {eq}`e:ising:hamiltonian` between two states is

```{math}
:label: e:ising:element

\langle s' \vert \hat{H} \vert s \rangle
  = J \, \delta_{s', s} \sum_{i=0}^{L-1}
      \bigl( 1 - 2\,(s_i \oplus s_{i+1}) \bigr)
  \; - \;
    h_x \sum_{i=0}^{L-1} \delta_{s',\, s \oplus 2^i} ,
```

where $\oplus$ is the bitwise exclusive-or. The coupling term sits on the
diagonal and merely counts bonds, $+J$ for each pair of neighbors that agree
and $-J$ for each pair that disagree. The field term connects each state to
the $L$ states that differ from it in a single bit, each with amplitude
$-h_x$. The Hamiltonian is therefore a real, symmetric, sparse
$2^L \times 2^L$ matrix with at most $L + 1$ nonzeros per row.

:::{note}
Written out for the four-spin ring solved on this page, the matrix of
Eq. {eq}`e:ising:element` splits into two-by-two blocks by the state of spin
3, the most significant bit. With rows labeled by bit pattern, columns by
state index, and a dot for a zero off the diagonal, it is

```{math}
:label: e:ising:matrix

\begin{gathered}
H = \begin{pmatrix}
      H_0 & -h_x I_{8 \times 8} \\
      -h_x I_{8 \times 8} & H_1
    \end{pmatrix}
\\[1ex]
H_0 = \begin{array}{r|cccccccc}
   & 0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 \\
  \hline
  0000 & 4J & -h_x & -h_x & \cdot & -h_x & \cdot & \cdot & \cdot \\
  0001 & -h_x & 0 & \cdot & -h_x & \cdot & -h_x & \cdot & \cdot \\
  0010 & -h_x & \cdot & 0 & -h_x & \cdot & \cdot & -h_x & \cdot \\
  0011 & \cdot & -h_x & -h_x & 0 & \cdot & \cdot & \cdot & -h_x \\
  0100 & -h_x & \cdot & \cdot & \cdot & 0 & -h_x & -h_x & \cdot \\
  0101 & \cdot & -h_x & \cdot & \cdot & -h_x & -4J & \cdot & -h_x \\
  0110 & \cdot & \cdot & -h_x & \cdot & -h_x & \cdot & 0 & -h_x \\
  0111 & \cdot & \cdot & \cdot & -h_x & \cdot & -h_x & -h_x & 0 \\
\end{array}
\\[1ex]
H_1 = \begin{array}{r|cccccccc}
   & 8 & 9 & 10 & 11 & 12 & 13 & 14 & 15 \\
  \hline
  1000 & 0 & -h_x & -h_x & \cdot & -h_x & \cdot & \cdot & \cdot \\
  1001 & -h_x & 0 & \cdot & -h_x & \cdot & -h_x & \cdot & \cdot \\
  1010 & -h_x & \cdot & -4J & -h_x & \cdot & \cdot & -h_x & \cdot \\
  1011 & \cdot & -h_x & -h_x & 0 & \cdot & \cdot & \cdot & -h_x \\
  1100 & -h_x & \cdot & \cdot & \cdot & 0 & -h_x & -h_x & \cdot \\
  1101 & \cdot & -h_x & \cdot & \cdot & -h_x & 0 & \cdot & -h_x \\
  1110 & \cdot & \cdot & -h_x & \cdot & -h_x & \cdot & 0 & -h_x \\
  1111 & \cdot & \cdot & \cdot & -h_x & \cdot & -h_x & -h_x & 4J \\
\end{array}
\end{gathered}
```

$H_0$ holds the eight patterns with spin 3 up (`0000` to `0111`, states 0
to 7) and $H_1$ the eight with spin 3 down (`1000` to `1111`, states 8 to
15). The off-diagonal blocks are the flips of spin 3, which pair every
pattern in $H_0$ with the one at the same position in $H_1$; within a block
only spins 0, 1, and 2 flip.

The diagonal carries the classical bond energy of each pattern: $4J$ for the
two uniform patterns `0000` and `1111`, $-4J$ for the two alternating
patterns `0101` and `1010`, and $0$ for the twelve patterns with two agreeing
and two disagreeing bonds (on a ring the number of disagreeing bonds is
always even). The diagonal of $H_1$ is that of $H_0$ read upward, because
flipping every spin leaves every bond as it was. Off the diagonal the two
blocks are identical: $-h_x$ times the adjacency matrix of the cube whose
eight corners are the patterns of spins 0, 1, and 2, three entries per row,
and the identity blocks add the fourth $-h_x$ of each row for the flip of
spin 3. The field part of each block splits the same way by spin 2, and so
on down; that is how the field wires the $2^L$ patterns into an
$L$-dimensional hypercube, one spin at a time. The matrix is symmetric
because flipping a bit is its own inverse: if $s'$ is one flip from $s$,
then $s$ is one flip from $s'$.
:::

A general state is a column vector $\psi$ whose entry $\psi_s$ is the
amplitude of configuration $s$. Applying $\hat{H}$ to it is a matrix-vector
product, and summing Eq. {eq}`e:ising:element` over the column index gives
entry $s$ of the result as

```{math}
:label: e:ising:matvec

(\hat{H} \psi)_s
  = J \, b(s) \, \psi_s
  - h_x \sum_{i=0}^{L-1} \psi_{s \oplus 2^i} ,
\qquad
b(s) = \sum_{i=0}^{L-1} \bigl( 1 - 2\,(s_i \oplus s_{i+1}) \bigr) ,
```

where $b(s)$ is the number of agreeing bonds minus the number of disagreeing
bonds in configuration $s$. Each entry is scaled by the bond count of its own
configuration, then $h_x$ times the entries of the $L$ configurations one
flip away is subtracted. Row `0101` of $H_0$ in Eq. {eq}`e:ising:matrix`,
state 5, has $-4J$ on the diagonal and $-h_x$ in columns 1, 4, and 7, and
the identity block adds $-h_x$ in column $13 = 5 + 8$, so

$$
(\hat{H} \psi)_5
  = -4J \, \psi_5 - h_x \,(\psi_1 + \psi_4 + \psi_7 + \psi_{13}) .
$$

The state the chain rests in, the ground state, is the eigenvector
$\lvert \psi_0 \rangle$ with the smallest eigenvalue $E_0$,

$$
\hat{H} \lvert \psi_0 \rangle = E_0 \lvert \psi_0 \rangle ,
\qquad
E_0 = \min_{\lVert \psi \rVert = 1}
      \langle \psi \vert \hat{H} \vert \psi \rangle ,
$$

and $E_0$ is the ground-state energy. This is the same shape of problem as
modal or buckling analysis, where the lowest eigenvalue of a matrix gives the
fundamental vibration mode or the critical load. Building the Hamiltonian
matrix and extracting its lowest eigenpair is called exact diagonalization
[^sandvik2010]. Its one hard part is size: the matrix grows as $4^L$, so a
dense solve is practical only up to a dozen or so spins. For $L = 4$ it is a
$16 \times 16$ matrix.

## Building the Hamiltonian

The matrix follows Eq. {eq}`e:ising:element` directly. Loop over every state
and every site, add the bond energy to the diagonal, and subtract $h_x$ in
the row of the flipped state:

```python
import numpy as np

import solvcon as sc


def build_hamiltonian(L, J, hx):
    n = 1 << L
    H = np.zeros((n, n), dtype='float64')
    for s in range(n):
        for i in range(L):
            j = (i + 1) % L
            H[s, s] += J * (1 - 2 * (((s >> i) & 1) ^ ((s >> j) & 1)))
            H[s ^ (1 << i), s] -= hx
    return H
```

The `(i + 1) % L` closes the ring. With $L = 4$, $J = 1$, and $h_x = 0.3$
the upper-left corner of $H_0$ in Eq. {eq}`e:ising:matrix` comes out as

```python
>>> H = build_hamiltonian(4, 1.0, 0.3)
>>> H[:4, :4]
array([[ 4. , -0.3, -0.3,  0. ],
       [-0.3,  0. ,  0. , -0.3],
       [-0.3,  0. ,  0. , -0.3],
       [ 0. , -0.3, -0.3,  0. ]])
```

State 0 is all four spins up. Every bond then has its spins agreeing, which
the antiferromagnet pays for, so its diagonal entry is $4J$. It connects to
states 1 and 2, which each have one spin flipped, with amplitude $-h_x$.

## Solving with solvcon

`sc.EigenSystem` diagonalizes a square `sc.SimpleArray`. It wraps the LAPACK
`*GEEV` driver for a general, not necessarily symmetric, matrix, so the
eigenvalues come back unsorted and split into real and imaginary parts, `wr`
and `wi`, and the right eigenvectors are the columns of `vr`. For the
symmetric Hamiltonian the imaginary parts vanish, and the ground state is the
column of `vr` at the position of the smallest `wr`:

```python
def ground_state(H):
    solver = sc.EigenSystem(sc.SimpleArray(H), do_vl=False)
    solver.run()
    wr, wi = np.array(solver.wr), np.array(solver.wi)
    assert np.allclose(wi, 0.0)
    order = np.argsort(wr)
    psi = np.array(solver.vr)[:, order[0]]
    return wr[order], psi / np.linalg.norm(psi)
```

`do_vl=False` skips the left eigenvectors, which a symmetric matrix does not
need. `sc.SimpleArray(H)` hands the numpy buffer to the solver without a
copy, and `np.array` on the result arrays copies them back out.
`sc.EigenSystem` is `None` on a build without a vendor LAPACK; every other
line runs the same on any build.

```python
>>> energies, psi = ground_state(H)
>>> print(energies[0])
-4.092961599426857
>>> print("%.1e" % np.linalg.norm(H @ psi - energies[0] * psi))
3.4e-15
```

The residual on the last line is the self-check that the recovered pair
satisfies $\hat{H} \lvert \psi_0 \rangle = E_0 \lvert \psi_0 \rangle$, with
`H @ psi` the product of Eq. {eq}`e:ising:matvec`; it needs no reference
solver. The full spectrum is symmetric about zero, and its
lowest two levels are nearly degenerate: the first excited state lies only
$0.0049$ above the ground state, the finite-size remnant of the two-fold
degenerate ordered state at $h_x = 0$.

## Checking Against the Exact Solution

The Jordan-Wigner transformation maps a spin chain onto free fermions
[^lieb1961]; applied to the transverse-field chain, it gives the spectrum in
closed form [^katsura1962][^pfeuty1970]. For a ring with an even number of
spins the ground-state energy is

```{math}
:label: e:ising:exact

E_0 = - \sum_{n=0}^{L-1}
        \sqrt{J^2 + h_x^2 - 2 J h_x \cos k_n} ,
\qquad k_n = \frac{(2n + 1)\pi}{L} ,
```

and Eq. {eq}`e:ising:exact` is one line of code:

```python
def exact_energy(L, J, hx):
    k = np.pi * (2 * np.arange(L) + 1) / L
    return -np.sum(np.sqrt(J**2 + hx**2 - 2 * J * hx * np.cos(k)))
```

For $L = 4$ the four cosines are $\pm 1/\sqrt{2}$ and the sum collapses to
$E_0 = -2\sqrt{2.18 + 2\sqrt{1.0081}} = -4.092961599426859$. The solvcon
result above differs from it by $1.8 \times 10^{-15}$, the rounding level of
double precision. The agreement holds across the field strength, from the
classical limit $h_x = 0$, where every bond is satisfied and $E_0 = -LJ$,
through the transition and into the field-dominated regime:

| $h_x$ | $E_0$ (solvcon)  | $E_0$ (exact)    |
|------:|-----------------:|-----------------:|
|   0.0 |  -4.000000000000 |  -4.000000000000 |
|   0.3 |  -4.092961599427 |  -4.092961599427 |
|   0.5 |  -4.271558410140 |  -4.271558410140 |
|   1.0 |  -5.226251859506 |  -5.226251859506 |
|   1.5 |  -6.760008550556 |  -6.760008550556 |
|   2.0 |  -8.543116820279 |  -8.543116820279 |
|   3.0 | -12.346784241457 | -12.346784241457 |

Two conventions are load-bearing. The ring must be closed: with open ends
the same four spins have $E_0 = -3.1433$, not $-4.0930$. And the check is
blind to the sign of $J$ on an even ring, because flipping every other spin
maps $J$ to $-J$ without changing the spectrum. On an odd ring the
antiferromagnet is frustrated because it cannot alternate all the way
around, so its energy differs from the ferromagnet's, and the closed form
then needs the parity bookkeeping worked out in [^heguo].

## Reading the Ground State

The eigenvector is where the physics is. Its entries are the amplitudes of
the $2^L$ configurations, here up to an overall sign:

```python
>>> np.round(psi, 4)
array([0.008 , 0.0541, 0.0541, 0.0159, 0.0541, 0.6984, 0.0159, 0.0541,
       0.0541, 0.0159, 0.6984, 0.0541, 0.0159, 0.0541, 0.0541, 0.008 ])
```

The two large entries are states 5 and 10, the bit patterns `0101` and
`1010`: the two perfectly alternating configurations. At $h_x = 0$ the ground
state would be either one of them. The weak field mixes them into an
equal-weight superposition, each with amplitude $0.6984$ where $1/\sqrt{2} =
0.7071$ would be the pure pattern, plus small admixtures of the
configurations one flip away.

Observables are expectation values in the ground state. The transverse
magnetization $m_x$ and the spin-spin correlation $C(r)$ at distance $r$
are

$$
m_x = \frac{1}{L} \sum_{i=0}^{L-1}
      \langle \psi_0 \vert \hat{\sigma}^x_i \vert \psi_0 \rangle ,
\qquad
C(r) = \frac{1}{L} \sum_{i=0}^{L-1} \langle \psi_0 \vert
       \hat{\sigma}^z_i \hat{\sigma}^z_{i+r} \vert \psi_0 \rangle .
$$

Both follow from the same bit rules that built the matrix. $\hat{\sigma}^x_i$
pairs each amplitude with the amplitude of the state with bit $i$ flipped,
and $\hat{\sigma}^z_i$ weights each amplitude by $1 - 2 s_i$:

```python
def observables(L, psi):
    states = np.arange(1 << L)

    def sz(i):
        return 1 - 2 * ((states >> i) & 1)

    mx = sum(psi @ psi[states ^ (1 << i)] for i in range(L)) / L
    corr = [np.mean([psi @ (sz(i) * sz((i + r) % L) * psi)
                     for i in range(L)]) for r in range(L)]
    return mx, corr
```

```python
>>> mx, corr = observables(4, psi)
>>> print(round(mx, 4))
0.1597
>>> np.round(corr, 4)
array([ 1.    , -0.9753,  0.9746, -0.9753])
```

The correlation alternates in sign with the distance and stays near one in
magnitude: neighbors point opposite ways, next-nearest neighbors the same
way, which is the signature of antiferromagnetic order. The weak field has
tilted every spin slightly along $x$, giving $m_x = 0.16$. The magnetization
along $z$ itself cannot show the order. For $h_x > 0$ every
$\langle \hat{\sigma}^z_i \rangle$ is zero at any finite $L$: the Hamiltonian
commutes with the global spin flip $\prod_i \hat{\sigma}^x_i$, which reverses
every $\hat{\sigma}^z_i$, so the nondegenerate ground state cannot favor one
direction. The staggered average, with alternating signs, vanishes too,
because the two alternating patterns are weighted equally. Order is therefore
read from the correlation.

Sweep the field and the two numbers trade places:

| $h_x$ | $m_x$  | $C(1)$  |
|------:|-------:|--------:|
|   0.0 | 0.0000 | -1.0000 |
|   0.3 | 0.1597 | -0.9753 |
|   0.5 | 0.2909 | -0.9224 |
|   1.0 | 0.6533 | -0.6533 |
|   1.5 | 0.8493 | -0.4160 |
|   2.0 | 0.9224 | -0.2909 |
|   3.0 | 0.9689 | -0.1799 |

At weak field the spins alternate along $z$ and barely lean along $x$; at
strong field they point along $x$ and the alternation along $z$ fades. The
two columns mirror each other, $m_x$ at $h_x = 0.5$ equals $-C(1)$ at
$h_x = 2$, and they cross at $h_x = J$. That mirror is the Kramers-Wannier
self-duality of the model, which exchanges the coupling with the field
[^sachdev], and the crossing is the four-spin shadow of the phase transition.

## Beyond Four Spins

The same code runs unchanged on a longer ring, and the energy per site
settles toward its value for an infinite chain:

| $L$ | $E_0 / L$ (solvcon) | $E_0 / L$ (exact) |
|----:|--------------------:|------------------:|
|   4 |     -1.023240399857 |   -1.023240399857 |
|   6 |     -1.022658330888 |   -1.022658330888 |
|   8 |     -1.022631167116 |   -1.022631167116 |
|  10 |     -1.022629620096 |   -1.022629620096 |

The ceiling is the dense matrix. $L = 10$ is $1024 \times 1024$; $L = 14$ is
$16384 \times 16384$, two gigabytes in double precision, and the
general-purpose `*GEEV` computes the whole spectrum when only the lowest pair
is wanted. Getting past that ceiling means never forming the matrix at all:
apply Eq. {eq}`e:ising:element` to a vector and feed the product to an
iterative eigensolver such as Lanczos [^sandvik2010]. The Cytnx
tensor-network library solves this same four-spin problem that way in its
exact-diagonalization example [^cytnx], from the same bit rules as
`build_hamiltonian`.

[^ising1925]: E. Ising, "Beitrag zur Theorie des Ferromagnetismus,"
    Zeitschrift fuer Physik 31(1):253-258, 1925.
    <https://doi.org/10.1007/BF02980577>

[^brush1967]: S. G. Brush, "History of the Lenz-Ising model," Reviews of
    Modern Physics 39(4):883-893, 1967.
    <https://doi.org/10.1103/RevModPhys.39.883>

[^kittel]: C. Kittel, *Introduction to Solid State Physics*, 8th ed., Wiley,
    2005. See the chapters on ferromagnetism and the Curie point.

[^onsager1944]: L. Onsager, "Crystal statistics. I. A two-dimensional model
    with an order-disorder transition," Physical Review 65(3-4):117-149,
    1944. <https://doi.org/10.1103/PhysRev.65.117>

[^feynman3]: R. P. Feynman, R. B. Leighton, and M. Sands, *The Feynman
    Lectures on Physics, Vol. III: Quantum Mechanics*, Addison-Wesley, 1965.
    See the chapters on two-state systems and on spin one-half.

[^sachdev]: S. Sachdev, *Quantum Phase Transitions*, 2nd ed., Cambridge
    University Press, 2011. See the chapters on the Ising chain in a
    transverse field.

[^lieb1961]: E. Lieb, T. Schultz, and D. Mattis, "Two soluble models of an
    antiferromagnetic chain," Annals of Physics 16(3):407-466, 1961.
    <https://doi.org/10.1016/0003-4916(61)90115-4>

[^katsura1962]: S. Katsura, "Statistical mechanics of the anisotropic linear
    Heisenberg model," Physical Review 127(5):1508-1518, 1962.
    <https://doi.org/10.1103/PhysRev.127.1508>

[^pfeuty1970]: P. Pfeuty, "The one-dimensional Ising model with a transverse
    field," Annals of Physics 57(1):79-90, 1970.
    <https://doi.org/10.1016/0003-4916(70)90270-8>

[^sandvik2010]: A. W. Sandvik, "Computational studies of quantum spin
    systems," AIP Conference Proceedings 1297:135-338, 2010.
    <https://doi.org/10.1063/1.3518900>

[^heguo]: Y. He and H. Guo, "The boundary effects of transverse field Ising
    model," Journal of Statistical Mechanics: Theory and Experiment
    2017:093101, 2017. <https://doi.org/10.1088/1742-5468/aa85b0>

[^cytnx]: Cytnx developers, *Cytnx*, `example/ED/ed_ising.py`, Apache
    License 2.0.
    <https://github.com/Cytnx-dev/Cytnx/blob/master/example/ED/ed_ising.py>

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
