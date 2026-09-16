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

[^katsura1962]: S. Katsura, "Statistical mechanics of the anisotropic linear
    Heisenberg model," Physical Review 127(5):1508-1518, 1962.
    <https://doi.org/10.1103/PhysRev.127.1508>

[^pfeuty1970]: P. Pfeuty, "The one-dimensional Ising model with a transverse
    field," Annals of Physics 57(1):79-90, 1970.
    <https://doi.org/10.1016/0003-4916(70)90270-8>

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
