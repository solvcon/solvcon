# Oblique-Shock Reflection

The oblique-shock reflection describes a supersonic stream that crosses an
oblique shock. The shock bounces off a slip wall, and the flow reaches
a steady state with three zones. The case has a closed-form solution for the
states of the three zones, the angles of the two shocks, and the reflection
point[^ywh1985]. A computed solution can be measured against it.

The configuration is shown in {numref}`f:obsrefl:domain`. A uniform supersonic
stream enters from the left. The top boundary has the second inflow behind the
first oblique shock. The shock stands and runs down to the bottom slip-wall
boundary. The first shock reflects off the wall. The flow turns back to
horizontal behind the reflected shock and leaves the domain on the right
boundary.

```{eval-rst}
.. pstake:: schematic/obsrefl_domain.tex
   :align: center
   :name: f:obsrefl:domain

   Three zones stand in the converged field. Zone 1 is the free stream, zone 2
   sits between the two shocks, and zone 3 sits behind the reflected shock. The
   incident shock runs from the upper-left corner to the reflection point on
   the wall, and the reflected shock leaves through the outflow.
```

(s:obsrefl:relations)=

## Oblique Shock Relations

An oblique shock separates the flow into two states: ahead and behind.
Subscript ${}_{\mathrm{U}}$ denotes flow states in the upstream (ahead of
the shock) and ${}_{\mathrm{D}}$ in the downstream (behind the shock). The
oblique-shock-reflection problem has two oblique shocks and three zones. The
oblique-shock relations are applied twice: $\mathrm{U}, \mathrm{D} = 1, 2$
across the incident shock, and $\mathrm{U}, \mathrm{D} = 2, 3$ across the
reflected one.

The flow states across an oblique shock are shown in
{numref}`f:obsrefl:oblique`. The straight shock stands at the angle $\beta$.
The flow is turned toward the shock front by angle $\theta$.

```{eval-rst}
.. pstake:: schematic/obsrefl_oblique.tex
   :align: center
   :name: f:obsrefl:oblique
   :width: 61%

   One oblique shock: the front stands at the angle :math:`\beta` from the
   incoming flow, which crosses it and leaves deflected by :math:`\theta`.
```

Split each velocity into a tangential component along the front and a normal
component across it. See {numref}`f:obsrefl:components`.

```{eval-rst}
.. pstake:: schematic/obsrefl_components.tex
   :align: center
   :name: f:obsrefl:components
   :width: 62%

   The same shock, with the velocity on each side split into its tangential and
   normal components. The tangential components are the same, so the jump is
   carried by the normal components.
```

The tangential component across the shock remains unchanged. The deflection
falls to the normal component. That component crosses what is, in its own
frame, a normal shock, at the normal upstream Mach number
(${M_{\mathrm{U}}}_n$)

```{math}
:label: e:obsrefl:mn1

{M_{\mathrm{U}}}_n = M_{\mathrm{U}}\sin\beta
```

so the jumps are the normal-shock relations[^naca1135] in ${M_{\mathrm{U}}}_n$,

```{math}
:label: e:obsrefl:jump

\frac{\rho_{\mathrm{D}}}{\rho_{\mathrm{U}}}
= \frac{(\gamma+1){M_{\mathrm{U}}}_n^2}{(\gamma-1){M_{\mathrm{U}}}_n^2 + 2},
\;
\frac{p_{\mathrm{D}}}{p_{\mathrm{U}}}
= 1 + \frac{2\gamma}{\gamma+1}\left({M_{\mathrm{U}}}_n^2 - 1\right),
\;
\frac{T_{\mathrm{D}}}{T_{\mathrm{U}}}
= \frac{p_{\mathrm{D}}}{p_{\mathrm{U}}}
  \frac{\rho_{\mathrm{U}}}{\rho_{\mathrm{D}}}
```

and the downstream Mach number follows from its own normal component,

```{math}
:label: e:obsrefl:mn2

{M_{\mathrm{D}}}_n^2
= \frac{(\gamma-1){M_{\mathrm{U}}}_n^2 + 2}
        {2\gamma {M_{\mathrm{U}}}_n^2 - (\gamma-1)},
\;
M_{\mathrm{D}} = \frac{{M_{\mathrm{D}}}_n}{\sin(\beta - \theta)}
```

The downstream velocity has magnitude
$M_{\mathrm{D}}\sqrt{\gamma p_{\mathrm{D}}/\rho_{\mathrm{D}}}$ and
points $\theta$ away from the upstream direction, toward the shock.

## Shock and Deflection Angles

The deflection angle $\theta$ and shock angle $\beta$ are related with the Mach
number $M$ (theta-beta-M relation),

```{math}
:label: e:obsrefl:tbm

\tan\theta = 2\cot\beta\,
\frac{M_{\mathrm{U}}^2\sin^2\beta - 1}
     {M_{\mathrm{U}}^2\left(\gamma + \cos 2\beta\right) + 2}
```

The problem provides the deflection angle $\theta$ and wants the shock angle
$\beta$. Rearrange {eq}`e:obsrefl:tbm` into a cubic in $\tan\beta$,

```{math}
\begin{aligned}
&\left(1 + \frac{\gamma-1}{2}M_{\mathrm{U}}^2\right)\tan\theta\,\tan^3\beta
 - \left(M_{\mathrm{U}}^2 - 1\right)\tan^2\beta \\
&\qquad + \left(1 + \frac{\gamma+1}{2}M_{\mathrm{U}}^2\right)
   \tan\theta\,\tan\beta + 1 = 0
\end{aligned}
```

Its three roots have the closed form[^rudd1998], indexed by $\delta = 0, 1,
2$,

```{math}
:label: e:obsrefl:beta

\tan\beta(M_{\mathrm{U}}, \theta) = \frac
{M_{\mathrm{U}}^2 - 1
 + 2\lambda\cos\left(\dfrac{4\pi\delta + \arccos\chi}{3}\right)}
{3\left(1 + \dfrac{\gamma-1}{2}M_{\mathrm{U}}^2\right)\tan\theta}
```

where $\lambda$ and $\chi$ depend only on $M_{\mathrm{U}}$ and $\theta$,

```{math}
:label: e:obsrefl:lambda

\begin{aligned}
\lambda &= \left[\left(M_{\mathrm{U}}^2-1\right)^2
  - 3\left(1 + \frac{\gamma-1}{2}M_{\mathrm{U}}^2\right)
     \left(1 + \frac{\gamma+1}{2}M_{\mathrm{U}}^2\right)
     \tan^2\theta\right]^{1/2} \\
\chi &= \frac{1}{\lambda^3}\left[\left(M_{\mathrm{U}}^2-1\right)^3
  - 9\left(1 + \frac{\gamma-1}{2}M_{\mathrm{U}}^2\right)
     \left(1 + \frac{\gamma-1}{2}M_{\mathrm{U}}^2
             + \frac{\gamma+1}{4}M_{\mathrm{U}}^4\right)
     \tan^2\theta\right]
\end{aligned}
```

$\delta = 0$ selects the strong shock standing at the larger angle. $\delta =
1$ selects the weak shock standing at the smaller angle. The reflection is the
weak branch, the one that occurs unless a raised back pressure downstream
forces the strong shock. See {numref}`f:obsrefl:tbm` for the weak and strong
shock in the relation chart between the shock angle $\beta$ and deflection
angle $\theta$ at $M_{\mathrm{U}} = 3$. $\delta = 2$ is a negative root without
physical meaning.

```{eval-rst}
.. pstake:: schematic/obsrefl_tbm.tex
   :align: center
   :name: f:obsrefl:tbm
   :width: 75%

   The deflection angle :math:`\theta` against the shock angle
   :math:`\beta`, Eq.
   :eq:`e:obsrefl:tbm`, for :math:`M_{\mathrm{U}} = 3`.  A deflection line
   crosses the curve twice, the weak root on the rising branch and the strong
   root on the falling one; the line drawn is the reference case's
   :math:`\theta = 10` degrees, whose weak root is the incident shock angle
   :math:`\beta_1`.  The curve rises from zero deflection at the Mach angle
   :math:`\mu = \arcsin(1/M_{\mathrm{U}})` and peaks at the largest deflection
   an attached shock can turn.
```

At the peak $\chi$ reaches $-1$ and the weak and the strong roots merge. It is
the largest deflection angle that the flow has a straight, attached shock. At
Mach 3 it is 34.07 degrees. For a deflection angle larger than the peak, there
is no attached shock.

## Shock Reflection

The flow states of the incident and reflected shocks shown in
{numref}`f:obsrefl:domain` can be determined by applying the {ref}`oblique
shock relations <s:obsrefl:relations>` twice. The incident shock uses
$\mathrm{U}, \mathrm{D} = 1, 2$. It deflects the free stream by angle $\theta$
toward the wall and stands at $\beta_1 = \beta (M_1, \theta)$. The reflected
shock uses $\mathrm{U}, \mathrm{D} = 2, 3$. It turns the flow back by the same
angle $\theta$, and the flow direction in zone 3 is along the $x$-coordinate
because no flow passes the wall. The shock angle is $\beta_2 = \beta (M_2,
\theta)$.

The point of shock reflection can be calculated from the angles. Over a domain
running from $(x_0, y_0)$ to $(x_1, y_1)$, the incident shock enters at the
upper-left corner and descends at $\beta_1$, so it reaches the wall at

```{math}
:label: e:obsrefl:xr

x_r = x_0 + \frac{y_1 - y_0}{\tan\beta_1}
```

The reflected shock leaves the wall at $x_r$, rises at $\beta_2 - \theta$, and
meets the right-hand-side outflow boundary at

```{math}
:label: e:obsrefl:ye

y_e = y_0 + (x_1 - x_r)\tan(\beta_2 - \theta)
```

If $x_r > x_1$, the domain is too short to hold the reflection: the incident
shock leaves through the outflow and there is no zone 3.

## Example Case

The example case uses a domain 4 units long and 1 unit tall, running from
$(x_0, y_0) = (0, 0)$ to $(x_1, y_1) = (4, 1)$. The inflow is a Mach 3 stream
of unit density and unit pressure, deflected 10 degrees: $M_1 = 3$,
$\rho_1 = p_1 = 1$, $\theta = 10^{\circ}$, and $\gamma = 1.4$. The flow
states in the three zones are:

|                 | zone 1 |  zone 2 | zone 3 |
|-----------------|-------:|--------:|-------:|
| density $\rho$  | 1.0000 |  1.6546 | 2.5651 |
| pressure $p$    | 1.0000 |  2.0545 | 3.8329 |
| Mach number $M$ | 3.0000 |  2.5050 | 2.0902 |
| $x$ velocity    | 3.5496 |  3.2526 | 3.0233 |
| $y$ velocity    | 0.0000 | -0.5735 | 0.0000 |

The two shocks stand at $\beta_1 = 27.383$ and $\beta_2 = 31.795$ degrees. The
second shock is 21.795 degrees above the wall ($\beta_2 - \theta =
21.795^{\circ}$). The reflection point is $x_r = 1.9306$, and the reflected
shock reaches the outflow at $y_e = 0.8275$.

[^ywh1985]: H. C. Yee, R. F. Warming, and A. Harten, "Implicit total variation
    diminishing (TVD) schemes for steady-state calculations," Journal of
    Computational Physics 57(3):327-360, 1985.
    <https://doi.org/10.1016/0021-9991(85)90183-4>

[^naca1135]: Ames Research Staff, "Equations, tables, and charts for
    compressible flow," NACA Report 1135, 1953, the canonical compilation of
    the normal- and oblique-shock relations.
    <https://ntrs.nasa.gov/citations/19930091059>

[^rudd1998]: L. von Eggers Rudd and M. J. Lewis, "Comparison of shock
    calculation methods," Journal of Aircraft 35(4):647-649, 1998, which
    weighs the closed-form root against iterative solutions of the
    theta-beta-M relation. <https://doi.org/10.2514/2.2349>

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->
