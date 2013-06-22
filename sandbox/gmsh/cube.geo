/*
 * A Gmsh template file for a cubic domain.
 */

lc = 2.0;

// vertices.
Point(1) = {1,1,-1,lc};
Point(2) = {-1,1,-1,lc};
Point(3) = {-1,-1,-1,lc};
Point(4) = {1,-1,-1,lc};
Point(5) = {1,1,1,lc};
Point(6) = {-1,1,1,lc};
Point(7) = {-1,-1,1,lc};
Point(8) = {1,-1,1,lc};

// lines.
Line(1) = {1,2}; // rear-lower line.
Line(2) = {2,3}; // left-lower line.
Line(3) = {3,4}; // front-lower line.
Line(4) = {4,1}; // right-lower line.
Line(5) = {5,6}; // rear-upper line.
Line(6) = {6,7}; // left-upper line.
Line(7) = {7,8}; // front-upper line.
Line(8) = {8,5}; // right-upper line.
Line(9) = {1,5}; // rear-right line.
Line(10) = {2,6}; // rear-left line.
Line(11) = {3,7}; // front-left line.
Line(12) = {4,8}; // front-right line.

// surface.
Line Loop(21) = {1,2,3,4}; // lower surface.
Plane Surface(22) = {21};
Line Loop(23) = {5,6,7,8}; // upper surface.
Plane Surface(24) = {23};
Periodic Surface 24 {5,6,7,8} = 22 {1,2,3,4};
Line Loop(25) = {-3,11,7,-12}; // front surface.
Plane Surface(26) = {25};
Line Loop(27) = {1,10,-5,-9}; // rear surface.
Plane Surface(28) = {27};
Periodic Surface 28 {1,10,-5,-9} = 26 {-3,11,7,-12};
Line Loop(29) = {-2,10,6,-11}; // left surface.
Plane Surface(30) = {29};
Line Loop(31) = {4,9,-8,-12}; // right surface.
Plane Surface(32) = {31};
Periodic Surface 32 {4,9,-8,-12} = 30 {-2,10,6,-11};

// volume.
Surface Loop(41) = {-22,-26,-30,24,28,32};
Volume(42) = {41};

// physics.
Physical Surface("lower") = {22};
Physical Surface("upper") = {24};
Physical Surface("front") = {26};
Physical Surface("rear") = {28};
Physical Surface("left") = {30};
Physical Surface("right") = {32};
Physical Volume("domain") = {42};

// vim: set ai et nu ff=unix ft=c:
