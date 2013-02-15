/*
 * A Gmsh template file for a rectangle domain.
 */
lc = 0.1;
// vertices.
Point(1) = {4,1,0,lc};
Point(2) = {2,2,0,lc};
Point(3) = {0,1,0,lc};
Point(4) = {0,0,0,lc};
Point(5) = {4,0,0,lc};
Point(6) = {3.5,  1,0,lc};
Point(7) = {  3,  1,0,lc};
Point(8) = {  3,0.5,0,lc};
Point(9) = {3.5,0.5,0,lc};
Point(10) = {  1,  1,0,lc};
Point(11) = {0.5,  1,0,lc};
Point(12) = {0.5,0.5,0,lc};
Point(13) = {  1,0.5,0,lc};
Point(14) = {  2,0.8,0,lc};
Point(15) = {1.5,0.2,0,lc};
Point(16) = {2.5,0.2,0,lc};
// lines.
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,5};
Line(5) = {5,1};
Line(6) = {6,7};
Line(7) = {7,8};
Line(8) = {8,9};
Line(9) = {9,6};
Line(10) = {10,11};
Line(11) = {11,12};
Line(12) = {12,13};
Line(13) = {13,10};
Line(14) = {14,15};
Line(15) = {15,16};
Line(16) = {16,14};
// surface.
Line Loop(1) = {1,2,3,4,5};
Line Loop(2) = {6,7,8,9};
Line Loop(3) = {10,11,12,13};
Line Loop(4) = {14,15,16};
Plane Surface(1) = {1,2,3,4};
// physics.
Physical Line("upper") = {1,2};
Physical Line("left") = {3};
Physical Line("lower") = {4};
Physical Line("right") = {5};
Physical Line("rwin") = {6,7,8,9};
Physical Line("lwin") = {10,11,12,13};
Physical Line("cwin") = {14,15,16};
Physical Surface("domain") = {1};
// vim: set ai et nu ff=unix ft=c:
