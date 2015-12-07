/*
 * Copyright (c) 2015, Yung-Yu Chen <yyc@solvcon.net>
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - Neither the name of the software nor the names of its contributors may be
 *   used to endorse or promote products derived from this software without
 *   specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

var camera, scene, renderer;
var geometry, material, mesh;

init();
animate();
//renderer.render(scene, camera);

function init() {

    camera = new THREE.PerspectiveCamera(10, window.innerWidth / window.innerHeight, 1, 10000);
    var shift = 0;
    camera.position.x = shift;
    camera.position.y = shift;
    camera.position.z = 50;
    //camera = new THREE.OrthographicCamera(width / - 2, width / 2, height / 2, height / - 2, 1, 10000);
    
    controls = new THREE.TrackballControls(camera);

    controls.rotateSpeed = 1.0;
    controls.zoomSpeed = 1.2;
    controls.panSpeed = 0.1;

    controls.noZoom = false;
    controls.noPan = false;

    controls.staticMoving = true;
    controls.dynamicDampingFactor = 0.3;

    controls.keys = [ 65, 83, 68 ];

    controls.addEventListener( 'change', render );    
  
    scene = new THREE.Scene();

    var triangle_material = new THREE.MeshBasicMaterial({
        color: 0x00ffff,
        wireframe: true
    });
    var triangle_geom = new get_geometry();

    var ball_material = new THREE.MeshBasicMaterial({color: 0xffffff});
    for (it = 0; it < triangle_geom.vertices.length; it++) {
        var ball_geom = new THREE.SphereGeometry(ball_radius, 16, 16);
        var ball = new THREE.Mesh(ball_geom, ball_material);
        vec = triangle_geom.vertices[it];
        ball.position.x = vec.x;
        ball.position.y = vec.y;
        ball.position.z = vec.z;
        scene.add(ball);
    }

    triangle_geom.computeBoundingSphere();
    mesh = new THREE.Mesh(triangle_geom, triangle_material);
    scene.add(mesh);
    
    var xcoord_material = new THREE.LineBasicMaterial({
        color: 0x0000ff
    });
    var xcoord_geom = new THREE.Geometry();
    xcoord_geom.vertices.push(
        new THREE.Vector3(0, 0, 0),
        new THREE.Vector3(1, 0, 0)
    );
    var xcoord_line = new THREE.Line(xcoord_geom, xcoord_material);
    scene.add(xcoord_line);

    var ycoord_material = new THREE.LineBasicMaterial({
        color: 0x00ff00
    });
    var ycoord_geom = new THREE.Geometry();
    ycoord_geom.vertices.push(
        new THREE.Vector3(0, 0, 0),
        new THREE.Vector3(0, 1, 0)
    );
    var ycoord_line = new THREE.Line(ycoord_geom, ycoord_material);
    scene.add(ycoord_line);

    var zcoord_material = new THREE.LineBasicMaterial({
        color: 0xffffff
    });
    var zcoord_geom = new THREE.Geometry();
    zcoord_geom.vertices.push(
        new THREE.Vector3(0, 0, 0),
        new THREE.Vector3(0, 0, 1)
    );
    var zcoord_line = new THREE.Line(zcoord_geom, zcoord_material);
    
    scene.add(zcoord_line);
    
    renderer = new THREE.WebGLRenderer();
    renderer.setSize(window.innerWidth, window.innerHeight);

    document.body.appendChild(renderer.domElement);

    renderer.render( scene, camera );
}

function animate() {
    requestAnimationFrame( animate );
    controls.update();
}

function render() {
    renderer.render( scene, camera );
    stats.update();
}

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:
