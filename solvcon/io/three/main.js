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


var SOLVCON = {
  VERSION: '0.1.4+'
}

SOLVCON.Viewer = function(container) {

  this.renderer = new THREE.WebGLRenderer();
  this.container = container;
  this.container.appendChild(this.renderer.domElement);

  this.renderer.setSize(window.innerWidth, window.innerHeight);

  this.scene = new THREE.Scene();
  this.camera = this.make_camera(0, 0, 10);

  this.controls = this.make_controls(this.camera);
  this.controls.addEventListener('change', this.render.bind(this));

}
SOLVCON.Viewer.prototype = {};
SOLVCON.Viewer.prototype.constructor = SOLVCON.Viewer;

SOLVCON.Viewer.prototype.make_camera = function(xval, yval, zval) {

  var camera = new THREE.PerspectiveCamera(
    10, window.innerWidth / window.innerHeight, 1, 10000);
  camera.position.x = xval;
  camera.position.y = yval;
  camera.position.z = zval;
  return camera;

}

SOLVCON.Viewer.prototype.make_controls = function(camera) {

  var controls = new THREE.TrackballControls(camera);

  controls.rotateSpeed = 1.0;
  controls.zoomSpeed = 1.2;
  controls.panSpeed = 0.1;

  controls.noZoom = false;
  controls.noPan = false;

  controls.staticMoving = true;
  controls.dynamicDampingFactor = 0.3;

  controls.keys = [ 65, 83, 68 ];

  return controls;

}

SOLVCON.Viewer.prototype.animate = function() {

  window.requestAnimationFrame(this.animate.bind(this));
  this.controls.update();

}

SOLVCON.Viewer.prototype.render = function() {
  this.renderer.render(this.scene, this.camera);
}

function make_coordinate() {

  var xcoord_material = new THREE.LineBasicMaterial({
      color: 0x0000ff
  });
  var xcoord_geom = new THREE.Geometry();
  xcoord_geom.vertices.push(
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(1, 0, 0)
  );
  var xcoord_line = new THREE.Line(xcoord_geom, xcoord_material);

  var ycoord_material = new THREE.LineBasicMaterial({
      color: 0x00ff00
  });
  var ycoord_geom = new THREE.Geometry();
  ycoord_geom.vertices.push(
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(0, 1, 0)
  );
  var ycoord_line = new THREE.Line(ycoord_geom, ycoord_material);

  var zcoord_material = new THREE.LineBasicMaterial({
      color: 0xffffff
  });
  var zcoord_geom = new THREE.Geometry();
  zcoord_geom.vertices.push(
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(0, 0, 1)
  );
  var zcoord_line = new THREE.Line(zcoord_geom, zcoord_material);

  return [xcoord_line, ycoord_line, zcoord_line];
}

function make_mesh(ndcrd, fcnds) {

  var mtrl = new THREE.MeshBasicMaterial({
    color: 0x00ffff,
    wireframe: true
  });

  var geom = new THREE.Geometry();
  var ndim = ndcrd[0].length;
  if (2 == ndim) {
    for (it = 0; it < ndcrd.length; it++) {
      crd = ndcrd[it];
      geom.vertices.push(new THREE.Vector3(crd[0], crd[1], 0));
    }
    for (it = 0; it < fcnds.length; it++) {
      nds = fcnds[it];
      geom.faces.push(new THREE.Face3(nds[1], nds[2], nds[3]));
    }
  } else { // 3 == ndim
    for (it = 0; it < ndcrd.length; it++) {
      crd = ndcrd[it];
      geom.vertices.push(new THREE.Vector3(crd[0], crd[1], crd[2]));
    }
    for (it = 0; it < fcnds.length; it++) {
      nds = fcnds[it];
      geom.faces.push(new THREE.Face3(nds[1], nds[2], nds[3]));
    }
  }

  var mesh = new THREE.Mesh(geom, mtrl);

  return mesh;

}

function add_ball(mesh, radius, scene) {

  var mtrl = new THREE.MeshBasicMaterial({color: 0xffffff});
  for (it = 0; it < mesh.geometry.vertices.length; it++) {
    var geom = new THREE.SphereGeometry(radius, 16, 16);
    var ball = new THREE.Mesh(geom, mtrl);
    vec = mesh.geometry.vertices[it];
    ball.position.x = vec.x;
    ball.position.y = vec.y;
    ball.position.z = vec.z;
    scene.add(ball);
  }

}

var Viewer = React.createClass({

  render: function() {

    return React.createElement(
      'div', {className: "container", ref: "container"},
      React.createElement(
        'div', {id: "information"},
        React.createElement('span', null, "Sample Text")
      )
    );

  },

  componentDidMount: function() {

    var viewer = new SOLVCON.Viewer(this.refs.container);
    viewer.animate();

    var coords = make_coordinate();
    viewer.scene.add(coords[0]);
    viewer.scene.add(coords[1]);
    viewer.scene.add(coords[2]);

    var mesh = make_mesh(ndcrd, fcnds);
    viewer.scene.add(mesh);
    add_ball(mesh, ball_radius, viewer.scene);

    viewer.render();

  }

});

function main() {

  ReactDOM.render(
    React.createElement(Viewer, null),
    document.getElementById('content')
  );

}

// vim: set ff=unix fenc=utf8 nobomb et sw=2 ts=2:
