/**
 * @author yungyuc / http://yyc.solvcon.net/
 */


SOLVCON.Surface = function(name, ndcrd, fcnds) {
  this.type = "SOLVCON.Surface";
  THREE.Object3D.call(this);

  this.name = name;
  this.SurfaceMaterialType = THREE.MeshBasicMaterial;
  // MeshLambertMaterial for light.

  // Vertices.
  this.vertices = [];
  var ndim = ndcrd[0].length;
  if (2 === ndim) {
    for (var it = 0; it < ndcrd.length; it++) {
      var crd = ndcrd[it];
      this.vertices.push(new THREE.Vector3(crd[0], crd[1], 0));
    }
  } else {
    for (var it = 0; it < ndcrd.length; it++) {
      var crd = ndcrd[it];
      this.vertices.push(new THREE.Vector3(crd[0], crd[1], crd[2]));
    }
  }

  // Surface.
  var geometry = new THREE.Geometry();
  geometry.vertices = this.vertices;
  var faces = [];
  for (var it = 0; it < fcnds.length; it++) {
    var nds = fcnds[it];
    if (2 === nds[0]) { // quadrilateral.
      faces.push(new THREE.Face3(nds[1], nds[2], nds[3]));
      faces.push(new THREE.Face3(nds[3], nds[4], nds[1]));
    } else if (3 === nds[0]) { // triangle.
      faces.push(new THREE.Face3(nds[1], nds[2], nds[3]));
    } else {
      throw "wat?!"
    }
  }
  geometry.faces = faces;
  geometry.computeFaceNormals();
  var material = new this.SurfaceMaterialType({
    color: 0xff00ff,
    side: THREE.DoubleSide,
  });
  this.surface = new THREE.Mesh(geometry, material);
  this.add(this.surface);

  // Wireframe.
  this.wireframe = new THREE.Object3D();
  var linem = new THREE.LineBasicMaterial({
    color: 0x0000ff,
    linewidth: 2,
  });
  for (var it = 0; it < fcnds.length; it++) {
    var nds = fcnds[it];
    var lineg = new THREE.Geometry();
    if (2 === nds[0]) { // quadrilateral.
      lineg.vertices.push(
        this.vertices[nds[1]],
        this.vertices[nds[2]],
        this.vertices[nds[3]],
        this.vertices[nds[4]],
        this.vertices[nds[1]]
      );
    } else if (3 === nds[0]) { // triangle.
      lineg.vertices.push(
        this.vertices[nds[1]],
        this.vertices[nds[2]],
        this.vertices[nds[3]],
        this.vertices[nds[1]]
      );
    } else {
      throw "wat?!"
    }
    this.wireframe.add(new THREE.Line(lineg, linem));
  }
  this.add(this.wireframe);
}

SOLVCON.Surface.prototype = Object.create(THREE.Object3D.prototype);
SOLVCON.Surface.prototype.constructor = SOLVCON.Surface;
SOLVCON.extend(SOLVCON.Surface.prototype, SOLVCON.CanvasClient.prototype);

SOLVCON.Surface.prototype.createReactClass = function () {

  var _this = this;
  return React.createClass({

    getInitialState: function() {
      return {
        surfaceVisible: true,
        wireframeVisible: true,
      };
    },

    render: function () {
      var _element = this;
      return React.createElement(
        'div',
        {
          className: "element",
        },
        React.createElement('span', {}, _this.name),
        React.createElement(SOLVCON.widget.ToggleStringReactClass, {
          toggler: function(evt) {
            _this.surface.visible = !_this.surface.visible;
            _element.setState({surfaceVisible: _this.surface.visible});
            _this.canvas.controller.publish("refresh", null);
          },
          message: "Surface: ".concat(this.state.surfaceVisible ? "on" : "off")
        }),
        React.createElement(SOLVCON.widget.ToggleStringReactClass, {
          toggler: function(evt) {
            _this.wireframe.visible = !_this.wireframe.visible;
            _element.setState({wireframeVisible: _this.wireframe.visible});
            _this.canvas.controller.publish("refresh", null);
          },
          message: "Wireframe: ".concat(this.state.wireframeVisible ? "on" : "off")
        })
      ); /* end React.createElement */
    },

  }); /* end React.createClass */

}; /* end createReactClass */
SOLVCON.makeCachedGetter(
  SOLVCON.Surface.prototype, "ReactClass", "createReactClass"
);

// vim: set ff=unix fenc=utf8 nobomb et sw=2 ts=2:
