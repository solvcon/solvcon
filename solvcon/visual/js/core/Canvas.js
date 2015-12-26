/**
 * @author yungyuc / http://yyc.solvcon.net/
 */


SOLVCON.Canvas = function(holderDomElement) {
  this.type = "SOLVCON.Canvas";
  this.holderDomElement = null;

  this._renderer = new THREE.WebGLRenderer();
  this._scene = new THREE.Scene();
  this._camera = new THREE.PerspectiveCamera(10, 1, 1, 10000);
  this._camera.position.set(0, 0, 20);
  this._scene.add(this._camera);

  this.controller = null;

  this.lights = this.createLights();
  for (var it=0; it<this.lights.length; it++) {
    this._scene.add(this.lights[it]);
  }
} /* end SOLVCON.Canvas */

SOLVCON.Canvas.prototype = Object.create(null);
SOLVCON.Canvas.prototype.constructor = SOLVCON.Canvas;

SOLVCON.Canvas.prototype.createLights = function () {
  var light, lights = [];
	light = new THREE.SpotLight(0xffffff);
	light.position.set(1000, 1000, 1000);
  lights.push(light);
	light = new THREE.SpotLight(0xffffff);
	light.position.set(-1000, 1000, -1000);
  lights.push(light);
	light = new THREE.SpotLight(0xffffff);
	light.position.set(1000, -1000, -1000);
  lights.push(light);
  return lights;
}

SOLVCON.Canvas.prototype.add = function (child) {
  this._scene.add(child);
  child.attachCanvas(this);
}

Object.defineProperty(SOLVCON.Canvas.prototype, "children", {
  get: function () {
    return this._scene.children;
  },
});

Object.defineProperty(SOLVCON.Canvas.prototype, "camera", {
  get: function () {
    return this._camera;
  },
});

SOLVCON.Canvas.prototype.createReactClass = function () {

  var _this = this;

  return React.createClass({

    componentDidMount: function () {

      _this.holderDomElement = ReactDOM.findDOMNode(this.refs.container);
      _this.holderDomElement.appendChild(_this._renderer.domElement);
      _this.controller = new SOLVCON.Controller(_this);
      _this.controller.updateScreen()
      _this.update();

      window.addEventListener(
        "resize",
        (function () {
          var timer;
          return function () {
            clearTimeout(timer);
            timer = setTimeout(_this.update.bind(_this), 100);
          }
        }())
      );

      _this.controller.subscribe(
        "refresh",
        function(data) {
          _this.update();
        }
      );

    },

    render: function() {

     elements = _this.children.map(function (it) {
        ReactClass = it.ReactClass;
        if (ReactClass) {
          return React.createElement(it.ReactClass, {key: it.id});
        } else {
          return null;
        }
      });

      return React.createElement(
        "div", {
          id: "content",
          ref: "content",
        },
        React.createElement(
          "div", {
            id: "elements",
            ref: "elements",
          },
          elements
        ),
        React.createElement(
          "div", {
            id: "container",
            ref: "container",
          }
        )
      );

    },

    componentDidUpdate: function() {
      _this.update();
    },

  }); /* end React.createClass */

}; /* end createReactClass */
SOLVCON.makeCachedGetter(
  SOLVCON.Canvas.prototype, "ReactClass", "createReactClass"
);

SOLVCON.Canvas.prototype.update = function () {
  var width = this.holderDomElement.clientWidth;
  var height = this.holderDomElement.clientHeight;

  this._renderer.setSize(width, height);
  this._camera.aspect = width/height;
  this._camera.updateProjectionMatrix();
  this._renderer.render(this._scene, this._camera);
}


SOLVCON.CanvasClient = function () {
} /* end SOLVCON.CanvasClient */

SOLVCON.CanvasClient.prototype = {
  attachCanvas: function (canvas) {
    if (! canvas instanceof SOLVCON.Canvas) {
      throw new Error("canvas should be SOLVCON.Canvas");
    }
    this.canvas = canvas;
    return this; // cascading.
  },
} /* end SOLVCON.CanvasClient.prototype */

// vim: set ff=unix fenc=utf8 nobomb et sw=2 ts=2:
