require.config({
    map: {
        "*": {
            "threejs": "nbextensions/pythreejs/three.js/build/three",
            "pythreejs": "nbextensions/pythreejs/pythreejs",
        }
    },
});

/* 
 * Module header start.
 */
define(
[
    "nbextensions/widgets/widgets/js/widget",
    "nbextensions/widgets/widgets/js/manager",
    "base/js/utils",
    "underscore",
    "threejs",
    "pythreejs"
],
function(widget, manager, utils, _, THREE, pythreejs) {
/*
 * Module header stop.
 * Begin module content, indentation level is zero.
 */

console.log("loading solvcon");
var register = {};

var CoordinateMark = function(camera, size, linewidth) {
    this.type = "CoordinateMark";
    THREE.Object3D.call(this);

    if (size === undefined) {
        size = 0.01;
    }
    if (linewidth === undefined) {
        linewidth = 3.0;
    }

    var createLine = function (vec1, vec2, color, linewidth) {
        var mtrl = new THREE.LineBasicMaterial({ color: color, linewidth: linewidth });
        var geom = new THREE.Geometry();
        geom.vertices.push(vec1, vec2);
        return new THREE.Line(geom, mtrl);
    } /* end createLine */

    this.add( 
        createLine(
            new THREE.Vector3(0, 0, 0), new THREE.Vector3(size, 0, 0), 0xff0000, linewidth
        ),
        createLine(
            new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, size, 0), 0x00ff00, linewidth
        ),
        createLine(
            new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, size), 0x0000ff, linewidth
        )
    );

    this.update = function() {
        // http://stackoverflow.com/a/13091694/1805420
        // http://stackoverflow.com/a/31835326/1805420
        var vector = new THREE.Vector3(-0.8, -0.8, 0.5);
        camera.updateMatrixWorld();
        vector.unproject(camera);
        vector.sub(camera.position).normalize();
        this.position.copy(camera.position);
        this.position.add(vector.multiplyScalar(camera.near*2));
    }; /* end update */

} /* end CoordinateMark */
CoordinateMark.prototype = Object.create(THREE.Object3D.prototype);
CoordinateMark.prototype.constructor = CoordinateMark;

var ViewerView = pythreejs.RendererView.extend({
    render: function() {
        var that = this;
        this.view_promises = pythreejs.RendererView.prototype.render.call(this);
        this.view_promises = Promise.resolve(this.view_promises).then(function(objs) {
            that.coordmark = new CoordinateMark(that.camera.obj);
            that.scene.obj.add(that.coordmark);
        });
        return this.view_promises;
    },

    animate: function() {
        this._animation_frame = false;
        if (this._update_loop) {
            this.schedule_update();
        }
        this.trigger('animate:update', this);
        if (this.coordmark) {
            this.coordmark.update();
        }
        if (this._render) {
            this.effectrenderer.render(this.scene.obj, this.camera.obj)
            this._render = false;
        }
    },
});
register['ViewerView'] = ViewerView;

return register;

/* 
 * module content ends.
 */
}); // end define
