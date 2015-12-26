/**
 * @author yungyuc / http://yyc.solvcon.net/
 */


SOLVCON.Coordinate = function(xcolor, ycolor, zcolor) {

  this.type = "SOLVCON.Coordinate";
  THREE.Object3D.call(this);

  this.canvas = null;

  this.add( 
    this.createLine(
      new THREE.Vector3(0, 0, 0), new THREE.Vector3(1, 0, 0), xcolor
    ),
    this.createLine(
      new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 1, 0), ycolor
    ),
    this.createLine(
      new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, 1), zcolor
    )
  );

} /* end SOLVCON.Cooordinate */

SOLVCON.Coordinate.prototype = Object.create(THREE.Object3D.prototype);
SOLVCON.Coordinate.prototype.constructor = SOLVCON.Coordinate;
SOLVCON.extend(SOLVCON.Coordinate.prototype, SOLVCON.CanvasClient.prototype);

SOLVCON.Coordinate.prototype.createLine = function (vec1, vec2, color) {

  var mtrl = new THREE.LineBasicMaterial({ color: color });
  var geom = new THREE.Geometry();
  geom.vertices.push(vec1, vec2);
  return new THREE.Line(geom, mtrl);

} /* end createLine */

SOLVCON.Coordinate.prototype.createReactClass = function () {

  var _this = this;

  return React.createClass({

    getInitialState: function() {
      return {
        visible: _this.visible,
      };
    },

    render: function () {
      var _element = this;
      return React.createElement(
        'div',
        {
          id: "coordinate",
          className: "element",
        },
        React.createElement('span', {}, "Coordinate"),
        React.createElement(SOLVCON.widget.ToggleStringReactClass, {
          toggler: function (evt) {
            _this.visible = !_this.visible;
            _element.setState({visible: _this.visible});
            _this.canvas.controller.publish("refresh", null);
          },
          message: this.state.visible ? "Displayed" : "Hidden",
        })
      ); /* end React.createElement */
    },

  }); /* end React.createClass */

}; /* end createReactClass */
SOLVCON.makeCachedGetter(
  SOLVCON.Coordinate.prototype, "ReactClass", "createReactClass"
);

// vim: set ff=unix fenc=utf8 nobomb et sw=2 ts=2:
