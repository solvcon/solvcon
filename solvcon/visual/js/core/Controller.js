/**
 * @author yungyuc / http://yyc.solvcon.net/
 */


SOLVCON.EventDispatcher = function () {

  var _this = this;

  this.queue = {};

  this.publish = function(evt, data) {

    var queue = _this.queue[evt];

    if (typeof queue === 'undefined') {
      return false;
    }

    for (var it=0; it<queue.length; it++) {
      queue[it](data);
    }

    return true;

  }

  this.subscribe = function(evt, callback) {

    if (typeof _this.queue[evt] === 'undefined') {
      _this.queue[evt] = [];
    }

    _this.queue[evt].push(callback);

  }

} /* end SOLVCON.EventDispatcher */
SOLVCON.EventDispatcher.prototype = {};
SOLVCON.EventDispatcher.prototype.constructor = SOLVCON.EventDispatcher;


SOLVCON.Controller = function (_canvas) {

  SOLVCON.EventDispatcher.call(this);

  var _this = this;

  this.rotateSpeed = 1.0;
  this.zoomSpeed = 1.2;
  this.dynamicDampingFactor = 0.2;

  var _screen = { left: 0, top: 0, width: 0, height: 0 };
  this.updateScreen = function () {
    var domElement = _canvas.holderDomElement;
    var box = domElement.getBoundingClientRect();
    // adjustments come from similar code in the jquery offset() function
    var doc = domElement.ownerDocument.documentElement;
    _screen.left = box.left + window.pageXOffset - doc.clientLeft;
    _screen.top = box.top + window.pageYOffset - doc.clientTop;
    _screen.width = box.width;
    _screen.height = box.height;
  }

  window.addEventListener(
    "resize",
    (function () {
      var resize_timer;
      return function () {
        clearTimeout(resize_timer);
        resize_timer = setTimeout(_this.updateScreen.bind(_this), 100);
      }
    }())
  );

  var getMouseOnCircle = (function () {
    var vector = new THREE.Vector2();
    return function (pageX, pageY) {
      vector.set(
        ( ( pageX - _screen.width * 0.5 - _screen.left ) / ( _screen.width * 0.5 ) ),
        ( ( _screen.height + 2 * ( _screen.top - pageY ) ) / _screen.width ) // screen.width intentional
      );
      return vector;
    };
  }());

  var _target = new THREE.Vector3(0, 0, 0),
  _eye = new THREE.Vector3(),

  _moveCurr = new THREE.Vector2(),
  _movePrev = new THREE.Vector2(),

  _zoomStart = new THREE.Vector2(),
  _zoomEnd = new THREE.Vector2();

  _eye.subVectors(_canvas.camera.position, _target);

  var rotateCamera = function () {

    var axis = new THREE.Vector3(),
        eyeDirection = new THREE.Vector3(),
        cameraUpDirection = new THREE.Vector3(),
        cameraSideDirection = new THREE.Vector3(),
        moveDirection = new THREE.Vector3(),
        quaternion = new THREE.Quaternion(),
        angle;

    return function () {

      moveDirection.set(_moveCurr.x - _movePrev.x, _moveCurr.y - _movePrev.y, 0);
      angle = moveDirection.length();

      _eye.copy(_canvas.camera.position).sub(_target);

      eyeDirection.copy(_eye).normalize();
      cameraUpDirection.copy(_canvas.camera.up).normalize();
      cameraSideDirection.crossVectors(cameraUpDirection, eyeDirection);

      cameraUpDirection.setLength(_moveCurr.y - _movePrev.y);
      cameraSideDirection.setLength(_moveCurr.x - _movePrev.x);

      moveDirection.copy(cameraUpDirection.add(cameraSideDirection));

      axis.crossVectors(moveDirection, _eye).normalize();

      angle *= _this.rotateSpeed;
      quaternion.setFromAxisAngle(axis, angle);

      _eye.applyQuaternion(quaternion);
      _canvas.camera.up.applyQuaternion(quaternion);
      _canvas.camera.position.addVectors(_target, _eye);
      _canvas.camera.lookAt(_target);

    }

  }();

  function zoomCamera () {

    var factor;

    factor = 1.0 + (_zoomEnd.y - _zoomStart.y) * _this.zoomSpeed;

    if (factor !== 1.0 && factor > 0.0) {

      _eye.multiplyScalar(factor);
      _zoomStart.y += (_zoomEnd.y - _zoomStart.y) * _this.dynamicDampingFactor;

    }

    _canvas.camera.position.addVectors(_target, _eye);
    _canvas.camera.lookAt(_target);

  };

  function mousedown (evt) {

    _canvas.holderDomElement.addEventListener(
      'mouseup', mouseup, false
    );
    _canvas.holderDomElement.addEventListener(
      'mousemove', mousemove, false
    );

    _moveCurr.copy(getMouseOnCircle(evt.pageX, evt.pageY));
    _movePrev.copy(_moveCurr);

    console.log("mousedown");

  }

  function mousemove (evt) {

    _moveCurr.copy(getMouseOnCircle(evt.pageX, evt.pageY));

    rotateCamera();

    _movePrev.copy(_moveCurr);

    console.log("mousemove");

    _this.publish("refresh", null);

  }

  function mouseup (evt) {

    _canvas.holderDomElement.removeEventListener(
      'mousemove', mousemove, false
    );

    console.log("mouseup");

  }

  _canvas.holderDomElement.addEventListener(
    'mousedown', mousedown, false
  );

  function mousewheel (evt) {

    _zoomEnd.copy(_zoomStart);

    var delta = 0;

    if (evt.wheelDelta) { // WebKit / Opera / Explorer 9

      delta = evt.wheelDelta / 40;

    } else if (evt.detail) { // Firefox

      delta = - evt.detail / 3;

    }

    _zoomStart.y += delta * 0.01;

    zoomCamera();

    console.log("mousewheel");

    _this.publish("refresh", null);

  }

  _canvas.holderDomElement.addEventListener(
    'mousewheel', mousewheel, false
  );

} /* end SOLVCON.Controller */
SOLVCON.Controller.prototype = Object.create(
  SOLVCON.EventDispatcher.prototype
);
SOLVCON.Controller.prototype.constructor = SOLVCON.Controller;

// vim: set ff=unix fenc=utf8 nobomb et sw=2 ts=2:
