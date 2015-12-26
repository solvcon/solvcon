/**
 * @author yungyuc / http://yyc.solvcon.net/
 */


var SOLVCON = {
  VERSION: '0.1.4+',
}

SOLVCON.widget = {}

SOLVCON.extend = function (target, source) {

  for (var name in source) {
    if (!target.hasOwnProperty(name)) {
      target[name] = source[name];
    }
  }

}

SOLVCON.makeCachedGetter = function (target, name, methodName) {

  Object.defineProperty(target, name, {
    get: function () {
      var cacheName = "_".concat(name);
      return function () {
        if (!this[cacheName]) {
          this[cacheName] = this[methodName]();
        }
        return this[cacheName];
      }
    }(),
  });

}

// vim: set ff=unix fenc=utf8 nobomb et sw=2 ts=2:
