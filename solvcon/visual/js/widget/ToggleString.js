/**
 * @author yungyuc / http://yyc.solvcon.net/
 */


SOLVCON.widget.ToggleStringReactClass = React.createClass({
  propTypes: {
    toggler: React.PropTypes.func,
    message: React.PropTypes.string,
  },

  render: function () {
    return React.createElement(
      "div",
      {
        onClick: this.props.toggler,
        style: {
          cursor: "pointer",
        },
      },
      React.createElement("span", null, this.props.message)
    );
  },
}); /* end ToggleStringReactClass */

// vim: set ff=unix fenc=utf8 nobomb et sw=2 ts=2:
