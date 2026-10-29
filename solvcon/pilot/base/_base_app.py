# Copyright (c) 2022, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


from PySide6.QtCore import Qt, QAbstractTableModel


class _Accessor(object):
    """
    Helper calss to access data within the configuration table using
    multiple dimensions, currenlty only support 2-dimensions table.

    This class allows users to access data by chaining multiple [],
    supporting string-based.

    :ivar _data: handle input data that allow user can using
        index or string to access it.
    :ivar _header: list of data's column header name.
    :vartype _header: list
    :ivar _dim_idx: using to indicate which dimension is
        currenlty being accessed by __getitem__.
    :vartype _dim_idx: int
    """

    def __init__(self, data, dim_idx=0, header=None):
        self._data = data
        self._header = header
        self._dim_idx = dim_idx

    def __getitem__(self, key):
        if self._dim_idx == 0:
            for row_idx, row in enumerate(self._data):
                if key == row[0]:
                    return _Accessor(self._data[row_idx],
                                     self._dim_idx + 1,
                                     self._header)
        else:
            for idx, ele in enumerate(self._header):
                if key == ele:
                    return self._data[idx]
        raise KeyError(f'Invaild key: {key} not found in table')


class GUIConfig(object):
    """
    Configuration class for the GUI.

    This class provides a configuration interface for the GUI, allowing
    users to set and retrieve parameters related to the simulation.

    :ivar: _tbl_content (list[list]): The content of the configuration
        table.
    :ivar: _col_header (list): The header for the configuration
        table columns.

    :Methods:
        - :py:meth:`data`: Get the value at a specific row and column
            in the configuration table.
        - :py:meth:`setData`: Set the value at a specific row
            and column in the configuration table.
        - :py:meth:`columnHeader`: Get the header for a specific column
            in the configuration table.
        - :py:meth:`editable`: Check if a cell in the configuration
            table is editable.
        - :py:meth:`rowCount`: Get the number of rows in the configuration
            table.
        - :py:meth:`columnCount`: Get the number of columns in the
            configuration table.
    """

    def __init__(self, input_data, col_headers):
        self._tbl_content = input_data
        self._col_header = col_headers

        expected_length = len(self._col_header)
        for i, row in enumerate(self._tbl_content):
            if len(row) != expected_length:
                raise ValueError(
                    f"Row {i} has length {len(row)}, "
                    f"expected {expected_length}"
                )

    def __getitem__(self, key):
        return _Accessor(self._tbl_content, 0, self._col_header)[key]

    def data(self, row, col):
        """
        Get the value at a specific row and column in the configuration table.

        :param row: Row index.
        :type row: int
        :param col: Column index.
        :type col: int
        :return:
            The value at the specified location in the configuration table.
        """
        return self._tbl_content[row][col]

    def setData(self, row, col, value):
        """
        Set the value at a specific row and column in the configuration table.

        :param row: Row index.
        :type row: int
        :param col: Column index.
        :type col: int
        :prarm value: Any
        :return: None
        """
        self._tbl_content[row][col] = value
        return True

    def columnHeader(self, col):
        """
        Get the specific column header in the configuration table.

        :param col: Column index.
        :type col: int
        :return: The header for the specific column.
        :rtype: list[str]
        """
        return self._col_header[col]

    def editable(self, row, col):
        """
        Check if the cell is editable.

        :param row: Row index.
        :type row: int
        :param col: Column index.
        :type col: int
        :return: True if the cell is editable, false otherwise.
        """
        if col == 1:
            return True
        return False

    def rowCount(self):
        """
        Get the number of rows in the configuration table.

        :return: The number of rows.
        """
        return len(self._tbl_content)

    def columnCount(self):
        """
        Get the number of columns in the configuration table.

        :return: The number of columns.
        """
        return len(self._tbl_content[0])


class SolverConfig(GUIConfig):
    """
    Configuration class for the solver.

    This class provides a configuration interface for the solver, allowing
    users to set and retrieve parameters related to the simulation.
    """

    def __init__(self, input_data):
        super().__init__(input_data, ["variable", "value", "description"])


class ConfigTableModel(QAbstractTableModel):
    """
    ConfigTableModel class for displaying configuration data.

    This class inherits from the QAbstractTableModel class and provides a model
    for displaying configuration data using table. It is used in the QTable
    class to display solver and plotting area configurations.

    :ivar config: Configuration object for the model.
    :vartype config: :class:`GUIConfig`

    :Methods:
        - :py:meth:`rowCount`: Get the number of rows in the model.
        - :py:meth:`columnCount`: Get the number of columns in the model.
        - :py:meth:`data`: Get the data at a specific index in the
          model.
        - :py:meth:`setData`: Set the data at a specific index
          in the model.
        - :py:meth:`flags`: Get the flags for a specific index in the
          model.
        - :py:meth:`headerData`: Get the header data
          for a specific section in the model.
        - :py:meth:`insertRow`: Insert a new row of data.
        - :py:meth:`deleteRow`: Delete a row of data.
    """

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config

    def rowCount(self, parent):
        return self.config.rowCount()

    def columnCount(self, parent):
        return self.config.columnCount()

    def data(self, index, role):
        if role in {Qt.DisplayRole, Qt.EditRole}:
            return self.config.data(index.row(), index.column())
        return None

    def setData(self, index, value, role):
        if role == Qt.EditRole:
            self.config.setData(index.row(), index.column(), value)
            return True

    def flags(self, index):
        if self.config.editable(index.row(), index.column()):
            return super().flags(index) | Qt.ItemIsEditable
        return super().flags(index)

    def headerData(self, section, orientation, role):
        if (role == Qt.DisplayRole
                and orientation == Qt.Horizontal):
            return self.config.columnHeader(section)
        return None

    def insertRow(self, data, position=None):
        """
        Insert data at specific row.

        :param data: data to be inserted
        :param position: row index of config._tbl_content
        :return: None
        """
        if len(data) != self.config.columnCount():
            raise ValueError(
                f"Row {position} has length {len(data)}, "
                f"expected {self.config.columnCount()}"
            )
        position = self.rowCount(None) if position is None else position
        if position < 0 or position > self.rowCount(None):
            raise IndexError(f"Row {position} out of range")
        self.config._tbl_content.insert(position, data)
        return None

    def deleteRow(self, position):
        """
        Delete data at specific row.

        :param position: row index of config._tbl_content
        :type position: int
        :return: None
        """
        if position < 0 or position >= self.rowCount(None):
            raise IndexError(f"Row {position} out of range")
        self.config._tbl_content.pop(position)
        return None


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
