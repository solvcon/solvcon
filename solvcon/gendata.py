# -*- coding: UTF-8 -*-
# Copyright (C) 2008-2009 by Yung-Yu Chen.  See LICENSE.txt for terms of usage.

"""
Generic data structures.
"""

class AttributeDict(dict):
    """
    Dictionary form which key can be assessed as attribute.
    """
    def __getattr__(self, name):
        """
        Consult self dictionary for attribute.  It's a shorthand.
        """
        return self[name]
    def __setattr__(self, name, value):
        """
        Save to self dictionary first, then self object table.

        @note: This method is overriden as a stupid-preventer.  It makes
        attribute setting consistent with attribute getting.
        """
        if name in self:
            self[name] = value
        else:
            super(AttributeDict, self).__setattr__(name, value)

class DefaultDict(dict):
    """
    Dictionary with default values.

    @cvar defdict: dictionary for default values.
    @type defdict: dict
    """
    defdict = {}
    def __init__(self, *args, **kw):
        """
        Assign default values to self after initiated.
        """
        super(DefaultDict, self).__init__(*args, **kw)
        for key in self.defdict:
            self[key] = self.defdict[key]

class SingleAssignDict(dict):
    """
    Dictionary in which key/value can only be assigned once.
    """
    def __setitem__(self, key, item):
        """
        Check for duplicated assignment.
        """
        if key in self:
            raise IndexError, "Cannot reset value for key=%s to override %s."%(
                str(key), str(self[key]))
        super(SingleAssignDict, self).__setitem__(key, item)

class Timer(dict):
    """
    Timer dictionary with increase method.
    """
    def __init__(self, *args, **kw):
        self.vtype = kw.pop('vtype', float)
        super(Timer, self).__init__(*args, **kw)
    def increase(self, key, delta):
        self[key] = self.get(key, self.vtype(0)) + self.vtype(delta)

# Define the base metaclass for classes want binders.
def bind(self):
    """
    Set up pointers.  All attributes which are pointers have to be
    initialized here.
    """
    pass
def unbind(self):
    """
    Release pointer.
    """
    for key in self._pointers_:
        setattr(self, key, None)
@property
def is_bound(self):
    """
    Determine if all the pointers are fully bound.
    """
    for key in self._pointers_:
        if getattr(self, key, None)==None:
            return False
    return True
@property
def is_unbound(self):
    """
    Determine if all the pointers are fully unbound.
    """
    for key in self._pointers_:
        if getattr(self, key, None)!=None:
            return False
    return True
class TypeWithBinder(type):
    """
    Meta class to make classes with ctypes pointers or containers with ctypes
    pointers.  The type will feather classes with bind/unbind methods along
    with is_bound/is_unbound properties.  The names of pointer variables have
    to be listed in _pointers_ class list variable.

    The bind/unbind methods are designed to be applied to pointers used by the
    instance.  is_bound/is_unbound properties can test for if pointers are
    fully bound or fully unbound to the instance, respectively.  You have to
    override the bind method and initiate pointers in it rather than in other
    method.  You can leave it alone if you don't need it.  Be sure to enter
    correct entries into the _pointers_ class variable.

    @cvar _pointers_: a list containing names of variables for ctypes pointers,
        ctypes structures, or containers that hold ctypes pointers.  The list
        would be used in binding/unbinding process.  Subclassing does not
        override the content of this list.  The names defined in the
        superclasses will be prepended in front of anything in the list defined
        in the subclass.
    @ctype _pointers_: list
    """
    def __new__(cls, name, bases, namespace):
        # incremental modification of _pointers_.
        pointers = []
        for base in bases:
            pointers.extend(getattr(base, '_pointers_', []))
        pointers.extend(namespace.get('_pointers_', []))
        namespace['_pointers_'] = pointers
        # supply the default binding methods and properties for classes without
        # their definitions.
        for key in ('bind', 'unbind', 'is_bound', 'is_unbound'):
            if namespace.get(key, None) != None:
                continue
            nokeyinbases = True
            for base in bases:
                if getattr(base, key, None) != None:
                    nokeyinbases = False
                    break
            if nokeyinbases:
                namespace[key] = globals()[key]
        # return.
        return super(TypeWithBinder, cls).__new__(cls, name, bases, namespace)
