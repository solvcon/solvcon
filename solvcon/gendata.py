# -*- coding: UTF-8 -*-
#
# Copyright (c) 2008, Yung-Yu Chen <yyc@solvcon.net>
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the SOLVCON nor the names of its contributors may be
#   used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

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
    Dictionary in which key can only be assigned to a value once.
    """
    def __setitem__(self, key, item):
        """
        >>> dct = SingleAssignDict()
        >>> # creating a new key is OK:
        >>> dct['a'] = 10
        >>> # resetting an existing key isn't allowed:
        >>> dct['a'] = 20
        Traceback (most recent call last):
          ...
        IndexError: Resetting key "a" (20 to 10) isn't allowed.
        >>> # even resetting a key to its current value isn't allowed:
        >>> dct['a'] = dct['a']
        Traceback (most recent call last):
          ...
        IndexError: Resetting key "a" (10 to 10) isn't allowed.
        """
        if key in self:
            raise IndexError(
                "Resetting key \"%s\" (%s to %s) isn't allowed." % (
                    str(key), str(item), str(self[key])))
        super(SingleAssignDict, self).__setitem__(key, item)

class TypeNameRegistry(SingleAssignDict, AttributeDict):
    """
    Registry class for the name of types.
    """
    def register(self, tobj):
        self[tobj.__name__] = tobj
        return tobj

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
