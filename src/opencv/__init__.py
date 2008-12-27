#!/usr/bin/env python
# ctypes-opencv - A Python wrapper for OpenCV using ctypes

# Copyright (c) 2008, Minh-Tri Pham
# All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

#    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#    * Neither the name of ctypes-opencv's copyright holders nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# For further inquiries, please contact Minh-Tri Pham at pmtri80@gmail.com.
# ----------------------------------------------------------------------------

from opencv.cxcore import *
from opencv.cv import *
from opencv.highgui import *
from opencv.interfaces import *

try:
    from opencv.ml import *
except ImportError:
    pass


# ----------------------------------------------------------------------------
# Begin of code section contributed by David Bolen
# ----------------------------------------------------------------------------
#
# Publish an optional "cv" namespace to provide access to names in this package
# minus any "cv"/"cv_" prefix.  For example, "cv.Foo" instead of "cvFoo".
# Names without a "cv" prefix remain intact in the new namespace.
#
# Names for which this would result in an invalid identifier (such as constant
# names beginning with numbers) retain a leading underscore (e.g., cv._32F)
#
# In cases (such as CV_SVD) where a structure would overlap a function (cvSVD)
# or where a factory function overlaps the data type (cvPoint2D3f/CvPoint2D3f)
# the structure or data type retains/receives a leading underscore.
#

class namespace(object):
    pass

nsp = namespace()


# Process names in reverse order so functions/factories cvXXX will show up
# before structures (CvXXX) or constants (CV_) and thus functions/factories
# get preference.

for sym, val in sorted(locals().items(), reverse=True):
    if sym.startswith('__'):
        continue

    if sym.lower().startswith('cv'):
        if sym[2:3] == '_' and not sym[3:4].isdigit():
            sname = sym[3:]
        else:
            sname = sym[2:]
    else:
        sname = sym

    # Use underscore to distinguish conflicts
    if hasattr(nsp, sname):
        sname = '_' + sname

    # If still have a conflict, punt and just install full name
    if not hasattr(nsp, sname):
        setattr(nsp, sname, val)
    else:
        setattr(nsp, sym, val)

cv = nsp
del nsp

# ----------------------------------------------------------------------------
# End of code section contributed by David Bolen
# ----------------------------------------------------------------------------
