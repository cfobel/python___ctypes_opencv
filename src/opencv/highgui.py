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

from ctypes import *
from cxcore import *
from cxcore import _cvReleaseImage


#-----------------------------------------------------------------------------
# Simple GUI
#-----------------------------------------------------------------------------


CV_WINDOW_AUTOSIZE = 1

# Creates window
cvNamedWindow = cfunc('cvNamedWindow', _hgDLL, c_int,
    ('name', c_char_p, 1), # const char* name
    ('flags', c_int, 1, 1), # int flags 
)
cvNamedWindow.__doc__ = """int cvNamedWindow(const char* name, int flags)

Creates window
"""

# Destroys a window
cvDestroyWindow = cfunc('cvDestroyWindow', _hgDLL, None,
    ('name', c_char_p, 1), # const char* name 
)
cvDestroyWindow.__doc__ = """void cvDestroyWindow(const char* name)

Destroys a window
"""

# Destroys all the HighGUI windows
cvDestroyAllWindows = cfunc('cvDestroyAllWindows', _hgDLL, None,
)
cvDestroyAllWindows.__doc__ = """void cvDestroyAllWindows(oi)

Destroys all the HighGUI windows
"""

# Sets window size
cvResizeWindow = cfunc('cvResizeWindow', _hgDLL, None,
    ('name', c_char_p, 1), # const char* name
    ('width', c_int, 1), # int width
    ('height', c_int, 1), # int height 
)
cvResizeWindow.__doc__ = """void cvResizeWindow(const char* name, int width, int height)

Sets window size
"""

# Sets window position
cvMoveWindow = cfunc('cvMoveWindow', _hgDLL, None,
    ('name', c_char_p, 1), # const char* name
    ('x', c_int, 1), # int x
    ('y', c_int, 1), # int y 
)
cvMoveWindow.__doc__ = """void cvMoveWindow(const char* name, int x, int y)

Sets window position
"""

# Gets window handle by name
cvGetWindowHandle = cfunc('cvGetWindowHandle', _hgDLL, c_void_p,
    ('name', c_char_p, 1), # const char* name 
)
cvGetWindowHandle.__doc__ = """void* cvGetWindowHandle(const char* name)

Gets window handle by name
"""

# Gets window name by handle
cvGetWindowName = cfunc('cvGetWindowName', _hgDLL, c_void_p,
    ('window_handle', c_void_p, 1), # void* window_handle 
)
cvGetWindowName.__doc__ = """constchar* cvGetWindowName(void* window_handle)

Gets window name by handle
"""

# Shows the image in the specified window
cvShowImage = cfunc('cvShowImage', _hgDLL, None,
    ('name', c_char_p, 1), # const char* name
    ('image', CvArr_p, 1), # const CvArr* image 
)
cvShowImage.__doc__ = """void cvShowImage(const char* name, const CvArr* image)

Shows the image in the specified window
"""

# Creates the trackbar and attaches it to the specified window
CvTrackbarCallback = CFUNCTYPE(None, # void
    c_int) # int pos

# Creates the trackbar and attaches it to the specified window
_cvCreateTrackbar = cfunc('cvCreateTrackbar', _hgDLL, c_int,
    ('trackbar_name', c_char_p, 1), # const char* trackbar_name
    ('window_name', c_char_p, 1), # const char* window_name
    ('value', ByRefArg(c_int), 1), # int* value
    ('count', c_int, 1), # int count
    ('on_change', CallableToFunc(CvTrackbarCallback), 1), # CvTrackbarCallback on_change 
)

def cvCreateTrackbar(trackbar_name, window_name, value, count, on_change):
    """int cvCreateTrackbar( const char* trackbar_name, const char* window_name, c_int_or_int value, int count, CvTrackbarCallback on_change )

    Creates the trackbar and attaches it to the specified window
    [ctypes-opencv] value can be a c_int or int, if it is a c_int, it holds the current value of the trackbar
    """
    if not isinstance(value, c_int):
        value = c_int(value)
    return _cvCreateTrackbar(trackbar_name, window_name, value, count, on_change)

# Retrieves trackbar position
cvGetTrackbarPos = cfunc('cvGetTrackbarPos', _hgDLL, c_int,
    ('trackbar_name', c_char_p, 1), # const char* trackbar_name
    ('window_name', c_char_p, 1), # const char* window_name 
)
cvGetTrackbarPos.__doc__ = """int cvGetTrackbarPos(const char* trackbar_name, const char* window_name)

Retrieves trackbar position
"""

# Sets trackbar position
cvSetTrackbarPos = cfunc('cvSetTrackbarPos', _hgDLL, None,
    ('trackbar_name', c_char_p, 1), # const char* trackbar_name
    ('window_name', c_char_p, 1), # const char* window_name
    ('pos', c_int, 1), # int pos 
)
cvSetTrackbarPos.__doc__ = """void cvSetTrackbarPos(const char* trackbar_name, const char* window_name, int pos)

Sets trackbar position
"""

# Assigns callback for mouse events
CV_EVENT_MOUSEMOVE = 0
CV_EVENT_LBUTTONDOWN = 1
CV_EVENT_RBUTTONDOWN = 2
CV_EVENT_MBUTTONDOWN = 3
CV_EVENT_LBUTTONUP = 4
CV_EVENT_RBUTTONUP = 5
CV_EVENT_MBUTTONUP = 6
CV_EVENT_LBUTTONDBLCLK = 7
CV_EVENT_RBUTTONDBLCLK = 8
CV_EVENT_MBUTTONDBLCLK = 9

CV_EVENT_FLAG_LBUTTON = 1
CV_EVENT_FLAG_RBUTTON = 2
CV_EVENT_FLAG_MBUTTON = 4
CV_EVENT_FLAG_CTRLKEY = 8
CV_EVENT_FLAG_SHIFTKEY = 16
CV_EVENT_FLAG_ALTKEY = 32

CvMouseCallback = CFUNCTYPE(None, # void
    c_int, # int event
    c_int, # int x
    c_int, # int y
    c_int, # int flags
    c_void_p) # void* param

# Assigns callback for mouse events
cvSetMouseCallback = cfunc('cvSetMouseCallback', _hgDLL, None,
    ('window_name', c_char_p, 1), # const char* window_name
    ('on_mouse', CallableToFunc(CvMouseCallback), 1), # CvMouseCallback on_mouse
    ('param', c_void_p, 1, None), # void* param
)
cvSetMouseCallback.__doc__ = """void cvSetMouseCallback( const char* window_name, CvMouseCallback on_mouse, void* param=NULL )

Assigns callback for mouse events
"""

# Waits for a pressed key
_cvWaitKey = cfunc('cvWaitKey', _hgDLL, c_int,
    ('delay', c_int, 1, 0), # int delay
)
def cvWaitKey(delay=0):
    """int cvWaitKey(int delay=0)

    Waits for a pressed key
    [ctypes-opencv] returns -1 if no key is pressed, or a string representing the key character
    """
    z = _cvWaitKey(delay)
    return z if z <= 0 else '%c' %z


#-----------------------------------------------------------------------------
# Loading and Saving Images
#-----------------------------------------------------------------------------


CV_LOAD_IMAGE_UNCHANGED = -1 # 8 bit, color or gray - deprecated, use CV_LOAD_IMAGE_ANYCOLOR
CV_LOAD_IMAGE_GRAYSCALE =  0 # 8 bit, gray
CV_LOAD_IMAGE_COLOR     =  1 # 8 bit unless combined with CV_LOAD_IMAGE_ANYDEPTH, color
CV_LOAD_IMAGE_ANYDEPTH  =  2 # any depth, if specified on its own gray by itself
                             # equivalent to CV_LOAD_IMAGE_UNCHANGED but can be modified
                             # with CV_LOAD_IMAGE_ANYDEPTH
CV_LOAD_IMAGE_ANYCOLOR  =  4


# ------ List of methods not to be called by a user ------

# Loads an image from file
_cvLoadImage = cfunc('cvLoadImage', _hgDLL, IplImage_p,
    ('filename', c_char_p, 1), # const char* filename
    ('iscolor', c_int, 1, 1), # int iscolor
)

# ------ List of methods a user should call ------


# Loads an image from file
def cvLoadImage(filename, iscolor=1):
    """IplImage* cvLoadImage(const char* filename, int iscolor=1)

    Loads an image from file
    """
    z = _cvLoadImage(filename, iscolor)
    sdAdd_autoclean(z, _cvReleaseImage)
    return z

# Saves an image to the file
cvSaveImage = cfunc('cvSaveImage', _hgDLL, c_int,
    ('filename', c_char_p, 1), # const char* filename
    ('image', CvArr_p, 1), # const CvArr* image 
)
cvSaveImage.__doc__ = """int cvSaveImage(const char* filename, const CvArr* image)

Saves an image to the file
"""


#-----------------------------------------------------------------------------
# Video I/O Functions
#-----------------------------------------------------------------------------


CV_CAP_ANY = 0     # autodetect
CV_CAP_MIL = 100     # MIL proprietary drivers
CV_CAP_VFW = 200     # platform native
CV_CAP_V4L = 200
CV_CAP_V4L2 = 200
CV_CAP_FIREWARE = 300     # IEEE 1394 drivers
CV_CAP_IEEE1394 = 300
CV_CAP_DC1394 = 300
CV_CAP_CMU1394 = 300
CV_CAP_STEREO = 400     # TYZX proprietary drivers
CV_CAP_TYZX = 400
CV_TYZX_LEFT = 400
CV_TYZX_RIGHT = 401
CV_TYZX_COLOR = 402
CV_TYZX_Z = 403
CV_CAP_QT = 500     # Quicktime

# CvCapture, supposed to be a black box
class CvCapture(_Structure):
    fields = []
    
    def __del__(self):
        _cvReleaseCapture(pointer(self))
        
CvCapture_p = POINTER(CvCapture)
CvCapture_r = ByRefArg(CvCapture)
    
# CvCapture
class CvVideoWriter(_Structure):
    _fields_ = [] # seriously, no field at all
CvVideoWriter_p = POINTER(CvVideoWriter)
    
# Minh-Tri's hacks
sdHack_del(CvVideoWriter_p)

_cvReleaseCapture = cfunc('cvReleaseCapture', _hgDLL, None,
    ('capture', ByRefArg(CvCapture_p), 1), # CvCapture** capture 
)

_cvCreateFileCapture = cfunc('cvCreateFileCapture', _hgDLL, CvCapture_p,
    ('filename', c_char_p, 1), # const char* filename 
)

# Initializes capturing video from file
def cvCreateFileCapture(filename):
    """CvCapture cvCreateFileCapture(const char* filename)

    Initializes capturing video from file
    [ctypes-opencv] returns None if no capture is created
    """
    z = _cvCreateFileCapture(filename)
    # sdAdd_autoclean(z, _cvReleaseCapture)
    # return z
    return z.content if bool(z) else None

cvCaptureFromFile = cvCreateFileCapture
cvCaptureFromAVI = cvCaptureFromFile

_cvCreateCameraCapture = cfunc('cvCreateCameraCapture', _hgDLL, CvCapture_p,
    ('index', c_int, 1), # int index 
)

# Initializes capturing video from camera
def cvCreateCameraCapture(index):
    """CvCapture cvCreateCameraCapture(int index)

    Initializes capturing video from camera
    [ctypes-opencv] returns None if no capture is created
    """
    z = _cvCreateCameraCapture(index)
    # sdAdd_autoclean(z, _cvReleaseCapture)
    # return z
    return z.contents if bool(z) else None
    
cvCaptureFromCAM = cvCreateCameraCapture

# Releases the CvCapture structure
cvReleaseCapture = cvFree

# Grabs frame from camera or file
cvGrabFrame = cfunc('cvGrabFrame', _hgDLL, c_int,
    ('capture', CvCapture_r, 1), # CvCapture* capture 
)
cvGrabFrame.__doc__ = """int cvGrabFrame(CvCapture capture)

Grabs frame from camera or file
"""

# Gets the image grabbed with cvGrabFrame
cvRetrieveFrame = cfunc('cvRetrieveFrame', _hgDLL, IplImage_p,
    ('capture', CvCapture_r, 1), # CvCapture* capture 
)
cvRetrieveFrame.__doc__ = """IplImage* cvRetrieveFrame(CvCapture capture)

Gets the image grabbed with cvGrabFrame
"""

# Grabs and returns a frame from camera or file
cvQueryFrame = cfunc('cvQueryFrame', _hgDLL, IplImage_p,
    ('capture', CvCapture_r, 1), # CvCapture* capture 
)
cvQueryFrame.__doc__ = """IplImage* cvQueryFrame(CvCapture capture)

Grabs and returns a frame from camera or file
"""

def CheckNonNull(result, func, args):
    if not result:
        raise RuntimeError, 'QueryFrame failed'
    return args

CV_CAP_PROP_POS_MSEC      = 0
CV_CAP_PROP_POS_FRAMES    = 1
CV_CAP_PROP_POS_AVI_RATIO = 2
CV_CAP_PROP_FRAME_WIDTH   = 3
CV_CAP_PROP_FRAME_HEIGHT  = 4
CV_CAP_PROP_FPS           = 5
CV_CAP_PROP_FOURCC        = 6
CV_CAP_PROP_FRAME_COUNT   = 7
CV_CAP_PROP_FORMAT        = 8
CV_CAP_PROP_MODE          = 9
CV_CAP_PROP_BRIGHTNESS    =10
CV_CAP_PROP_CONTRAST      =11
CV_CAP_PROP_SATURATION    =12
CV_CAP_PROP_HUE           =13
CV_CAP_PROP_GAIN          =14
CV_CAP_PROP_CONVERT_RGB   =15

def CV_FOURCC(c1,c2,c3,c4):
    return (((ord(c1))&255) + (((ord(c2))&255)<<8) + (((ord(c3))&255)<<16) + (((ord(c4))&255)<<24))

# Gets video capturing properties
cvGetCaptureProperty = cfunc('cvGetCaptureProperty', _hgDLL, c_double,
    ('capture', CvCapture_r, 1), # CvCapture* capture
    ('property_id', c_int, 1), # int property_id 
)
cvGetCaptureProperty.__doc__ = """double cvGetCaptureProperty(CvCapture capture, int property_id)

Gets video capturing properties
"""

# Sets video capturing properties
cvSetCaptureProperty = cfunc('cvSetCaptureProperty', _hgDLL, c_int,
    ('capture', CvCapture_r, 1), # CvCapture* capture
    ('property_id', c_int, 1), # int property_id
    ('value', c_double, 1), # double value 
)
cvSetCaptureProperty.__doc__ = """int cvSetCaptureProperty(CvCapture capture, int property_id, double value)

Sets video capturing properties
"""

_cvReleaseVideoWriter = cfunc('cvReleaseVideoWriter', _hgDLL, None,
    ('writer', ByRefArg(CvVideoWriter_p), 1), # CvVideoWriter** writer 
)

_cvCreateVideoWriter = cfunc('cvCreateVideoWriter', _hgDLL, CvVideoWriter_p,
    ('filename', c_char_p, 1), # const char* filename
    ('fourcc', c_int, 1), # int fourcc
    ('fps', c_double, 1), # double fps
    ('frame_size', CvSize, 1), # CvSize frame_size
    ('is_color', c_int, 1, 1), # int is_color
)

# Creates video file writer
def cvCreateVideoWriter(filename, fourcc, fps, frame_size, is_color=1):
    """CvVideoWriter* cvCreateVideoWriter(const char* filename, int fourcc, double fps, CvSize frame_size, int is_color=1)

    Creates video file writer
    """
    z = _cvCreateVideoWriter(filename, fourcc, fps, frame_size, is_color)
    sdAdd_autoclean(z, _cvReleaseVideoWriter)
    return z

cvCreateAVIWriter = cvCreateVideoWriter

# Releases AVI writer
cvReleaseVideoWriter = cvFree

# Writes a frame to video file
cvWriteFrame = cfunc('cvWriteFrame', _hgDLL, c_int,
    ('writer', CvVideoWriter_p, 1), # CvVideoWriter* writer
    ('image', IplImage_p, 1), # const IplImage* image 
)
cvWriteFrame.__doc__ = """int cvWriteFrame(CvVideoWriter* writer, const IplImage* image)

Writes a frame to video file
"""

cvWriteToAVI = cvWriteFrame


#-----------------------------------------------------------------------------
# Utility and System Functions
#-----------------------------------------------------------------------------


# Initializes HighGUI
cvInitSystem = cfunc('cvInitSystem', _hgDLL, c_int,
    ('argc', c_int, 1), # int argc
    ('argv', POINTER(c_char_p), 1), # char** argv 
)
cvInitSystem.__doc__ = """int cvInitSystem(int argc, char** argv)

Initializes HighGUI
"""

CV_CVTIMG_FLIP = 1
CV_CVTIMG_SWAP_RB = 2

# Converts one image to another with optional vertical flip
cvConvertImage = cfunc('cvConvertImage', _hgDLL, None,
    ('src', CvArr_p, 1), # const CvArr* src
    ('dst', CvArr_p, 1), # CvArr* dst
    ('flags', c_int, 1, 0), # int flags
)
cvConvertImage.__doc__ = """void cvConvertImage(const CvArr* src, CvArr* dst, int flags=0)

Converts one image to another with optional vertical flip
"""

# Start a new thread in X Window
cvStartWindowThread = cfunc('cvStartWindowThread', _hgDLL, c_int,
)
cvStartWindowThread.__doc__ = """int cvStartWindowThread()

Starts a new thread for rendering in X Window
"""


# --- 1 Simple GUI -----------------------------------------------------------

# --- 2 Loading and Saving Images --------------------------------------------

# --- 3 Video I/O functions --------------------------------------------------

# --- 4 Utility and System Functions -----------------------------------------




#=============================================================================
# Wrap up all the functions and constants into __all__
#=============================================================================
__all__ = [x for x in locals().keys() \
    if  x.startswith('CV') or \
        x.startswith('cv') or \
        x.startswith('Cv') or \
        x.startswith('IPL') or \
        x.startswith('Ipl') or \
        x.startswith('ipl')]

