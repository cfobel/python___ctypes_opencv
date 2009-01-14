#!/usr/bin/env python
# highgui_win32
# Copyright (c) 2009, David Bolen
# All rights reserved.

"""
highgui_win32

Background UI thread implementation for Windows, since OpenCV doesn't
support it natively.  Wraps (and is safe for import * from) the following
OpenCV functions:

    cvStartWindowThread
    cvNamedWindow
    cvDestroyWindow
    cvDestroyAllWindows
    cvShowImage
    cvSetMouseCallback
    cvCreateTrackbar
    cvWaitKey

If cvStartWindowThread is not called, the other functions are just pass
through to their normal wrappers in the opencv.highgui module.  However,
once cvStartWindowThread is called, those functions execute calls to the
equivalent opencv.highgui methods from within a background thread, and that
thread maintains an event loop, calling cvWaitKey periodically.
"""

import sys
import traceback
from threading import Thread
from Queue import Queue, Empty
from ctypes import windll
from functools import wraps

import opencv.highgui as hg

#
# --------------------------------------------------------------------------
#

def threadmethod(func):
    """Wraps a method so that it is executed within the image thread"""

    @wraps(func)
    def execute(self, *args, **kwargs):
        self.cmds.put((func, (self,)+args, kwargs))
    return execute


class ImageThread(Thread):

    def __init__(self):
        Thread.__init__(self)
        self.cmds = Queue()  # Requests functions to run in image thread
        self.keys = Queue()  # Holds recent keys read within the image thread
        self.stopping = False

    def run(self):
        while 1:
            if self.stopping:
                hg.cvDestroyAllWindows()
                return
            try:
                self.func, self.args, self.kwargs = self.cmds.get_nowait()
                try:
                    self.func(*self.args, **self.kwargs)
                except Exception, e:
                    sys.stderr.write('ImageThread exception - ignoring:')
                    traceback.print_exc()
            except Empty:
                rc = hg.cvWaitKey(100)
                # Permit buffering of approximately the most recent 10 keys
                if rc >= 0:
                    if self.keys.qsize() >= 10:
                        self.keys.get()
                    self.keys.put(rc)
            except:
                sys.stderr.write('ImageThread exception - shutting down:')
                traceback.print_exc()

    def join(self, *args, **kwargs):
        # Given we never exit, assume that someone waiting for us to complete
        # is actually an indication that we should shut down
        self.stopping = True
        return Thread.join(self, *args, **kwargs)

    def _ensurewindow(self, name, foreground=False):
        if not (self.stopping or hg.cvGetWindowHandle(name)):
            # First image, or window closed manually - open window
            hg.cvNamedWindow(name)
            # Bring to front same as if the foreground thread had created it
            if foreground:
                windll.user32.SetForegroundWindow(hg.cvGetWindowHandle(name))

    #
    # HighGUI methods (run in image thread)
    #

    @threadmethod
    def NamedWindow(self, name, foreground=True):
        self._ensurewindow(name, foreground)

    @threadmethod
    def DestroyWindow(self, name):
        hg.cvDestroyWindow(name)

    @threadmethod
    def DestroyAllWindows(self):
        hg.cvDestroyAllWindows()

    @threadmethod
    def ShowImage(self, name, image):
        hg.cvShowImage(name, image)

    @threadmethod
    def SetMouseCallback(self, name, callback, param=None):
        hg.cvSetMouseCallback(name, callback, param)

    @threadmethod
    def CreateTrackbar(self, tb_name, win_name, value, count, on_change=None):
        hg.cvCreateTrackbar(tb_name, win_name, value, count, on_change)

    def WaitKey(self, wait=0):
        # Note that this runs in caller's thread
        try:
            result = self.keys.get(timeout=wait/1000.0 if wait > 0 else None)
        except Empty:
            result = -1
            
        return result

    #
    # Convenience methods
    #

    @threadmethod
    def show(self, name, image, callback=None, foreground=True):
        self._ensurewindow(name, foreground)
        hg.cvShowImage(name, image)
        if callback is not None:
            hg.cvSetMouseCallback(name, callback, image)


#
# --------------------------------------------------------------------------

#
# Module global instance of the Image thread and support for creating it
#

_image_thread = None

def cvStartWindowThread():
    global _image_thread

    if _image_thread is None or not _image_thread.isAlive():
        _image_thread = ImageThread()
        _image_thread.start()

    return _image_thread

#
# Wrapper functions using thread if present, falling back to original functions.
#

def if_imagethread(method, original):
    """Run method in image thread if present, otherwise original"""

    def execute(*args, **kwargs):
        if _image_thread:
            func = getattr(_image_thread, method)
        else:
            func = original
        return func(*args, **kwargs)

    # Raw CFUNCTYPE wrappers don't have __name__, so fudge a bit
    setattr(execute, '__name__', (getattr(original, '__name__', None) or
                                  'cv' + method))
    setattr(execute, '__doc__', getattr(original, '__doc__'))
    return execute


cvNamedWindow       = if_imagethread('NamedWindow',       hg.cvNamedWindow)
cvDestroyWindow     = if_imagethread('DestroyWindow',     hg.cvDestroyWindow)
cvDestroyAllWindows = if_imagethread('DestroyAllWindows', hg.cvDestroyWindow)
cvShowImage         = if_imagethread('ShowImage',         hg.cvShowImage)
cvSetMouseCallback  = if_imagethread('SetMouseCallback',  hg.cvSetMouseCallback)
cvCreateTrackbar    = if_imagethread('CreateTrackbar',    hg.cvCreateTrackbar)
cvWaitKey           = if_imagethread('WaitKey',           hg.cvWaitKey)

#
#
# --------------------------------------------------------------------------
#

__all__ = [x for x in locals().keys() if x.startswith('cv')]
