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


#=============================================================================
# Begin of of cv/cvtypes.h
#=============================================================================


# spatial and central moments
class CvMoments(_Structure):
    _fields_ = [
        # spatial moments
        ('m00', c_double),
        ('m10', c_double),
        ('m20', c_double),
        ('m11', c_double),
        ('m02', c_double),
        ('m30', c_double),
        ('m21', c_double),
        ('m12', c_double),
        ('m03', c_double),
        
        # central moments
        ('mu20', c_double),
        ('mu11', c_double),
        ('mu02', c_double),
        ('mu30', c_double),
        ('mu21', c_double),
        ('mu12', c_double),
        ('mu03', c_double),
        
        # m00 != 0 ? 1/sqrt(m00) : 0
        ('inv_sqrt_m00', c_double),
    ]
CvMoments_p = POINTER(CvMoments)
CvMoments_r = ByRefArg(CvMoments)
    
# Hu invariants
class CvHuMoments(_Structure):
    _fields_ = [ # Hu invariants
        ('hu1', c_double),
        ('hu2', c_double),
        ('hu3', c_double),
        ('hu4', c_double),
        ('hu5', c_double),
        ('hu6', c_double),
        ('hu7', c_double),
    ]
CvHuMoments_p = POINTER(CvHuMoments)
CvHuMoments_r = ByRefArg(CvHuMoments)
    
# Connected Component
class CvConnectedComp(_Structure):
    _fields_ = [('area', c_double), # area of the connected component
                ('value', CvScalar), # average color of the connected component
                ('rect', CvRect), # ROI of the component
                ('contour', CvSeq_p)] # optional component boundary
CvConnectedComp_p = POINTER(CvConnectedComp)
CvConnectedComp_r = ByRefArg(CvConnectedComp)
                
#Viji Periapoilan 4/16/2007 (start)
#Added constants for contour retrieval mode - Apr 19th
CV_RETR_EXTERNAL = 0
CV_RETR_LIST     = 1
CV_RETR_CCOMP    = 2
CV_RETR_TREE     = 3

#Added constants for contour approximation method  - Apr 19th
CV_CHAIN_CODE               = 0
CV_CHAIN_APPROX_NONE        = 1
CV_CHAIN_APPROX_SIMPLE      = 2
CV_CHAIN_APPROX_TC89_L1     = 3
CV_CHAIN_APPROX_TC89_KCOS   = 4
CV_LINK_RUNS                = 5
#Viji Periapoilan 4/16/2007(end)

# this structure is supposed to be treated like a blackbox, OpenCV's design
class CvContourScanner(_Structure):
    _fields_ = []

# Freeman chain reader state
class CvChainPtReader(_Structure):
    _fields_ = CV_SEQ_READER_FIELDS() + [
        ('code', c_char),
        ('pt', CvPoint),
        ('deltas', ((c_char*2)*8)),
    ]
CvChainPtReader_p = POINTER(CvChainPtReader)
CvChainPtReader_r = ByRefArg(CvChainPtReader)
    
# Contour tree header
class CvContourTree(_Structure):
    _fields_ = CV_SEQUENCE_FIELDS() + [
        ('p1', CvPoint), # the first point of the binary tree root segment
        ('p2', CvPoint), # the last point of the binary tree root segment
    ]
CvContourTree_p = POINTER(CvContourTree)
    
# Minh-Tri's hacks
sdHack_cvseq(CvContourTree_p)

# Finds a sequence of convexity defects of given contour
class CvConvexityDefect(_Structure):
    _fields_ = [
        ('start', CvPoint_p), # point of the contour where the defect begins
        ('end', CvPoint_p), # point of the contour where the defect ends
        ('depth_point', CvPoint_p), # the farthest from the convex hull point within the defect
        ('depth', c_float), # distance between the farthest point and the convex hull
    ]

# Data structures and related enumerations for Planar Subdivisions
CvSubdiv2DEdge = c_size_t

class CvSubdiv2DPoint(_Structure):
    pass
CvSubdiv2DPoint_p = POINTER(CvSubdiv2DPoint)
CvSubdiv2DPoint_r = ByRefArg(CvSubdiv2DPoint)
    
def CV_QUADEDGE2D_FIELDS():
    return [
        ('flags', c_int),
        ('pt', CvSubdiv2DPoint_p*4),
        ('next', CvSubdiv2DEdge*4),
    ]

def CV_SUBDIV2D_POINT_FIELDS():
    return [
        ('flags', c_int),
        ('first', CvSubdiv2DEdge),
        ('pt', CvPoint2D32f),
    ]

CV_SUBDIV2D_VIRTUAL_POINT_FLAG = (1 << 30)

class CvQuadEdge2D(_Structure):
    _fields_ = CV_QUADEDGE2D_FIELDS()
CvQuadEdge2D_p = POINTER(CvQuadEdge2D)
CvQuadEdge2D_r = ByRefArg(CvQuadEdge2D)
    
CvSubdiv2DPoint._fields_ = CV_SUBDIV2D_POINT_FIELDS()

def CV_SUBDIV2D_FIELDS():
    return CV_GRAPH_FIELDS() + [
        ('quad_edges', c_int),
        ('is_geometry_valid', c_int),
        ('recent_edge', CvSubdiv2DEdge),
        ('topleft', CvPoint2D32f),
        ('bottomright', CvPoint2D32f),
    ]

class CvSubdiv2D(_Structure):
    _fields_ = CV_SUBDIV2D_FIELDS()
CvSubdiv2D_p = POINTER(CvSubdiv2D)    
CvSubdiv2D_r = ByRefArg(CvSubdiv2D)    
    
# Minh-Tri's hacks
sdHack_contents_getattr(CvSubdiv2D_p)
sdHack_cvseq(CvSubdiv2D_p)

CvSubdiv2DPointLocation = c_int
CV_PTLOC_ERROR = -2
CV_PTLOC_OUTSIDE_RECT = -1
CV_PTLOC_INSIDE = 0
CV_PTLOC_VERTEX = 1
CV_PTLOC_ON_EDGE = 2

CvNextEdgeType = c_int
CV_NEXT_AROUND_ORG   = 0x00
CV_NEXT_AROUND_DST   = 0x22
CV_PREV_AROUND_ORG   = 0x11
CV_PREV_AROUND_DST   = 0x33
CV_NEXT_AROUND_LEFT  = 0x13
CV_NEXT_AROUND_RIGHT = 0x31
CV_PREV_AROUND_LEFT  = 0x20
CV_PREV_AROUND_RIGHT = 0x02

# Gets the next edge with the same origin point (counterwise)
def CV_SUBDIV2D_NEXT_EDGE( edge ):
    """CvSubdiv2DEdge CV_SUBDIV2D_NEXT_EDGE(CvSubdiv2DEdge edge)
    
    Gets the next edge with the same origin point (counterwise)
    """
    ev = edge.value
    return cast(c_void_p(ev & ~3), CvQuadEdge2D_p).contents.next[ev&3]


# Defines for Distance Transform
CV_DIST_USER    = -1  # User defined distance
CV_DIST_L1      = 1   # distance = |x1-x2| + |y1-y2|
CV_DIST_L2      = 2   # the simple euclidean distance
CV_DIST_C       = 3   # distance = max(|x1-x2|,|y1-y2|)
CV_DIST_L12     = 4   # L1-L2 metric: distance = 2(sqrt(1+x*x/2) - 1))
CV_DIST_FAIR    = 5   # distance = c^2(|x|/c-log(1+|x|/c)), c = 1.3998
CV_DIST_WELSCH  = 6   # distance = c^2/2(1-exp(-(x/c)^2)), c = 2.9846
CV_DIST_HUBER   = 7   # distance = |x|<c ? x^2/2 : c(|x|-c/2), c=1.345

CvFilter = c_int
CV_GAUSSIAN_5x5 = 7

# Older definitions
CvVect32f = c_float_p
CvMatr32f = c_float_p
CvVect64d = c_double_p
CvMatr64d = c_double_p

class CvMatrix3(_Structure):
    _fields_ = [('m', (c_float*3)*3)]

# Computes "minimal work" distance between two weighted point configurations
CvDistanceFunction = CFUNCTYPE(c_float, # float
    c_float_p, # const float* f1
    c_float_p, # const float* f2
    c_void_p) # void* userdata

# CvRandState
class CvRandState(_Structure):
    _fields_ = [
        ('state', CvRNG), # RNG state (the current seed and carry)
        ('disttype', c_int), # distribution type
        ('param', CvScalar*2), # parameters of RNG
    ]
CvRandState_p = POINTER(CvRandState)
CvRandState_r = ByRefArg(CvRandState)
    
# CvConDensation
class CvConDensation(_Structure):
    _fields_ = [
        ('MP', c_int),
        ('DP', c_int),
        ('DynamMatr', c_float_p), # Matrix of the linear Dynamics system
        ('State', c_float_p), # Vector of State
        ('SamplesNum', c_int), # Number of the Samples
        ('flSamples', POINTER(c_float_p)), # arr of the Sample Vectors
        ('flNewSamples', POINTER(c_float_p)), # temporary array of the Sample Vectors
        ('flConfidence', c_float_p), # Confidence for each Sample
        ('flCumulative', c_float_p), # Cumulative confidence
        ('Temp', c_float_p), # Temporary vector
        ('RandomSample', c_float_p), # RandomVector to update sample set
        ('RandS', CvRandState_p), # Array of structures to generate random vectors
    ]
    
    def __del__(self):
        _cvReleaseCondensation(pointer(self))
        
CvConDensation_p = POINTER(CvConDensation)
CvConDensation_r = ByRefArg(CvConDensation)
    
# standard Kalman filter (in G. Welch' and G. Bishop's notation):
#
#  x(k)=A*x(k-1)+B*u(k)+w(k)  p(w)~N(0,Q)
#  z(k)=H*x(k)+v(k),   p(v)~N(0,R)
class CvKalman(_Structure):
    _fields_ = [
        ('MP', c_int), # number of measurement vector dimensions
        ('DP', c_int), # number of state vector dimensions
        ('CP', c_int), # number of control vector dimensions
        
        # backward compatibility fields
        ('PosterState', c_float_p), # =state_pre->data.fl
        ('PriorState', c_float_p), # =state_post->data.fl
        ('DynamMatr', c_float_p), # =transition_matrix->data.fl
        ('MeasurementMatr', c_float_p), # =measurement_matrix->data.fl
        ('MNCovariance', c_float_p), # =measurement_noise_cov->data.fl
        ('PNCovariance', c_float_p), # =process_noise_cov->data.fl
        ('KalmGainMatr', c_float_p), # =gain->data.fl
        ('PriorErrorCovariance', c_float_p), # =error_cov_pre->data.fl
        ('PosterErrorCovariance', c_float_p), # =error_cov_post->data.fl
        ('Temp1', c_float_p), # temp1->data.fl
        ('Temp2', c_float_p), # temp2->data.fl
        
        ('state_pre', CvMat_p), # predicted state (x'(k))
        ('state_post', CvMat_p), # corrected state (x(k))
        ('transition_matrix', CvMat_p), # state transition matrix (A)
        ('control_matrix', CvMat_p), # control matrix (B)
        ('measurement_matrix', CvMat_p), # measurement matrix (H)
        ('process_noise_cov', CvMat_p), # process noise covariance matrix (Q)
        ('measurement_noise_cov', CvMat_p), # measurement noise covariance matrix (R)
        ('error_cov_pre', CvMat_p), # priori error estimate covariance matrix (P'(k))
        ('gain', CvMat_p), # Kalman gain matrix (K(k))
        ('error_cov_post', CvMat_p), # posteriori error estimate covariance matrix (P(k))
        ('temp1', CvMat_p),
        ('temp2', CvMat_p),
        ('temp3', CvMat_p),
        ('temp4', CvMat_p),
        ('temp5', CvMat_p),
    ]
    
    def __del__(self):
        _cvReleaseKalman(pointer(self))
        
CvKalman_p = POINTER(CvKalman)
CvKalman_r = ByRefArg(CvKalman)
    
# Haar-like Object Detection structures

CV_HAAR_MAGIC_VAL    = 0x42500000
CV_TYPE_NAME_HAAR    = "opencv-haar-classifier"
CV_HAAR_FEATURE_MAX  = 3

class CvHaarFeatureRect(_Structure):
    _fields_ = [
        ('r', CvRect),
        ('weight', c_float),
    ]    

class CvHaarFeature(_Structure):
    _fields_ = [
        ('titled', c_int),
        ('rect', CvHaarFeatureRect*CV_HAAR_FEATURE_MAX),
    ]
CvHaarFeature_p = POINTER(CvHaarFeature)

class CvHaarClassifier(_Structure):
    _fields_ = [
        ('count', c_int),
        ('haar_feature', CvHaarFeature_p),
        ('threshold', c_float_p),
        ('left', c_int_p),
        ('right', c_int_p),
        ('alpha', c_float_p),
    ]
CvHaarClassifier_p = POINTER(CvHaarClassifier)

class CvHaarStageClassifier(_Structure):
    _fields_ = [
        ('count', c_int),
        ('threshold', c_float),
        ('classifier', CvHaarClassifier_p),
        ('next', c_int),
        ('child', c_int),
        ('parent', c_int),
    ]
CvHaarStageClassifier_p = POINTER(CvHaarStageClassifier)

class CvHidHaarClassifierCascade(_Structure): # not implemented yet
    _fields_ = []
CvHidHaarClassifierCascade_p = POINTER(CvHidHaarClassifierCascade)

class CvHaarClassifierCascade(_Structure):
    _fields_ = [
        ('flags', c_int),
        ('count', c_int),
        ('orig_window_size', CvSize),
        ('real_window_size', CvSize),
        ('scale', c_double),
        ('stage_classifier', CvHaarStageClassifier_p),
        ('hid_cascade', CvHidHaarClassifierCascade_p),
    ]
    
    def __del__(self):
        _cvReleaseHaarClassifierCascade(pointer(self))
        
CvHaarClassifierCascade_p = POINTER(CvHaarClassifierCascade)
CvHaarClassifierCascade_r = ByRefArg(CvHaarClassifierCascade)

class CvAvgComp(_Structure):
    _fields_ = [
        ('rect', CvRect),
        ('neighbors', c_int),
    ]

    
#=============================================================================
# End of of cv/cvtypes.h
#=============================================================================




#=============================================================================
# Begin of of cv/cv.h
#=============================================================================


#-----------------------------------------------------------------------------
# Image Processing: Gradients, Edges and Corners
#-----------------------------------------------------------------------------


CV_SCHARR = -1
CV_MAX_SOBEL_KSIZE = 7

# Calculates first, second, third or mixed image derivatives using extended Sobel operator
cvSobel = cfunc('cvSobel', _cvDLL, None,
    ('src', CvArr_p, 1), # const CvArr* src
    ('dst', CvArr_p, 1), # CvArr* dst
    ('xorder', c_int, 1), # int xorder
    ('yorder', c_int, 1), # int yorder
    ('aperture_size', c_int, 1, 3), # int aperture_size
)
cvSobel.__doc__ = """void cvSobel(const CvArr* src, CvArr* dst, int xorder, int yorder, int aperture_size=3)

Calculates first, second, third or mixed image derivatives using extended Sobel operator
"""

# Calculates Laplacian of the image
cvLaplace = cfunc('cvLaplace', _cvDLL, None,
    ('src', CvArr_p, 1), # const CvArr* src
    ('dst', CvArr_p, 1), # CvArr* dst
    ('aperture_size', c_int, 1, 3), # int aperture_size
)
cvLaplace.__doc__ = """void cvLaplace(const CvArr* src, CvArr* dst, int aperture_size=3)

Calculates Laplacian of the image
"""

CV_CANNY_L2_GRADIENT = 1 << 31

# Implements Canny algorithm for edge detection
cvCanny = cfunc('cvCanny', _cvDLL, None,
    ('image', CvArr_p, 1), # const CvArr* image
    ('edges', CvArr_p, 1), # CvArr* edges
    ('threshold1', c_double, 1), # double threshold1
    ('threshold2', c_double, 1), # double threshold2
    ('aperture_size', c_int, 1, 3), # int aperture_size
)
cvCanny.__doc__ = """void cvCanny(const CvArr* image, CvArr* edges, double threshold1, double threshold2, int aperture_size=3)

Implements Canny algorithm for edge detection
"""

# Calculates feature map for corner detection
cvPreCornerDetect = cfunc('cvPreCornerDetect', _cvDLL, None,
    ('image', CvArr_p, 1), # const CvArr* image
    ('corners', CvArr_p, 1), # CvArr* corners
    ('aperture_size', c_int, 1, 3), # int aperture_size
)
cvPreCornerDetect.__doc__ = """void cvPreCornerDetect(const CvArr* image, CvArr* corners, int aperture_size=3)

Calculates feature map for corner detection
"""

# Calculates eigenvalues and eigenvectors of image blocks for corner detection
cvCornerEigenValsAndVecs = cfunc('cvCornerEigenValsAndVecs', _cvDLL, None,
    ('image', CvArr_p, 1), # const CvArr* image
    ('eigenvv', CvArr_p, 1), # CvArr* eigenvv
    ('block_size', c_int, 1), # int block_size
    ('aperture_size', c_int, 1, 3), # int aperture_size
)
cvCornerEigenValsAndVecs.__doc__ = """void cvCornerEigenValsAndVecs(const CvArr* image, CvArr* eigenvv, int block_size, int aperture_size=3)

Calculates eigenvalues and eigenvectors of image blocks for corner detection
"""

# Calculates minimal eigenvalue of gradient matrices for corner detection
cvCornerMinEigenVal = cfunc('cvCornerMinEigenVal', _cvDLL, None,
    ('image', CvArr_p, 1), # const CvArr* image
    ('eigenval', CvArr_p, 1), # CvArr* eigenval
    ('block_size', c_int, 1), # int block_size
    ('aperture_size', c_int, 1, 3), # int aperture_size
)
cvCornerMinEigenVal.__doc__ = """void cvCornerMinEigenVal(const CvArr* image, CvArr* eigenval, int block_size, int aperture_size=3)

Calculates minimal eigenvalue of gradient matrices for corner detection
"""

# Harris edge detector
cvCornerHarris = cfunc('cvCornerHarris', _cvDLL, None,
    ('image', CvArr_p, 1), # const CvArr* image
    ('harris_responce', CvArr_p, 1), # CvArr* harris_responce
    ('block_size', c_int, 1), # int block_size
    ('aperture_size', c_int, 1, 3), # int aperture_size
    ('k', c_double, 1, 0), # double k
)
cvCornerHarris.__doc__ = """void cvCornerHarris(const CvArr* image, CvArr* harris_responce, int block_size, int aperture_size=3, double k=0.04)

Harris edge detector
"""

# Refines corner locations
_cvFindCornerSubPix = cfunc('cvFindCornerSubPix', _cvDLL, None,
    ('image', CvArr_p, 1), # const CvArr* image
    ('corners', c_void_p, 1), # CvPoint2D32f* corners
    ('count', c_int, 1), # int count
    ('win', CvSize, 1), # CvSize win
    ('zero_zone', CvSize, 1), # CvSize zero_zone
    ('criteria', CvTermCriteria, 1), # CvTermCriteria criteria 
)

def cvFindCornerSubPix(image, corners, win, zero_zone, criteria):
    """list_or_tuple_of_CvPoint2D32f cvFindCornerSubPix(const CvArr* image, list_or_tuple_of_CvPoint2D32f corners, CvSize win, CvSize zero_zone, CvTermCriteria criteria)

    Refines corner locations
    [ctypes-opencv] Parameter 'corners' is returned with points updated.
    """
    mat = cvCreateMatFromCvPoint2D32fList(corners)
    n = len(corners)
    _cvFindCornerSubPix(image, mat.data.ptr, n, win, zero_zone, criteria)
    for i in xrange(n):
        x = mat[0,i]
        y = corners[i]
        y.x = x[0]
        y.y = x[1]        
    return corners

# Determines strong corners on image
_cvGoodFeaturesToTrack = cfunc('cvGoodFeaturesToTrack', _cvDLL, None,
    ('image', CvArr_p, 1), # const CvArr* image
    ('eig_image', CvArr_p, 1), # CvArr* eig_image
    ('temp_image', CvArr_p, 1), # CvArr* temp_image
    ('corners', c_void_p, 1), # CvPoint2D32f* corners
    ('corner_count', ByRefArg(c_int), 1), # int* corner_count
    ('quality_level', c_double, 1), # double quality_level
    ('min_distance', c_double, 1), # double min_distance
    ('mask', CvArr_p, 1, None), # const CvArr* mask
    ('block_size', c_int, 1, 3), # int block_size
    ('use_harris', c_int, 1, 0), # int use_harris
    ('k', c_double, 1, 0), # double k
)

def cvGoodFeaturesToTrack(image, eig_image, temp_image, max_corner_count, quality_level, min_distance, mask=None, block_size=3, use_harris=0, k=0):
    """list_of_CvPoint2D32f cvGoodFeaturesToTrack(const CvArr* image, CvArr* eig_image, CvArr* temp_image, int max_corner_count, double quality_level, double min_distance, const CvArr* mask=NULL, int block_size=3, int use_harris=0, double k=0.04)

    Determines strong corners on image
    [ctypes-opencv] 'max_corner_count' is the maximum number of corners to detect
    [ctypes-opencv] Returns a list of detected points
    """
    corners = cvCreateMat(1, max_corner_count, CV_32FC2)
    count = c_int(max_corner_count)
    _cvGoodFeaturesToTrack(image, eig_image, temp_image, corners.data.ptr, count, quality_level, min_distance, mask, block_size, use_harris, k)
    z = []
    for i in xrange(count.value):
        x = corners[0,i]
        z.append(CvPoint2D32f(x[0], x[1]))
    return z

#-----------------------------------------------------------------------------
# Image Processing: Sampling, Interpolation and Geometrical Transforms
#-----------------------------------------------------------------------------


# Reads raster line to buffer
cvSampleLine = cfunc('cvSampleLine', _cvDLL, c_int,
    ('image', CvArr_p, 1), # const CvArr* image
    ('pt1', CvPoint, 1), # CvPoint pt1
    ('pt2', CvPoint, 1), # CvPoint pt2
    ('buffer', c_void_p, 1), # void* buffer
    ('connectivity', c_int, 1, 8), # int connectivity
)
cvSampleLine.__doc__ = """int cvSampleLine(const CvArr* image, CvPoint pt1, CvPoint pt2, void* buffer, int connectivity=8)

Reads raster line to buffer
"""

# Retrieves pixel rectangle from image with sub-pixel accuracy
cvGetRectSubPix = cfunc('cvGetRectSubPix', _cvDLL, None,
    ('src', CvArr_p, 1), # const CvArr* src
    ('dst', CvArr_p, 1), # CvArr* dst
    ('center', CvPoint2D32f, 1), # CvPoint2D32f center 
)
cvGetRectSubPix.__doc__ = """void cvGetRectSubPix(const CvArr* src, CvArr* dst, CvPoint2D32f center)

Retrieves pixel rectangle from image with sub-pixel accuracy
"""

# Retrieves pixel quadrangle from image with sub-pixel accuracy
cvGetQuadrangleSubPix = cfunc('cvGetQuadrangleSubPix', _cvDLL, None,
    ('src', CvArr_p, 1), # const CvArr* src
    ('dst', CvArr_p, 1), # CvArr* dst
    ('map_matrix', CvMat_p, 1), # const CvMat* map_matrix 
)
cvGetQuadrangleSubPix.__doc__ = """void cvGetQuadrangleSubPix(const CvArr* src, CvArr* dst, const CvMat* map_matrix)

Retrieves pixel quadrangle from image with sub-pixel accuracy
"""

#Viji Periapoilan 4/16/2007 (start)
# Added the following constants to work with facedetect sample
CV_INTER_NN     = 0 #nearest-neigbor interpolation, 
CV_INTER_LINEAR = 1 #bilinear interpolation (used by default) 
CV_INTER_CUBIC  = 2 # bicubic interpolation. 
CV_INTER_AREA = 3 #resampling using pixel area relation. It is preferred method for image decimation that gives moire-free results. In case of zooming it is similar to CV_INTER_NN method.
#Viji Periapoilan 4/16/2007(end)

# Resizes image
cvResize = cfunc('cvResize', _cvDLL, None,
    ('src', CvArr_p, 1), # const CvArr* src
    ('dst', CvArr_p, 1), # CvArr* dst
    ('interpolation', c_int, 1), # int interpolation
)
cvResize.__doc__ = """void cvResize(const CvArr* src, CvArr* dst, int interpolation=CV_INTER_LINEAR)

Resizes image
"""

CV_WARP_FILL_OUTLIERS = 8
CV_WARP_INVERSE_MAP = 16

# Applies affine transformation to the image
cvWarpAffine = cfunc('cvWarpAffine', _cvDLL, None,
    ('src', CvArr_p, 1), # const CvArr* src
    ('dst', CvArr_p, 1), # CvArr* dst
    ('map_matrix', CvMat_p, 1), # const CvMat* map_matrix
    ('flags', c_int, 1), # int flags
    ('fillval', CvScalar, 1), # CvScalar fillval
)
cvWarpAffine.__doc__ = """void cvWarpAffine(const CvArr* src, CvArr* dst, const CvMat* map_matrix, int flags=CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS, CvScalar fillval=cvScalarAll(0)

Applies affine transformation to the image
"""

# Calculates affine transform from 3 corresponding points
cvGetAffineTransform = cfunc('cvGetAffineTransform', _cvDLL, CvMat_p,
    ('src', CvPoint2D32f_r, 1), # const CvPoint2D32f* src
    ('dst', CvPoint2D32f_r, 1), # const CvPoint2D32f* dst
    ('map_matrix', CvMat_p, 1), # CvMat* map_matrix
)
cvGetAffineTransform.__doc__ = """CvMat* cvGetAffineTransform(const CvPoint2D32f src, const CvPoint2D32f dst, CvMat* map_matrix)

Calculates affine transform from 3 corresponding points
"""

# Calculates affine matrix of 2d rotation
cv2DRotationMatrix = cfunc('cv2DRotationMatrix', _cvDLL, CvMat_p,
    ('center', CvPoint2D32f, 1), # CvPoint2D32f center
    ('angle', c_double, 1), # double angle
    ('scale', c_double, 1), # double scale
    ('map_matrix', CvMat_p, 1), # CvMat* map_matrix 
)
cv2DRotationMatrix.__doc__ = """CvMat* cv2DRotationMatrix(CvPoint2D32f center, double angle, double scale, CvMat* map_matrix)

Calculates affine matrix of 2d rotation
"""

# Applies perspective transformation to the image
cvWarpPerspective = cfunc('cvWarpPerspective', _cvDLL, None,
    ('src', CvArr_p, 1), # const CvArr* src
    ('dst', CvArr_p, 1), # CvArr* dst
    ('map_matrix', CvMat_p, 1), # const CvMat* map_matrix
    ('flags', c_int, 1), # int flags
    ('fillval', CvScalar, 1), # CvScalar fillval
)
cvWarpPerspective.__doc__ = """void cvWarpPerspective(const CvArr* src, CvArr* dst, const CvMat* map_matrix, int flags=CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS, CvScalar fillval=cvScalarAll(0)

Applies perspective transformation to the image
"""

# Calculates perspective transform from 4 corresponding points
cvGetPerspectiveTransform = cfunc('cvGetPerspectiveTransform', _cvDLL, CvMat_p,
    ('src', CvPoint2D32f_r, 1), # const CvPoint2D32f* src
    ('dst', CvPoint2D32f_r, 1), # const CvPoint2D32f* dst
    ('map_matrix', CvMat_p, 1), # CvMat* map_matrix 
)
cvGetPerspectiveTransform.__doc__ = """CvMat* cvGetPerspectiveTransform(const CvPoint2D32f src, const CvPoint2D32f dst, CvMat* map_matrix)

Calculates perspective transform from 4 corresponding points
"""

# Applies generic geometrical transformation to the image
cvRemap = cfunc('cvRemap', _cvDLL, None,
    ('src', CvArr_p, 1), # const CvArr* src
    ('dst', CvArr_p, 1), # CvArr* dst
    ('mapx', CvArr_p, 1), # const CvArr* mapx
    ('mapy', CvArr_p, 1), # const CvArr* mapy
    ('flags', c_int, 1), # int flags
    ('fillval', CvScalar, 1), # CvScalar fillval
)
cvRemap.__doc__ = """void cvRemap(const CvArr* src, CvArr* dst, const CvArr* mapx, const CvArr* mapy, int flags=CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS, CvScalar fillval=cvScalarAll(0)

Applies generic geometrical transformation to the image
"""

# Remaps image to log-polar space
cvLogPolar = cfunc('cvLogPolar', _cvDLL, None,
    ('src', CvArr_p, 1), # const CvArr* src
    ('dst', CvArr_p, 1), # CvArr* dst
    ('center', CvPoint2D32f, 1), # CvPoint2D32f center
    ('M', c_double, 1), # double M
    ('flags', c_int, 1), # int flags
)
cvLogPolar.__doc__ = """void cvLogPolar(const CvArr* src, CvArr* dst, CvPoint2D32f center, double M, int flags=CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS)

Remaps image to log-polar space
"""


#-----------------------------------------------------------------------------
# Image Processing: Morphological Operations
#-----------------------------------------------------------------------------


CV_SHAPE_RECT = 0
CV_SHAPE_CROSS = 1
CV_SHAPE_ELLIPSE = 2
CV_SHAPE_CUSTOM = 100

_cvReleaseStructuringElement = cfunc('cvReleaseStructuringElement', _cvDLL, None,
    ('element', ByRefArg(IplConvKernel_p), 1), # IplConvKernel** element 
)

_cvCreateStructuringElementEx = cfunc('cvCreateStructuringElementEx', _cvDLL, IplConvKernel_p,
    ('cols', c_int, 1), # int cols
    ('rows', c_int, 1), # int rows
    ('anchor_x', c_int, 1), # int anchor_x
    ('anchor_y', c_int, 1), # int anchor_y
    ('shape', c_int, 1), # int shape
    ('values', c_int_p, 1, None), # int* values
)

# Creates structuring element
def cvCreateStructuringElementEx(cols, rows, anchor_x, anchor_y, shape, values=None):
    """IplConvKernel* cvCreateStructuringElementEx(int cols, int rows, int anchor_x, int anchor_y, int shape, int* values=NULL)

    Creates structuring element
    """
    z = _cvCreateStructuringElementEx(cols, rows, anchor_x, anchor_y, shape, values)
    sdAdd_autoclean(z, _cvReleaseStructuringElement)
    return z

# Deletes structuring element
cvReleaseStructuringElement = cvFree

# Erodes image by using arbitrary structuring element
cvErode = cfunc('cvErode', _cvDLL, None,
    ('src', CvArr_p, 1), # const CvArr* src
    ('dst', CvArr_p, 1), # CvArr* dst
    ('element', IplConvKernel_p, 1, None), # IplConvKernel* element
    ('iterations', c_int, 1, 1), # int iterations
)
cvErode.__doc__ = """void cvErode(const CvArr* src, CvArr* dst, IplConvKernel* element=NULL, int iterations=1)

Erodes image by using arbitrary structuring element
"""

# Dilates image by using arbitrary structuring element
cvDilate = cfunc('cvDilate', _cvDLL, None,
    ('src', CvArr_p, 1), # const CvArr* src
    ('dst', CvArr_p, 1), # CvArr* dst
    ('element', IplConvKernel_p, 1, None), # IplConvKernel* element
    ('iterations', c_int, 1, 1), # int iterations
)
cvDilate.__doc__ = """void cvDilate(const CvArr* src, CvArr* dst, IplConvKernel* element=NULL, int iterations=1)

Dilates image by using arbitrary structuring element
"""

CV_MOP_OPEN = 2
CV_MOP_CLOSE = 3
CV_MOP_GRADIENT = 4
CV_MOP_TOPHAT = 5
CV_MOP_BLACKHAT = 6

# Performs advanced morphological transformations
cvMorphologyEx = cfunc('cvMorphologyEx', _cvDLL, None,
    ('src', CvArr_p, 1), # const CvArr* src
    ('dst', CvArr_p, 1), # CvArr* dst
    ('temp', CvArr_p, 1), # CvArr* temp
    ('element', IplConvKernel_p, 1), # IplConvKernel* element
    ('operation', c_int, 1), # int operation
    ('iterations', c_int, 1, 1), # int iterations
)
cvMorphologyEx.__doc__ = """void cvMorphologyEx(const CvArr* src, CvArr* dst, CvArr* temp, IplConvKernel* element, int operation, int iterations=1)

Performs advanced morphological transformations
"""


#-----------------------------------------------------------------------------
# Image Processing: Filters and Color Conversion
#-----------------------------------------------------------------------------


CV_BLUR_NO_SCALE = 0
CV_BLUR = 1
CV_GAUSSIAN = 2
CV_MEDIAN = 3
CV_BILATERAL = 4

# Smooths the image in one of several ways
cvSmooth = cfunc('cvSmooth', _cvDLL, None,
    ('src', CvArr_p, 1), # const CvArr* src
    ('dst', CvArr_p, 1), # CvArr* dst
    ('smoothtype', c_int, 1), # int smoothtype
    ('param1', c_int, 1, 3), # int param1
    ('param2', c_int, 1, 0), # int param2
    ('param3', c_double, 1, 0), # double param3
)
cvSmooth.__doc__ = """void cvSmooth(const CvArr* src, CvArr* dst, int smoothtype=CV_GAUSSIAN, int param1=3, int param2=0, double param3=0)

Smooths the image in one of several ways
"""

# Convolves image with the kernel
cvFilter2D = cfunc('cvFilter2D', _cvDLL, None,
    ('src', CvArr_p, 1), # const CvArr* src
    ('dst', CvArr_p, 1), # CvArr* dst
    ('kernel', CvMat_p, 1), # const CvMat* kernel
    ('anchor', CvPoint, 1), # CvPoint anchor
)
cvFilter2D.__doc__ = """void cvFilter2D(const CvArr* src, CvArr* dst, const CvMat* kernel, CvPoint anchor=cvPoint(-1, -1)

Convolves image with the kernel
"""

# Copies image and makes border around it
cvCopyMakeBorder = cfunc('cvCopyMakeBorder', _cvDLL, None,
    ('src', CvArr_p, 1), # const CvArr* src
    ('dst', CvArr_p, 1), # CvArr* dst
    ('offset', CvPoint, 1), # CvPoint offset
    ('bordertype', c_int, 1), # int bordertype
    ('value', CvScalar, 1), # CvScalar value
)
cvCopyMakeBorder.__doc__ = """void cvCopyMakeBorder(const CvArr* src, CvArr* dst, CvPoint offset, int bordertype, CvScalar value=cvScalarAll(0)

Copies image and makes border around it
"""

# Calculates integral images
cvIntegral = cfunc('cvIntegral', _cvDLL, None,
    ('image', CvArr_p, 1), # const CvArr* image
    ('sum', CvArr_p, 1), # CvArr* sum
    ('sqsum', CvArr_p, 1, None), # CvArr* sqsum
    ('tilted_sum', CvArr_p, 1, None), # CvArr* tilted_sum
)
cvIntegral.__doc__ = """void cvIntegral(const CvArr* image, CvArr* sum, CvArr* sqsum=NULL, CvArr* tilted_sum=NULL)

Calculates integral images
"""


CV_BGR2BGRA =   0
CV_RGB2RGBA =   CV_BGR2BGRA

CV_BGRA2BGR =   1
CV_RGBA2RGB =   CV_BGRA2BGR

CV_BGR2RGBA =   2
CV_RGB2BGRA =   CV_BGR2RGBA

CV_RGBA2BGR =   3
CV_BGRA2RGB =   CV_RGBA2BGR

CV_BGR2RGB  =   4
CV_RGB2BGR  =   CV_BGR2RGB

CV_BGRA2RGBA =  5
CV_RGBA2BGRA =  CV_BGRA2RGBA

CV_BGR2GRAY =   6
CV_RGB2GRAY =   7
CV_GRAY2BGR =   8
CV_GRAY2RGB =   CV_GRAY2BGR
CV_GRAY2BGRA =  9
CV_GRAY2RGBA =  CV_GRAY2BGRA
CV_BGRA2GRAY =  10
CV_RGBA2GRAY =  11

CV_BGR2BGR565 = 12
CV_RGB2BGR565 = 13
CV_BGR5652BGR = 14
CV_BGR5652RGB = 15
CV_BGRA2BGR565 = 16
CV_RGBA2BGR565 = 17
CV_BGR5652BGRA = 18
CV_BGR5652RGBA = 19

CV_GRAY2BGR565 = 20
CV_BGR5652GRAY = 21

CV_BGR2BGR555  = 22
CV_RGB2BGR555  = 23
CV_BGR5552BGR  = 24
CV_BGR5552RGB  = 25
CV_BGRA2BGR555 = 26
CV_RGBA2BGR555 = 27
CV_BGR5552BGRA = 28
CV_BGR5552RGBA = 29

CV_GRAY2BGR555 = 30
CV_BGR5552GRAY = 31

CV_BGR2XYZ =    32
CV_RGB2XYZ =    33
CV_XYZ2BGR =    34
CV_XYZ2RGB =    35

CV_BGR2YCrCb =  36
CV_RGB2YCrCb =  37
CV_YCrCb2BGR =  38
CV_YCrCb2RGB =  39

CV_BGR2HSV =    40
CV_RGB2HSV =    41

CV_BGR2Lab =    44
CV_RGB2Lab =    45

CV_BayerBG2BGR = 46
CV_BayerGB2BGR = 47
CV_BayerRG2BGR = 48
CV_BayerGR2BGR = 49

CV_BayerBG2RGB = CV_BayerRG2BGR
CV_BayerGB2RGB = CV_BayerGR2BGR
CV_BayerRG2RGB = CV_BayerBG2BGR
CV_BayerGR2RGB = CV_BayerGB2BGR

CV_BGR2Luv =    50
CV_RGB2Luv =    51
CV_BGR2HLS =    52
CV_RGB2HLS =    53

CV_HSV2BGR =    54
CV_HSV2RGB =    55

CV_Lab2BGR =    56
CV_Lab2RGB =    57
CV_Luv2BGR =    58
CV_Luv2RGB =    59
CV_HLS2BGR =    60
CV_HLS2RGB =    61

CV_COLORCVT_MAX = 100

# Converts image from one color space to another
cvCvtColor = cfunc('cvCvtColor', _cvDLL, None,
    ('src', CvArr_p, 1), # const CvArr* src
    ('dst', CvArr_p, 1), # CvArr* dst
    ('code', c_int, 1), # int code 
)
cvCvtColor.__doc__ = """void cvCvtColor(const CvArr* src, CvArr* dst, int code)

Converts image from one color space to another
"""

CV_THRESH_BINARY = 0      # value = (value > threshold) ? max_value : 0
CV_THRESH_BINARY_INV = 1  # value = (value > threshold) ? 0 : max_value
CV_THRESH_TRUNC = 2       # value = (value > threshold) ? threshold : value
CV_THRESH_TOZERO = 3      # value = (value > threshold) ? value : 0
CV_THRESH_TOZERO_INV = 4  # value = (value > threshold) ? 0 : value
CV_THRESH_MASK = 7
CV_THRESH_OTSU = 8        # use Otsu algorithm to choose the optimal threshold value

# Applies fixed-level threshold to array elements
cvThreshold = cfunc('cvThreshold', _cvDLL, None,
    ('src', CvArr_p, 1), # const CvArr* src
    ('dst', CvArr_p, 1), # CvArr* dst
    ('threshold', c_double, 1), # double threshold
    ('max_value', c_double, 1), # double max_value
    ('threshold_type', c_int, 1), # int threshold_type 
)
cvThreshold.__doc__ = """void cvThreshold(const CvArr* src, CvArr* dst, double threshold, double max_value, int threshold_type)

Applies fixed-level threshold to array elements
"""

CV_ADAPTIVE_THRESH_MEAN_C = 0
CV_ADAPTIVE_THRESH_GAUSSIAN_C = 1

# Applies adaptive threshold to array
cvAdaptiveThreshold = cfunc('cvAdaptiveThreshold', _cvDLL, None,
    ('src', CvArr_p, 1), # const CvArr* src
    ('dst', CvArr_p, 1), # CvArr* dst
    ('max_value', c_double, 1), # double max_value
    ('adaptive_method', c_int, 1), # int adaptive_method
    ('threshold_type', c_int, 1), # int threshold_type
    ('block_size', c_int, 1, 3), # int block_size
    ('param1', c_double, 1, 5), # double param1
)
cvAdaptiveThreshold.__doc__ = """void cvAdaptiveThreshold(const CvArr* src, CvArr* dst, double max_value, int adaptive_method=CV_ADAPTIVE_THRESH_MEAN_C, int threshold_type=CV_THRESH_BINARY, int block_size=3, double param1=5)

Applies adaptive threshold to array
"""


#-----------------------------------------------------------------------------
# Image Processing: Pyramids
#-----------------------------------------------------------------------------


# Downsamples image
cvPyrDown = cfunc('cvPyrDown', _cvDLL, None,
    ('src', CvArr_p, 1), # const CvArr* src
    ('dst', CvArr_p, 1), # CvArr* dst
    ('filter', c_int, 1), # int filter
)
cvPyrDown.__doc__ = """void cvPyrDown(const CvArr* src, CvArr* dst, int filter=CV_GAUSSIAN_5x5)

Downsamples image
"""

# Upsamples image
cvPyrUp = cfunc('cvPyrUp', _cvDLL, None,
    ('src', CvArr_p, 1), # const CvArr* src
    ('dst', CvArr_p, 1), # CvArr* dst
    ('filter', c_int, 1), # int filter
)
cvPyrUp.__doc__ = """void cvPyrUp(const CvArr* src, CvArr* dst, int filter=CV_GAUSSIAN_5x5)

Upsamples image
"""


#-----------------------------------------------------------------------------
# Image Processing: Connected Components
#-----------------------------------------------------------------------------



# Fills a connected component with given color
cvFloodFill = cfunc('cvFloodFill', _cvDLL, None,
    ('image', CvArr_p, 1), # CvArr* image
    ('seed_point', CvPoint, 1), # CvPoint seed_point
    ('new_val', CvScalar, 1), # CvScalar new_val
    ('lo_diff', CvScalar, 1), # CvScalar lo_diff
    ('up_diff', CvScalar, 1), # CvScalar up_diff
    ('comp', CvConnectedComp_r, 1, None), # CvConnectedComp* comp
    ('flags', c_int, 1, 4), # int flags
    ('mask', CvArr_p, 1, None), # CvArr* mask
)
cvFloodFill.__doc__ = """void cvFloodFill(CvArr* image, CvPoint seed_point, CvScalar new_val, CvScalar lo_diff=cvScalarAll(0), CvScalar up_diff=cvScalarAll(0), CvConnectedComp comp=None, int flags=4, CvArr* mask=NULL)

Fills a connected component with given color
"""

CV_FLOODFILL_FIXED_RANGE = 1 << 16
CV_FLOODFILL_MASK_ONLY = 1 << 17

# Finds contours in binary image
_cvFindContours = cfunc('cvFindContours', _cvDLL, c_int,
    ('image', CvArr_p, 1), # CvArr* image
    ('storage', CvMemStorage_p, 1), # CvMemStorage* storage
    ('first_contour', POINTER(CvSeq_p), 1), # CvSeq** first_contour
    ('header_size', c_int, 1, sizeof(CvContour)), # int header_size
    ('mode', c_int, 1, CV_RETR_LIST), # int mode
    ('method', c_int, 1, CV_CHAIN_APPROX_SIMPLE), # int method
    ('offset', CvPoint, 1, cvPoint(0,0)), # CvPoint offset
)

# Finds contours in binary image
def cvFindContours(image, storage, header_size=sizeof(CvContour), mode=CV_RETR_LIST, method=CV_CHAIN_APPROX_SIMPLE, offset=cvPoint(0,0)):
    """(int ncontours, CvSeq* first_contour) = cvFindContours(CvArr* image, CvMemStorage* storage, int header_size=sizeof(CvContour), int mode=CV_RETR_LIST, int method=CV_CHAIN_APPROX_SIMPLE, CvPoint offset=cvPoint(0, 0)

    Finds contours in binary image
    """
    first = CvSeq_p()
    nc = _cvFindContours(image, storage, first, header_size, mode, method, offset)
    return nc, first

# Initializes contour scanning process
_cvStartFindContours = cfunc('cvStartFindContours', _cvDLL, CvContourScanner,
    ('image', CvArr_p, 1), # CvArr* image
    ('storage', CvMemStorage_p, 1), # CvMemStorage* storage
    ('header_size', c_int, 1, sizeof(CvContour)), # int header_size
    ('mode', c_int, 1, CV_RETR_LIST), # int mode
    ('method', c_int, 1, CV_CHAIN_APPROX_SIMPLE), # int method
    ('offset', CvPoint, 1, cvPoint(0,0)), # CvPoint offset
)

def cvStartFindContours(image, storage, header_size=sizeof(CvContour), mode=CV_RETR_LIST, method=CV_CHAIN_APPROX_SIMPLE, offset=CvPoint(0,0)):
    """CvContourScanner cvStartFindContours(CvArr* image, CvMemStorage* storage, int header_size=sizeofCvContour, int mode=CV_RETR_LIST, int method=CV_CHAIN_APPROX_SIMPLE, CvPoint offset=cvPoint(0, 0)

    Initializes contour scanning process
    """
    z = _cvStartFindContours(image, storage, header_size, mode, method, offset)
    z._depends = (storage,) # to make sure storage is always deleted after z is deleted
    return z

# Finds next contour in the image
cvFindNextContour = cfunc('cvFindNextContour', _cvDLL, CvSeq_p,
    ('scanner', CvContourScanner, 1), # CvContourScanner scanner 
)
cvFindNextContour.__doc__ = """CvSeq* cvFindNextContour(CvContourScanner scanner)

Finds next contour in the image
"""

# Replaces retrieved contour
cvSubstituteContour = cfunc('cvSubstituteContour', _cvDLL, None,
    ('scanner', CvContourScanner, 1), # CvContourScanner scanner
    ('new_contour', CvSeq_p, 1), # CvSeq* new_contour 
)
cvSubstituteContour.__doc__ = """void cvSubstituteContour(CvContourScanner scanner, CvSeq* new_contour)

Replaces retrieved contour
"""

# Finishes scanning process
cvEndFindContours = cfunc('cvEndFindContours', _cvDLL, CvSeq_p,
    ('scanner', ByRefArg(CvContourScanner), 1), # CvContourScanner* scanner 
)
cvEndFindContours.__doc__ = """CvSeq* cvEndFindContours(CvContourScanner* scanner)

Finishes scanning process
"""


#-----------------------------------------------------------------------------
# Segmentation
#-----------------------------------------------------------------------------


# Implements image segmentation by pyramids
_cvPyrSegmentation = cfunc('cvPyrSegmentation', _cvDLL, None,
    ('src', IplImage_p, 1), # IplImage* src
    ('dst', IplImage_p, 1), # IplImage* dst
    ('storage', CvMemStorage_p, 1), # CvMemStorage* storage
    ('comp', ByRefArg(CvSeq_p), 1), # CvSeq** comp
    ('level', c_int, 1), # int level
    ('threshold1', c_double, 1), # double threshold1
    ('threshold2', c_double, 1), # double threshold2 
)

def cvPyrSegmentation(src, dst, storage, level, threshold1, threshold2):
    """CvSeq* cvPyrSegmentation(IplImage* src, IplImage* dst, CvMemStorage* storage, int level, double threshold1, double threshold2)

    Implements image segmentation by pyramids
    """
    comp = CvSeq_p()
    _cvPyrSegmentation(src, dst, storage, comp, level, threshold1, threshold2)
    return comp

_default_cvTermCriteria = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 5, 1)

# Does meanshift image segmentation
cvPyrMeanShiftFiltering = cfunc('cvPyrMeanShiftFiltering', _cvDLL, None,
    ('src', IplImage_p, 1), # IplImage* src
    ('dst', IplImage_p, 1), # IplImage* dst
    ('sp', c_double, 1), # double sp
    ('sr', c_double, 1), # double sr
    ('max_level', c_int, 1, 1), # int max_level=1
    ('termcrit', CvTermCriteria, 1, _default_cvTermCriteria), # CvTermCriteria termcrit=cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,5,1)
)
cvPyrMeanShiftFiltering.__doc__ = """void cvPyrMeanShiftFiltering( const CvArr* src, CvArr* dst, double sp, double sr, int max_level=1, CvTermCriteria termcrit=cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,5,1))

Does meanshift image segmentation
"""

# Does watershed segmentation
cvWatershed = cfunc('cvWatershed', _cvDLL, None,
    ('image', CvArr_p, 1), # const CvArr* image
    ('markers', CvArr_p, 1), # CvArr* markers
)
cvWatershed.__doc__ = """void cvWatershed( const CvArr* image, CvArr* markers )

Does watershed segmentation
"""    


#-----------------------------------------------------------------------------
# Image and Contour moments
#-----------------------------------------------------------------------------


# Calculates all moments up to third order of a polygon or rasterized shape
cvMoments = cfunc('cvMoments', _cvDLL, None,
    ('arr', CvArr_p, 1), # const CvArr* arr
    ('moments', CvMoments_r, 1), # CvMoments* moments
    ('binary', c_int, 1, 0), # int binary
)
cvMoments.__doc__ = """void cvMoments(const CvArr* arr, CvMoments moments, int binary=0)

Calculates all moments up to third order of a polygon or rasterized shape
"""

# Retrieves spatial moment from moment state structure
cvGetSpatialMoment = cfunc('cvGetSpatialMoment', _cvDLL, c_double,
    ('moments', CvMoments_r, 1), # CvMoments* moments
    ('x_order', c_int, 1), # int x_order
    ('y_order', c_int, 1), # int y_order 
)
cvGetSpatialMoment.__doc__ = """double cvGetSpatialMoment(CvMoments moments, int x_order, int y_order)

Retrieves spatial moment from moment state structure
"""

# Retrieves central moment from moment state structure
cvGetCentralMoment = cfunc('cvGetCentralMoment', _cvDLL, c_double,
    ('moments', CvMoments_r, 1), # CvMoments* moments
    ('x_order', c_int, 1), # int x_order
    ('y_order', c_int, 1), # int y_order 
)
cvGetCentralMoment.__doc__ = """double cvGetCentralMoment(CvMoments moments, int x_order, int y_order)

Retrieves central moment from moment state structure
"""

# Retrieves normalized central moment from moment state structure
cvGetNormalizedCentralMoment = cfunc('cvGetNormalizedCentralMoment', _cvDLL, c_double,
    ('moments', CvMoments_r, 1), # CvMoments* moments
    ('x_order', c_int, 1), # int x_order
    ('y_order', c_int, 1), # int y_order 
)
cvGetNormalizedCentralMoment.__doc__ = """double cvGetNormalizedCentralMoment(CvMoments moments, int x_order, int y_order)

Retrieves normalized central moment from moment state structure
"""

# Calculates seven Hu invariants
cvGetHuMoments = cfunc('cvGetHuMoments', _cvDLL, None,
    ('moments', CvMoments_r, 1), # CvMoments* moments
    ('hu_moments', CvHuMoments_r, 1), # CvHuMoments* hu_moments 
)
cvGetHuMoments.__doc__ = """void cvGetHuMoments(CvMoments moments, CvHuMoments hu_moments)

Calculates seven Hu invariants
"""


#-----------------------------------------------------------------------------
# Special Image Transforms
#-----------------------------------------------------------------------------


CV_HOUGH_STANDARD = 0
CV_HOUGH_PROBABILISTIC = 1
CV_HOUGH_MULTI_SCALE = 2
CV_HOUGH_GRADIENT = 3

# Finds lines in binary image using Hough transform
cvHoughLines2 = cfunc('cvHoughLines2', _cvDLL, CvSeq_p,
    ('image', CvArr_p, 1), # CvArr* image
    ('line_storage', c_void_p, 1), # void* line_storage
    ('method', c_int, 1), # int method
    ('rho', c_double, 1), # double rho
    ('theta', c_double, 1), # double theta
    ('threshold', c_int, 1), # int threshold
    ('param1', c_double, 1, 0), # double param1
    ('param2', c_double, 1, 0), # double param2
)
cvHoughLines2.__doc__ = """CvSeq* cvHoughLines2(CvArr* image, void* line_storage, int method, double rho, double theta, int threshold, double param1=0, double param2=0)

Finds lines in binary image using Hough transform
"""

# Finds circles in grayscale image using Hough transform
cvHoughCircles = cfunc('cvHoughCircles', _cvDLL, CvSeq_p,
    ('image', CvArr_p, 1), # CvArr* image
    ('circle_storage', c_void_p, 1), # void* circle_storage
    ('method', c_int, 1), # int method
    ('dp', c_double, 1), # double dp
    ('min_dist', c_double, 1), # double min_dist
    ('param1', c_double, 1, 100), # double param1
    ('param2', c_double, 1, 100), # double param2
)
cvHoughCircles.__doc__ = """CvSeq* cvHoughCircles(CvArr* image, void* circle_storage, int method, double dp, double min_dist, double param1=100, double param2=100)

Finds circles in grayscale image using Hough transform
"""

CV_DIST_MASK_3 = 3
CV_DIST_MASK_5 = 5
CV_DIST_MASK_PRECISE = 0

# Calculates distance to closest zero pixel for all non-zero pixels of source image
cvDistTransform = cfunc('cvDistTransform', _cvDLL, None,
    ('src', CvArr_p, 1), # const CvArr* src
    ('dst', CvArr_p, 1), # CvArr* dst
    ('distance_type', c_int, 1), # int distance_type
    ('mask_size', c_int, 1, 3), # int mask_size
    ('mask', c_float_p, 1, None), # const float* mask
    ('labels', CvArr_p, 1, None), # CvArr* labels
)
cvDistTransform.__doc__ = """void cvDistTransform(const CvArr* src, CvArr* dst, int distance_type=CV_DIST_L2, int mask_size=3, const float* mask=NULL, CvArr* labels=NULL)

Calculates distance to closest zero pixel for all non-zero pixels of source image
"""

CV_INPAINT_NS = 0
CV_INPAINT_TELEA = 1

# Inpaints the selected region in the image
cvInpaint = cfunc('cvInpaint', _cvDLL,  None,
    ('src', CvArr_p, 1), # const CvArr* src
    ('mask', CvArr_p, 1), # const CvArr* mask
    ('dst', CvArr_p, 1), # const CvArr* dst
    ('flags', c_int, 1), # int flags
    ('inpaintRadius', c_double, 1), # double inpaintRadius
)
cvInpaint.__doc__ = """void cvInpaint( const CvArr* src, const CvArr* mask, CvArr* dst, int flags, double inpaintRadius )

Inpaints the selected region in the image
"""


#-----------------------------------------------------------------------------
# Histograms
#-----------------------------------------------------------------------------


# Releases histogram
_cvReleaseHist = cfunc('cvReleaseHist', _cvDLL, None,
    ('hist', ByRefArg(CvHistogram_p), 1), # CvHistogram** hist 
)

_cvCreateHist = cfunc('cvCreateHist', _cvDLL, CvHistogram_p,
    ('dims', c_int, 1), # int dims
    ('sizes', ListPOINTER(c_int), 1), # int* sizes
    ('type', c_int, 1), # int type
    ('ranges', ListPOINTER2(c_float), 1, None), # float** ranges=NULL
    ('uniform', c_int, 1, 1), # int uniform=1
)

# Creates histogram
def cvCreateHist(sizes, hist_type, ranges=None, uniform=1):
    """CvHistogram cvCreateHist(list_or_tuple_of_int sizes, int type, list_of_list_of_float ranges (default=None), int uniform=1)

    Creates histogram
    """
    z = deref(_cvCreateHist(len(sizes), sizes, hist_type, ranges, uniform))
    if z is not None:
        z._owner = True
    return z

# Sets bounds of histogram bins
cvSetHistBinRanges = cfunc('cvSetHistBinRanges', _cvDLL, None,
    ('hist', CvHistogram_r, 1), # CvHistogram* hist
    ('ranges', ListPOINTER2(c_float), 1), # float** ranges
    ('uniform', c_int, 1, 1), # int uniform
)
cvSetHistBinRanges.__doc__ = """void cvSetHistBinRanges(CvHistogram hist, float** ranges, int uniform=1)

Sets bounds of histogram bins
"""

# Clears histogram
cvClearHist = cfunc('cvClearHist', _cvDLL, None,
    ('hist', CvHistogram_r, 1), # CvHistogram* hist 
)
cvClearHist.__doc__ = """void cvClearHist(CvHistogram hist)

Clears histogram
"""

# Makes a histogram out of array
_cvMakeHistHeaderForArray = cfunc('cvMakeHistHeaderForArray', _cvDLL, CvHistogram_p,
    ('dims', c_int, 1), # int dims
    ('sizes', c_int_p, 1), # int* sizes
    ('hist', CvHistogram_p, 1), # CvHistogram* hist
    ('data', c_float_p, 1), # float* data
    ('ranges', ListPOINTER2(c_float), 1, None), # float** ranges
    ('uniform', c_int, 1, 1), # int uniform
)

def cvMakeHistHeaderForArray(dims, sizes, data, ranges=None, uniform=1):
    if z is not None:
        z._owner = True
    return z

.__doc__ = """CvHistogram* cvMakeHistHeaderForArray(int dims, int* sizes, CvHistogram* hist, float* data, float** ranges=NULL, int uniform=1)

Makes a histogram out of array
"""

# Finds minimum and maximum histogram bins
cvGetMinMaxHistValue = cfunc('cvGetMinMaxHistValue', _cvDLL, None,
    ('hist', CvHistogram_p, 1), # const CvHistogram* hist
    ('min_value', c_float_p, 2), # float* min_value
    ('max_value', c_float_p, 2), # float* max_value
    ('min_idx', c_int_p, 1, None), # int* min_idx
    ('max_idx', c_int_p, 1, None), # int* max_idx
)
cvGetMinMaxHistValue.__doc__ = """float min_value, max_value = cvGetMinMaxHistValue(const CvHistogram* hist, int* min_idx=NULL, int* max_idx=NULL)

    Finds minimum and maximum histogram bins
    [ctypes-opencv] Note that both min_value and max_value are returned.
    """

# Normalizes histogram
cvNormalizeHist = cfunc('cvNormalizeHist', _cvDLL, None,
    ('hist', CvHistogram_p, 1), # CvHistogram* hist
    ('factor', c_double, 1), # double factor 
)
cvNormalizeHist.__doc__ = """void cvNormalizeHist(CvHistogram* hist, double factor)

Normalizes histogram
"""

# Thresholds histogram
cvThreshHist = cfunc('cvThreshHist', _cvDLL, None,
    ('hist', CvHistogram_p, 1), # CvHistogram* hist
    ('threshold', c_double, 1), # double threshold 
)
cvThreshHist.__doc__ = """void cvThreshHist(CvHistogram* hist, double threshold)

Thresholds histogram
"""

CV_COMP_CORREL       = 0
CV_COMP_CHISQR       = 1
CV_COMP_INTERSECT    = 2
CV_COMP_BHATTACHARYYA= 3

# Compares two dense histograms
cvCompareHist = cfunc('cvCompareHist', _cvDLL, c_double,
    ('hist1', CvHistogram_p, 1), # const CvHistogram* hist1
    ('hist2', CvHistogram_p, 1), # const CvHistogram* hist2
    ('method', c_int, 1), # int method 
)
cvCompareHist.__doc__ = """double cvCompareHist(const CvHistogram* hist1, const CvHistogram* hist2, int method)

Compares two dense histograms
"""

_cvCopyHist = cfunc('cvCopyHist', _cvDLL, None,
    ('src', CvHistogram_p, 1), # const CvHistogram* src
    ('dst', ByRefArg(CvHistogram_p), 1, CvHistogram_p()), # CvHistogram** dst 
)

# Copies histogram
def cvCopyHist(src, dst=None):
    """CvHistogram* cvCopyHist(const CvHistogram* src, CvHistogram* dst=NULL)

    Copies histogram
    [ctypes-opencv] If dst is NULL, a new CvHistogram is created and the address of it is returned. Otherwise, dst supplies the address of the CvHistogram to be copied to. Warning: I haven't tested this function.
    """
    if dst is None:
        z = _cvCopyHist(src)
        sdAdd_autoclean(z, _cvReleaseHist)
        return z
    _cvCopyHist(src, dst)
    return dst
    
# Calculate the histogram
cvCalcHist = cfunc('cvCalcArrHist', _cvDLL, None,
    ('image', FlexibleListPOINTER(IplImage_p), 1), # IplImage** image
    ('hist', CvHistogram_p, 1), # CvHistogram* hist
    ('accumulate', c_int, 1, 0), # int accumulate
    ('mask', CvArr_p, 1, None), # CvArr* mask
)
cvCalcHist.__doc = """void cvCalcHist( CvArr** arr, CvHistogram* hist, int accumulate=0, const CvArr* mask=NULL )

Calculates array histogram
"""

cvCalcArrHist = cvCalcHist

# Calculates back projection
cvCalcBackProject = cfunc('cvCalcArrBackProject', _cvDLL, None,
    ('image', ListPOINTER(IplImage_p), 1), # IplImage** image
    ('back_project', IplImage_p, 1), # IplImage* back_project
    ('hist', CvHistogram_p, 1), # CvHistogram* hist
)
cvCalcBackProject.__doc = """void cvCalcBackProject( CvArr** image, CvArr* dst, const CvHistogram* hist )

Calculates back projection
"""

# Calculates back projection
cvCalcBackProjectPatch = cfunc('cvCalcArrBackProjectPatch', _cvDLL, None,
    ('image', ListPOINTER(IplImage_p), 1), # IplImage** image
    ('dst', CvArr_p, 1), # CvArr* dst
    ('range', CvSize, 1), # CvSize range
    ('hist', CvHistogram_p, 1), # CvHistogram* hist
    ('method', c_int, 1), # int method
    ('factor', c_double, 1), # double factor
)
cvCalcBackProjectPatch.__doc = """void cvCalcBackProjectPatch( CvArr** image, CvArr* dst, CvSize range, CvHistogram* hist, int method, double factor )

Calculates back projection
"""

cvCalcArrBackProjectPatch = cvCalcBackProjectPatch

# Divides one histogram by another
cvCalcProbDensity = cfunc('cvCalcProbDensity', _cvDLL, None,
    ('hist1', CvHistogram_p, 1), # const CvHistogram* hist1
    ('hist2', CvHistogram_p, 1), # const CvHistogram* hist2
    ('dst_hist', CvHistogram_p, 1), # CvHistogram* dst_hist
    ('scale', c_double, 1, 255), # double scale
)
cvCalcProbDensity.__doc__ = """void cvCalcProbDensity(const CvHistogram* hist1, const CvHistogram* hist2, CvHistogram* dst_hist, double scale=255)

Divides one histogram by another
"""

def cvQueryHistValue_1D(hist, i1, i2):
    """Queries value of histogram bin"""
    return cvGetReal1D(hist.contents.bins, i1)

def cvQueryHistValue_2D(hist, i1, i2):
    """Queries value of histogram bin"""
    return cvGetReal2D(hist.contents.bins, i1, i2)

def cvQueryHistValue_3D(hist, i1, i2, i3):
    """Queries value of histogram bin"""
    return cvGetReal2D(hist.contents.bins, i1, i2, i3)

# Equalizes histogram of grayscale image
cvEqualizeHist = cfunc('cvEqualizeHist', _cvDLL, None,
    ('src', CvArr_p, 1), # const CvArr* src
    ('dst', CvArr_p, 1), # CvArr* dst 
)
cvEqualizeHist.__doc__ = """void cvEqualizeHist(const CvArr* src, CvArr* dst)

Equalizes histogram of grayscale image
"""

def cvGetHistValue_1D(hist, i1, i2):
    """Returns pointer to histogram bin"""
    return cvPtr1D(hist.contents.bins, i1, 0)

def cvQueryHistValue_2D(hist, i1, i2):
    """Returns pointer to histogram bin"""
    return cvPtr2D(hist.contents.bins, i1, i2, 0)

def cvQueryHistValue_3D(hist, i1, i2, i3):
    """Returns pointer to histogram bin"""
    return cvPtr3D(hist.contents.bins, i1, i2, i3, 0)


#-----------------------------------------------------------------------------
# Matching
#-----------------------------------------------------------------------------


CV_TM_SQDIFF = 0
CV_TM_SQDIFF_NORMED = 1
CV_TM_CCORR = 2
CV_TM_CCORR_NORMED = 3
CV_TM_CCOEFF = 4
CV_TM_CCOEFF_NORMED = 5

# Compares template against overlapped image regions
cvMatchTemplate = cfunc('cvMatchTemplate', _cvDLL, None,
    ('image', CvArr_p, 1), # const CvArr* image
    ('templ', CvArr_p, 1), # const CvArr* templ
    ('result', CvArr_p, 1), # CvArr* result
    ('method', c_int, 1), # int method 
)
cvMatchTemplate.__doc__ = """void cvMatchTemplate(const CvArr* image, const CvArr* templ, CvArr* result, int method)

Compares template against overlapped image regions
"""

CV_CONTOURS_MATCH_I1 = 1
CV_CONTOURS_MATCH_I2 = 2
CV_CONTOURS_MATCH_I3 = 3
CV_CONTOUR_TREES_MATCH_I1 = 1
CV_CLOCKWISE = 1
CV_COUNTER_CLOCKWISE = 2

# Compares two shapes
cvMatchShapes = cfunc('cvMatchShapes', _cvDLL, c_double,
    ('object1', c_void_p, 1), # const void* object1
    ('object2', c_void_p, 1), # const void* object2
    ('method', c_int, 1), # int method
    ('parameter', c_double, 1, 0), # double parameter
)
cvMatchShapes.__doc__ = """double cvMatchShapes(const void* object1, const void* object2, int method, double parameter=0)

Compares two shapes
"""

cvCalcEMD2 = cfunc('cvCalcEMD2', _cvDLL, c_float,
    ('signature1', CvArr_p, 1), # const CvArr* signature1
    ('signature2', CvArr_p, 1), # const CvArr* signature2
    ('distance_type', c_int, 1), # int distance_type
    ('distance_func', CvDistanceFunction, 1, None), # CvDistanceFunction distance_func
    ('cost_matrix', c_void_p, 1, None), # const CvArr* cost_matrix
    ('flow', CvArr_p, 1, None), # CvArr* flow
    ('lower_bound', c_float_p, 1, None), # float* lower_bound
    ('userdata', c_void_p, 1, None), # void* userdata
)
cvCalcEMD2.__doc__ = """float cvCalcEMD2( const CvArr* signature1, const CvArr* signature2, int distance_type, CvDistanceFunction distance_func=NULL, const CvArr* cost_matrix=NULL, CvArr* flow=NULL, float* lower_bound=NULL, void* userdata=NULL )

Computes earth mover distance between two weighted point sets (called signatures)
"""


#-----------------------------------------------------------------------------
# Contour Processing Functions
#-----------------------------------------------------------------------------


# Approximates Freeman chain(s) with polygonal curve
cvApproxChains = cfunc('cvApproxChains', _cvDLL, CvSeq_p,
    ('src_seq', CvSeq_p, 1), # CvSeq* src_seq
    ('storage', CvMemStorage_p, 1), # CvMemStorage* storage
    ('method', c_int, 1), # int method
    ('parameter', c_double, 1, 0), # double parameter
    ('minimal_perimeter', c_int, 1, 0), # int minimal_perimeter
    ('recursive', c_int, 1, 0), # int recursive
)
cvApproxChains.__doc__ = """CvSeq* cvApproxChains(CvSeq* src_seq, CvMemStorage* storage, int method=CV_CHAIN_APPROX_SIMPLE, double parameter=0, int minimal_perimeter=0, int recursive=0)

Approximates Freeman chain(s) with polygonal curve
"""

# Initializes chain reader
_cvStartReadChainPoints = cfunc('cvStartReadChainPoints', _cvDLL, None,
    ('chain', CvChain_p, 1), # CvChain* chain
    ('reader', CvChainPtReader_r, 1), # CvChainPtReader* reader 
)

def cvStartReadChainPoints(chain):
    """CvChainPtReader cvStartReadChainPoints(CvChain* chain)

    Initializes chain reader
    [ctypes-opencv] *creates* a chain reader before initializing it
    """
    z = CvChainPtReader()
    _cvStartReadChainPoints(chain, z)
    z._depends = (chain,) # to make sure chain is always deleted after z is deleted
    return z

# Gets next chain point
cvReadChainPoint = cfunc('cvReadChainPoint', _cvDLL, CvPoint,
    ('reader', CvChainPtReader_r, 1), # CvChainPtReader* reader 
)
cvReadChainPoint.__doc__ = """CvPoint cvReadChainPoint(CvChainPtReader reader)

Gets next chain point
"""

CV_POLY_APPROX_DP = 0

# Approximates polygonal curve(s) with desired precision
cvApproxPoly = cfunc('cvApproxPoly', _cvDLL, CvSeq_p,
    ('src_seq', c_void_p, 1), # const void* src_seq
    ('header_size', c_int, 1), # int header_size
    ('storage', CvMemStorage_p, 1), # CvMemStorage* storage
    ('method', c_int, 1), # int method
    ('parameter', c_double, 1), # double parameter
    ('parameter2', c_int, 1, 0), # int parameter2
)
cvApproxPoly.__doc__ = """CvSeq* cvApproxPoly(const void* src_seq, int header_size, CvMemStorage* storage, int method, double parameter, int parameter2=0)

Approximates polygonal curve(s) with desired precision
"""

CV_DOMINANT_IPAN = 1

# Finds high-curvature points of the contour
cvFindDominantPoints = cfunc('cvFindDominantPoints', _cvDLL, CvSeq_p,
    ('contour', CvSeq_p, 1), # CvSeq* contour
    ('storage', CvMemStorage_p, 1), # CvMemStorage* storage
    ('method', c_int, 1, CV_DOMINANT_IPAN), # int header_size
    ('parameter1', c_double, 1, 0), # double parameter1
    ('parameter2', c_double, 1, 0), # double parameter2
    ('parameter3', c_double, 1, 0), # double parameter3
    ('parameter4', c_double, 1, 0), # double parameter4
)
cvFindDominantPoints.__doc__ = """CvSeq* cvFindDominantPoints( CvSeq* contour, CvMemStorage* storage, int method=CV_DOMINANT_IPAN, double parameter1=0, double parameter2=0, double parameter3=0, double parameter4=0)

Finds high-curvature points of the contour
"""

# Calculates up-right bounding rectangle of point set
cvBoundingRect = cfunc('cvBoundingRect', _cvDLL, CvRect,
    ('points', CvArr_p, 1), # CvArr* points
    ('update', c_int, 1, 0), # int update
)
cvBoundingRect.__doc__ = """CvRect cvBoundingRect(CvArr* points, int update=0)

Calculates up-right bounding rectangle of point set
"""

# Calculates area of the whole contour or contour section
cvContourArea = cfunc('cvContourArea', _cvDLL, c_double,
    ('contour', CvArr_p, 1), # const CvArr* contour
    ('slice', CvSlice, 1, CV_WHOLE_SEQ), # CvSlice slice
)
cvContourArea.__doc__ = """double cvContourArea(const CvArr* contour, CvSlice slice=CV_WHOLE_SEQ)

Calculates area of the whole contour or contour section
"""

# Calculates contour perimeter or curve length
cvArcLength = cfunc('cvArcLength', _cvDLL, c_double,
    ('curve', c_void_p, 1), # const void* curve
    ('slice', CvSlice, 1), # CvSlice slice
    ('is_closed', c_int, 1), # int is_closed
)
cvArcLength.__doc__ = """double cvArcLength(const void* curve, CvSlice slice=CV_WHOLE_SEQ, int is_closed=-1)

Calculates contour perimeter or curve length
"""

def cvContourPerimeter(contour):
    """Calculates the contour perimeter of a contour."""
    return cvArcLength( contour, CV_WHOLE_SEQ, 1 )

# Creates hierarchical representation of contour
cvCreateContourTree = cfunc('cvCreateContourTree', _cvDLL, CvContourTree_p,
    ('contour', CvSeq_p, 1), # const CvSeq* contour
    ('storage', CvMemStorage_p, 1), # CvMemStorage* storage
    ('threshold', c_double, 1), # double threshold 
)
cvCreateContourTree.__doc__ = """CvContourTree* cvCreateContourTree(const CvSeq* contour, CvMemStorage* storage, double threshold)

Creates hierarchical representation of contour
"""

# Restores contour from tree
cvContourFromContourTree = cfunc('cvContourFromContourTree', _cvDLL, CvSeq_p,
    ('tree', CvContourTree_p, 1), # const CvContourTree* tree
    ('storage', CvMemStorage_p, 1), # CvMemStorage* storage
    ('criteria', CvTermCriteria, 1), # CvTermCriteria criteria 
)
cvContourFromContourTree.__doc__ = """CvSeq* cvContourFromContourTree(const CvContourTree* tree, CvMemStorage* storage, CvTermCriteria criteria)

Restores contour from tree
"""

# Compares two contours using their tree representations
cvMatchContourTrees = cfunc('cvMatchContourTrees', _cvDLL, c_double,
    ('tree1', CvContourTree_p, 1), # const CvContourTree* tree1
    ('tree2', CvContourTree_p, 1), # const CvContourTree* tree2
    ('method', c_int, 1), # int method
    ('threshold', c_double, 1), # double threshold 
)
cvMatchContourTrees.__doc__ = """double cvMatchContourTrees(const CvContourTree* tree1, const CvContourTree* tree2, int method, double threshold)

Compares two contours using their tree representations
"""


#-----------------------------------------------------------------------------
# Computational Geometry
#-----------------------------------------------------------------------------


# Finds bounding rectangle for two given rectangles
cvMaxRect = cfunc('cvMaxRect', _cvDLL, CvRect,
    ('rect1', CvRect_r, 1), # const CvRect* rect1
    ('rect2', CvRect_r, 1), # const CvRect* rect2 
)
cvMaxRect.__doc__ = """CvRect cvMaxRect(const CvRect rect1, const CvRect rect2)

Finds bounding rectangle for two given rectangles
"""

# Initializes point sequence header from a point vector
cvPointSeqFromMat = cfunc('cvPointSeqFromMat', _cvDLL, CvSeq_p,
    ('seq_kind', c_int, 1), # int seq_kind
    ('mat', CvArr_p, 1), # const CvArr* mat
    ('contour_header', CvContour_p, 1), # CvContour* contour_header
    ('block', CvSeqBlock_p, 1), # CvSeqBlock* block 
)
cvPointSeqFromMat.__doc__ = """CvSeq* cvPointSeqFromMat(int seq_kind, const CvArr* mat, CvContour* contour_header, CvSeqBlock* block)

Initializes point sequence header from a point vector
"""

# Finds box vertices
_cvBoxPoints = cfunc('cvBoxPoints', _cvDLL, None,
    ('box', CvBox2D, 1), # CvBox2D box
    ('pt', CvPoint2D32f*4, 1), # CvPoint2D32f pt
)

def cvBoxPoints(box):
    """CvPoint2D32f pt[4] cvBoxPoints(CvBox2D box)

    Finds box vertices
    """
    pts = (CvPoint2D32f*4)()
    _cvBoxPoints(box, pts)
    return pts

# Fits ellipse to set of 2D points
cvFitEllipse2 = cfunc('cvFitEllipse2', _cvDLL, CvBox2D,
    ('points', CvArr_p, 1), # const CvArr* points 
)
cvFitEllipse2.__doc__ = """CvBox2D cvFitEllipse2(const CvArr* points)

Fits ellipse to set of 2D points
"""

# Fits line to 2D or 3D point set
cvFitLine = cfunc('cvFitLine', _cvDLL, None,
    ('points', CvArr_p, 1), # const CvArr* points
    ('dist_type', c_int, 1), # int dist_type
    ('param', c_double, 1), # double param
    ('reps', c_double, 1), # double reps
    ('aeps', c_double, 1), # double aeps
    ('line', c_float_p, 1), # float* line 
)
cvFitLine.__doc__ = """void cvFitLine(const CvArr* points, int dist_type, double param, double reps, double aeps, float* line)

Fits line to 2D or 3D point set
"""

# Finds convex hull of point set
_cvConvexHull2 = cfunc('cvConvexHull2', _cvDLL, CvSeq_p,
    ('input', CvArr_p, 1), # const CvArr* input
    ('hull_storage', c_void_p, 1, None), # void* hull_storage
    ('orientation', c_int, 1, CV_CLOCKWISE), # int orientation
    ('return_points', c_int, 1, 0), # int return_points
)

def cvConvexHull2(input, orientation=CV_CLOCKWISE, return_points=0):
    """list_or_tuple_of_int cvConvexHull2(list_or_tuple_of_CvPoint input, int orientation=CV_CLOCKWISE, int return_points=0)

    Finds convex hull of point set
    """
    n = len(input)
    hull_mat = cvCreateMat(1, n, CV_32SC1)
    
    point_mat = cvCreateMatFromCvPointList(input)
    _cvConvexHull2(point_mat, hull_mat, orientation, return_points)
    
    hull = []
    for i in xrange(hull_mat.contents.cols):
        hull.append(hull_mat[0,i])
        
    return hull

# Tests contour convex
cvCheckContourConvexity = cfunc('cvCheckContourConvexity', _cvDLL, c_int,
    ('contour', CvArr_p, 1), # const CvArr* contour 
)
cvCheckContourConvexity.__doc__ = """int cvCheckContourConvexity(const CvArr* contour)

Tests contour convex
"""

# Finds convexity defects of contour
cvConvexityDefects = cfunc('cvConvexityDefects', _cvDLL, CvSeq_p,
    ('contour', CvArr_p, 1), # const CvArr* contour
    ('convexhull', CvArr_p, 1), # const CvArr* convexhull
    ('storage', CvMemStorage_p, 1, None), # CvMemStorage* storage
)
cvConvexityDefects.__doc__ = """CvSeq* cvConvexityDefects(const CvArr* contour, const CvArr* convexhull, CvMemStorage* storage=NULL)

Finds convexity defects of contour
"""

# Point in contour test
cvPointPolygonTest = cfunc('cvPointPolygonTest', _cvDLL, c_double,
    ('contour', CvArr_p, 1), # const CvArr* contour
    ('pt', CvPoint2D32f, 1), # CvPoint2D32f pt
    ('measure_dist', c_int, 1), # int measure_dist 
)
cvPointPolygonTest.__doc__ = """double cvPointPolygonTest(const CvArr* contour, CvPoint2D32f pt, int measure_dist)

Point in contour test
"""

# Finds circumscribed rectangle of minimal area for given 2D point set
cvMinAreaRect2 = cfunc('cvMinAreaRect2', _cvDLL, CvBox2D,
    ('points', CvArr_p, 1), # const CvArr* points
    ('storage', CvMemStorage_p, 1, None), # CvMemStorage* storage
)
cvMinAreaRect2.__doc__ = """CvBox2D cvMinAreaRect2(const CvArr* points, CvMemStorage* storage=NULL)

Finds circumscribed rectangle of minimal area for given 2D point set
"""

# Finds circumscribed circle of minimal area for given 2D point set
_cvMinEnclosingCircle = cfunc('cvMinEnclosingCircle', _cvDLL, c_int,
    ('points', CvArr_p, 1), # const CvArr* points
    ('center', ByRefArg(CvPoint2D32f), 1), # CvPoint2D32f* center
    ('radius', ByRefArg(c_float), 1), # float* radius 
)

def cvMinEnclosingCircle(points):
    """(int success, CvPoint2D32f center, float radius) = cvMinEnclosingCircle(const CvArr* points)

    Finds circumscribed circle of minimal area for given 2D point set
    """
    center = CvPoint2D32f()
    radius = c_float()
    success = _cvMinEnclosingCircle(points, center, radius)
    return success, center, radius.value

# Calculates pair-wise geometrical histogram for contour
cvCalcPGH = cfunc('cvCalcPGH', _cvDLL, None,
    ('contour', CvSeq_p, 1), # const CvSeq* contour
    ('hist', CvHistogram_p, 1), # CvHistogram* hist 
)
cvCalcPGH.__doc__ = """void cvCalcPGH(const CvSeq* contour, CvHistogram* hist)

Calculates pair-wise geometrical histogram for contour
"""


#-----------------------------------------------------------------------------
# Planar Subdivisions
#-----------------------------------------------------------------------------


cvSubdiv2DNextEdge = CV_SUBDIV2D_NEXT_EDGE

# Returns one of edges related to given
def cvSubdiv2DGetEdge(edge, next_edge_type):
    """CvSubdiv2DEdge  cvSubdiv2DGetEdge( CvSubdiv2DEdge edge, CvNextEdgeType type )

    Returns one of edges related to given
    """
    ev = edge.value
    e = cast(c_void_p(ev & ~3), POINTER(CvQuadEdge2D))
    ev = e.contents.next[(ev + next_edge_type) & 3]
    return CvSubdiv2DEdge((ev & ~3) + ((ev + (next_edge_type >> 4)) & 3))

# Returns another edge of the same quad-edge
def cvSubdiv2DRotateEdge(edge, rotate):
    """CvSubdiv2DEdge  cvSubdiv2DRotateEdge( CvSubdiv2DEdge edge, int rotate )

    Returns another edge of the same quad-edge
    """
    ev = edge.value
    return  CvSubdiv2DEdge((ev & ~3) + ((ev + rotate) & 3))

# Returns edge origin
def cvSubdiv2DEdgeOrg(edge):
    """CvSubdiv2DPoint cvSubdiv2DEdgeOrg( CvSubdiv2DEdge edge )

    Returns edge origin
    [ctypes-opencv] returns None if no point is found
    """
    ev = edge.value
    e = pointer(CvQuadEdge2D.from_address(ev & ~3))
    return deref(e.contents.pt[ev & 3])

# Returns edge destination
def cvSubdiv2DEdgeDst(edge):
    """CvSubdiv2DPoint cvSubdiv2DEdgeDst( CvSubdiv2DEdge edge )

    Returns edge destination
    [ctypes-opencv] returns None if no point is found
    """
    ev = edge.value
    e = cast(c_void_p(ev & ~3), POINTER(CvQuadEdge2D))
    return deref(e.contents.pt[(ev + 2) & 3])

# Initializes Delaunay triangulation
cvInitSubdivDelaunay2D = cfunc('cvInitSubdivDelaunay2D', _cvDLL, None,
    ('subdiv', CvSubdiv2D_p, 1), # CvSubDiv2D* subdiv
    ('rect', CvRect, 1), # CvRect rect
)
cvInitSubdivDelaunay2D.__doc__ = """void cvInitSubdivDelaunay2D( CvSubdiv2D* subdiv, CvRect rect )

Initializes Delaunay triangulation
"""

# Creates new subdivision
cvCreateSubdiv2D = cfunc('cvCreateSubdiv2D', _cvDLL, CvSubdiv2D_p,
    ('subdiv_type', c_int, 1), # int subdiv_type
    ('header_size', c_int, 1), # int header_size
    ('vtx_size', c_int, 1), # int vtx_size
    ('quadedge_size', c_int, 1), # int quadedge_size
    ('storage', CvMemStorage_p, 1), # CvMemStorage* storage
)
cvCreateSubdiv2D.__doc__ = """CvSubdiv2D* cvCreateSubdiv2D( int subdiv_type, int header_size, int vtx_size, int quadedge_size, CvMemStorage* storage )

Creates new subdivision
"""


# Simplified Delaunay diagram creation
def cvCreateSubdivDelaunay2D(rect, storage):
    """CvSubdiv2D* cvCreateSubdivDelaunay2D(CvRect rect, CvMemStorage* storage)
    
    Simplified Delaunay diagram creation
    """
    subdiv = cvCreateSubdiv2D(CV_SEQ_KIND_SUBDIV2D, sizeof(CvSubdiv2D), sizeof(CvSubdiv2DPoint), sizeof(CvQuadEdge2D), storage)
    cvInitSubdivDelaunay2D(subdiv, rect)
    return subdiv
    
# Inserts a single point to Delaunay triangulation
_cvSubdivDelaunay2DInsert = cfunc('cvSubdivDelaunay2DInsert', _cvDLL, CvSubdiv2DPoint_p,
    ('subdiv', CvSubdiv2D_p, 1), # CvSubdiv2D* subdiv
    ('pt', CvPoint2D32f, 1), # CvPoint2D32f pt
)
def cvSubdivDelaunay2DInsert(subdiv, pt):
    """CvSubdiv2DPoint cvSubdivDelaunay2DInsert(CvSubdiv2D* subdiv, CvPoint2D32f p)

    Inserts a single point to Delaunay triangulation
    [ctypes-opencv] returns None if no subdiv2dpoint is inserted
    """
    return deref(_cvSubdivDelaunay2DInsert(subdiv, pt), subdiv)

# Inserts a single point to Delaunay triangulation
_cvSubdiv2DLocate = cfunc('cvSubdiv2DLocate', _cvDLL, CvSubdiv2DPointLocation,
    ('subdiv', CvSubdiv2D_p, 1), # CvSubdiv2D* subdiv
    ('pt', CvPoint2D32f, 1), # CvPoint2D32f pt
    ('edge', ByRefArg(CvSubdiv2DEdge), 1), # CvSubdiv2DEdge* edge
    ('vertex', ByRefArg(CvSubdiv2DPoint_p), 1, None), # CvSubdiv2DPoint** vertex
)

def cvSubdiv2DLocate(subdiv, pt):
    """(CvSubdiv2DPointLocation res[, CvSubdiv2DEdge edge][, CvSubdiv2DPoint vertex]) = cvSubdiv2DLocate(CvSubdiv2D* subdiv, CvPoint2D32f pt)

    Inserts a single point to Delaunay triangulation
    [ctypes-opencv] Depending on the value of 'res', addtional objects are returned. But the returning object is always a tuple.
    """
    edge = CvSubdiv2DEdge()
    vertex = CvSubdiv2DPoint_p()
    z = _cvSubdiv2DLocate(subdiv, pt, edge, vertex)
    return \
        (z, edge) if z == CV_PTLOC_INSIDE or z == CV_PTLOC_ONEDGE else \
        (z, vertex.contents) if z == CV_PTLOC_VERTEX else \
        (z,)

# Finds the closest subdivision vertex to given point
_cvFindNearestPoint2D = cfunc('cvFindNearestPoint2D', _cvDLL, CvSubdiv2DPoint_p,
    ('subdiv', CvSubdiv2D_p, 1), # CvSubdiv2D* subdiv
    ('pt', CvPoint2D32f, 1), # CvPoint2D32f pt 
)

def cvFindNearestPoint2D(subdiv, pt):
    """CvSubdiv2DPoint cvFindNearestPoint2D(CvSubdiv2D* subdiv, CvPoint2D32f pt)

    Finds the closest subdivision vertex to given point
    [ctypes-opencv] returns None if no subdiv2dpoint is found
    """
    return deref(_cvFindNearestPoint2D(subdiv, pt), subdiv)

# Calculates coordinates of Voronoi diagram cells
cvCalcSubdivVoronoi2D = cfunc('cvCalcSubdivVoronoi2D', _cvDLL, None,
    ('subdiv', CvSubdiv2D_p, 1), # CvSubdiv2D* subdiv 
)
cvCalcSubdivVoronoi2D.__doc__ = """void cvCalcSubdivVoronoi2D(CvSubdiv2D* subdiv)

Calculates coordinates of Voronoi diagram cells
"""

# Removes all virtual points
cvClearSubdivVoronoi2D = cfunc('cvClearSubdivVoronoi2D', _cvDLL, None,
    ('subdiv', CvSubdiv2D_p, 1), # CvSubdiv2D* subdiv 
)
cvClearSubdivVoronoi2D.__doc__ = """void cvClearSubdivVoronoi2D(CvSubdiv2D* subdiv)

Removes all virtual points
"""


#-----------------------------------------------------------------------------
# Accumulation of Background Statistics
#-----------------------------------------------------------------------------


# Adds frame to accumulator
cvAcc = cfunc('cvAcc', _cvDLL, None,
    ('image', CvArr_p, 1), # const CvArr* image
    ('sum', CvArr_p, 1), # CvArr* sum
    ('mask', CvArr_p, 1, None), # const CvArr* mask
)
cvAcc.__doc__ = """void cvAcc(const CvArr* image, CvArr* sum, const CvArr* mask=NULL)

Adds frame to accumulator
"""

# Adds the square of source image to accumulator
cvSquareAcc = cfunc('cvSquareAcc', _cvDLL, None,
    ('image', CvArr_p, 1), # const CvArr* image
    ('sqsum', CvArr_p, 1), # CvArr* sqsum
    ('mask', CvArr_p, 1, None), # const CvArr* mask
)
cvSquareAcc.__doc__ = """void cvSquareAcc(const CvArr* image, CvArr* sqsum, const CvArr* mask=NULL)

Adds the square of source image to accumulator
"""

# Adds product of two input images to accumulator
cvMultiplyAcc = cfunc('cvMultiplyAcc', _cvDLL, None,
    ('image1', CvArr_p, 1), # const CvArr* image1
    ('image2', CvArr_p, 1), # const CvArr* image2
    ('acc', CvArr_p, 1), # CvArr* acc
    ('mask', CvArr_p, 1, None), # const CvArr* mask
)
cvMultiplyAcc.__doc__ = """void cvMultiplyAcc(const CvArr* image1, const CvArr* image2, CvArr* acc, const CvArr* mask=NULL)

Adds product of two input images to accumulator
"""

# Updates running average
cvRunningAvg = cfunc('cvRunningAvg', _cvDLL, None,
    ('image', CvArr_p, 1), # const CvArr* image
    ('acc', CvArr_p, 1), # CvArr* acc
    ('alpha', c_double, 1), # double alpha
    ('mask', CvArr_p, 1, None), # const CvArr* mask
)
cvRunningAvg.__doc__ = """void cvRunningAvg(const CvArr* image, CvArr* acc, double alpha, const CvArr* mask=NULL)

Updates running average
"""


#-----------------------------------------------------------------------------
# Motion Templates
#-----------------------------------------------------------------------------


# Updates motion history image by moving silhouette
cvUpdateMotionHistory = cfunc('cvUpdateMotionHistory', _cvDLL, None,
    ('silhouette', CvArr_p, 1), # const CvArr* silhouette
    ('mhi', CvArr_p, 1), # CvArr* mhi
    ('timestamp', c_double, 1), # double timestamp
    ('duration', c_double, 1), # double duration 
)
cvUpdateMotionHistory.__doc__ = """void cvUpdateMotionHistory(const CvArr* silhouette, CvArr* mhi, double timestamp, double duration)

Updates motion history image by moving silhouette
"""

# Calculates gradient orientation of motion history image
cvCalcMotionGradient = cfunc('cvCalcMotionGradient', _cvDLL, None,
    ('mhi', CvArr_p, 1), # const CvArr* mhi
    ('mask', CvArr_p, 1), # CvArr* mask
    ('orientation', CvArr_p, 1), # CvArr* orientation
    ('delta1', c_double, 1), # double delta1
    ('delta2', c_double, 1), # double delta2
    ('aperture_size', c_int, 1, 3), # int aperture_size
)
cvCalcMotionGradient.__doc__ = """void cvCalcMotionGradient(const CvArr* mhi, CvArr* mask, CvArr* orientation, double delta1, double delta2, int aperture_size=3)

Calculates gradient orientation of motion history image
"""

# Calculates global motion orientation of some selected region
cvCalcGlobalOrientation = cfunc('cvCalcGlobalOrientation', _cvDLL, c_double,
    ('orientation', CvArr_p, 1), # const CvArr* orientation
    ('mask', CvArr_p, 1), # const CvArr* mask
    ('mhi', CvArr_p, 1), # const CvArr* mhi
    ('timestamp', c_double, 1), # double timestamp
    ('duration', c_double, 1), # double duration 
)
cvCalcGlobalOrientation.__doc__ = """double cvCalcGlobalOrientation(const CvArr* orientation, const CvArr* mask, const CvArr* mhi, double timestamp, double duration)

Calculates global motion orientation of some selected region
"""

# Segments whole motion into separate moving parts
cvSegmentMotion = cfunc('cvSegmentMotion', _cvDLL, CvSeq_p,
    ('mhi', CvArr_p, 1), # const CvArr* mhi
    ('seg_mask', CvArr_p, 1), # CvArr* seg_mask
    ('storage', CvMemStorage_p, 1), # CvMemStorage* storage
    ('timestamp', c_double, 1), # double timestamp
    ('seg_thresh', c_double, 1), # double seg_thresh 
)
cvSegmentMotion.__doc__ = """CvSeq* cvSegmentMotion(const CvArr* mhi, CvArr* seg_mask, CvMemStorage* storage, double timestamp, double seg_thresh)

Segments whole motion into separate moving parts
"""


#-----------------------------------------------------------------------------
# Object Tracking
#-----------------------------------------------------------------------------


# Finds object center on back projection
cvMeanShift = cfunc('cvMeanShift', _cvDLL, c_int,
    ('prob_image', CvArr_p, 1), # const CvArr* prob_image
    ('window', CvRect, 1), # CvRect window
    ('criteria', CvTermCriteria, 1), # CvTermCriteria criteria
    ('comp', CvConnectedComp_r, 1), # CvConnectedComp* comp 
)
cvMeanShift.__doc__ = """int cvMeanShift(const CvArr* prob_image, CvRect window, CvTermCriteria criteria, CvConnectedComp comp)

Finds object center on back projection
"""

# Finds object center, size, and orientation
cvCamShift = cfunc('cvCamShift', _cvDLL, c_int,
    ('prob_image', CvArr_p, 1), # const CvArr* prob_image
    ('window', CvRect, 1), # CvRect window
    ('criteria', CvTermCriteria, 1), # CvTermCriteria criteria
    ('comp', CvConnectedComp_r, 1), # CvConnectedComp* comp
    ('box', CvBox2D_p, 1), # CvBox2D* box
)
cvCamShift.__doc__ = """int cvCamShift(const CvArr* prob_image, CvRect window, CvTermCriteria criteria, CvConnectedComp comp, CvBox2D* box=NULL)

Finds object center, size, and orientation
"""

CV_VALUE = 1
CV_ARRAY = 2

# Changes contour position to minimize its energy
_cvSnakeImage = cfunc('cvSnakeImage', _cvDLL, None,
    ('image', IplImage_p, 1), # const IplImage* image
    ('points', ListPOINTER(CvPoint), 1), # CvPoint* points
    ('length', c_int, 1), # int length
    ('alpha', ListPOINTER(c_float), 1), # float* alpha
    ('beta', ListPOINTER(c_float), 1), # float* beta
    ('gamma', ListPOINTER(c_float), 1), # float* gamma
    ('coeff_usage', c_int, 1), # int coeff_usage
    ('win', CvSize, 1), # CvSize win
    ('criteria', CvTermCriteria, 1), # CvTermCriteria criteria
    ('calc_gradient', c_int, 1, 1), # int calc_gradient
)

def cvSnakeImage(image, points, alpha, beta, gamma, coeff_usage, win, criteria, calc_gradient=1):
    """void cvSnakeImage(const IplImage* image, list_or_tuple_of_CvPoint points, list_or_tuple_of_float alpha, list_or_tuple_of_float beta, list_or_tuple_of_float gamma, int coeff_usage, CvSize win, CvTermCriteria criteria, int calc_gradient=1)

    Changes contour position to minimize its energy
    """
    _cvSnakeImage(image, points, len(points), alpha, beta, gamma, coeff_usage, win, criteria, calc_gradient)


#-----------------------------------------------------------------------------
# Optical Flow
#-----------------------------------------------------------------------------


CV_LKFLOW_PYR_A_READY = 1
CV_LKFLOW_PYR_B_READY = 2
CV_LKFLOW_INITIAL_GUESSES = 4

# Calculates optical flow for two images
cvCalcOpticalFlowHS = cfunc('cvCalcOpticalFlowHS', _cvDLL, None,
    ('prev', CvArr_p, 1), # const CvArr* prev
    ('curr', CvArr_p, 1), # const CvArr* curr
    ('use_previous', c_int, 1), # int use_previous
    ('velx', CvArr_p, 1), # CvArr* velx
    ('vely', CvArr_p, 1), # CvArr* vely
    ('lambda', c_double, 1), # double lambda
    ('criteria', CvTermCriteria, 1), # CvTermCriteria criteria 
)
cvCalcOpticalFlowHS.__doc__ = """void cvCalcOpticalFlowHS(const CvArr* prev, const CvArr* curr, int use_previous, CvArr* velx, CvArr* vely, double lambda, CvTermCriteria criteria)

Calculates optical flow for two images
"""

# Calculates optical flow for two images
cvCalcOpticalFlowLK = cfunc('cvCalcOpticalFlowLK', _cvDLL, None,
    ('prev', CvArr_p, 1), # const CvArr* prev
    ('curr', CvArr_p, 1), # const CvArr* curr
    ('win_size', CvSize, 1), # CvSize win_size
    ('velx', CvArr_p, 1), # CvArr* velx
    ('vely', CvArr_p, 1), # CvArr* vely 
)
cvCalcOpticalFlowLK.__doc__ = """void cvCalcOpticalFlowLK(const CvArr* prev, const CvArr* curr, CvSize win_size, CvArr* velx, CvArr* vely)

Calculates optical flow for two images
"""

# Calculates optical flow for two images by block matching method
cvCalcOpticalFlowBM = cfunc('cvCalcOpticalFlowBM', _cvDLL, None,
    ('prev', CvArr_p, 1), # const CvArr* prev
    ('curr', CvArr_p, 1), # const CvArr* curr
    ('block_size', CvSize, 1), # CvSize block_size
    ('shift_size', CvSize, 1), # CvSize shift_size
    ('max_range', CvSize, 1), # CvSize max_range
    ('use_previous', c_int, 1), # int use_previous
    ('velx', CvArr_p, 1), # CvArr* velx
    ('vely', CvArr_p, 1), # CvArr* vely 
)
cvCalcOpticalFlowBM.__doc__ = """void cvCalcOpticalFlowBM(const CvArr* prev, const CvArr* curr, CvSize block_size, CvSize shift_size, CvSize max_range, int use_previous, CvArr* velx, CvArr* vely)

Calculates optical flow for two images by block matching method
"""

# Calculates optical flow for a sparse feature set using iterative Lucas-Kanade method in   pyramids
_cvCalcOpticalFlowPyrLK = cfunc('cvCalcOpticalFlowPyrLK', _cvDLL, None,
    ('prev', CvArr_p, 1), # const CvArr* prev
    ('curr', CvArr_p, 1), # const CvArr* curr
    ('prev_pyr', CvArr_p, 1), # CvArr* prev_pyr
    ('curr_pyr', CvArr_p, 1), # CvArr* curr_pyr
    ('prev_features', c_void_p, 1), # const CvPoint2D32f* prev_features
    ('curr_features', c_void_p, 1), # CvPoint2D32f* curr_features
    ('count', c_int, 1), # int count
    ('win_size', CvSize, 1), # CvSize win_size
    ('level', c_int, 1), # int level
    ('status', c_char_p, 1), # char* status
    ('track_error', c_float_p, 1), # float* track_error
    ('criteria', CvTermCriteria, 1), # CvTermCriteria criteria
    ('flags', c_int, 1), # int flags 
)

def cvCalcOpticalFlowPyrLK(prev, curr, prev_pyr, curr_pyr, pref_features, win_size, level, track_error, criteria, flags):
    """(curr_features, status) = cvCalcOpticalFlowPyrLK(const CvArr* prev, const CvArr* curr, CvArr* prev_pyr, CvArr* curr_pyr, const CvPoint2D32f* prev_features, CvSize win_size, int level, float* track_error, CvTermCriteria criteria, int flags)

    Calculates optical flow for a sparse feature set using iterative Lucas-Kanade method in pyramids
    [ctypes-opencv] The number of features to be tracked is len(pref_features). 
    [ctypes-opencv] Returns a new list of features (curr_features) and a new status array.
    """
    n = len(pref_features)
    inmat = cvCreateMatFromCvPoint2D32fList(pref_features)
    outmat = cvCreateMat(1, n, CV_32FC2)
    status = (c_char*n)()
    _cvCalcOpticalFlowPyrLK(prev, curr, prev_pyr, curr_pyr, inmat.data.ptr, outmat.data.ptr, n, win_size, level, status, track_error, criteria, flags)
    curr_features = []
    z = (CvPoint2D32f*n)()
    memmove(z, outmat.data.ptr, n*sizeof(CvPoint2D32f))
    return z, status


#-----------------------------------------------------------------------------
# Estimators
#-----------------------------------------------------------------------------


# Deallocates Kalman filter structure
_cvReleaseKalman = cfunc('cvReleaseKalman', _cvDLL, None,
    ('kalman', ByRefArg(CvKalman_p), 1), # CvKalman** kalman 
)

# Allocates Kalman filter structure
_cvCreateKalman = cfunc('cvCreateKalman', _cvDLL, CvKalman_p,
    ('dynam_params', c_int, 1), # int dynam_params
    ('measure_params', c_int, 1), # int measure_params
    ('control_params', c_int, 1, 0), # int control_params
)

def cvCreateKalman(dynam_params, measure_params, control_params=0):
    """CvKalman cvCreateKalman(int dynam_params, int measure_params, int control_params=0)

    Allocates Kalman filter structure
    [ctypes-opencv] returns None if no CvKalman is created
    """
    return deref(_cvCreateKalman(dynam_params, measure_params, control_params))

# Estimates subsequent model state
cvKalmanPredict = cfunc('cvKalmanPredict', _cvDLL, CvMat_p,
    ('kalman', CvKalman_r, 1), # CvKalman* kalman
    ('control', CvMat_p, 1, None), # const CvMat* control
)
cvKalmanPredict.__doc__ = """const CvMat* cvKalmanPredict(CvKalman kalman, const CvMat* control=NULL)

Estimates subsequent model state
"""

cvKalmanUpdateByTime = cvKalmanPredict

# Adjusts model state
cvKalmanCorrect = cfunc('cvKalmanCorrect', _cvDLL, CvMat_p,
    ('kalman', CvKalman_r, 1), # CvKalman* kalman
    ('measurement', CvMat_p, 1), # const CvMat* measurement 
)
cvKalmanCorrect.__doc__ = """const CvMat* cvKalmanCorrect(CvKalman kalman, const CvMat* measurement)

Adjusts model state
"""

cvKalmanUpdateByMeasurement = cvKalmanCorrect

# Deallocates ConDensation filter structure
_cvReleaseConDensation = cfunc('cvReleaseConDensation', _cvDLL, None,
    ('condens', ByRefArg(CvConDensation_p), 1), # CvConDensation** condens 
)

_cvCreateConDensation = cfunc('cvCreateConDensation', _cvDLL, CvConDensation_p,
    ('dynam_params', c_int, 1), # int dynam_params
    ('measure_params', c_int, 1), # int measure_params
    ('sample_count', c_int, 1), # int sample_count 
)

# Allocates ConDensation filter structure
def cvCreateConDensation(dynam_params, measure_params, sample_count):
    """CvConDensation cvCreateConDensation(int dynam_params, int measure_params, int sample_count)

    Allocates ConDensation filter structure
    [ctypes-opencv] returns None if no CvConDensation is created
    """
    return deref(_cvCreateConDensation(dynam_params, measure_params, sample_count))

# Initializes sample set for ConDensation algorithm
cvConDensInitSampleSet = cfunc('cvConDensInitSampleSet', _cvDLL, None,
    ('condens', CvConDensation_r, 1), # CvConDensation* condens
    ('lower_bound', CvMat_p, 1), # CvMat* lower_bound
    ('upper_bound', CvMat_p, 1), # CvMat* upper_bound 
)
cvConDensInitSampleSet.__doc__ = """void cvConDensInitSampleSet(CvConDensation condens, CvMat* lower_bound, CvMat* upper_bound)

Initializes sample set for ConDensation algorithm
"""

# Estimates subsequent model state
cvConDensUpdateByTime = cfunc('cvConDensUpdateByTime', _cvDLL, None,
    ('condens', CvConDensation_r, 1), # CvConDensation* condens 
)
cvConDensUpdateByTime.__doc__ = """void cvConDensUpdateByTime(CvConDensation condens)

Estimates subsequent model state
"""


#-----------------------------------------------------------------------------
# Object Detection
#-----------------------------------------------------------------------------


# Releases haar classifier cascade
_cvReleaseHaarClassifierCascade = cfunc('cvReleaseHaarClassifierCascade', _cvDLL, None,
    ('cascade', ByRefArg(CvHaarClassifierCascade_p), 1), # CvHaarClassifierCascade** cascade 
)

# Loads a trained cascade classifier from file or the classifier database embedded in OpenCV
_cvLoadHaarClassifierCascade = cfunc('cvLoadHaarClassifierCascade', _cvDLL, CvHaarClassifierCascade_p,
    ('directory', c_char_p, 1), # const char* directory
    ('orig_window_size', CvSize, 1), # CvSize orig_window_size 
)

def cvLoadHaarClassifierCascade(directory, orig_window_size):
    """CvHaarClassifierCascade cvLoadHaarClassifierCascade(const char* directory, CvSize orig_window_size)

    Loads a trained cascade classifier from file or the classifier database embedded in OpenCV
    [ctypes-opencv] returns None if no cascade is loaded
    """
    return deref(_cvLoadHaarClassifierCascade(directory, orig_window_size))


CV_HAAR_DO_CANNY_PRUNING = 1
CV_HAAR_SCALE_IMAGE = 2

# Detects objects in the image
cvHaarDetectObjects = cfunc('cvHaarDetectObjects', _cvDLL, CvSeq_p,
    ('image', CvArr_p, 1), # const CvArr* image
    ('cascade', CvHaarClassifierCascade_r, 1), # CvHaarClassifierCascade* cascade
    ('storage', CvMemStorage_p, 1), # CvMemStorage* storage
    ('scale_factor', c_double, 1, 1), # double scale_factor
    ('min_neighbors', c_int, 1, 3), # int min_neighbors
    ('flags', c_int, 1, 0), # int flags
    ('min_size', CvSize, 1), # CvSize min_size
)
cvHaarDetectObjects.__doc__ = """CvSeq* cvHaarDetectObjects(const CvArr* image, CvHaarClassifierCascade cascade, CvMemStorage* storage, double scale_factor=1.1, int min_neighbors=3, int flags=0, CvSize min_size=cvSize(0, 0)

Detects objects in the image
"""
def ChangeCvSeqToCvRect(result, func, args):
    '''Handle the casting to extract a list of Rects from the Seq returned'''
    res = []
    for i in xrange(result[0].total):
        f = cvGetSeqElem(result, i)
        r = cast(f, CvRect_p)[0]
        res.append(r)
    return res
cvHaarDetectObjects.errcheck = ChangeCvSeqToCvRect

# Assigns images to the hidden cascade
cvSetImagesForHaarClassifierCascade = cfunc('cvSetImagesForHaarClassifierCascade', _cvDLL, None,
    ('cascade', CvHaarClassifierCascade_r, 1), # CvHaarClassifierCascade* cascade
    ('sum', CvArr_p, 1), # const CvArr* sum
    ('sqsum', CvArr_p, 1), # const CvArr* sqsum
    ('tilted_sum', CvArr_p, 1), # const CvArr* tilted_sum
    ('scale', c_double, 1), # double scale 
)
cvSetImagesForHaarClassifierCascade.__doc__ = """void cvSetImagesForHaarClassifierCascade(CvHaarClassifierCascade cascade, const CvArr* sum, const CvArr* sqsum, const CvArr* tilted_sum, double scale)

Assigns images to the hidden cascade
"""

# Runs cascade of boosted classifier at given image location
cvRunHaarClassifierCascade = cfunc('cvRunHaarClassifierCascade', _cvDLL, c_int,
    ('cascade', CvHaarClassifierCascade_r, 1), # CvHaarClassifierCascade* cascade
    ('pt', CvPoint, 1), # CvPoint pt
    ('start_stage', c_int, 1, 0), # int start_stage
)
cvRunHaarClassifierCascade.__doc__ = """int cvRunHaarClassifierCascade(CvHaarClassifierCascade cascade, CvPoint pt, int start_stage=0)

Runs cascade of boosted classifier at given image location
"""


#-----------------------------------------------------------------------------
# Camera Calibration
#-----------------------------------------------------------------------------


CV_CALIB_USE_INTRINSIC_GUESS = 1
CV_CALIB_FIX_ASPECT_RATIO = 2
CV_CALIB_FIX_PRINCIPAL_POINT = 4
CV_CALIB_ZERO_TANGENT_DIST = 8
CV_CALIB_CB_ADAPTIVE_THRESH = 1
CV_CALIB_CB_NORMALIZE_IMAGE = 2
CV_CALIB_CB_FILTER_QUADS = 4



# Projects 3D points to image plane
cvProjectPoints2 = cfunc('cvProjectPoints2', _cvDLL, None,
    ('object_points', CvMat_p, 1), # const CvMat* object_points
    ('rotation_vector', CvMat_p, 1), # const CvMat* rotation_vector
    ('translation_vector', CvMat_p, 1), # const CvMat* translation_vector
    ('intrinsic_matrix', CvMat_p, 1), # const CvMat* intrinsic_matrix
    ('distortion_coeffs', CvMat_p, 1), # const CvMat* distortion_coeffs
    ('image_points', CvMat_p, 1), # CvMat* image_points
    ('dpdrot', CvMat_p, 1, None), # CvMat* dpdrot
    ('dpdt', CvMat_p, 1, None), # CvMat* dpdt
    ('dpdf', CvMat_p, 1, None), # CvMat* dpdf
    ('dpdc', CvMat_p, 1, None), # CvMat* dpdc
    ('dpddist', CvMat_p, 1, None), # CvMat* dpddist
)
cvProjectPoints2.__doc__ = """void cvProjectPoints2(const CvMat* object_points, const CvMat* rotation_vector, const CvMat* translation_vector, const CvMat* intrinsic_matrix, const CvMat* distortion_coeffs, CvMat* image_points, CvMat* dpdrot=NULL, CvMat* dpdt=NULL, CvMat* dpdf=NULL, CvMat* dpdc=NULL, CvMat* dpddist=NULL)

Projects 3D points to image plane
"""

# Finds perspective transformation between two planes
cvFindHomography = cfunc('cvFindHomography', _cvDLL, None,
    ('src_points', CvMat_p, 1), # const CvMat* src_points
    ('dst_points', CvMat_p, 1), # const CvMat* dst_points
    ('homography', CvMat_p, 1), # CvMat* homography 
)
cvFindHomography.__doc__ = """void cvFindHomography(const CvMat* src_points, const CvMat* dst_points, CvMat* homography)

Finds perspective transformation between two planes
"""

# Finds intrinsic and extrinsic camera parameters using calibration pattern
cvCalibrateCamera2 = cfunc('cvCalibrateCamera2', _cvDLL, None,
    ('object_points', CvMat_p, 1), # const CvMat* object_points
    ('image_points', CvMat_p, 1), # const CvMat* image_points
    ('point_counts', CvMat_p, 1), # const CvMat* point_counts
    ('image_size', CvSize, 1), # CvSize image_size
    ('intrinsic_matrix', CvMat_p, 1), # CvMat* intrinsic_matrix
    ('distortion_coeffs', CvMat_p, 1), # CvMat* distortion_coeffs
    ('rotation_vectors', CvMat_p, 1, None), # CvMat* rotation_vectors
    ('translation_vectors', CvMat_p, 1, None), # CvMat* translation_vectors
    ('flags', c_int, 1, 0), # int flags
)
cvCalibrateCamera2.__doc__ = """void cvCalibrateCamera2(const CvMat* object_points, const CvMat* image_points, const CvMat* point_counts, CvSize image_size, CvMat* intrinsic_matrix, CvMat* distortion_coeffs, CvMat* rotation_vectors=NULL, CvMat* translation_vectors=NULL, int flags=0)

Finds intrinsic and extrinsic camera parameters using calibration pattern
"""

# Finds extrinsic camera parameters for particular view
cvFindExtrinsicCameraParams2 = cfunc('cvFindExtrinsicCameraParams2', _cvDLL, None,
    ('object_points', CvMat_p, 1), # const CvMat* object_points
    ('image_points', CvMat_p, 1), # const CvMat* image_points
    ('intrinsic_matrix', CvMat_p, 1), # const CvMat* intrinsic_matrix
    ('distortion_coeffs', CvMat_p, 1), # const CvMat* distortion_coeffs
    ('rotation_vector', CvMat_p, 1), # CvMat* rotation_vector
    ('translation_vector', CvMat_p, 1), # CvMat* translation_vector 
)
cvFindExtrinsicCameraParams2.__doc__ = """void cvFindExtrinsicCameraParams2(const CvMat* object_points, const CvMat* image_points, const CvMat* intrinsic_matrix, const CvMat* distortion_coeffs, CvMat* rotation_vector, CvMat* translation_vector)

Finds extrinsic camera parameters for particular view
"""

# Converts rotation matrix to rotation vector or vice versa
cvRodrigues2 = cfunc('cvRodrigues2', _cvDLL, c_int,
    ('src', CvMat_p, 1), # const CvMat* src
    ('dst', CvMat_p, 1), # CvMat* dst
    ('jacobian', CvMat_p, 1, 0), # CvMat* jacobian
)
cvRodrigues2.__doc__ = """int cvRodrigues2(const CvMat* src, CvMat* dst, CvMat* jacobian=0)

Converts rotation matrix to rotation vector or vice versa
"""

# Transforms image to compensate lens distortion
cvUndistort2 = cfunc('cvUndistort2', _cvDLL, None,
    ('src', CvArr_p, 1), # const CvArr* src
    ('dst', CvArr_p, 1), # CvArr* dst
    ('intrinsic_matrix', CvMat_p, 1), # const CvMat* intrinsic_matrix
    ('distortion_coeffs', CvMat_p, 1), # const CvMat* distortion_coeffs 
)
cvUndistort2.__doc__ = """void cvUndistort2(const CvArr* src, CvArr* dst, const CvMat* intrinsic_matrix, const CvMat* distortion_coeffs)

Transforms image to compensate lens distortion
"""

# Computes undistorion map
cvInitUndistortMap = cfunc('cvInitUndistortMap', _cvDLL, None,
    ('intrinsic_matrix', CvMat_p, 1), # const CvMat* intrinsic_matrix
    ('distortion_coeffs', CvMat_p, 1), # const CvMat* distortion_coeffs
    ('mapx', CvArr_p, 1), # CvArr* mapx
    ('mapy', CvArr_p, 1), # CvArr* mapy 
)
cvInitUndistortMap.__doc__ = """void cvInitUndistortMap(const CvMat* intrinsic_matrix, const CvMat* distortion_coeffs, CvArr* mapx, CvArr* mapy)

Computes undistorion map
"""

_cvFindChessboardCorners = cfunc('cvFindChessboardCorners', _cvDLL, c_int,
    ('image', c_void_p, 1), # const void* image
    ('pattern_size', CvSize, 1), # CvSize pattern_size
    ('corners', CvPoint2D32f_p, 1), # CvPoint2D32f* corners
    ('corner_count', c_int_p, 1, None), # int* corner_count
    ('flags', c_int, 1, CV_CALIB_CB_ADAPTIVE_THRESH), # int flags
)

# Finds positions of internal corners of the chessboard
def cvFindChessboardCorners(image, pattern_size, flags=CV_CALIB_CB_ADAPTIVE_THRESH):
    """ctypes_array_of_CvPoint2D32f cvFindChessboardCorners(const void* image, CvSize pattern_size, int flags=CV_CALIB_CB_ADAPTIVE_THRESH)

    Finds positions of internal corners of the chessboard
    """
    MAX_CORNER_COUNT = 32768
    corners = (CvPoint2D32f*MAX_CORNER_COUNT)()
    corner_count = c_int(0)
    z = _cvFindChessboardCorners(image, pattern_size, corners, pointer(corner_count), flags)
        
    if z:
        n = corner_count.value
        corners = (CvPoint2D32f*n)(*corners[:n])
    else:
        corners = (CvPoint2D32f*0)()
        
    corners.pattern_found = z
    return corners
    
_cvDrawChessboardCorners = cfunc('cvDrawChessboardCorners', _cvDLL, None,
    ('image', CvArr_p, 1), # CvArr* image
    ('pattern_size', CvSize, 1), # CvSize pattern_size
    ('corners', CvPoint2D32f_p, 1), # CvPoint2D32f* corners
    ('count', c_int, 1), # int count
    ('pattern_was_found', c_int, 1), # int pattern_was_found 
)

# Renders the detected chessboard corners
def cvDrawChessboardCorners(image, pattern_size, corners):
    """void cvDrawChessboardCorners(CvArr* image, CvSize pattern_size, ctypes_array_of_CvPoint2D32f corners)

    Renders the detected chessboard corners
    """
    _cvDrawChessboardCorners(image, pattern_size, corners, len(corners), corners.pattern_found)


#-----------------------------------------------------------------------------
# Camera Calibration
#-----------------------------------------------------------------------------


class CvPOSITObject(_Structure):
    _fields_ = [
        ('N', c_int), # int N
        ('inv_matr', c_float_p), # float* inv_matr
        ('obj_vecs', c_float_p), # float* obj_vecs
        ('img_vecs', c_float_p), # float* img_vecs
    ]
    
    def __del__(self):
        _cvReleasePOSITObject(pointer(self))
        
CvPOSITObject_p = POINTER(CvPOSITObject)
CvPOSITObject_r = ByRefArg(CvPOSITObject)

# Deallocates 3D object structure
_cvReleasePOSITObject = cfunc('cvReleasePOSITObject', _cvDLL, None,
    ('posit_object', ByRefArg(CvPOSITObject_p), 1), # CvPOSITObject** posit_object 
)

_cvCreatePOSITObject = cfunc('cvCreatePOSITObject', _cvDLL, CvPOSITObject_p,
    ('points', ListPOINTER(CvPoint3D32f), 1), # CvPoint3D32f* points
    ('point_count', c_int, 1), # int point_count 
)

# Initializes structure containing object information
def cvCreatePOSITObject(points):
    """CvPOSITObject cvCreatePOSITObject(list_or_tupleof_CvPoint3D32f points)

    Initializes structure containing object information
    [ctypes-opencv] returns None if no posit object is created
    """
    return deref(_cvCreatePOSITObject(points, len(points)))

# Implements POSIT algorithm
cvPOSIT = cfunc('cvPOSIT', _cvDLL, None,
    ('posit_object', CvPOSITObject_r, 1), # CvPOSITObject* posit_object
    ('image_points', CvPoint2D32f_r, 1), # CvPoint2D32f* image_points
    ('focal_length', c_double, 1), # double focal_length
    ('criteria', CvTermCriteria, 1), # CvTermCriteria criteria
    ('rotation_matrix', CvMatr32f, 1), # CvMatr32f rotation_matrix
    ('translation_vector', CvVect32f, 1), # CvVect32f translation_vector 
)
cvPOSIT.__doc__ = """void cvPOSIT(CvPOSITObject posit_object, CvPoint2D32f image_points, double focal_length, CvTermCriteria criteria, CvMatr32f rotation_matrix, CvVect32f translation_vector)

Implements POSIT algorithm
"""

# Calculates homography matrix for oblong planar object (e.g. arm)
cvCalcImageHomography = cfunc('cvCalcImageHomography', _cvDLL, None,
    ('line', c_float_p, 1), # float* line
    ('center', CvPoint3D32f_r, 1), # CvPoint3D32f* center
    ('intrinsic', c_float_p, 1), # float* intrinsic
    ('homography', c_float_p, 1), # float* homography 
)
cvCalcImageHomography.__doc__ = """void cvCalcImageHomography(float* line, CvPoint3D32f center, float* intrinsic, float* homography)

Calculates homography matrix for oblong planar object (e.g. arm)
"""


#-----------------------------------------------------------------------------
# Epipolar Geometry
#-----------------------------------------------------------------------------


CV_FM_7POINT = 1
CV_FM_8POINT = 2
CV_FM_LMEDS_ONLY = 4
CV_FM_RANSAC_ONLY = 8
CV_FM_LMEDS = CV_FM_LMEDS_ONLY + CV_FM_8POINT
CV_FM_RANSAC = CV_FM_RANSAC_ONLY + CV_FM_8POINT

# Calculates fundamental matrix from corresponding points in two images
cvFindFundamentalMat = cfunc('cvFindFundamentalMat', _cvDLL, c_int,
    ('points1', CvMat_p, 1), # const CvMat* points1
    ('points2', CvMat_p, 1), # const CvMat* points2
    ('fundamental_matrix', CvMat_p, 1), # CvMat* fundamental_matrix
    ('method', c_int, 1), # int method
    ('param1', c_double, 1, 1), # double param1
    ('param2', c_double, 1, 0), # double param2
    ('status', CvMat_p, 1, None), # CvMat* status
)
cvFindFundamentalMat.__doc__ = """int cvFindFundamentalMat(const CvMat* points1, const CvMat* points2, CvMat* fundamental_matrix, int method=CV_FM_RANSAC, double param1=1., double param2=0.99, CvMat* status=NUL)

Calculates fundamental matrix from corresponding points in two images
"""

# For points in one image of stereo pair computes the corresponding epilines in the other image
cvComputeCorrespondEpilines = cfunc('cvComputeCorrespondEpilines', _cvDLL, None,
    ('points', CvMat_p, 1), # const CvMat* points
    ('which_image', c_int, 1), # int which_image
    ('fundamental_matrix', CvMat_p, 1), # const CvMat* fundamental_matrix
    ('correspondent_lines', CvMat_p, 1), # CvMat* correspondent_lines
)
cvComputeCorrespondEpilines.__doc__ = """void cvComputeCorrespondEpilines(const CvMat* points, int which_image, const CvMat* fundamental_matrix, CvMat* correspondent_line)

For points in one image of stereo pair computes the corresponding epilines in the other image
"""

# Convert points to/from homogeneous coordinates
if _cvver == 110:
    cvConvertPointsHomogeneous = cfunc('cvConvertPointsHomogeneous', _cvDLL, None,
        ('src', CvMat_p, 1), # const CvMat* src
        ('dst', CvMat_p, 1), # CvMat* dst 
    )
    cvConvertPointsHomogeneous.__doc__ = """void cvConvertPointsHomogeneous(const CvMat* src, CvMat* dst)

    Convert points to/from homogeneous coordinates
    """
    
    cvConvertPointsHomogenious = cvConvertPointsHomogeneous
elif _cvver == 100:
    cvConvertPointsHomogenious = cfunc('cvConvertPointsHomogenious', _cvDLL, None,
        ('src', CvMat_p, 1), # const CvMat* src
        ('dst', CvMat_p, 1), # CvMat* dst 
    )
    cvConvertPointsHomogenious.__doc__ = """void cvConvertPointsHomogeneous(const CvMat* src, CvMat* dst)

    Convert points to/from homogeneous coordinates
    """
    
    cvConvertPointsHomogeneous = cvConvertPointsHomogenious



# --- 1 Image Processing -----------------------------------------------------

# --- 1.1 Gradients, Edges and Corners ---------------------------------------

# --- 1.2 Sampling, Interpolation and Geometrical Transforms -----------------

# --- 1.3 Morphological Operations -------------------------------------------

# --- 1.4 Filters and Color Conversion ---------------------------------------

# --- 1.5 Pyramids and the Applications --------------------------------------

# --- 1.6 Connected Components -----------------------------------------------

# --- 1.7 Image and Contour moments ------------------------------------------

# --- 1.8 Special Image Transforms -------------------------------------------

# --- 1.9 Histograms ---------------------------------------------------------

# --- 1.10 Matching ----------------------------------------------------------

# --- 2 Structural Analysis --------------------------------------------------

# --- 2.1 Contour Processing Functions ---------------------------------------

# --- 2.2 Computational Geometry ---------------------------------------------

# --- 2.3 Planar Subdivisions ------------------------------------------------

# --- 3 Motion Analysis and Object Tracking ----------------------------------

# --- 3.1 Accumulation of Background Statistics ------------------------------

# --- 3.2 Motion Templates ---------------------------------------------------

# --- 3.3 Object Tracking ----------------------------------------------------

# --- 3.4 Optical Flow -------------------------------------------------------

# --- 3.5 Estimators ---------------------------------------------------------

# --- 4 Pattern Recognition --------------------------------------------------

# --- 4.1 Object Detection ---------------------------------------------------

# --- 5 Camera Calibration and 3D Reconstruction -----------------------------

# --- 5.1 Camera Calibration -------------------------------------------------

# --- 5.2 Pose Estimation ----------------------------------------------------

# --- 5.3 Epipolar Geometry --------------------------------------------------


#=============================================================================
# End of of cv/cv.h
#=============================================================================




#=============================================================================
# Wrap up all the functions and constants into __all__
#=============================================================================
__all__ = [x for x in locals().keys() \
    if  x.startswith('CV') or \
        x.startswith('cv') or \
        x.startswith('Cv') or \
        x.startswith('IPL') or \
        x.startswith('Ipl') or \
        x.startswith('ipl') or \
        x.startswith('sizeof_')]
