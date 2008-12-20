#!/usr/bin/python
# OpenCV's Python demo
# -- adapted by Minh-Tri Pham to work with ctypes-opencv
from opencv import *
import sys

if __name__ == "__main__":
    cvNamedWindow("win")
    filename = sys.argv[1]
    im = cvLoadImage(filename, CV_LOAD_IMAGE_GRAYSCALE)
    im3 = cvLoadImage(filename, CV_LOAD_IMAGE_COLOR)
    chessboard_dim = cvSize( 5, 6 )
    
    corners = cvFindChessboardCorners( im, chessboard_dim )
    cvDrawChessboardCorners( im3, chessboard_dim, corners )
    
    cvShowImage("win", im3);
    cvWaitKey()
