#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Juan Montesinos"
__year__ = "2019"
__version__ = "0.1"
__maintainer__ = "Juan Montesinos"
__email__ = "juanfelipe.montesinos@upf.edu"
__status__ = "Prototype"
import cv2
import numpy as np
class window():
    def __init__(self,flow,name):
        self.flow = flow
        self.name = name
        self.N = self.flow.N
        self.idx = 0
        self._reset()        
        cv2.namedWindow(self.name,cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.name, self.click_and_crop)
        # keep looping until the 'q' key is pressed
    def main(self):
        while True:
            # display the image and wait for a keypress
            cv2.imshow(self.name, self.image)
            key = cv2.waitKey(1) & 0xFF
        	# if the 'r' key is pressed, reset the cropping region
            flag = self.run_key(key)
            if flag.all() == self.key2mask(ord('q')).all():
                break
        cv2.destroyAllWindows()
        return self.flow
    def key2mask(self,key):
        ascii_ = ord('a')
        tmp = np.zeros(25,dtype=bool)
        tmp[key-ascii_] = True
        return tmp
    def run_key(self,key):
        if key == ord('r') and self.allow_key(key):
            self._reset()        
    	# if the 'c' key is pressed, break from the loop
        elif key == ord('c') and self.allow_key(key):
            self.state = self.state & (~self.key2mask(ord('q')) | ~self.key2mask(ord('a')) | ~self.key2mask(ord('s')) | ~self.key2mask(ord('c')))
            # if there are two reference points, then crop the region of interest
            # from teh image and display it
            if len(self.refPt) == 2:
            	roi = self.clone[self.refPt[0][1]:self.refPt[1][1], self.refPt[0][0]:self.refPt[1][0]]
            	cv2.imshow('ROI', roi)
            	cv2.waitKey(0)
            self._reset()
        elif key == ord('q') and self.allow_key(key): 
            return self.key2mask(key)
        elif key == ord('a') and self.allow_key(key):
            self.overflow_idx(-1)
            self.display(self.idx)
        elif key == ord('s') and self.allow_key(key):
            self.overflow_idx(+1)
            self.display(self.idx)
        elif key == ord('l') and self.allow_key(key):
            speed = 125
            pause = False
            while True:
                cv2.imshow(self.name, self.image)
                if not pause:
                    self.overflow_idx(1)
                    self.display(self.idx)
                key = cv2.waitKey(int(speed)) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('a'):
                    speed *= 1.1
                elif key == ord('s'):
                    speed *= 0.9
                elif key == ord('p'):
                    pause = not pause
                    
        return np.ones(25,dtype=bool)
    def overflow_idx(self,i):
        idx_ = self.idx + i
        if idx_ >= self.N:
            self.idx = 0
        elif idx_ < 0:
            self.idx = self.N-1
        else:
            self.idx += i
    def allow_key(self,key):
        k = self.key2mask(key)
        return k.all() == (k.all() and self.state.all()).all()
    def _reset_state(self):
        self.state = np.ones(25,dtype=bool) 
    def _reset(self):
        self.refPt = []
        self.cropping = False  
        self.display(self.idx)
        self._reset_state()
        cv2.namedWindow('ROI')
        cv2.destroyWindow('ROI')
    def display(self,idx):
        self.image = self.flow(idx).asimage()[...,::-1].copy()
        self.clone = self.flow(idx).asimage()[...,::-1].copy()
    def click_and_crop(self,event, x, y, flags, param):
    	# grab references to the global variables
    
    	# if the left mouse button was clicked, record the starting
    	# (x, y) coordinates and indicate that self.cropping is being
    	# performed
    	if event == cv2.EVENT_LBUTTONDOWN:
    		self.refPt = [(x, y)]
    		self.cropping = True
    	elif event == cv2.EVENT_MOUSEMOVE and self.cropping:
    		self.image = self.clone.copy()
    		cv2.rectangle(self.image, self.refPt[0], (x,y), (0, 255, 0), 2)  
    	# check to see if the left mouse button was released
    	elif event == cv2.EVENT_LBUTTONUP:
    		# record the ending (x, y) coordinates and indicate that
    		# the self.cropping operation is finished
    		self.refPt.append((x, y))
    		self.cropping = False
    
    		# draw a rectangle around the region of interest
    		cv2.rectangle(self.image, self.refPt[0], self.refPt[1], (0, 255, 0), 2)