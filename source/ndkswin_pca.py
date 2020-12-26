#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 13:07:20 2020

@author: maurrastogbe
"""

import numpy as np
from scipy import stats
import random
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
import pandas as pd


class NDKSWIN(BaseDriftDetector):
    r""" Kolmogorov-Smirnov Windowing method for concept drift detection.

    Parameters
    ----------
    alpha: float (default=0.005)
        Probability for the test statistic of the Kolmogorov-Smirnov-Test
        The alpha parameter is very sensitive, therefore should be set
        below 0.01.

    window_size: float (default=100)
        Size of the sliding window

    stat_size: float (default=30)
        Size of the statistic window

    ---data: numpy.ndarray of shape (n_samples, 1) (default=None,optional)
        Already collected data to avoid cold start.---
    data: numpy.ndarray of shape (n_samples, n_attributes) (default=None,optional)
        Already collected data to avoid cold start.

    n_dimensions = the number of random dimensions to consider when computing 
        stats and detecting the drift
    
    Notes
    -----
    KSWIN (Kolmogorov-Smirnov Windowing) [1]_ is a concept change detection method based
    on the Kolmogorov-Smirnov (KS) statistical test. KS-test is a statistical test with
    no assumption of underlying data distribution. KSWIN can monitor data or performance
    distributions. ----Note that the detector accepts one dimensional input as array.---
    Note that this version is free of dimensional number input as array

    KSWIN maintains a sliding window :math:`\Psi` of fixed size :math:`n` (window_size). The
    last :math:`r` (stat_size) samples of :math:`\Psi` are assumed to represent the last
    concept considered as :math:`R`. From the first :math:`n-r` samples of :math:`\Psi`,
    :math:`r` samples are uniformly drawn, representing an approximated last concept :math:`W`.

    The KS-test is performed on the windows :math:`R` and :math:`W` of the same size. KS
    -test compares the distance of the empirical cumulative data distribution :math:`dist(R,W)`.

    A concept drift is detected by KSWIN if:

    * :math:`dist(R,W) > \sqrt{-\frac{ln\alpha}{r}}`

    -> The difference in empirical data distributions between the windows :math:`R` and :math:`W`
    is too large as that R and W come from the same distribution.

    References
    ----------
    .. [1] Christoph Raab, Moritz Heusinger, Frank-Michael Schleif, Reactive
       Soft Prototype Computing for Concept Drift Streams, Neurocomputing, 2020,

    Examples
    --------
    >>> # Imports
    >>> import numpy as np
    >>> from skmultiflow.data.sea_generator import SEAGenerator
    >>> from skmultiflow.drift_detection import KSWIN
    >>> import numpy as np
    >>> # Initialize KSWIN and a data stream
    >>> kswin = KSWIN(alpha=0.01)
    >>> stream = SEAGenerator(classification_function = 2,
    >>>     random_state = 112, balance_classes = False,noise_percentage = 0.28)
    >>> # Store detections
    >>> detections = []
    >>> # Process stream via KSWIN and print detections
    >>> for i in range(1000):
    >>>         data = stream.next_sample(10)
    >>>         batch = data[0][0][0]
    >>>         kswin.add_element(batch)
    >>>         if kswin.detected_change():
    >>>             print("\rIteration {}".format(i))
    >>>             print("\r KSWINReject Null Hyptheses")
    >>>             detections.append(i)
    >>> print("Number of detections: "+str(len(detections)))
    """

    def __init__(self, alpha=0.005, window_size=100, stat_size=30, data=None, 
                 n_dimensions:int=2, n_tested_samples=0.01,
                 fixed_checked_dimension=False, fixed_checked_sample=False):
        super().__init__()
        self.window_size = window_size
        self.n_tested_samples = n_tested_samples
        self.fixed_checked_dimension = fixed_checked_dimension
        self.fixed_checked_sample = fixed_checked_sample
        self.n_dimensions = n_dimensions
        self.stat_size = stat_size
        self.alpha = alpha
        self.change_detected = False
        self.p_value = 0
        self.n = 0
        if self.alpha < 0 or self.alpha > 1:
            raise ValueError("Alpha must be between 0 and 1")

        if self.window_size < 0:
            raise ValueError("window_size must be greater than 0")

        if self.window_size < self.stat_size:
            raise ValueError("stat_size must be smaller than window_size")
        
        if self.n_dimensions <= 0 or (data is not None and self.n_dimensions > data.shape[1]):
            print("Warning: n_dimensions must be between 1 and <= input_value.shape[1]. We will consider all dimensions to compute the drift detection.")
            #raise ValueError("n_dimensions must be between 1 and <= data.shape[1]")
            self.n_dimensions = data.shape[1]
        #else:
        #    self.n_dimensions = n_dimensions

        if self.n_tested_samples <= 0.0 or self.n_tested_samples > 1.0 :
            raise ValueError("n_tested_samples must be between > 0 and <= 1")
        else:
            self.n_samples_to_test = int(self.window_size*self.n_tested_samples)

        #if not isinstance(data, np.ndarray) or data is None:
            #self.window = np.array([])
        #else:
        #    self.window = data
        
        from sklearn.decomposition import PCA
        pca = PCA(n_components=self.n_dimensions)
        pca_input_value = pca.fit_transform(data)
        self.window = pca_input_value
        
    def add_element(self, input_value):
        """ Add element to sliding window

        Adds an element on top of the sliding window and removes
        the oldest one from the window. Afterwards, the KS-test
        is performed.

        Parameters
        ----------
        input_value: ndarray
            New data sample the sliding window should add.
        """
        
        #print("input_value = ")
        #print(input_value)
        self.change_detected = False
            
        if self.fixed_checked_dimension:
            sample_dimensions=list(range(self.n_dimensions))
        else:
            if self.n_dimensions > input_value.shape[1]:
                print("n_dimensions must be between 1 and <= input_value.shape[1]. We will consider the first dimension only to compute the drift detection.")
                sample_dimensions = [0]
            else:
                #sample_dimensions = random.sample(list(range(input_value.shape[1])), self.n_dimensions)
                from sklearn.decomposition import PCA
                pca = PCA(n_components=self.n_dimensions)
                pca_input_value = pca.fit_transform(input_value)
                sample_dimensions=list(range(pca_input_value.shape[1]))
        
        
        
        print("sample_dimensions")
        print(sample_dimensions)
        
        if self.fixed_checked_sample:
            sample_test_data = pca_input_value[list(range(self.n_samples_to_test))]
        else:
            if self.n_samples_to_test > pca_input_value.shape[0]:
                #print("self.n_samples_to_test = "+str(self.n_samples_to_test))
                #print("input_value.shape[0] = "+str(input_value.shape[0]))
                print("Not enough data in input_value to pick "+str(self.n_samples_to_test)+" We will use 100% of input_value.")
                sample_test_data = pca_input_value
            else:
                sample_test_data = pca_input_value[random.sample(list(range(pca_input_value.shape[0])), self.n_samples_to_test)]
        
        print("sample_test_data")
        print(sample_test_data)
        for value in sample_test_data:
            #print("self.change_detected = "+str(self.change_detected))
            if self.change_detected == False:
                self.n += 1
                currentLength = self.window.shape[0]
                if currentLength >= self.window_size:
                    #print(type(self.window))
                    #print(type(input_value))
                    #print("self.window 1 = ")
                    #print(self.window)
                    #self.window = np.delete(self.window, 0)
                    self.window = np.delete(self.window, 0,0)
                    #print("self.window = np.delete(self.window, 0) = ")
                    #print(self.window)
                    
                    for i in sample_dimensions:
                        #rnd_window = np.random.choice(self.window[:,i][:-self.stat_size], self.stat_size)
                        rnd_window = np.random.choice(np.array(pd.DataFrame(self.window)[i])[:-self.stat_size], self.stat_size)
                        
                        #print("rnd_window = ")
                        #print(rnd_window)
                        #print("np.array(pd.DataFrame(self.window)[i])[:-self.stat_size] = ")
                        #print(np.array(pd.DataFrame(self.window)[i])[:-self.stat_size])
                        #print("np.array(pd.DataFrame(self.window)[i]) = ")
                        #print(np.array(pd.DataFrame(self.window)[i]))
                        #print("np.array(pd.DataFrame(self.window)[i])[-self.stat_size:] = ")
                        #print(np.array(pd.DataFrame(self.window)[i])[-self.stat_size:])
                        
                        #(st, self.p_value) = stats.ks_2samp(rnd_window,
                         #                                   self.window[:,i][-self.stat_size:], mode="exact")
                        (st, self.p_value) = stats.ks_2samp(rnd_window,
                                                            np.array(pd.DataFrame(self.window)[i])[-self.stat_size:], mode="exact")
                        #print("self.p_value = ")
                        #print(self.p_value)
                        #print("st = ")
                        #print(st)
            
                        if self.p_value <= self.alpha and st > 0.1:
                            self.change_detected = True
                            self.window = self.window[-self.stat_size:]
                            #print("Change_detected in dimension "+str(i)+" on data "+str(value))
                            break
                        else:
                            self.change_detected = False
                            #print("self.change_detected = False")
                else:  # Not enough samples in sliding window for a valid test
                    #raise ValueError("Not enough samples in sliding window for a valid test")
                    #print("Not enough samples in sliding window for a valid test")
                    self.change_detected = False
        
                self.window = np.concatenate([self.window, [value]])
                #print(self.window)
            else:
                #print("break execution")
                break

    def detected_change(self):
        """ Get detected change

        Returns
        -------
        bool
            Whether or not a drift occurred

        """
        return self.change_detected

    def reset(self):
        """ reset

        Resets the change detector parameters.
        """
        self.p_value = 0
        self.window = np.array([])
        self.change_detected = False
