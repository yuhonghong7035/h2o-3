# -*- encoding: utf-8 -*-
"""
H2O TargetEncoder.

:copyright: (c) 2016 H2O.ai
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from h2o.expr import ExprNode
from h2o.frame import H2OFrame
from h2o.utils.typechecks import (assert_is_type)

__all__ = ("TargetEncoder", )

class TargetEncoder(object):

    """
    Status: alpha version

    This is a main class that provides Python's API to the Java implementation of the target encoding.

    In general target encoding could be applied to three types of problems, namely:
    
         1) Binary classification (supported)
         2) Multi-class classification (not supported yet)
         3) Regression (not supported yet)

    Sample usage:

    >>> targetEncoder = TargetEncoder(x=e_columns, y=responseColumnName, blending=True, inflection_point=3, smoothing=1)
    >>> targetEncoder.fit(frame) 
    >>> encodedValid = targetEncoder.transform(frame=frame, holdout_type="kfold", seed=1234)
    >>> encodedTest = targetEncoder.transform(frame=testFrame, holdout_type="none", noise=0.0, seed=1234)
    """

    #-------------------------------------------------------------------------------------------------------------------
    # Construction
    #-------------------------------------------------------------------------------------------------------------------

    def __init__(self, x=None, y=None, fold_column='', blending_avg=True, inflection_point=3, smoothing=1):
        """
        Creates instance of the TargetEncoder class and setting parameters that will be used in both `train` and `transform` methods.

        :param List[str] x: List of categorical column names or indices that we want apply target encoding to.

        :param str y: the name or column index of the response variable in the data.
        :param str fold_column: the name or column index of the fold column in the data.
        :param boolean blending_avg: whether to perform blended average. Defaults to TRUE.
        :param double inflection_point: parameter for blending. Used to calculate `lambda`. Determines half of the minimal sample size
            for which we completely trust the estimate based on the sample in the particular level of categorical variable.
        :param double smoothing: parameter for blending. Used to calculate `lambda`. Controls the rate of transition between
            the particular level's posterior probability and the prior probability. For smoothing values approaching infinity it becomes a hard
            threshold between the posterior and the prior probability.

        """

        self._teColumns = x
        self._responseColumnName = y
        self._foldColumnName = fold_column
        self._blending = blending_avg
        self._inflectionPoint = inflection_point
        self._smoothing = smoothing


    def fit(self, frame = None):
        """
        Returns encoding map as an object that maps 'column_name' -> 'frame_with_encoding_map_for_this_column_name'

        :param frame frame: An H2OFrame object with which to create the target encoding map
        """
        self._encodingMap = ExprNode("target.encoder.fit", frame, self._teColumns, self._responseColumnName,
                                     self._foldColumnName)._eager_map_frame()

        return self._encodingMap

    def transform(self, frame=None, holdout_type=None, noise=-1, seed=-1):
        """
        Apply transformation to `te_columns` based on the encoding maps generated during `TargetEncoder.fit()` call.
        You must not pass encodings manually from `.fit()` method because they are being stored internally
        after `.fit()' had been called.

        :param frame frame: to which frame we are applying target encoding transformations.
        :param str holdout_type: Supported options:

                1) "kfold" - encodings for a fold are generated based on out-of-fold data.
                2) "loo" - leave one out. Current row's response value is subtracted from the pre-calculated per-level frequencies.
                3) "none" - we do not holdout anything. Using whole frame for training
                
        :param float noise: the amount of random noise added to the target encoding.  This helps prevent overfitting. Defaults to 0.01 * range of y.
        :param int seed: a random seed used to generate draws from the uniform distribution for random noise. Defaults to -1.
        """
        assert_is_type(holdout_type, "kfold", "loo", "none")

        # We need to make sure that frames are being sent in the same order
        assert self._encodingMap.map_keys['string'] == self._teColumns
        encodingMapKeys = self._encodingMap.map_keys['string']
        encodingMapFramesKeys = list(map(lambda x: x['key']['name'], self._encodingMap.frames))
        return H2OFrame._expr(expr=ExprNode("target.encoder.transform", encodingMapKeys, encodingMapFramesKeys, frame, self._teColumns, holdout_type,
                                            self._responseColumnName, self._foldColumnName,
                                            self._blending, self._inflectionPoint, self._smoothing,
                                            noise, seed))
