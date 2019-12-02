#
# Copyright (C) 2019 Luca Pasqualini
# University of Siena - Artificial Intelligence Laboratory - SAILab
#
#
# USienaRL is licensed under a BSD 3-Clause.
#
# You should have received a copy of the license along with this
# work. If not, see <https://opensource.org/licenses/BSD-3-Clause>.

# Import packages

import numpy


def softmax(sequence):
    """
    Compute the softmax of a sequence of values, i.e. the resulting probability distribution over that sequence.

    :param sequence: the sequence of which to compute the softmax
    :return: the probability distribution over the sequence
    """
    # Make sure the sequence is a numpy array
    if not isinstance(sequence, numpy.ndarray):
        sequence = numpy.array(sequence)
    # Make sure the length of the shape of the given array is 2
    assert len(sequence.shape) == 2
    # Get the element with max value in the given array as the normalization factor
    normalization_factor = numpy.max(sequence, axis=1)
    # Increase the shape size of the normalization factor to allow broadcasting
    normalization_factor = normalization_factor[:, numpy.newaxis]
    # Apply the normalization
    numerator = numpy.exp(sequence - normalization_factor)
    # Compute the denominator by summing all the normalized numerator elements
    denominator = numpy.sum(numerator, axis=1)
    # Increase the shape size of the denominator to allow broadcasting
    denominator = denominator[:, numpy.newaxis]
    # Return the softmax result probability distribution
    return numerator / denominator
