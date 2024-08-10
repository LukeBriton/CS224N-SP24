#!/usr/bin/env python

import argparse
import numpy as np
import random

from utils.gradcheck import gradcheck_naive, grad_tests_softmax, grad_tests_negsamp
from utils.utils import normalizeRows, softmax


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    Arguments:
    x -- A scalar or numpy array.
    Return:
    s -- sigmoid(x)
    """

    ### YOUR CODE HERE (~1 Line)
    # Vectorization & Broadcasting
    s = 1/(1 + np.exp(-x))
    ### END YOUR CODE

    return s


def naiveSoftmaxLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset
):
    """ Naive Softmax loss & gradient function for word2vec models

    Implement the naive softmax loss and gradients between a center word's 
    embedding and an outside word's embedding. This will be the building block
    for our word2vec models. For those unfamiliar with numpy notation, note 
    that a numpy ndarray with a shape of (x, ) is a one-dimensional array, which
    you can effectively treat as a vector with length x.

    Arguments:
    centerWordVec -- numpy ndarray, center word's embedding
                    in shape (word vector length, )
                    (v_c in the pdf handout)
    outsideWordIdx -- integer, the index of the outside word
                    (o of u_o in the pdf handout)
    outsideVectors -- outside vectors is
                    in shape (num words in vocab, word vector length) 
                    for all words in vocab (tranpose of U in the pdf handout)
    dataset -- needed for negative sampling, unused here.

    Return:
    loss -- naive softmax loss
    gradCenterVec -- the gradient with respect to the center word vector
                     in shape (word vector length, )
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length) 
                    (dJ / dU)
    """

    ### YOUR CODE HERE (~6-8 Lines)
    # F**K numpy!
    # numpy doesn't differentiate column and row vectors by default... 
    # https://stackoverflow.com/questions/17428621/python-differentiating-between-row-and-column-vectors
    # row vector: shape(1, -1)
    # column vector: shape(-1, 1)
    v_c = centerWordVec.reshape(-1, 1) # v_c, column

    # Compute the softmax function for each row of the U^T.
    # softmaxed[w] = softmax(u_w^T * v_c)
    # Dammit, it softmaxes along each row!!!
    hat_y = softmax((outsideVectors @ v_c).reshape(-1)).reshape(-1, 1) # \hat{y}, column

    # loss = -log\hat{y}_o = -log softmax(u_o^T v_c)
    loss = -np.log(hat_y[outsideWordIdx][0]) # x or [x] ? -> x
    
    # dJ / dv_c = -u_o^T + \hat{y}^TU^T
    # Almost drop the minus sign here!!!
    gradCenterVec = -outsideVectors[outsideWordIdx].reshape(1, -1) + (np.transpose(hat_y) @ outsideVectors)
    gradCenterVec = gradCenterVec.reshape(-1) # We need to return a 1d-array.
    # print(gradCenterVec)
    # dJ/du_i = v_c^T\hat{y}_i
    # dJ/du_o = -v_c^T + v_c^T\hat{y}_o
    gradOutsideVecs = np.empty((v_c.shape[0], 0))
    for hat_y_i in hat_y.reshape(-1):
        gradOutsideVecs = np.c_[gradOutsideVecs, v_c * hat_y_i]
    gradOutsideVecs = np.transpose(gradOutsideVecs)
    gradOutsideVecs[outsideWordIdx] -= centerWordVec
    # print(gradOutsideVecs)

    ### Please use the provided softmax function (imported earlier in this file)
    ### This numerically stable implementation helps you avoid issues pertaining
    ### to integer overflow. 

    ### END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs


def getNegativeSamples(outsideWordIdx, dataset, K):
    """ Samples K indexes which are not the outsideWordIdx """

    negSampleWordIndices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == outsideWordIdx:
            newidx = dataset.sampleTokenIdx()
        negSampleWordIndices[k] = newidx
    return negSampleWordIndices


def negSamplingLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset,
    K=10
):
    """ Negative sampling loss function for word2vec models

    Implement the negative sampling loss and gradients for a centerWordVec
    and a outsideWordIdx word vector as a building block for word2vec
    models. K is the number of negative samples to take.

    Note: The same word may be negatively sampled multiple times. For
    example if an outside word is sampled twice, you shall have to
    double count the gradient with respect to this word. Thrice if
    it was sampled three times, and so forth.

    Arguments/Return Specifications: same as naiveSoftmaxLossAndGradient
    """

    # Negative sampling of words is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    negSampleWordIndices = getNegativeSamples(outsideWordIdx, dataset, K)
    indices = [outsideWordIdx] + negSampleWordIndices

    ### YOUR CODE HERE (~10 Lines)
    # 1. At first I haven't used indices etc. above, and passed "Gradient check for negSamplingLossAndGradient".
    # 2. Later I changed to use samples, which should output the correct answer.
    # 3. However it seems the "gradOutsideVecs" to return needs to give all words in "outsideVectors" a value,
    # -> even if the unsampled ones should just be zero. (This explains why 1. could pass the Gradient check.)

    v_c = centerWordVec.reshape(-1, 1) # v_c, column
    u_o_T = outsideVectors[outsideWordIdx].reshape(1, -1)
    # https://stackoverflow.com/questions/47540800/how-to-select-all-elements-in-a-numpy-array-except-for-a-sequence-of-indices
    # U_neg = np.delete(outsideVectors, outsideWordIdx, 0) # without u_o
    ## Misused the whole vocabulary for negative sampling before... 
    U_neg = np.take(outsideVectors, negSampleWordIndices, 0)
    
    # sigmoided = sigmoid(- outsideVectors @ v_c) # \sigma(-U v_c), with o, 1d
    sigmoided_o = sigmoid(u_o_T @ v_c)[0][0] # Be aware of sign!!!
    ## sigmoided_sampled = np.take(sigmoided_neg.reshape(-1), negSampleWordIndices)
    sigmoided_neg = sigmoid(- U_neg @ v_c) # \sigma(-U_neg v_c), without o, 1d

    # loss=-\log(\sigma(u_o^T v_c))-\sum_{s=1}^K\log(\sigma(-u_{w_s}^T v_c))
    # loss = -np.log(sigmoided_o) - np.sum(np.log(sigmoided_neg.reshape(-1)))
    # loss = -np.log(1 - sigmoided[outsideWordIdx][0]) - np.sum(np.log(sigmoided_neg.reshape(-1)))
    loss = -np.log(sigmoided_o) - np.sum(np.log(sigmoided_neg))
    
    # dJ / dv_c = -u_o^T\sigma(-u_o^T v_c) + \sum u_{w_s}^T \sigma{u_{w_s}^T v_c)}
    # https://stackoverflow.com/questions/40034993/how-to-get-element-wise-matrix-multiplication-hadamard-product-in-numpy
    # The default, axis=None, will sum all of the elements of the input array.
    # print(np.sum((U_neg * (1 - sigmoided_neg)))) -> Sums all to a single float
    # print(np.sum((U_neg * (1 - sigmoided_neg)), 0)) -> The vector we want
    gradCenterVec = - (u_o_T * (1 - sigmoided_o)) + np.sum((U_neg * (1 - sigmoided_neg)), 0)
    gradCenterVec = gradCenterVec.reshape(-1) # We need to return a 1d-array.

    # dJ/du_i = v_c^T\sigma(u_i^T v_c) 
    # dJ/du_o = v_c^T\sigma(u_o^T v_c) - v_c^T
    # sigmoided_all = (1 - np.insert(sigmoided_neg, outsideWordIdx, 1 - sigmoided_o)) # 1d
    sigmoided_all = (1 - np.insert(sigmoided_neg, 0, 1 - sigmoided_o)) # 1d

    gradOutsideVecs = np.zeros(outsideVectors.shape)
    for i, sig in enumerate(sigmoided_all):
        idx = indices[i]
        gradOutsideVecs[idx] = (np.count_nonzero(sigmoided_all == sig) * v_c * sig).reshape(-1) # (3, 1) 2d -> (3, 1) 1d
    gradOutsideVecs[outsideWordIdx] -= centerWordVec
    
    '''
    # The plausible answer corresponding to the written assignment, in terms of U_{w_1, \dots, o, \dots, w_K}^T
    # Maybe I should have put o as the first term... never mind because its shape should also be "wrong".
    gradOutsideVecs = np.empty((v_c.shape[0], 0))
    for sig in sigmoided_all:
        # https://stackoverflow.com/questions/28663856/how-do-i-count-the-occurrence-of-a-certain-item-in-an-ndarray
        gradOutsideVecs = np.c_[gradOutsideVecs, np.count_nonzero(sigmoided_all == sig) * v_c * sig]
    gradOutsideVecs = np.transpose(gradOutsideVecs)
    # https://stackoverflow.com/questions/1903462/how-can-i-zip-sort-parallel-numpy-arrays
    gradOutsideVecs = gradOutsideVecs[np.array(indices).argsort()]
    # https://stackoverflow.com/questions/15637336/numpy-unique-with-order-preserved
    _, idx = np.unique(gradOutsideVecs, return_index=True, axis = 0)
    gradOutsideVecs = gradOutsideVecs[np.sort(idx)]
    gradOutsideVecs[outsideWordIdx] -= centerWordVec # This should be excecuted after being deduplicated.
    exit()
    '''
    ### Please use your implementation of sigmoid in here.

    ### END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs


def skipgram(currentCenterWord, windowSize, outsideWords, word2Ind,
             centerWordVectors, outsideVectors, dataset,
             word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currentCenterWord -- a string of the current center word
    windowSize -- integer, context window size
    outsideWords -- list of no more than 2*windowSize strings, the outside words
    word2Ind -- a dictionary that maps words to their indices in
              the word vector list
    centerWordVectors -- center word vectors (as rows) is in shape 
                        (num words in vocab, word vector length) 
                        for all words in vocab (V in pdf handout)
    outsideVectors -- outside vectors is in shape 
                        (num words in vocab, word vector length) 
                        for all words in vocab (transpose of U in the pdf handout)
    word2vecLossAndGradient -- the loss and gradient function for
                               a prediction vector given the outsideWordIdx
                               word vectors, could be one of the two
                               loss functions you implemented above.

    Return:
    loss -- the loss function value for the skip-gram model
            (J in the pdf handout)
    gradCenterVecs -- the gradient with respect to the center word vector
                     in shape (num words in vocab, word vector length)
                     (dJ / dv_c in the pdf handout)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length) 
                    (dJ / dU)
    """

    loss = 0.0
    gradCenterVecs = np.zeros(centerWordVectors.shape)
    gradOutsideVectors = np.zeros(outsideVectors.shape)

    ### YOUR CODE HERE (~8 Lines)
    ## At first I thought it was to 'append' each vector to the vectors 
    ## gradCenterVecs, gradOutsideVectors initially already have zeros!!!
    ## gradCenterVecs = np.empty((0, gradCenterVecs.shape[1]))
    ## gradOutsideVectors = np.empty((0, gradOutsideVectors.shape[1]))
    centerWordIdx = word2Ind[currentCenterWord]
    centerWordVec = centerWordVectors[centerWordIdx]
    for outsideWord in outsideWords:
        outsideWordIdx = word2Ind[outsideWord]
        _loss, gradCenterVec, gradOutsideVector = word2vecLossAndGradient(centerWordVec, outsideWordIdx, outsideVectors, dataset)
        loss += _loss
        gradCenterVecs[centerWordIdx] += gradCenterVec
        gradOutsideVectors += gradOutsideVector
        ## np.append() will turn 2d-array into 1d!!!
        ## gradCenterVecs = np.append(gradCenterVecs, gradCenterVec)
        ## gradCenterVecs = np.r_[gradCenterVecs, gradCenterVec.reshape(1, -1)]
        ## gradOutsideVectors = np.r_[gradOutsideVectors, gradOutsideVector]
    ### END YOUR CODE
    
    return loss, gradCenterVecs, gradOutsideVectors


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, word2Ind, wordVectors, dataset,
                         windowSize,
                         word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    batchsize = 50
    loss = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    centerWordVectors = wordVectors[:int(N/2),:]
    outsideVectors = wordVectors[int(N/2):,:]
    for i in range(batchsize):
        windowSize1 = random.randint(1, windowSize)
        centerWord, context = dataset.getRandomContext(windowSize1)

        c, gin, gout = word2vecModel(
            centerWord, windowSize1, context, word2Ind, centerWordVectors,
            outsideVectors, dataset, word2vecLossAndGradient
        )
        loss += c / batchsize
        grad[:int(N/2), :] += gin / batchsize
        grad[int(N/2):, :] += gout / batchsize

    return loss, grad

def test_sigmoid():
    """ Test sigmoid function """
    print("=== Sanity check for sigmoid ===")
    assert sigmoid(0) == 0.5
    assert np.allclose(sigmoid(np.array([0])), np.array([0.5]))
    assert np.allclose(sigmoid(np.array([1,2,3])), np.array([0.73105858, 0.88079708, 0.95257413]))
    print("Tests for sigmoid passed!")

def getDummyObjects():
    """ Helper method for naiveSoftmaxLossAndGradient and negSamplingLossAndGradient tests """

    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in range(2*C)]

    dataset = type('dummy', (), {})()
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])

    return dataset, dummy_vectors, dummy_tokens

def test_naiveSoftmaxLossAndGradient():
    """ Test naiveSoftmaxLossAndGradient """
    dataset, dummy_vectors, dummy_tokens = getDummyObjects()

    print("==== Gradient check for naiveSoftmaxLossAndGradient ====")
    def temp(vec):
        loss, gradCenterVec, gradOutsideVecs = naiveSoftmaxLossAndGradient(vec, 1, dummy_vectors, dataset)
        return loss, gradCenterVec
    gradcheck_naive(temp, np.random.randn(3), "naiveSoftmaxLossAndGradient gradCenterVec")

    centerVec = np.random.randn(3)
    def temp(vec):
        loss, gradCenterVec, gradOutsideVecs = naiveSoftmaxLossAndGradient(centerVec, 1, vec, dataset)
        return loss, gradOutsideVecs
    gradcheck_naive(temp, dummy_vectors, "naiveSoftmaxLossAndGradient gradOutsideVecs")

def test_negSamplingLossAndGradient():
    """ Test negSamplingLossAndGradient """
    dataset, dummy_vectors, dummy_tokens = getDummyObjects()

    print("==== Gradient check for negSamplingLossAndGradient ====")
    def temp(vec):
        loss, gradCenterVec, gradOutsideVecs = negSamplingLossAndGradient(vec, 1, dummy_vectors, dataset)
        return loss, gradCenterVec
    gradcheck_naive(temp, np.random.randn(3), "negSamplingLossAndGradient gradCenterVec")

    centerVec = np.random.randn(3)
    def temp(vec):
        loss, gradCenterVec, gradOutsideVecs = negSamplingLossAndGradient(centerVec, 1, vec, dataset)
        return loss, gradOutsideVecs
    gradcheck_naive(temp, dummy_vectors, "negSamplingLossAndGradient gradOutsideVecs")

def test_skipgram():
    """ Test skip-gram with naiveSoftmaxLossAndGradient """
    dataset, dummy_vectors, dummy_tokens = getDummyObjects()

    print("==== Gradient check for skip-gram with naiveSoftmaxLossAndGradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, naiveSoftmaxLossAndGradient),
        dummy_vectors, "naiveSoftmaxLossAndGradient Gradient")
    grad_tests_softmax(skipgram, dummy_tokens, dummy_vectors, dataset)

    print("==== Gradient check for skip-gram with negSamplingLossAndGradient ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingLossAndGradient),
        dummy_vectors, "negSamplingLossAndGradient Gradient")
    grad_tests_negsamp(skipgram, dummy_tokens, dummy_vectors, dataset, negSamplingLossAndGradient)

def test_word2vec():
    """ Test the two word2vec implementations, before running on Stanford Sentiment Treebank """
    test_sigmoid()
    test_naiveSoftmaxLossAndGradient()
    test_negSamplingLossAndGradient()
    test_skipgram()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test your implementations.')
    parser.add_argument('function', nargs='?', type=str, default='all',
                        help='Name of the function you would like to test.')

    args = parser.parse_args()
    if args.function == 'sigmoid':
        test_sigmoid()
    elif args.function == 'naiveSoftmaxLossAndGradient':
        test_naiveSoftmaxLossAndGradient()
    elif args.function == 'negSamplingLossAndGradient':
        test_negSamplingLossAndGradient()
    elif args.function == 'skipgram':
        test_skipgram()
    elif args.function == 'all':
        test_word2vec()
