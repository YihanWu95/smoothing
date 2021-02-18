import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint


class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, sigma: float):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> (int, float):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.

        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch, device='cuda') * self.sigma
                predictions = self.base_classifier(batch + noise).argmax(1)
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]

class Gaussian(Smooth):
    
    def __init__(self, base_classifier: torch.nn.Module, num_classes: int,sigma,L):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        :param L: interval [0,L] for searching r_x
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma=sigma
        self.L=L
        
    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            counts = np.zeros((100,self.num_classes), dtype=int)
            for i in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size
                batch = x.repeat((this_batch_size, 1, 1, 1))
                if i == 0:
                    noise,dis = gaussian_sampling(x,this_batch_size,self.sigma)
                    predictions = self.base_classifier(batch + noise).argmax(1)
                    distance_array = dis.cpu().numpy()
                    prediction_array = predictions.cpu().numpy()
                    pred = self._count_arr(predictions.cpu().numpy(), self.num_classes)
                else:
                    noise,dis = gaussian_sampling(x,this_batch_size,self.sigma)
                    predictions = self.base_classifier(batch + noise).argmax(1)
                    distance_array = np.concatenate((distance_array,dis.cpu().numpy()),axis=None)
                    prediction_array = np.concatenate((prediction_array,predictions.cpu().numpy()),axis=None)
                    pred += self._count_arr(predictions.cpu().numpy(), self.num_classes)
                counts += self._count_radius(predictions.cpu().numpy(),dis.cpu().numpy(), self.num_classes)
            return distance_array, prediction_array, counts, pred

    def _count_radius(self, arr, dis, length: int,) -> np.ndarray:
        """Calculating the count vector mentioned in our paper
        :param (arr, dis): predictions with respect to perturbed samples
        :return: a count table
        """
        counts = np.zeros((100,length), dtype=int)
        space = np.linspace(0,self.L,100)
        for i,idx in enumerate(arr):
            for j,r in enumerate(space):
#                 print(r,dis[i])
                if dis[i]>r:
                    counts[j,idx] += 1
        return counts 
    
    def radius_x(self, counts, target) -> np.ndarray:
        """Calculating R_x
        :param counts: a count table
        :param target: ground truth label
        :return: (R_x, g_1) defined in our algorithm
        """
        
        space = np.linspace(0,self.L,100)
        g1=0
        for i in range(counts.shape[0]):
            s=sum(counts[i,:])
            # flag is the value of g_2 in our algorithm
            flag=proportion_confint(counts[i,target], s, alpha=2 * 0.001/200, method="beta")[1]
            if flag<0.5:
                all_1=counts[0,:]-counts[i,:]
                if i>0:
                    g1=proportion_confint(all_1[target], sum(all_1), alpha=2 * 0.001/200, method="beta")[1]
                else:
                    g1=0
                return space[i],g1
            elif i==np.shape(counts)[0]-1:
                all_1=counts[0,:]
                g1=proportion_confint(all_1[target], sum(all_1), alpha=2 * 0.001/200, method="beta")[1]
                return self.L,g1
            else:
                continue
        
        
    
    
    
    def certify(self, x: torch.tensor,y: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> (int, float):
            """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
            With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
            robust within a L2 ball of radius R around x.

            :param x: the input [channel x height x width]
            :param n0: the number of Monte Carlo samples to use for selection
            :param n: the number of Monte Carlo samples to use for estimation
            :param alpha: the failure probability
            :param batch_size: batch size to use when evaluating the base classifier
            :return: (R_x, g_1, certified radius)
                     in the case of abstention, the class will be ABSTAIN and the radius 0.
            """
            self.base_classifier.eval()
            
            _, _, counts,pred = self._sample_noise(x, n, batch_size)
#             print(counts.shape)
            R_x,g1 = self.radius_x(counts, y)
            nA = pred[y].item()
            pABar = self._lower_confidence_bound(nA, n, alpha)
            if pABar < 0.5:
                radius=0
            else:
                radius = self.sigma * norm.ppf(pABar)
            return  R_x ,g1 ,radius


    


def gaussian_sampling(x: torch.tensor, n: int, sigma: float):
    """ Sample noise from C * e^{||x||_2^2/2sigma^2}/||x||_2^{d-1}.
        
        :param x: the input [channel x width x height]
        :param n: number of sample
        :param sigma: standard deviation
        :param d: shape of a sample
        :return: an ndarray[int] (n,d[0],d[1],d[2],...)
        """
    batch = x.repeat((n, 1, 1, 1))
    shape = batch.shape
    noise = torch.randn_like(batch, device='cuda')*sigma
    noise = noise.reshape([n,-1])
    r = noise.norm(2, 1, keepdim=False)
    output = noise.reshape(shape)
    
    return output, r    