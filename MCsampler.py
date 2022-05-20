
import numpy as np
import torch


class Uniform:
    def __init__(self, dim, dist='uniform', support=torch.tensor([[-4, 4], [-4, 4]]), device=torch.device('cpu')):

        self.dist_name = dist
        self.support = support
        self.dim = dim
        self.device = device

        self.x_min = -4.
        self.x_max = 4.

        self.dist = torch.distributions.Uniform(self.x_min, self.x_max)


    def logpx_norm(self, x):
        if self.device == torch.device('cuda'):
            return torch.ones_like(x).mul_(-np.log(self.x_max - self.x_min)).sum(dim=1)
        else:
            return self.dist.log_prob(x).sum(dim=1)

    def logpx(self, x):
        if self.device == torch.device('cuda'):
            return torch.ones_like(x).mul_(-np.log(self.x_max - self.x_min)).sum(dim=1)
        else:
            #return torch.ones_like(x).mul_(-np.log(self.x_max - self.x_min)).sum(dim=1)
            return self.dist.log_prob(x).sum(dim=1)

    def sample(self, nsamples, beta=1.0):
        if self.device == torch.device('cuda'):
            samples = torch.cuda.FloatTensor(nsamples, self.dim).uniform_().mul_(self.x_max - self.x_min)
            samples.add_(self.x_min)
        else:
            samples = self.dist.sample([nsamples, self.dim])
        return samples

class ImportanceSampling:
    def __init__(self, p_distribution, dim, device=torch.device('cpu')):

        self.pdist = p_distribution
        self.qdist = Uniform(dim=dim, support=torch.tensor([[-5, 5], [-5, 5]]), device=device)
        self.dim = dim
        self.device = device

    def sample(self, nsamples, breweight=True, beta=1.0):
        if beta != 1.0:
            raise NotImplementedError('This case is not considered yet.')

        nsamples_internal = 10 * nsamples
        samples_q = self.qdist.sample(nsamples_internal)
        log_w = self.log_w(samples_q)

        if breweight:
            m = torch.distributions.Multinomial(nsamples, logits=log_w)
            msample = m.sample()
            index_q = torch.zeros(nsamples, device=self.device).long()
            count = 0
            for idx, amount in enumerate(msample):
                if amount > 0:
                    for j in range(int(amount.item())):
                        index_q[int(count)] = idx
                        count += 1
            return samples_q.index_select(0, index_q)
        else:
            return samples_q, log_w

    def mean(self, N=1000000):
        x, log_w = self.sample(N, breweight=False)
        m = (x*log_w.exp().unsqueeze(1)).sum(dim=0)
        return m

    def variance(self):
        raise NotImplementedError('Variance estimation is so far not implemented.')


    def log_w(self, x):
        log_q = self.qdist.logpx(x)
        log_p = self.pdist.logpx(x, beta=1.0, bsum=False)
        log_w = log_p - log_q
        log_w_norm = self.get_normalized_log_w(log_w)
        return log_w_norm

    def get_normalized_log_w(self, log_w):
        c = self.logsumexp(log_w)
        return log_w - c

    def logsumexp(self, log_x):
        """Perform the log-sum-exp of the weights."""
        max_exp = log_x.max()
        my_sum = torch.exp(log_x - max_exp).sum()
        return torch.log(my_sum) + max_exp

    def q_distribution_mog(self, N):
        '''
        This function is the proposing distribution q(x). For more effective sampling approach, this should
        be as close as possible to p(x) - the target distribution.
        :return:
        '''

        x_prop = torch.random.randn(N, self.dim)
        sigma = torch.ones(N)
        sigma[:int(N/2)] = 1.
        sigma[int(N/2):] = 0.7
        sigma = sigma.unsqueeze(1).repeat(1, 2)
        mean = torch.ones(N)
        mean[:int(N/2)] = -1.5
        mean[int(N/2):] = 1.5
        mean = mean.unsqueeze(1).repeat(1, 2)
        mean[:, 1] = 0.

        x_prop = x_prop * sigma + mean

        return x_prop

class MetropolisHastings:
    def __init__(self, distribution, dim, sigmaSq=None, xinit=None, bgpu=False):

        # device setting
        if bgpu:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.dist = distribution

        # dimension of the distribution
        self.dim = dim
        self.sigmasq = 1.

        # set sigma for proposal
        if sigmaSq == None:
            self.updatesigmasq(0.1)
        else:
            self.updatesigmasq(sigmaSq)

        # set initial
        if xinit == None:
            self.xinit = torch.rand(dim, device=self.device)
        else:
            self.xinit = xinit.to(self.device)

        self.nburnin = 100
        self.nskip = 10
        self.minaccept = 0.15
        self.maxaxxept = 0.5

    def updatesigmasq(self, _newsigsq):
        self.sigmasq = _newsigsq
        self.sigma = np.sqrt(self.sigmasq)
        # TODO
        #self._cov = _cov = np.eye(self.dim) * self.sigmaSq

    def updateXinit(self, xinit):
        self.xinit = xinit

    #def LogQtrans(self, x, meanXgiven):
    #    return stat.multivariate_normal.logpdf(x, mean=meanXgiven, cov=self._cov)

    def sample(self, nsamples, beta=1.0):
        bRestartSampling = True
        sampleStorage = torch.zeros([nsamples, self.dim], device=self.device)
        nsamplestot = self.nburnin + nsamples * self.nskip
        #
        while bRestartSampling:
            iaccept = 0
            icountsample = 0
            xn = self.xinit
            logpn = self.dist.logpx(x=xn, beta=beta)
            for i in range(0, nsamplestot):
                # calculate proposal
                xp = xn + self.sigma * torch.randn(self.dim, device=self.device)
                # calculate log likelihood of proposal
                logpp = self.dist.logpx(x=xp, beta=beta)
                # calculate acceptance ratio
                a = torch.exp(logpp - logpn)
                #
                # print a
                if a >= 1. or np.random.uniform() <= a:
                    iaccept = iaccept + 1
                    xn = xp
                    logpn = logpp
                #
                if ((i - self.nburnin) % self.nskip) == 0 and i >= self.nburnin:
                    # print i
                    # print icountsample
                    if icountsample < nsamples:
                        #print(xn)
                        #print(sampleStorage)
                        sampleStorage[icountsample, :] = xn
                    icountsample = icountsample + 1
            #
            totalacceptance = float(iaccept) / nsamplestot
            print('Total acceptance ratio:')
            print(totalacceptance)
            # this is only when wants to have an adaptive method
            if False:
                if totalacceptance < self.minaccept:
                    # decrease sigma
                    sigsqnew = 0.8 * self.sigmasq
                    self.updatesigmasq(sigsqnew)
                    bRestartSampling = True
                    print('SigmaSq has been decreased.')
                elif totalacceptance > self.maxaccept:
                    # increase sigma
                    sigSqNew = 1.2 * self.sigmasq
                    self.updatesigmasq(sigSqNew)
                    bRestartSampling = True
                    print('SigmaSq has been increased.')
                else:
                    bRestartSampling = False
                    print('Sampling successful.')
                print(bRestartSampling)
                print(self.sigmasq)
            else:
                bRestartSampling = False

        return sampleStorage
