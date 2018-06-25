<script src="https://storage.googleapis.com/quickdraw-models/sketchRNN/world_models/demo/lib/template.v1.js"></script>
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>


*Abstract*. 

We propose a novel method to protect privacy for Bayesian inference.  We first transform the log-likelihood function into a bounded function using a sanitizer function. Then, we use mean field variational inference to approximate the posterior distribution. Next, we use the exponential mechanism with Kullback–Leibler divergence as the utility function to define a probability distribution regarding all mean-field approximations. 
Finally, we use stochastic gradient Langevin dynamics to draw a random mean field approximation from the probability distribution defined by the exponential mechanism. The random mean field approximation guarantees differential privacy for individuals in the dataset.



## Introduction



The objective of Bayesian inference (previously known as inverse probability) is to compute the posterior distribution of latent parameters given the observed data. Let $$\theta$$ be the parameter vector of the model and $$x$$ be the observed data. By Bayes' theorem, we have:

$$
P(\theta \mid x) = \frac{P(x \mid \theta) \times P(\theta)}{P(x)}
$$

Here $$P(\theta \mid x)$$ is the posterior distribution of $$\theta$$, $$P(\theta)$$ is the prior distribution, $$P(x \mid \theta)$$ is the likelihood function and $$P(x)$$ is  the evidence. $$P(x)$$ can be computed as follows:

$$
P(x) =  \int_\theta P(x \mid \theta) P(\theta) d\theta
$$

However,  this integral is untractable in general. There are two general approaches to tackle this problem: (1) monte carlo markov chain (MCMC) and (2) variational inference (VI).  MCMC algorithm constructs a markov chain whose statationary distribution is $$P(\theta \mid x)$$. Meanwhile, VI approximates $$P(\theta \mid x)$$ by a parametrized  and tractable distribution $$Q(\theta \mid z)$$ such that $$Q(\theta \mid z)$$ is as close as possible to $$P(\theta \mid x)$$.

Here, we aim to protect the privacy of individuals in the dataset, which is essentially the observed data $$x$$. Especially, we aim to guarantee differential privacy for Bayesian inference.  Differential privacy is a robust and mathematical definition of privacy protection which guarantees that each individual's data does not much influent on the useful information which is extracted from the dataset of these individuals. Let $
$ be the information which is extracted from the dataset $$x$$.  Differential privacy constructs a probability distribution $$P(z \mid x)$$  of $$z$$ given $$x$$. The information, which will be published, is a random sample of this distribution:

$$
z_{priv} \sim P(z \mid x)
$$

Differential privacy guarantees that for any two neighbouring datasets$$x$$ and $$x^\prime$$, which differ at only one data record, $$P(z\mid x)$$ and $$P(z \mid x^\prime)$$ are close to each other. Formally, for any neighbouring datasets $$x$$ and $$x^\prime$$, and for any $$z$$:

$$
P(z \mid x) \leq \exp(\epsilon) \cdot P(z \mid x^\prime)
$$

Here, $$\epsilon$$ is called the privacy budget. Low $$\epsilon$$ guarantees strong privacy, otherwise, high $$\epsilon$$ guarantees weak privacy.

In this work, $$z$$,  which is extracted from dataset $$x$$, is actually the parameter vector $$z$$ of the mean field variational distribution $$Q(\theta \mid z)$$. In other words, we aim to guarantee that publishing parameters of the approximated posterior distribution will not compromise the privacy of individuals in dataset $$x$$.

There are previous works related to this problem. Wang *et al.* <dt-cite key="wang2015privacy"></dt-cite> are the first to prove that sampling a single random sample from the posterior distribution guarantees differential privacy under some conditions. The authors use stochastic gradient Monte-Carlo to sample the posterior distribution. Park *et al.* <dt-cite key="park2016variational"></dt-cite> proposed that for variational inference with conjugate exponential (CE) family, it is enough for guaranteeing differential privacy by perturbing the expected sufficient statistics of the complete-data likelihood. Jalko *et al.* <dt-cite key="jalko2017differentially"></dt-cite>  proposed to guarantee differential privacy for variational inference with non-conjugate models by a Gaussian mechanism which adds Gaussian noise to the gradient vector at each step of the optimization procedure.   


<script type="text/front-matter">
  title: "Differential Privacy for Bayesian Inference"
  description: "Using stochastic gradient Langevin dynamics to guarantee differential privacy for variational inference"
  authors:
  - Chris Olah: Thông T. Nguyễn
  affiliations:
  - Nanyang Technological University: http://www.ntu.edu.sg
</script>

<script type="text/bibliography"> 
@inproceedings{wang2015privacy,
  title={Privacy for free: Posterior sampling and stochastic gradient monte carlo},
  author={Wang, Yu-Xiang and Fienberg, Stephen and Smola, Alex},
  booktitle={International Conference on Machine Learning},
  pages={2493--2502},
  year={2015}
}

@article{park2016variational,
  title={Variational Bayes in private settings (VIPS)},
  author={Park, Mijung and Foulds, James and Chaudhuri, Kamalika and Welling, Max},
  journal={arXiv preprint arXiv:1611.00340},
  year={2016}
}

@inproceedings{jalko2017differentially,
  title={Differentially Private Variational Inference for Non-conjugate Models},
  author={J{\"a}lk{\"o}, Joonas and Dikmen, Onur and Honkela, Antti and others},
  booktitle={Uncertainty in Artificial Intelligence 2017 Proceedings of the 33rd Conference, UAI 2017},
  year={2017},
  organization={The Association for Uncertainty in Artificial Intelligence}
}
</script>

## Method

The main idea of our method is to use the exponential mechanism to define a probability distribution of  $$z$$ with the utility function $$U(z; x)$$ based on the Kullback–Leibler divergence score. The probability density function of $$z$$ is defined as follows:

$$
f_Z(z) \propto \exp\left( \frac{U(z; x)}{2\Delta_U} \cdot \epsilon\right) 
$$

where $$\Delta_U$$ is the sensitivity of the utility function $$U$$.  For any $$z$$ and any pair of neighbouring datasets $$x$$ and $$x^\prime$$,

$$
\Delta_U = \max_{z, x, x^\prime} \left| U(z; x) - U(z; x^\prime) \right|
$$

It is proven that a random sample from the above distribution guarantees $$\epsilon$$-differential privacy. 

Here, we use the variational lowerbound as the utility function: 

$$
\begin{align}
U(z; x) &= \log\left( P(x) \right) - D_{KL} \left( Q(\theta \mid z) ~\|~ P(\theta \mid x) \right) \\
&= \mathrm{E}_Q \left[   \log\left(P(x \mid \theta)\right) + \log\left( P(\theta)\right)  - \log\left(Q(\theta \mid z)\right) \right]
\end{align}
$$

Then, the sensitivity of $$U$$ is

$$
\Delta_U = \max_{z, x, x^\prime}  \left| \mathrm{E}_Q \left[  \log\left(P(x \mid \theta\right) - \log\left(P(x^\prime \mid \theta) \right)\right] \right|
$$

Let $$\ell(\theta; x_i)$$ be the log-likelihood of function of $$i^{th}$$ data record. Then, 

$$ P(x \mid \theta) = \sum_{i=1}^n \ell(\theta; x_i)$$ 

where $$n$$ is the size of dataset $$x$$. Now, assuming that two neightbouring datasets $$x$$ and $$x^\prime$$ are differ at only $$x_n$$ and $$x^\prime_n$$,

$$
\Delta_U = \max_{z, x, x^\prime} \left | \mathrm{E}_Q \left[ \ell(\theta; x_n) - \ell(\theta; x_n^\prime)\right] \right|
$$

So, the objective here is to bound $$\Delta_U$$ by a finite number. We  achieve that by approximating $$\ell(\theta; x_i)$$ by a finite log-likelihood function  $$\gamma(\theta; x_i)$$ as follows:

$$
\gamma(\theta; x_i) = \tau \cdot \textrm{tanh}\left( \frac{\ell(\theta; x_i)}{\tau} \right)
$$ 

where $$[-\tau; \tau]$$ is the range of $$\gamma(\cdot)$$. The below figure shows the effect of the approximation to an identity function with $$\tau = 2.0$$:

<details>
  
<summary>Click to see code</summary>

<div markdown="1">
```python
#@title
# http://pytorch.org/
from os import path
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())

accelerator = 'cu80' if path.exists('/opt/bin/nvidia-smi') else 'cpu'

!pip install -q http://download.pytorch.org/whl/{accelerator}/torch-0.4.0-{platform}-linux_x86_64.whl torchvision
import torch
import tensorflow as tf

#@title
from pylab import rcParams
import matplotlib.pyplot as plt
rcParams['figure.figsize'] = 4,3
rcParams['font.size'] = 14
plt.style.use('ggplot')

plt.rcParams['font.serif'] = 'dejavuserif'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.titlesize'] = 16
```

    


    [31mtorch-0.4.0-cp36-cp36m-linux_x86_64.whl is not a supported wheel on this platform.[0m



```python
#@title Figure 1. Bounded log-likelihood function

from pylab import rcParams
import matplotlib.pyplot as plt

import torch

rcParams['figure.figsize'] = 4,3
x = torch.linspace(-3, 3, 100)
y1 = x.clone()
tau = 2.0
y2 = tau * torch.tanh(y1 / tau)
plt.plot(x.numpy(), y1.numpy(), "r-")
plt.plot(x.numpy(), y2.numpy(), "b-")
plt.legend(['$\ell(\cdot)$', '$\gamma(\cdot)$' ])
plt.show()
```
</div>

</details>

![png](Variational_Inference_files/Variational_Inference_9_0.png)


The main idea here is to keep the shape of the log-likelihood function $$\ell(\cdot)$$ at *important values* near $$0$$ and bending the function at other values to keep it finite.

It is then guaranteed that by using $$\gamma$$ instead of $$\ell$$ as the log-likelihood function,

$$
\Delta_U \leq 2\tau
$$

Next, we now aim to draw a random sample from $$f_Z(z)$$. We will use stochastic gradient Langevin dynamics (SGLD) to achieve this goal. The basic idea of SGLD is simulating a particle in a physical system. At each timestep, the particle follows a noisy gradient of the potential energy function at its current location.  The probability distribution of the particle will converge to the distribution specified by the potential energy function. In our case, we have:

$$
z_{t+1} = z_t + \frac{\epsilon}{2\Delta_U}\cdot \nabla U(z; x) \cdot \delta_t + r \cdot \sqrt{2 \delta_t}
$$

where $$r \sim \mathcal N(0, 1)$$ and $$\delta_t$$ is the step size at time $$t$$.

We here assume that we can use the reparameterization trick for the probability $$Q(\theta \mid z )$$, that we can represent $$Q$$ as follows:

$$
\begin{align}
r &\sim  P(r) \\
\theta  &= g(z, r)
\end{align}
$$

Therefore, $$U(z; x)$$ can be approximate by:

$$
U(z; x) \approx  \frac 1 L \sum_{i=1}^L  \left(\log\left(P(x \mid \theta_i)\right) + \log\left( P(\theta_i)\right)  - \log\left(Q(\theta_i \mid z)\right)  \right)
$$

where $$\theta_i = g(z, r_i)$$ and $$r_i \sim P(r)$$. This approximation allows us to estimate the gradient of $$U(z; x)$$.

## Experiments

### Data 


$$
\begin{align}
Y &= a X_1 + b X_2 + intercept + r \\
r &\sim \mathcal N(0, \sigma^2)
\end{align}
$$

We generate a dataset with 
$$
\begin{align}
a  &= +2.0 \\
b &= -3.5 \\
intercept &= -5.0 \\
\sigma &= +0.1
\end{align}
$$



<details>
<summary>Click to see code</summary>
<div markdown="1">

```python
#@title Generate Data
import torch
N = 4000

isCuda = False
if torch.cuda.device_count() > 0:
    isCuda = True

    
isCuda = False

a = 2.0
b = -3.5
intercept = -5.0
noise = 0.1

### 

#@title
def generateData():
    X = torch.rand(N, 2) - 0.5
    theta = torch.zeros(2, 1)
    theta[0, 0] = a
    theta[1, 0] = b
    Y = torch.mm(X, theta) + intercept + noise * torch.randn(N, 1)
    data = torch.Tensor( N, 3 )
    data[:, 0:2] = X
    data[:, 2] = Y[:,0]
    return data, X, Y

#@title
data, X, Y = generateData()
if isCuda:
    data = data.cuda()
    X = X.cuda()
    Y = Y.cuda()
    
import seaborn as sns
_ = plt.scatter(X[:, 1], Y[:,0])
```
</div>
</details>

![png](Variational_Inference_files/Variational_Inference_14_0.png)


<details>
<summary>Click to see code</summary>
<div markdown="1">
  
```python
#@title Posterior function

def log_prior(theta):
    coeff = theta[0:3]
    sigma = theta[3]
    hc = HalfCauchy(0.0, 10.0)

    sigma1 = 10.0
    
    return torch.sum(  -torch.pow(coeff, 2) * 0.5 / (sigma1**2) \
                     -  math.log(sigma1) )  + hc.log_pdf_param(sigma)

def log_likelihood(theta, data):
    X = data[:, 0:2]
    Y = data[:, 2:3]
    slope = theta[0:2]
    intercept = theta[2]
    sigma = torch.exp(theta[3])
    return torch.sum(- torch.pow(Y - (torch.mm(X, slope) + intercept), 2) * 0.5\
                          / torch.pow(sigma, 2) - theta[3]  )

def log_posterior(theta, data):
    return log_likelihood(theta, data) + log_prior(theta)


import math
class HalfCauchy:
    def __init__(self, median, scale):
        self.x0 = median
        self.gamma = scale
    def log_pdf(self, x):
        inv =   math.log(math.pi) + math.log(self.gamma) + \
                    torch.log( 1.0 + torch.pow(x - self.x0, 2) / self.gamma)
        
        return math.log(2.0) - inv
    
        
    def get(self, param):    
        return torch.exp(param)
    
    def pdf(self, param):
        return torch.exp(  self.log_pdf(self.get(param)) )
    
    def log_pdf_param(self, param):
        x = torch.exp(param)
        return self.log_pdf(x) + param
        
    def pdf_param(self, param):
        #self.param.zero_()
        #self.param[0] = param
        x = torch.exp(param)
        return self.pdf(x) * torch.exp(param)     
    
    
theta = torch.zeros(4, 1, requires_grad=True)
if isCuda:
    theta = theta.data.cuda()
theta.requires_grad= True
```
</div>
</details>

### Maximum-a-posteriori (MAP) estimation


```
#@title
optimizer = torch.optim.SGD([ theta ], lr=1e-2)
for t in tqdm.tqdm(range(100000)):
    loss = -log_posterior(theta, data) / N
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(loss.item())
```


    100%|██████████| 100000/100000 [01:16<00:00, 1311.62it/s]
    -1.8227145671844482


    

```python
#@title
print( "a = %f\nb=%f\nintercept=%f\nsigma=%f" % (theta[0,0].item(),
                                                  theta[1,0].item(),
                                                  theta[2,0].item(),
                                                  math.exp(theta[3,0].item()) ))
```

    a = 1.993653
    b=-3.496528
    intercept=-5.000208
    sigma=0.097722



```python
#@title
!pip install pystan
```



```python
#@title
import pystan

linreg_code = """
data {
    int<lower=0> N;
    matrix[N, 2] x; 
    real y[N]; 
}
parameters {
    real intercept;
    vector[2] theta;
    real<lower=0> sigma;
}


model {
    vector[N] yhat;
    theta ~ normal(0.0, 10.0);
    intercept ~ normal(0.0, 10.0);
    sigma ~ cauchy(0.0, 1.0);
    
    yhat = x * theta + intercept;    
    
    y ~ normal(yhat, sigma);
}
"""
```

<details>
<summary>Click to see code</summary>
<div markdown="1">


```python
## sm = pystan.StanModel(model_code=linreg_code)
#@title
X = data[:, 0:2]
Y = data[:, 2:3]

dat = {'N':N, 
       'x': X.cpu().numpy().reshape( (N, 2)),
       'y': Y.cpu().numpy().reshape( (N) )  }

fit = sm.sampling(data=dat, iter=10000, chains=1)
```



```python
#@title
rcParams['figure.figsize'] = 10,10
fit.plot()
rcParams['figure.figsize'] = 4,3
samples = fit.extract()

mytheta = torch.empty(len(samples['theta']), 4)
mytheta[:, 0:2] = torch.from_numpy(samples['theta'])
mytheta[:, 2]   = torch.from_numpy(samples['intercept'])
mytheta[:, 3]   = torch.from_numpy(samples['sigma'])
```

</div>
</details>

![png](Variational_Inference_files/Variational_Inference_22_0.png)


### Variational Inference


<details>
<summary>Click to see code</summary>
<div markdown="1">
  
```python
#@title

import torch 
def log_pdf_VI(theta, tau, sigma):
     #return torch.sum(- torch.pow((theta - tau) / sigma, 2) * 0.5 - torch.log(sigma))
     return torch.sum(- torch.log(sigma))

def generate_noise(n):
    return torch.randn(n, 4)


noise = generate_noise(10)
tau = torch.zeros(4, 1, requires_grad=True)
logsigma = torch.zeros(4, 1, requires_grad=True)
##noise * sigma.expand_as(noise) + tau.expand_as(noise)

if isCuda:
    tau = tau.cuda().data
    logsigma = logsigma.cuda().data
    tau.requires_grad=True
    logsigma.requires_grad=True
```


```python
#@title
X = data[:, 0:2]
Y = data[:, 2:3]

#Auto-Encoding Variational Inference
def AEVI():
    ###theta = torch.zeros(4, 1, requires_grad=False)
    global tau, logsigma
    #tau = torch.randn(4, 1, requires_grad=True)
    #sigma = torch.ones(4, 1, requires_grad=True)
    optimizer = torch.optim.Adam([ tau, logsigma ], lr=1e-3)
    
    n = 1
    NN = 100000
    for it in tqdm.tqdm(range(NN)):
        if it == NN * 8 // 10:
            optimizer = torch.optim.Adam([ tau, logsigma ], lr=1e-4)
        loss = torch.zeros(1)
        if isCuda:
            loss = loss.cuda()
        mysigma = torch.exp(logsigma)
        ###sigma.data = torch.max(sigma, 1e-8*torch.ones_like(sigma))
        ##if isCuda:
        ##    sigma = sigma.cuda()
            
        for i in range(n):
            noise = torch.randn(4,1).data
            if isCuda:
                noise = noise.cuda()
            theta = noise * mysigma + tau
            loss = loss + 1.0 / n *  ( log_pdf_VI(theta, tau, mysigma) - log_posterior(theta, data))
        loss = loss / N
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #if it % 1000 == 0:
        #    print(loss.item())
    print(N*loss.item())
    return tau, logsigma
```

</div>
</details>


```python
tau, logsigma = AEVI()
```

    100%|██████████| 100000/100000 [01:29<00:00, 1113.57it/s]

    -7269.19412612915


    

<details>
<summary>Click to see code</summary>
<div markdown="1">


```python
#@title
import seaborn as sns
name = ["a", "b", "intercept", "sigma"]


sigma = torch.exp(logsigma.data)
for id in range(4):
    ##myt = [ s[id,0] for s in Xs[0:]]



    ### myt = [ s[id,0] for s in Xs[0:]]




    xx = torch.linspace(float(min(mytheta[:, id].numpy())), float(max(mytheta[:, id].numpy())), 100)
    if id == 3:
        xx = torch.linspace(math.log(float(min(mytheta[:, id].numpy()))), math.log(float(max(mytheta[:, id].numpy()))), 100)
        
    yy =  torch.exp( - 0.5 * torch.pow ( (xx - tau.cpu()[id,0])\
           / sigma.cpu()[id,0], 2)) / math.sqrt( 2. * math.pi ) \
    / sigma.cpu()[id,0]

    
    if id == 3:
        ###myt = [math.exp(t) for t in myt]
        xx = torch.exp(xx)
        yy = yy / xx
        
    _ = sns.distplot(mytheta[:, id].numpy(), hist=False)

   # _ = plt.legend(['Variational Inference'])        
        
    _ = plt.plot(xx.cpu().data.numpy(), yy.cpu().data.numpy(), "b-")

    #_ = plt.legend(['Variational Inference', 'Hamiltonian Monte Carlo'])
    plt.ylabel(name[id])
    plt.show()

```

</div>
</details>


![png](Variational_Inference_files/Variational_Inference_27_0.png)



![png](Variational_Inference_files/Variational_Inference_27_1.png)



![png](Variational_Inference_files/Variational_Inference_27_2.png)



![png](Variational_Inference_files/Variational_Inference_27_3.png)




<details>
<summary>Click to see code</summary>
<div markdown="1">
  
  
```python
#@title
logsigma_bak = logsigma.clone()
tau_bak = tau.clone()
logsigma = logsigma.data
tau = tau.data
```


```python
#@title
### noise = generate_noise(10)
tau = tau_bak.data # torch.zeros(4, 1, requires_grad=True)
#sigma = sigma_bak.data # torch.ones(4, 1, requires_grad=True)
logsigma =  logsigma_bak.data # torch.log(sigma_bak.data)
##noise * sigma.expand_as(noise) + tau.expand_as(noise)

if isCuda:
    tau = tau.cuda().data
    logsigma = logsigma.cuda().data
    tau.requires_grad=True
    logsigma.requires_grad=True
```


```python
#@title
#Auto-Encoding Variational Inference - Stochastic Gradient Langevin Dynamics
def AEVI_SGLD():
    global tau, logsigma
    #tau = torch.zeros(4, 1, requires_grad=True)
    #sigma = torch.ones(4, 1, requires_grad=True)

    if isCuda:
        tau = tau.cuda().data
        logsigma = logsigma.cuda().data
        tau.requires_grad=True
        logsigma.requires_grad=True
    
    et = 1e-6
    
    samples = []
    
    n = 1
    var_tau = torch.zeros_like(tau)
    var_sigma = torch.zeros_like(logsigma)
    s_tau = torch.ones_like(tau)
    s_sigma = et*torch.ones_like(logsigma)
    s2_tau = et*math.sqrt(2.0*et)*torch.ones_like(tau)
    s2_sigma = math.sqrt(2.0*et)*torch.ones_like(logsigma)
    NN = 300000
    lim = NN // 4
    for it in tqdm.tqdm(range(NN)):
        logsigma.requires_grad = True
        tau.requires_grad = True
        loss = torch.zeros(1)
        if isCuda:
            loss = loss.cuda()
            
        for i in range(n):
            noise = torch.randn(4,1, requires_grad=False)
            if isCuda:
                noise = noise.cuda()
                
            mysigma = torch.exp(logsigma)
            qtheta = noise * mysigma + tau
            loss = loss + 1.0 / n *  ( log_pdf_VI(qtheta, tau, mysigma) - log_posterior(qtheta, data))
        loss = loss 
        
        
        loss.backward()
        
        #if it % 100 == 0:
        #    print(logsigma.grad.data, tau.grad.data)

        if it < lim:
            var_tau = var_tau +  torch.pow(tau.grad.data, 2)
            var_sigma = var_sigma + torch.pow(logsigma.grad.data, 2)
            tau_noise = torch.randn_like(tau) * math.sqrt(2*et) 
            sigma_noise = torch.randn_like(logsigma) * math.sqrt(2*et) 

            tau = tau.data - et * tau.grad.data + tau_noise
            logsigma = logsigma.data - et * logsigma.grad.data + sigma_noise 
            
        elif it == lim:
            et = 4e-3
            s_tau = et / torch.sqrt(var_tau/lim)
            s_sigma = et / torch.sqrt(var_sigma/lim)
            s2_tau = torch.sqrt(2*s_tau.data)
            s2_sigma = torch.sqrt(2*s_sigma.data)
        else:
            tau_noise = torch.randn_like(tau) * s2_tau
            sigma_noise = torch.randn_like(logsigma) * s2_sigma

            tau = tau.data - s_tau * tau.grad.data + tau_noise
            logsigma = logsigma.data - s_sigma * logsigma.grad.data + sigma_noise 

            samples.append(  ( tau.data.numpy(), logsigma.data.numpy() ) )
        #if it % 1000 == 0:
        #    print(loss.item())
    return samples
```

</div>
</details>



```python
samples = AEVI_SGLD()
```

    100%|██████████| 300000/300000 [04:04<00:00, 1228.04it/s]



```python
#@title
rcParams['figure.figsize'] = 10,10

for id in range(4):
    for yy in range(2):
        plt.subplot(4,2,id*2 + yy+1)
        ss = [ sample[yy][id] for sample in samples if math.isfinite(sample[yy][id])]
        sns.distplot(ss)
```


![png](Variational_Inference_files/Variational_Inference_32_1.png)



<details>
<summary>Click to see code</summary>
<div markdown="1">
  
```python
import tensorflow as tf
import math
import numpy as np
import tqdm
sess = tf.Session()

taubound = 4.0

def log_prior(co, logsigma):
    sigma = tf.exp(logsigma)
    gamma = 1.0
    x0 = 0.0

    log_cauchy_sigma = math.log(math.pi) + math.log(gamma) + tf.log(  1.0 + tf.pow( sigma - x0, 2)/gamma)

    log_cauchy_logsigma = math.log(2.0) - log_cauchy_sigma + logsigma

    log_prior =   -tf.reduce_sum( tf.pow(co / 10.0, 2)  * 0.5  ) +  log_cauchy_logsigma
    return log_prior

def sanitizer(loglikelihood):
    return taubound * tf.tanh( loglikelihood / taubound)

def log_likelihood(coeff, intercept, logsigma, X, Y):
    tau = 2.0
    sigma = tf.exp(logsigma)
    Yhat =  tf.matmul(X, coeff) + intercept
    llh = -tf.pow((Y-Yhat)/sigma , 2) * 0.5 - logsigma
    return tf.reduce_sum(  sanitizer ( llh ) )

def log_pos(X, Y, theta):
    co = theta[0:3,:]
    coeff = theta[0:2,:]
    intercept = theta[2,0]
    logsigma = theta[3,0]

    return log_prior(co, logsigma) + log_likelihood(coeff, intercept, logsigma, X, Y)
```


```python
tf.reset_default_graph()

with tf.device('/cpu:0'):
    D = tf.constant(data.numpy())
    X = D[:, 0:2]
    Y = D[:, 2:3]

    theta = tf.Variable(tf.zeros([4, 1]))

    log_posterior = log_pos(X, Y, theta)
    
    loss = -log_posterior / N

    ## grad_theta = tf.gradients(loss, theta)

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss, var_list=[theta])

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
for i in tqdm.tqdm(range(200000)):
    sess.run(train)

tttheta = sess.run(theta)
print(tttheta)
```

    100%|██████████| 200000/200000 [01:15<00:00, 2644.57it/s]

    [[ 1.9943571]
     [-3.496611 ]
     [-5.000044 ]
     [-2.275206 ]]


    



```python
#@title
tf.reset_default_graph()

with tf.device('/cpu:0'):
    D = tf.constant(data.numpy())
    X = D[:, 0:2]
    Y = D[:, 2:3]
    
    vi_mean  = tf.Variable( tf.zeros([4,1]) )
    vi_logsigma = tf.Variable( tf.zeros([4,1]))
    vi_sigma = tf.exp(vi_logsigma)

    noi = tf.random_normal( [4, 1])
    sample = noi * vi_sigma + vi_mean
    log_posterior = log_pos(X, Y, sample)

    vi_loss = -(log_posterior + 10*tf.reduce_sum(vi_logsigma)) / N


    optimizer = tf.train.AdamOptimizer(1e-3)
    train = optimizer.minimize(vi_loss, var_list=[vi_mean, vi_logsigma])
    
    optimizer1 = tf.train.AdamOptimizer(1e-4)
    train1 = optimizer1.minimize(vi_loss, var_list=[vi_mean, vi_logsigma])

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
NN = 100000
for i in tqdm.tqdm(range(NN)):
    if i == (NN * 8) // 10:
        train = train1
    sess.run(train)

mm, ms =  sess.run( (vi_mean, vi_logsigma) )
print(mm)
print(ms)

```

    100%|██████████| 100000/100000 [00:44<00:00, 2249.02it/s]

    [[ 1.9936111]
     [-3.4969277]
     [-5.0000567]
     [-2.270344 ]]
    [[-3.9523826]
     [-3.968117 ]
     [-5.1965833]
     [-3.2393332]]


</div>
</details>
    



```python
#@title
tf.reset_default_graph()


et = 1e-6
gg = 10.0
deltaUtility = 2 * taubound

privacyBudget = 1.0
def privacyController(utility):
    return privacyBudget * utility / (2*deltaUtility)
     

with tf.device('/cpu:0'):
    D = tf.constant(data.numpy())
    X = D[:, 0:2]
    Y = D[:, 2:3]
    
    vi_mean  = tf.Variable( mm )
    vi_logsigma = tf.Variable( ms )
    vi_sigma = tf.exp(vi_logsigma)

    vi_mean_var  = tf.Variable( tf.zeros([4,1]) )
    vi_logsigma_var = tf.Variable( tf.zeros([4,1]))

    noi = tf.random_normal( [4, 1])
    sample = noi * vi_sigma + vi_mean
    log_posterior = log_pos(X, Y, sample)    

    vi_loss = -(log_posterior + gg * tf.reduce_sum(vi_logsigma)) 
    vi_loss = privacyController(vi_loss)
    
    vi_mean_grad, vi_logsigma_grad = tf.gradients(vi_loss, [vi_mean, vi_logsigma] )
    vi_mean_noise = tf.random_normal( [4, 1]) * math.sqrt(2*et)
    vi_logsigma_noise = tf.random_normal( [4, 1]) * math.sqrt(2*et)
    
    vi_mean_ops = tf.assign_sub(vi_mean, vi_mean_grad * et + vi_mean_noise )
    vi_logsigma_ops = tf.assign_sub(vi_logsigma, vi_logsigma_grad * et + vi_logsigma_noise )
    
    vi_mean_var_ops = tf.assign_add(vi_mean_var,  tf.pow(vi_mean_grad, 2) )
    vi_logsigma_var_ops = tf.assign_add(vi_logsigma_var,  tf.pow(vi_logsigma_grad, 2) )

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
NN = 500000
for i in tqdm.tqdm(range(NN//4)):
    _,_,ll,_,_ = sess.run( (vi_mean_var_ops, vi_logsigma_var_ops, vi_loss, vi_mean_ops, vi_logsigma_ops) )
    #if i % 1000 == 0:
    #    print(ll)

mmm, mms =  sess.run( (vi_mean_var, vi_logsigma_var) )
mv = np.sqrt(mmm / (NN//4))
mls = np.sqrt(mms / (NN//4))
print(mv)
print(mls)

```

    100%|██████████| 125000/125000 [01:11<00:00, 1760.40it/s]

    [[101.93008176]
     [117.83236038]
     [474.97880158]
     [ 61.1510518 ]]
    [[10.23276229]
     [15.7553059 ]
     [31.95735458]
     [11.47932333]]


    



```python
#@title
tf.reset_default_graph()


et = 1e-2


with tf.device('/cpu:0'):
    D = tf.constant(data.numpy())
    X = D[:, 0:2]
    Y = D[:, 2:3]
    tfmv = tf.constant(mv, dtype=tf.float32)
    tflsv = tf.constant(mls, dtype=tf.float32)
    
    vi_mean  = tf.Variable( mm )
    vi_logsigma = tf.Variable( ms )
    vi_sigma = tf.exp(vi_logsigma)

    noi = tf.random_normal( [4, 1])
    sample = noi * vi_sigma + vi_mean
    log_posterior = log_pos(X, Y, sample)    

    vi_loss = -(log_posterior + gg * tf.reduce_sum(vi_logsigma)) 
    vi_loss = privacyController(vi_loss)
    
    vi_mean_grad, vi_logsigma_grad = tf.gradients(vi_loss, [vi_mean, vi_logsigma] )
    vi_mean_noise = tf.random_normal( [4, 1]) * tf.sqrt(2*et / tfmv)
    vi_logsigma_noise = tf.random_normal( [4, 1]) * tf.sqrt(2*et / tflsv)
    
    vi_mean_ops = tf.assign_sub(vi_mean, vi_mean_grad * et / tfmv + vi_mean_noise )
    vi_logsigma_ops = tf.assign_sub(vi_logsigma, vi_logsigma_grad * et / tflsv + vi_logsigma_noise )
    

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

NN = 1000000

LLS = []
for i in tqdm.tqdm(range(NN)):
    ll,_,_, vivimean, vivilogsigma = \
    sess.run( (vi_loss, vi_mean_ops, vi_logsigma_ops, vi_mean, vi_logsigma) )
    LLS.append(  (vivimean, vivilogsigma) )
    #if i % 1000 == 0:
    #    print(ll)

```

    100%|██████████| 1000000/1000000 [09:34<00:00, 1740.07it/s]



```python
#@title
rcParams['figure.figsize'] = 10,10

for id in range(4):
    for yy in range(2):
        plt.subplot(4,2,id*2 + yy+1)
        ss = [ sample[yy][id] for sample in LLS if not math.isnan(sample[yy][id])]
        sns.distplot(ss)
```

![png](Variational_Inference_files/Variational_Inference_39_1.png)

