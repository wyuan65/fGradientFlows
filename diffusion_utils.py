from abc import ABC, abstractmethod
from pickletools import int4
from typing import Optional, List, Type, Tuple, Dict
import math

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.axes._axes import Axes
import torch
import torch.distributions as D
from torch.func import vmap, jacrev
from tqdm import tqdm
import seaborn as sns
from sklearn.datasets import make_moons, make_circles
import ot
import copy

# Constants for the duration of our use of Gaussian conditional probability paths, to avoid polluting the namespace...
PARAMS = {
    "scale": 10.0,
    "target_scale": 5.0,
    "target_std": 1.0,
}
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

class Sampleable(ABC):
    """
    Distribution which can be sampled from
    """
    @property
    @abstractmethod
    def dim(self) -> int:
        """
        Returns:
            - Dimensionality of the distribution
        """
        pass
        
    @abstractmethod
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (batch_size, dim)
        """
        pass
class Density(ABC):
    """
    Distribution with tractable density
    """
    @abstractmethod
    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the log density at x.
        Args:
            - x: shape (batch_size, dim)
        Returns:
            - log_density: shape (batch_size, 1)
        """
        pass
class Gaussian(torch.nn.Module, Sampleable, Density):
    """
    Multivariate Gaussian distribution
    """
    def __init__(self, mean: torch.Tensor, cov: torch.Tensor):
        """
        mean: shape (dim,)
        cov: shape (dim,dim)
        """
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("cov", cov)

    @property
    def dim(self) -> int:
        return self.mean.shape[0]

    @property
    def distribution(self):
        return D.MultivariateNormal(self.mean, self.cov, validate_args=False)

    def sample(self, num_samples) -> torch.Tensor:
        return self.distribution.sample((num_samples,))
        
    def log_density(self, x: torch.Tensor):
        return self.distribution.log_prob(x).view(-1, 1)

    @classmethod
    def isotropic(cls, dim: int, std: float) -> "Gaussian":
        mean = torch.zeros(dim)
        cov = torch.eye(dim) * std ** 2
        return cls(mean, cov)
class GaussianMixture(torch.nn.Module, Sampleable, Density):
    """
    Two-dimensional Gaussian mixture model, and is a Density and a Sampleable. Wrapper around torch.distributions.MixtureSameFamily.
    """
    def __init__(
        self,
        means: torch.Tensor,  # nmodes x data_dim
        covs: torch.Tensor,  # nmodes x data_dim x data_dim
        weights: torch.Tensor,  # nmodes
    ):
        """
        means: shape (nmodes, 2)
        covs: shape (nmodes, 2, 2)
        weights: shape (nmodes, 1)
        """
        super().__init__()
        self.nmodes = means.shape[0]
        self.register_buffer("means", means)
        self.register_buffer("covs", covs)
        self.register_buffer("weights", weights)
        self.P = torch.randn((20,self.dim)).to(device)
        self.P = self.P/torch.norm(self.P, dim=0, keepdim=True)

    @property
    def dim(self) -> int:
        return self.means.shape[1]

    @property
    def distribution(self):
        return D.MixtureSameFamily(
                mixture_distribution=D.Categorical(probs=self.weights, validate_args=False),
                component_distribution=D.MultivariateNormal(
                    loc=self.means,
                    covariance_matrix=self.covs,
                    validate_args=False,
                ),
                validate_args=False,
            )

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(x).view(-1, 1)

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.distribution.sample(torch.Size((num_samples,)))

    def sample_projected(self, num_samples: int) -> torch.Tensor:
        sample = self.distribution.sample(torch.Size((num_samples,))) 
        return (self.P[None,:,:]@sample[:,:,None]).squeeze()

    @classmethod
    def random_2D(
        cls, nmodes: int, std: float, scale: float = 10.0, x_offset: float = 0.0, seed = 0.0
    ) -> "GaussianMixture":
        torch.manual_seed(seed)
        means = (torch.rand(nmodes, 2) - 0.5) * scale + x_offset * torch.Tensor([1.0, 0.0])
        covs = torch.diag_embed(torch.ones(nmodes, 2)) * std ** 2
        weights = torch.ones(nmodes)
        return cls(means, covs, weights)

    @classmethod
    def symmetric_2D(
        cls, nmodes: int, std: float, scale: float = 10.0, x_offset: float = 0.0
    ) -> "GaussianMixture":
        angles = torch.linspace(0, 2 * np.pi, nmodes + 1)[:nmodes]
        means = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * scale + torch.Tensor([1.0, 0.0]) * x_offset
        covs = torch.diag_embed(torch.ones(nmodes, 2) * std ** 2)
        weights = torch.ones(nmodes) / nmodes
        return cls(means, covs, weights)

    @classmethod
    def symmetric_4D(
        cls, nmodes: int, std: float
    ) -> "GaussianMixture":
        # generate points uniformly on the 4D hypersphere
        a = torch.tensor([3.0,3.0,3.0,3.0])
        means = torch.stack([a,-a],dim=0)

        # isotropic covariance
        covs = torch.diag_embed(torch.ones(nmodes, 4) * std ** 2)
        #weights = torch.ones(nmodes) / nmodes
        weights = torch.tensor([0.3,0.7])
        return cls(means, covs, weights)

class ODE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Returns the drift coefficient of the ODE.
        Args:
            - xt: state at time t, shape (bs, dim)
            - t: time, shape (batch_size, 1)
        Returns:
            - drift_coefficient: shape (batch_size, dim)
        """
        pass
class SDE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Returns the drift coefficient of the ODE.
        Args:
            - xt: state at time t, shape (batch_size, dim)
            - t: time, shape (batch_size, 1)
        Returns:
            - drift_coefficient: shape (batch_size, dim)
        """
        pass

    @abstractmethod
    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Returns the diffusion coefficient of the ODE.
        Args:
            - xt: state at time t, shape (batch_size, dim)
            - t: time, shape (batch_size, 1)
        Returns:
            - diffusion_coefficient: shape (batch_size, dim)
        """
        pass
class Simulator(ABC):
    @abstractmethod
    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor):
        """
        Takes one simulation step
        Args:
            - xt: state at time t, shape (bs, dim)
            - t: time, shape (bs,1)
            - dt: time, shape (bs,1)
        Returns:
            - nxt: state at time t + dt (bs, dim)
        """
        pass

    @torch.no_grad()
    def simulate(self, x: torch.Tensor, ts: torch.Tensor):
        """
        Simulates using the discretization gives by ts
        Args:
            - x_init: initial state at time ts[0], shape (batch_size, dim)
            - ts: timesteps, shape (bs, num_timesteps,1)
        Returns:
            - x_final: final state at time ts[-1], shape (batch_size, dim)
        """
        for t_idx in range(len(ts) - 1):
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h)
        return x

    @torch.no_grad()
    def simulate_with_trajectory(self, x: torch.Tensor, ts: torch.Tensor):
        """
        Simulates using the discretization gives by ts
        Args:
            - x_init: initial state at time ts[0], shape (bs, dim)
            - ts: timesteps, shape (bs, num_timesteps, 1)
        Returns:
            - xs: trajectory of xts over ts, shape (batch_size, num
            _timesteps, dim)
        """
        xs = [x.clone()]
        nts = ts.shape[1]
        for t_idx in tqdm(range(nts - 1)):
            t = ts[:,t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h)
            xs.append(x.clone())
        return torch.stack(xs, dim=1)
class EulerSimulator(Simulator):
    def __init__(self, ode: ODE):
        self.ode = ode
        
    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor):
        return xt + self.ode.drift_coefficient(xt,t) * h
class GDSimulator(Simulator):
    def __init__(self, ode: ODE):
        self.ode = ode
        
    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor):
        return xt + self.ode.drift_coefficient(xt) * h
class NesterovSimulator(Simulator):
    def __init__(self, ode: ODE):
        self.ode = ode
        
    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor):
        return xt + self.ode.drift_coefficient(xt) * h
class EulerMaruyamaSimulator(Simulator):
    def __init__(self, sde: SDE):
        self.sde = sde
        
    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor):
        return xt + self.sde.drift_coefficient(xt,t) * h + self.sde.diffusion_coefficient(xt,t) * torch.sqrt(h) * torch.randn_like(xt)

class ConditionalProbabilityPath(torch.nn.Module, ABC):
    """
    Abstract base class for conditional probability paths
    """
    def __init__(self, p_simple: Sampleable, p_data: Sampleable):
        super().__init__()
        self.p_simple = p_simple
        self.p_data = p_data

    def sample_marginal_path(self, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the marginal distribution p_t(x) = p_t(x|z) p(z)
        Args:
            - t: time (num_samples, 1)
        Returns:
            - x: samples from p_t(x), (num_samples, dim)
        """
        num_samples = t.shape[0]
        # Sample conditioning variable z ~ p(z)
        z = self.sample_conditioning_variable(num_samples) # (num_samples, dim)
        # Sample conditional probability path x ~ p_t(x|z)
        x = self.sample_conditional_path(z, t) # (num_samples, dim)
        return x

    @abstractmethod
    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
        """
        Samples the conditioning variable z
        Args:
            - num_samples: the number of samples
        Returns:
            - z: samples from p(z), (num_samples, dim)
        """
        pass
    
    @abstractmethod
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the conditional distribution p_t(x|z)
        Args:
            - z: conditioning variable (num_samples, dim)
            - t: time (num_samples, 1)
        Returns:
            - x: samples from p_t(x|z), (num_samples, dim)
        """
        pass
        
    @abstractmethod
    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional vector field u_t(x|z)
        Args:
            - x: position variable (num_samples, dim)
            - z: conditioning variable (num_samples, dim)
            - t: time (num_samples, 1)
        Returns:
            - conditional_vector_field: conditional vector field (num_samples, dim)
        """ 
        pass

    @abstractmethod
    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional score of p_t(x|z)
        Args:
            - x: position variable (num_samples, dim)
            - z: conditioning variable (num_samples, dim)
            - t: time (num_samples, 1)
        Returns:
            - conditional_score: conditional score (num_samples, dim)
        """ 
        pass
def record_every(num_timesteps: int, record_every: int) -> torch.Tensor:
    """
    Compute the indices to record in the trajectory given a record_every parameter
    """
    if record_every == 1:
        return torch.arange(num_timesteps)
    return torch.cat(
        [
            torch.arange(0, num_timesteps - 1, record_every),
            torch.tensor([num_timesteps - 1]),
        ]
    )
class Alpha(ABC):
    def __init__(self):
        # Check alpha_t(0) = 0
        assert torch.allclose(
            self(torch.zeros(1,1)), torch.zeros(1,1)
        )
        # Check alpha_1 = 1
        assert torch.allclose(
            self(torch.ones(1,1)), torch.ones(1,1)
        )
        
    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates alpha_t. Should satisfy: self(0.0) = 0.0, self(1.0) = 1.0.
        Args:
            - t: time (num_samples, 1)
        Returns:
            - alpha_t (num_samples, 1)
        """ 
        pass

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: time (num_samples, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1)
        """ 
        t = t.unsqueeze(1) # (num_samples, 1, 1)
        dt = vmap(jacrev(self))(t) # (num_samples, 1, 1, 1, 1)
        return dt.view(-1, 1)  
class Beta(ABC):
    def __init__(self):
        # Check beta_0 = 1
        assert torch.allclose(
            self(torch.zeros(1,1)), torch.ones(1,1)
        )
        # Check beta_1 = 0
        assert torch.allclose(
            self(torch.ones(1,1)), torch.zeros(1,1)
        )
        
    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates alpha_t. Should satisfy: self(0.0) = 1.0, self(1.0) = 0.0.
        Args:
            - t: time (num_samples, 1)
        Returns:
            - beta_t (num_samples, 1)
        """ 
        pass 

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt beta_t.
        Args:
            - t: time (num_samples, 1)
        Returns:
            - d/dt beta_t (num_samples, 1)
        """ 
        t = t.unsqueeze(1) # (num_samples, 1, 1)
        dt = vmap(jacrev(self))(t) # (num_samples, 1, 1, 1, 1)
        return dt.view(-1, 1)
class LinearAlpha(Alpha):
    """
    Implements alpha_t = t
    """
    
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - t: time (num_samples, 1)
        Returns:
            - alpha_t (num_samples, 1)
        """ 
        return t 
        
    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: time (num_samples, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1)
        """ 
        return torch.ones_like(t)
class SquareRootBeta(Beta):
    """
    Implements beta_t = rt(1-t)
    """
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - t: time (num_samples, 1)
        Returns:
            - beta_t (num_samples, 1)
        """ 
        return torch.sqrt(1-t)

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: time (num_samples, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1)
        """ 
        return - 0.5 / (torch.sqrt(1 - t) + 1e-4)
class GaussianConditionalProbabilityPath(ConditionalProbabilityPath):
    def __init__(self, p_data: Sampleable, alpha: Alpha, beta: Beta):
        aux_dim = 2 
        p_simple = Gaussian.isotropic(aux_dim, 1.0)
        super().__init__(p_simple, p_data)
        self.alpha = alpha
        self.beta = beta
        self.p_data = p_data

    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
        """
        Samples the conditioning variable z ~ p_data(x)
        Args:
            - num_samples: the number of samples
        Returns:
            - z: samples from p(z), (num_samples, dim)
        """
        return self.p_data.sample(num_samples) 
    
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Samples from the conditional distribution p_t(x|z) = N(alpha_t * z, beta_t**2 * I_d)
        Args:
            - z: conditioning variable (num_samples, dim)
            - t: time (num_samples, 1)
        Returns:
            - x: samples from p_t(x|z), (num_samples, dim)
        """
        return self.alpha(t)*z+self.beta(t)*torch.randn_like(z) 
        
    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional vector field u_t(x|z)
        Note: Only defined on t in [0,1)
        Args:
            - x: position variable (num_samples, dim)
            - z: conditioning variable (num_samples, dim)
            - t: time (num_samples, 1)
        Returns:
            - conditional_vector_field: conditional vector field (num_samples, dim)
        """ 
        return (self.alpha.dt(t)-self.beta.dt(t)/self.beta(t)*self.alpha(t))*z + self.beta.dt(t)/self.beta(t)*x

    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the conditional score of p_t(x|z) = N(alpha_t * z, beta_t**2 * I_d)
        Note: Only defined on t in [0,1)
        Args:
            - x: position variable (num_samples, dim)
            - z: conditioning variable (num_samples, dim)
            - t: time (num_samples, 1)
        Returns:
            - conditional_score: conditional score (num_samples, dim)
        """ 
        return -(x - self.alpha(t)*z)/self.beta(t)**2

class ConditionalVectorFieldODE(ODE):
    def __init__(self, path: ConditionalProbabilityPath, z: torch.Tensor):
        """
        Args:
        - path: the ConditionalProbabilityPath object to which this vector field corresponds
        - z: the conditioning variable, (1, dim)
        """
        super().__init__()
        self.path = path
        self.z = z

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Returns the conditional vector field u_t(x|z)
        Args:
            - x: state at time t, shape (bs, dim)
            - t: time, shape (bs,.)
        Returns:
            - u_t(x|z): shape (batch_size, dim)
        """
        bs = x.shape[0]
        z = self.z.expand(bs, *self.z.shape[1:])
        return self.path.conditional_vector_field(x,z,t)
class ConditionalVectorFieldSDE(SDE):
    def __init__(self, path: ConditionalProbabilityPath, z: torch.Tensor, sigma: float):
        """
        Args:
        - path: the ConditionalProbabilityPath object to which this vector field corresponds
        - z: the conditioning variable, (1, ...)
        """
        super().__init__()
        self.path = path
        self.z = z
        self.sigma = sigma

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Returns the conditional vector field u_t(x|z)
        Args:
            - x: state at time t, shape (bs, dim)
            - t: time, shape (bs,.)
        Returns:
            - u_t(x|z): shape (batch_size, dim)
        """
        bs = x.shape[0]
        z = self.z.expand(bs, *self.z.shape[1:])
        return self.path.conditional_vector_field(x,z,t) + 0.5 * self.sigma**2 * self.path.conditional_score(x,z,t)

    def diffusion_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: state at time t, shape (bs, dim)
            - t: time, shape (bs,.)
        Returns:
            - u_t(x|z): shape (batch_size, dim)
        """
        return self.sigma * torch.randn_like(x)
def build_mlp(dims: List[int], activation: Type[torch.nn.Module] = torch.nn.SiLU):
        mlp = []
        for idx in range(len(dims) - 1):
            mlp.append(torch.nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                mlp.append(activation())
        return torch.nn.Sequential(*mlp)
class MLPVectorField(torch.nn.Module):
    """
    MLP-parameterization of the learned vector field u_t^theta(x)
    """
    def __init__(self, dim: int, hiddens: List[int]):
        super().__init__()
        self.dim = dim
        self.net = build_mlp([dim + 1] + hiddens + [dim])

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Args:
        - x: (bs, dim)
        Returns:
        - u_t^theta(x): (bs, dim)
        """
        xt = torch.cat([x,t], dim=-1)
        return self.net(xt)                
class MLPScore(torch.nn.Module):
    """
    MLP-parameterization of the learned score field
    """
    def __init__(self, dim: int, hiddens: List[int]):
        super().__init__()
        self.dim = dim
        self.net = build_mlp([dim + 1] + hiddens + [dim])

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Args:
        - x: (bs, dim)
        Returns:
        - s_t^theta(x): (bs, dim)
        """
        xt = torch.cat([x,t], dim=-1)
        return self.net(xt)        
class MLPEquilibrium(torch.nn.Module):
    """
    MLP-parameterization of the learned equilibrium field
    """
    def __init__(self, dim: int, hiddens: List[int]):
        super().__init__()
        self.dim = dim
        self.net = build_mlp([dim] + hiddens + [dim])

    def forward(self, x: torch.Tensor):
        """
        Args:
        - x: (bs, dim)
        Returns:
        - u^theta(x): (bs, dim)
        """
        return self.net(x)    
class MLPTransport(torch.nn.Module):
    """
    Transport map T_θ: R^d -> R^d
    Implemented as a residual map: T(x) = x + f_θ(x).
    """
    def __init__(self, dim: int, hiddens: List[int]):
        super().__init__()
        self.dim = dim
        self.base = build_mlp([dim] + hiddens + [dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.base(x)
class MLPCritic(torch.nn.Module):
    """
    Scalar critic h_λ: R^d -> (0,1) or R
    """
    def __init__(self, dim: int, hiddens: List[int]):
        super().__init__()
        self.net = build_mlp([dim] + hiddens + [1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)**2
        #return torch.sigmoid(self.net(x))
        #return torch.tanh(self.net(x))/2

# remove gamma
class VariationalObjective(ABC):
    """
    Interface for the variational representation of F(P).

    It bundles:
      - A_h(y) term evaluated at y = T_θ(x)
      - B_h(z) term evaluated at z ~ Γ
      - Γ sampler
    """
    def __init__(self, gamma: Sampleable):
        self.gamma = gamma

    def sample_gamma(self, num_samples: int) -> torch.Tensor:
        return self.gamma.sample(num_samples)

    @abstractmethod
    def A(self, y: torch.Tensor, h: torch.nn.Module) -> torch.Tensor:
        """
        A_h(y) term, returns shape (bs,)
        """
        pass

    @abstractmethod
    def B(self, h_z: torch.Tensor) -> torch.Tensor:
        """
        B_h(z) term, returns shape (bs,)
        """
        pass
class KLObjective(VariationalObjective):
    r"""
    F(P) = KL(P || Q)

    Requires h \in (0,\infty)
    """
    def __init__(self, gamma: Sampleable):
        super().__init__(gamma)

    def A(self, y: torch.Tensor, h: torch.nn.Module) -> torch.Tensor:
        return torch.log(torch.clamp(h(y), min=1e-8))*torch.exp(torch.tensor(1.0))

    def B(self, h_z: torch.Tensor) -> torch.Tensor:
        return h_z 
class RKLObjective(VariationalObjective):
    r"""
    F(P) = RKL(P || Q)

    Requires h \in (0,1)
    """
    def __init__(self, gamma: Sampleable):
        super().__init__(gamma)

    def A(self, y: torch.Tensor, h: torch.nn.Module) -> torch.Tensor:
        return torch.log(torch.clamp(h(y), min=1e-8))

    def B(self, h_z: torch.Tensor) -> torch.Tensor:
        return -torch.log(-torch.log(torch.clamp(h_z, min=1e-8)))
class JSObjective(VariationalObjective):
    r"""
    F(P) = JS(P || Q)

    Requires h \in (0,1)
    """
    def __init__(self, gamma: Sampleable):
        super().__init__(gamma)

    def Ans(self, y: torch.Tensor, h: torch.nn.Module) -> torch.Tensor:
        return -torch.log(torch.clamp(h(y), min=1e-8))

    def A(self, y: torch.Tensor, h: torch.nn.Module) -> torch.Tensor:
        return torch.log(torch.clamp(1.0 - h(y), min=1e-8))

    def B(self, h_z: torch.Tensor) -> torch.Tensor:
        return -torch.log(torch.clamp(h_z, min=1e-8))
class HellingerObjective(VariationalObjective):
    r"""
    F(P) = Hellinger(P || Q)

    Requires h \in (0,\infty)
    """
    def __init__(self, gamma: Sampleable):
        super().__init__(gamma)

    def A(self, y: torch.Tensor, h: torch.nn.Module) -> torch.Tensor:
        return -1/h(y)

    def B(self, h_z: torch.Tensor) -> torch.Tensor:
        return h_z
class PearsonChiSquaredObjective(VariationalObjective):
    r"""
    F(P) = PearsonChiSquared(P || Q)

    Requires h \in (0,\infty)
    """
    def __init__(self, gamma: Sampleable):
        super().__init__(gamma)

    def A(self, y: torch.Tensor, h: torch.nn.Module) -> torch.Tensor:
        return h(y)

    def B(self, h_z: torch.Tensor) -> torch.Tensor:
        return h_z**2
class TVObjective(VariationalObjective):
    r"""
    F(P) = TV(P || Q)

    Requires h \in (-1/2,1/2)
    """
    def __init__(self, gamma: Sampleable):
        super().__init__(gamma)

    def A(self, y: torch.Tensor, h: torch.nn.Module) -> torch.Tensor:
        return h(y)

    def B(self, h_z: torch.Tensor) -> torch.Tensor:
        return h_z
class JeffreysObjective(VariationalObjective):
    r"""
    F(P) = J(P || Q)

    Requires h \in (-1/2,1/2)
    """
    def __init__(self, gamma: Sampleable):
        super().__init__(gamma)

    def A(self, y: torch.Tensor, h: torch.nn.Module) -> torch.Tensor:
        return h(y)

    def B(self, h_z: torch.Tensor) -> torch.Tensor:
        return h_z


class Trainer(ABC):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass

    def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, num_epochs: int, device: torch.device, lr: float = 1e-3, **kwargs) -> torch.Tensor:
        # Start
        self.model.to(device)
        opt = self.get_optimizer(lr)
        self.model.train()

        # Train loop
        pbar = tqdm(enumerate(range(num_epochs)))
        for idx, epoch in pbar:
            opt.zero_grad()
            loss = self.get_train_loss(**kwargs)
            loss.backward()
            opt.step()
            pbar.set_description(f'Epoch {idx}, loss: {loss.item()}')

        # Finish
        self.model.eval()            
class ConditionalScoreMatchingTrainer(Trainer):
    def __init__(self, path: ConditionalProbabilityPath, model: MLPScore, **kwargs):
        super().__init__(model, **kwargs)
        self.path = path

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        z = self.path.p_data.sample(batch_size)
        t = torch.rand(batch_size,1).to(device)
        x = self.path.sample_conditional_path(z,t)
        loss = torch.nn.MSELoss()
        return loss(self.model(x,t), self.path.conditional_score(x,z,t))
class ConditionalFlowMatchingTrainer(Trainer):
    def __init__(self, path: ConditionalProbabilityPath, model: MLPVectorField, **kwargs):
        super().__init__(model, **kwargs)
        self.path = path

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        z = self.path.p_data.sample(batch_size)
        t = torch.rand(batch_size,1).to(device)
        x = self.path.sample_conditional_path(z,t)
        loss = torch.nn.MSELoss()
        return loss(self.model(x,t), self.path.conditional_vector_field(x,z,t))
class ConditionalEquilibriumMatchingTrainer(Trainer):
    def __init__(self, path: ConditionalProbabilityPath, model: MLPEquilibrium, **kwargs):
        super().__init__(model, **kwargs)
        self.path = path

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        z = self.path.p_data.sample(batch_size)
        t = torch.rand(batch_size,1).to(device)
        x = self.path.sample_conditional_path(z,t)
        gamma = 1
        a = 0.9
        b = 1
        loss = torch.nn.MSELoss()
        #c = gamma*self.c_linear(t)
        #c = gamma*self.c_trunc(t,a)
        c = gamma*self.c_piecewise(t,a,b)
        return loss(self.model(x), c*self.path.conditional_vector_field(x,z,t))

    def c_linear(self, t):
        return 1 - t
    
    def c_trunc(self, t, a):
        left = torch.ones_like(t) 
        right = (1 - t) / (1 - a)
        return torch.where(t <= a,left,right)
    
    def c_piecewise(self, t, a, b):
        left = b - (b - 1) / a * t
        right = (1 - t) / (1 - a)
        return torch.where(t <= a, left, right)
class WGFTrainer():
    def __init__(self, T: MLPTransport, h: MLPCritic, P0:Sampleable, Q:Sampleable, V: VariationalObjective, a: int, **kwargs):
        self.T = T
        self.h = h
        self.T_list = []

        self.P0 = P0
        self.Q = Q
        self.V = V
        self.a = a

    def get_lambda_loss(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        for T in self.T_list:   
            with torch.no_grad():
                x = T(x)
        return -torch.mean(self.V.A(self.T(x),self.h) - self.V.B(self.h(z)))

    def get_theta_loss(self, x: torch.Tensor) -> torch.Tensor:
        for T in self.T_list:   
            with torch.no_grad():
                x = T(x)
        return torch.mean(torch.norm(self.T(x) - x,dim=1)**2/(2*self.a) + self.V.A(self.T(x),self.h))

    def get_lambda_optimizer(self, lr: float):
        return torch.optim.Adam(self.h.parameters(), lr=lr)
    
    def get_theta_optimizer(self, lr: float):
        return torch.optim.Adam(self.T.parameters(), lr=lr) 

    def train(self, K: int, num_epochs: int, num_lambda_steps: int, num_theta_steps: int, device: torch.device, batch_size: int, lr: float = 1e-3) -> torch.Tensor:
        # Start
        self.T.to(device)
        self.h.to(device)

        self.T.train()
        self.h.train()

        # Train loop
        kbar = tqdm(enumerate(range(K)), position=0)
        for _ in kbar:
            opt_lambda = self.get_lambda_optimizer(lr)
            opt_theta = self.get_theta_optimizer(lr)
            pbar = tqdm(enumerate(range(num_epochs)), leave=False, position=1)
            for idx, _ in pbar:
                x = self.P0.sample(batch_size)
                z = self.Q.sample(batch_size)

                for _ in range(num_lambda_steps):
                    opt_lambda.zero_grad()
                    lambda_loss = self.get_lambda_loss(x,z)
                    lambda_loss.backward()
                    opt_lambda.step()

                for _ in range(num_theta_steps): 
                    opt_theta.zero_grad()
                    theta_loss = self.get_theta_loss(x)
                    theta_loss.backward()
                    opt_theta.step()
                
                # ============================================================
                # PLOTTING EVERY 100 EPOCHS
                # ============================================================
                """
                if idx % 100 == 0:
                    with torch.no_grad():
                        # 1. Scatter plot of T(x)
                        x_vis = self.P0.sample(500).to(device)    # 2D samples
                        T_x = self.T(x_vis).cpu().numpy()

                        # 2. Real samples from Q
                        z_vis = self.Q.sample(500).cpu().numpy()

                        # 3. Heatmap of h(x) on grid
                        grid_x = torch.linspace(-10, 10, 200)
                        grid_y = torch.linspace(-10, 10, 200)
                        Xg, Yg = torch.meshgrid(grid_x, grid_y, indexing="xy")
                        grid_points = torch.stack([Xg.flatten(), Yg.flatten()], dim=1).to(device)
                        h_vals = self.h(grid_points).cpu().numpy().reshape(200, 200)

                    fig, ax = plt.subplots(figsize=(5, 5))
                    im = ax.imshow(h_vals.T, origin="lower",
                                    extent=(-10, 10, -10, 10),
                                    cmap="viridis",
                                    vmin=0, vmax=1)
                    fig.colorbar(im, ax=ax)
                    ax.scatter(T_x[:, 0], T_x[:, 1], s=5, alpha=0.6)
                    ax.scatter(z_vis[:, 0], z_vis[:, 1], s=5, alpha=0.6)
                    ax.set_title(f"T(x) at epoch {idx}")
                    ax.set_xlim(-10, 10)
                    ax.set_ylim(-10, 10)

                    plt.savefig(f"./epoch_{idx}.pdf")
                """
                pbar.set_postfix({"Epoch": idx, "lambda loss": lambda_loss.item(), "theta loss": theta_loss.item()})
            T_k = copy.deepcopy(self.T) 
            T_k.eval()
            self.T_list.append(T_k)
        self.T.eval()    
        self.h.eval()
        return self.T_list

class LearnedVectorFieldODE(ODE):

    def __init__(self, net: MLPVectorField):
        self.net = net

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: (bs, dim)
            - t: (bs, dim)
        Returns:
            - u_t: (bs, dim)
        """
        return self.net(x, t)
class LearnedEquilibriumODE(ODE):

    def __init__(self, net: MLPVectorField):
        self.net = net

    def drift_coefficient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: (bs, dim)
        Returns:
            - u: (bs, dim)
        """
        return self.net(x)
class LangevinFlowSDE(SDE):
    def __init__(self, flow_model: MLPVectorField, score_model: MLPScore, sigma: float):
        """
        Args:
        - path: the ConditionalProbabilityPath object to which this vector field corresponds
        - z: the conditioning variable, (1, dim)
        """
        super().__init__()
        self.flow_model = flow_model
        self.score_model = score_model
        self.sigma = sigma

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: state at time t, shape (bs, dim)
            - t: time, shape (bs,.)
        Returns:
            - u_t(x|z): shape (batch_size, dim)
        """
        return self.flow_model(x,t) + 0.5 * self.sigma ** 2 * self.score_model(x, t)

    def diffusion_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: state at time t, shape (bs, dim)
            - t: time, shape (bs,.)
        Returns:
            - u_t(x|z): shape (batch_size, dim)
        """
        return self.sigma * torch.randn_like(x)
class ScoreFromVectorField(torch.nn.Module):
    """
    Parameterization of score via learned vector field (for the special case of a Gaussian conditional probability path)
    """
    def __init__(self, vector_field: MLPVectorField, alpha: Alpha, beta: Beta):
        super().__init__()
        self.vector_field = vector_field
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Args:
        - x: (bs, dim)
        Returns:
        - score: (bs, dim)
        """
        return (self.alpha(t)*self.vector_field(x,t)-self.alpha.dt(t)*x)/(self.beta(t)**2*self.alpha.dt(t)-self.alpha(t)*self.beta.dt(t)*self.beta(t))
class VectorFieldFromScore(torch.nn.Module):
    """
    Parameterization of score via learned vector field (for the special case of a Gaussian conditional probability path)
    """
    def __init__(self, score: MLPScore, alpha: Alpha, beta: Beta):
        super().__init__()
        self.score = score
        self.alpha = alpha
        self.beta = beta

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Args:
        - x: (bs, dim)
        Returns:
        - vector_field: (bs, dim)
        """
        return self.alpha.dt(t)/self.alpha(t)*x + (self.beta(t)**2*self.alpha.dt(t)/self.alpha(t)-self.beta.dt(t)*self.beta(t))*self.score(x,t)

def hist2d_samples(samples, ax: Optional[Axes] = None, bins: int = 200, scale: float = 5.0, percentile: int = 99, **kwargs):
    H, xedges, yedges = np.histogram2d(samples[:, 0], samples[:, 1], bins=bins, range=[[-scale, scale], [-scale, scale]])
    
    # Determine color normalization based on the 99th percentile
    cmax = np.percentile(H, percentile)
    cmin = 0.0
    norm = cm.colors.Normalize(vmax=cmax, vmin=cmin)
    
    # Plot using imshow for more control
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    ax.imshow(H.T, extent=extent, origin='lower', norm=norm, **kwargs)
def hist2d_sampleable(sampleable: Sampleable, num_samples: int, ax: Optional[Axes] = None, bins=200, scale: float = 5.0, percentile: int = 99, **kwargs):
    assert sampleable.dim == 2
    if ax is None:
        ax = plt.gca()
    samples = sampleable.sample(num_samples).detach().cpu() # (ns, 2)
    hist2d_samples(samples, ax, bins, scale, percentile, **kwargs)
def scatter_sampleable(sampleable: Sampleable, num_samples: int, ax: Optional[Axes] = None, **kwargs):
    assert sampleable.dim == 2
    if ax is None:
        ax = plt.gca()
    samples = sampleable.sample(num_samples) # (ns, 2)
    ax.scatter(samples[:,0].cpu(), samples[:,1].cpu(), **kwargs)
def kdeplot_sampleable(sampleable: Sampleable, num_samples: int, ax: Optional[Axes] = None, **kwargs):
    assert sampleable.dim == 2
    if ax is None:
        ax = plt.gca()
    samples = sampleable.sample(num_samples) # (ns, 2)
    sns.kdeplot(x=samples[:,0].cpu(), y=samples[:,1].cpu(), ax=ax, **kwargs)
def imshow_density(density: Density, x_bounds: Tuple[float, float], y_bounds: Tuple[float, float], bins: int, ax: Optional[Axes] = None, x_offset: float = 0.0, **kwargs):
    if ax is None:
        ax = plt.gca()
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    x = torch.linspace(x_min, x_max, bins).to(device) + x_offset
    y = torch.linspace(y_min, y_max, bins).to(device)
    X, Y = torch.meshgrid(x, y)
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    density = density.log_density(xy).reshape(bins, bins).T
    im = ax.imshow(density.cpu(), extent=[x_min, x_max, y_min, y_max], origin='lower', **kwargs)
def contour_density(density: Density, bins: int, scale: float, ax: Optional[Axes] = None, x_offset:float = 0.0, **kwargs):
    if ax is None:
        ax = plt.gca()
    x = torch.linspace(-scale + x_offset, scale + x_offset, bins).to(device)
    y = torch.linspace(-scale, scale, bins).to(device)
    X, Y = torch.meshgrid(x, y)
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    density = density.log_density(xy).reshape(bins, bins).T
    im = ax.contour(density.cpu(), origin='lower', **kwargs)

def compare_scores(path,flow_model,score_model):
    #######################
    # Change these values #
    #######################
    num_bins = 30
    num_marginals = 4

    #########################
    # Define score networks #
    #########################
    learned_score_model = score_model
    flow_score_model = ScoreFromVectorField(flow_model, path.alpha, path.beta)


    ###############################
    # Plot score fields over time #
    ###############################
    fig, axes = plt.subplots(2, num_marginals, figsize=(6 * num_marginals, 12))
    axes = axes.reshape((2, num_marginals))

    scale = PARAMS["scale"]
    x_bounds = [-scale,scale]
    y_bounds = [-scale,scale]

    ts = torch.linspace(0.01, 0.9999, num_marginals).to(device)
    xs = torch.linspace(-scale, scale, num_bins).to(device)
    ys = torch.linspace(-scale, scale, num_bins).to(device)
    xx, yy = torch.meshgrid(xs, ys)
    xx = xx.reshape(-1,1)
    yy = yy.reshape(-1,1)
    xy = torch.cat([xx,yy], dim=-1)

    axes[0,0].set_ylabel("Learned with Score Matching", fontsize=12)
    axes[1,0].set_ylabel("Computed from $u_t^{{\\theta}}(x)$", fontsize=12)
    for idx in range(num_marginals):
        t = ts[idx]
        bs = num_bins ** 2
        tt = t.view(1,1).expand(bs, 1)
        
        # Learned scores
        learned_scores = learned_score_model(xy, tt)
        learned_scores_x = learned_scores[:,0]
        learned_scores_y = learned_scores[:,1]

        ax = axes[0, idx]
        ax.quiver(xx.detach().cpu(), yy.detach().cpu(), learned_scores_x.detach().cpu(), learned_scores_y.detach().cpu(), scale=125, alpha=0.5)
        imshow_density(density=path.p_simple, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Reds'))
        imshow_density(density=path.p_data, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Blues'))
        ax.set_title(f'$s_{{t}}^{{\\theta}}$ at t={t.item():.2f}')
        ax.set_xticks([])
        ax.set_yticks([])
        

        # Flow score model
        ax = axes
        flow_scores = flow_score_model(xy,tt)
        flow_scores_x = flow_scores[:,0]
        flow_scores_y = flow_scores[:,1]

        ax = axes[1, idx]
        ax.quiver(xx.detach().cpu(), yy.detach().cpu(), flow_scores_x.detach().cpu(), flow_scores_y.detach().cpu(), scale=125, alpha=0.5)
        imshow_density(density=path.p_simple, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Reds'))
        imshow_density(density=path.p_data, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Blues'))
        ax.set_title(f'$\\tilde{{s}}_{{t}}^{{\\theta}}$ at t={t.item():.2f}')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig("compare_scores.pdf", bbox_inches="tight")
def compare_vector_fields(path,flow_model,score_model):
    #######################
    # Change these values #
    #######################
    num_bins = 30
    num_marginals = 4

    #########################
    # Define score networks #
    #########################
    learned_flow_model = flow_model 
    score_flow_model = VectorFieldFromScore(score_model, path.alpha, path.beta)

    ###############################
    # Plot score fields over time #
    ###############################
    fig, axes = plt.subplots(2, num_marginals, figsize=(6 * num_marginals, 12))
    axes = axes.reshape((2, num_marginals))

    scale = PARAMS["scale"]
    x_bounds = [-scale,scale]
    y_bounds = [-scale,scale]

    ts = torch.linspace(0.01, 0.9999, num_marginals).to(device)
    xs = torch.linspace(-scale, scale, num_bins).to(device)
    ys = torch.linspace(-scale, scale, num_bins).to(device)
    xx, yy = torch.meshgrid(xs, ys)
    xx = xx.reshape(-1,1)
    yy = yy.reshape(-1,1)
    xy = torch.cat([xx,yy], dim=-1)

    axes[0,0].set_ylabel("Learned with Flow Matching", fontsize=12)
    axes[1,0].set_ylabel("Computed from $s_t^{{\\theta}}(x)$", fontsize=12)
    for idx in range(num_marginals):
        t = ts[idx]
        bs = num_bins ** 2
        tt = t.view(1,1).expand(bs, 1)
        
        # Learned scores
        learned_vector_field = learned_flow_model(xy, tt)
        learned_vector_field_x = learned_vector_field[:,0]
        learned_vector_field_y = learned_vector_field[:,1]

        ax = axes[0, idx]
        ax.quiver(xx.detach().cpu(), yy.detach().cpu(), learned_vector_field_x.detach().cpu(), learned_vector_field_y.detach().cpu(), scale=125, alpha=0.5)
        imshow_density(density=path.p_simple, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Reds'))
        imshow_density(density=path.p_data, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Blues'))
        ax.set_title(f'$u_{{t}}^{{\\theta}}$ at t={t.item():.2f}')
        ax.set_xticks([])
        ax.set_yticks([])
        

        # Flow score model
        ax = axes
        score_flows = score_flow_model(xy,tt)
        score_flows_x = score_flows[:,0]
        score_flows_y = score_flows[:,1]

        ax = axes[1, idx]
        ax.quiver(xx.detach().cpu(), yy.detach().cpu(), score_flows_x.detach().cpu(), score_flows_y.detach().cpu(), scale=125, alpha=0.5)
        imshow_density(density=path.p_simple, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Reds'))
        imshow_density(density=path.p_data, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Blues'))
        ax.set_title(f'$\\tilde{{u}}_{{t}}^{{\\theta}}$ at t={t.item():.2f}')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig("compare_vector_fields.pdf", bbox_inches="tight")
def compare_vector_equilibrium(path,flow_model,equilibrium_model):
    #######################
    # Change these values #
    #######################
    num_bins = 30
    num_marginals = 4

    ###############################
    # Plot fields over time #
    ###############################
    fig, axes = plt.subplots(2, num_marginals, figsize=(6 * num_marginals, 12))
    axes = axes.reshape((2, num_marginals))

    scale = PARAMS["scale"]
    x_bounds = [-scale,scale]
    y_bounds = [-scale,scale]

    ts = torch.linspace(0.0, 0.9999, num_marginals).to(device)
    xs = torch.linspace(-scale, scale, num_bins).to(device)
    ys = torch.linspace(-scale, scale, num_bins).to(device)
    xx, yy = torch.meshgrid(xs, ys)
    xx = xx.reshape(-1,1)
    yy = yy.reshape(-1,1)
    xy = torch.cat([xx,yy], dim=-1)

    axes[0,0].set_ylabel("Learned with Flow Matching", fontsize=12)
    axes[1,0].set_ylabel("Learned with Equilibrium Matching", fontsize=12)
    for idx in range(num_marginals):
        t = ts[idx]
        bs = num_bins ** 2
        tt = t.view(1,1).expand(bs, 1)
        
        # Learned scores
        learned_vector_field = flow_model(xy, tt)
        learned_vector_field_x = learned_vector_field[:,0]
        learned_vector_field_y = learned_vector_field[:,1]

        ax = axes[0, idx]
        ax.quiver(xx.detach().cpu(), yy.detach().cpu(), learned_vector_field_x.detach().cpu(), learned_vector_field_y.detach().cpu(), scale=125, alpha=0.5)
        imshow_density(density=path.p_simple, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Reds'))
        imshow_density(density=path.p_data, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Blues'))
        ax.set_title(f'$u_{{t}}^{{\\theta}}$ at t={t.item():.2f}')
        ax.set_xticks([])
        ax.set_yticks([])
        

        # Equilibrium model
        ax = axes
        learned_equilibrium = equilibrium_model(xy)
        learned_equilibrium_x = learned_equilibrium[:,0]
        learned_equilibrium_y = learned_equilibrium[:,1]

        ax = axes[1, idx]
        ax.quiver(xx.detach().cpu(), yy.detach().cpu(), learned_equilibrium_x.detach().cpu(), learned_equilibrium_y.detach().cpu(), scale=125, alpha=0.5)
        imshow_density(density=path.p_simple, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Reds'))
        imshow_density(density=path.p_data, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Blues'))
        ax.set_title(f'$\\tilde{{u}}_{{t}}^{{\\theta}}$ at t={t.item():.2f}')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig("compare_vector_equilibrium.pdf", bbox_inches="tight")

def plot_flow(path,flow_model,num_timesteps,output_file):
    #######################
    # Change these values #
    #######################
    num_samples = 1000
    num_marginals = 3

    ##############
    # Setup Plot #
    ##############
    scale = PARAMS["scale"]
    x_bounds = [-scale,scale]
    y_bounds = [-scale,scale]
    legend_size=24
    markerscale=1.8

    # Setup figure
    fig, axes = plt.subplots(1,3, figsize=(36, 12))

    ###########################################
    # Graph Samples from Learned Marginal ODE #
    ###########################################
    ax = axes[1]

    ax.set_xlim(*x_bounds)
    ax.set_ylim(*y_bounds)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Samples from Learned Marginal ODE", fontsize=20)

    # Plot source and target
    imshow_density(density=path.p_simple, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Reds'))
    imshow_density(density=path.p_data, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Blues'))

    # Construct integrator and plot trajectories
    ode = LearnedVectorFieldODE(flow_model)
    simulator = EulerSimulator(ode)
    x0 = path.p_simple.sample(num_samples) # (num_samples, 2)
    ts = torch.linspace(0.0, 1.0, num_timesteps).view(1,-1,1).expand(num_samples,-1,1).to(device) # (num_samples, nts, 1)
    xts = simulator.simulate_with_trajectory(x0, ts) # (bs, nts, dim)

    # Extract every n-th integration step to plot
    every_n = record_every(num_timesteps=num_timesteps, record_every=num_timesteps // num_marginals)
    xts_every_n = xts[:,every_n,:] # (bs, nts // n, dim)
    ts_every_n = ts[0,every_n] # (nts // n,)
    for plot_idx in range(xts_every_n.shape[1]):
        tt = ts_every_n[plot_idx].item()
        ax.scatter(xts_every_n[:,plot_idx,0].detach().cpu(), xts_every_n[:,plot_idx,1].detach().cpu(), marker='o', alpha=0.5, label=f't={tt:.2f}')

    ax.legend(prop={'size': legend_size}, loc='upper right', markerscale=markerscale)

    ##############################################
    # Graph Trajectories of Learned Marginal ODE #
    ##############################################
    ax = axes[2]
    ax.set_title("Trajectories of Learned Marginal ODE", fontsize=20)
    ax.set_xlim(*x_bounds)
    ax.set_ylim(*y_bounds)
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot source and target
    imshow_density(density=path.p_simple, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Reds'))
    imshow_density(density=path.p_data, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Blues'))

    for traj_idx in range(num_samples // 10):
        ax.plot(xts[traj_idx,:,0].detach().cpu(), xts[traj_idx,:,1].detach().cpu(), alpha=0.5, color='black')

    ################################################
    # Graph Ground-Truth Marginal Probability Path #
    ################################################
    ax = axes[0]
    ax.set_title("Ground-Truth Marginal Probability Path", fontsize=20)
    ax.set_xlim(*x_bounds)
    ax.set_ylim(*y_bounds)
    ax.set_xticks([])
    ax.set_yticks([])

    for plot_idx in range(xts_every_n.shape[1]):
        tt = ts_every_n[plot_idx].unsqueeze(0).expand(num_samples, 1)
        marginal_samples = path.sample_marginal_path(tt)
        ax.scatter(marginal_samples[:,0].detach().cpu(), marginal_samples[:,1].detach().cpu(), marker='o', alpha=0.5, label=f't={tt[0,0].item():.2f}')

    # Plot source and target
    imshow_density(density=path.p_simple, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Reds'))
    imshow_density(density=path.p_data, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Blues'))

    ax.legend(prop={'size': legend_size}, loc='upper right', markerscale=markerscale)
        
    plt.savefig(output_file, bbox_inches="tight")
def plot_equilibrium(path,equilibrium_model,num_timesteps,output_file):
    #######################
    # Change these values #
    #######################
    num_samples = 1000
    num_marginals = 3

    ##############
    # Setup Plot #
    ##############
    scale = PARAMS["scale"]
    x_bounds = [-scale,scale]
    y_bounds = [-scale,scale]
    legend_size=24
    markerscale=1.8

    # Setup figure
    fig, axes = plt.subplots(1,3, figsize=(36, 12))

    ###########################################
    # Graph Samples from Learned Marginal ODE #
    ###########################################
    ax = axes[1]

    ax.set_xlim(*x_bounds)
    ax.set_ylim(*y_bounds)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Samples from Learned Marginal ODE", fontsize=20)

    # Plot source and target
    imshow_density(density=path.p_simple, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Reds'))
    imshow_density(density=path.p_data, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Blues'))

    # Construct integrator and plot trajectories
    ode = LearnedEquilibriumODE(equilibrium_model)
    simulator = GDSimulator(ode)
    x0 = path.p_simple.sample(num_samples) # (num_samples, 2)
    ts = torch.linspace(0.0, 1.0, num_timesteps).view(1,-1,1).expand(num_samples,-1,1).to(device) # (num_samples, nts, 1)
    xts = simulator.simulate_with_trajectory(x0, ts) # (bs, nts, dim)

    # Extract every n-th integration step to plot
    every_n = record_every(num_timesteps=num_timesteps, record_every=num_timesteps // num_marginals)
    xts_every_n = xts[:,every_n,:] # (bs, nts // n, dim)
    ts_every_n = ts[0,every_n] # (nts // n,)
    for plot_idx in range(xts_every_n.shape[1]):
        tt = ts_every_n[plot_idx].item()
        ax.scatter(xts_every_n[:,plot_idx,0].detach().cpu(), xts_every_n[:,plot_idx,1].detach().cpu(), marker='o', alpha=0.5, label=f't={tt:.2f}')

    ax.legend(prop={'size': legend_size}, loc='upper right', markerscale=markerscale)

    ##############################################
    # Graph Trajectories of Learned Marginal ODE #
    ##############################################
    ax = axes[2]
    ax.set_title("Trajectories of Learned Marginal ODE", fontsize=20)
    ax.set_xlim(*x_bounds)
    ax.set_ylim(*y_bounds)
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot source and target
    imshow_density(density=path.p_simple, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Reds'))
    imshow_density(density=path.p_data, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Blues'))

    for traj_idx in range(num_samples // 10):
        ax.plot(xts[traj_idx,:,0].detach().cpu(), xts[traj_idx,:,1].detach().cpu(), alpha=0.5, color='black')

    ################################################
    # Graph Ground-Truth Marginal Probability Path #
    ################################################
    ax = axes[0]
    ax.set_title("Ground-Truth Marginal Probability Path", fontsize=20)
    ax.set_xlim(*x_bounds)
    ax.set_ylim(*y_bounds)
    ax.set_xticks([])
    ax.set_yticks([])

    for plot_idx in range(xts_every_n.shape[1]):
        tt = ts_every_n[plot_idx].unsqueeze(0).expand(num_samples, 1)
        marginal_samples = path.sample_marginal_path(tt)
        ax.scatter(marginal_samples[:,0].detach().cpu(), marginal_samples[:,1].detach().cpu(), marker='o', alpha=0.5, label=f't={tt[0,0].item():.2f}')

    # Plot source and target
    imshow_density(density=path.p_simple, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Reds'))
    imshow_density(density=path.p_data, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Blues'))

    ax.legend(prop={'size': legend_size}, loc='upper right', markerscale=markerscale)
        
    plt.savefig(output_file, bbox_inches="tight")
def plot_vwgf(P0,Q,vwgf_model,output_file):
    #######################
    # Change these values #
    #######################
    num_samples = 1000
    num_marginals = 3

    ##############
    # Setup Plot #
    ##############
    scale = PARAMS["scale"]
    x_bounds = [-scale,scale]
    y_bounds = [-scale,scale]
    legend_size=24
    markerscale=1.8

    # Setup figure
    fig, axes = plt.subplots(1,2, figsize=(24, 12))

    ###########################################
    # Graph Samples from Learned Marginal ODE #
    ###########################################
    ax = axes[0]

    ax.set_xlim(*x_bounds)
    ax.set_ylim(*y_bounds)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Samples from Learned Marginal ODE", fontsize=20)

    # Plot source and target
    imshow_density(density=P0, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.5, cmap=plt.get_cmap('Reds'))
    imshow_density(density=Q, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.5, cmap=plt.get_cmap('Blues'))

    # Construct integrator and plot trajectories
    x = P0.sample(num_samples).to(device)
    trajectory = [x.clone()]

    for T_k in vwgf_model:
        with torch.no_grad():
            x = T_k(x)
        trajectory.append(x.clone())

    for k in range(0, len(trajectory), 25):     # step by 5
        xk = trajectory[k]
        ax.scatter(
            xk[:,0].cpu(),
            xk[:,1].cpu(),
            alpha=0.4,
            label=f"T_{k}"
        )

    ax.legend(prop={'size': legend_size}, loc='upper right', markerscale=markerscale)

    ##############################################
    # Graph Trajectories of Learned Marginal ODE #
    ##############################################
    ax = axes[1]
    ax.set_title("Trajectories of Learned Marginal ODE", fontsize=20)
    ax.set_xlim(*x_bounds)
    ax.set_ylim(*y_bounds)
    ax.set_xticks([])
    ax.set_yticks([])

    num_samples = trajectory[0].shape[0]

    # Convert list of tensors into a big tensor of shape (num_steps, num_samples, dim)
    traj_tensor = torch.stack(trajectory, dim=0)   # (num_steps, num_samples, 2)

    # Plot each particle trajectory
    for i in range(0, num_samples, 10):
        xs = traj_tensor[:, i, 0].cpu()
        ys = traj_tensor[:, i, 1].cpu()
        ax.plot(xs, ys, alpha=0.4, color="black")

    # Plot source and target
    imshow_density(density=P0, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.5, cmap=plt.get_cmap('Reds'))
    imshow_density(density=Q, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.5, cmap=plt.get_cmap('Blues'))
        
    plt.savefig(output_file, bbox_inches="tight")
def plot_score(path,flow_model,score_model,num_timesteps,output_file):
    #######################
    # Change these values #
    #######################
    num_samples = 1000
    num_marginals = 3
    sigma = 2.0 # Don't set sigma too large or you'll get numerical issues!

    ##############
    # Setup Plot #
    ##############
    scale = PARAMS["scale"]
    x_bounds = [-scale,scale]
    y_bounds = [-scale,scale]
    legend_size = 24
    markerscale = 1.8

    # Setup figure
    fig, axes = plt.subplots(1,3, figsize=(36, 12))

    ###########################################
    # Graph Samples from Learned Marginal SDE #
    ###########################################
    ax = axes[1]
    ax.set_title("Samples from Learned Marginal SDE", fontsize=20)
    ax.set_xlim(*x_bounds)
    ax.set_ylim(*y_bounds)
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot source and target
    imshow_density(density=path.p_simple, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Reds'))
    imshow_density(density=path.p_data, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Blues'))


    # Construct integrator and plot trajectories
    sde = LangevinFlowSDE(flow_model, score_model, sigma)
    simulator = EulerMaruyamaSimulator(sde)
    x0 = path.p_simple.sample(num_samples) # (num_samples, 2)
    ts = torch.linspace(0.01, 1.0, num_timesteps).view(1,-1,1).expand(num_samples,-1,1).to(device) # (num_samples, nts, 1)
    xts = simulator.simulate_with_trajectory(x0, ts) # (bs, nts, dim)

    # Extract every n-th integration step to plot
    every_n = record_every(num_timesteps=num_timesteps, record_every=num_timesteps // num_marginals)
    xts_every_n = xts[:,every_n,:] # (bs, nts // n, dim)
    ts_every_n = ts[0,every_n] # (nts // n,)
    for plot_idx in range(xts_every_n.shape[1]):
        tt = ts_every_n[plot_idx].item()
        ax.scatter(xts_every_n[:,plot_idx,0].detach().cpu(), xts_every_n[:,plot_idx,1].detach().cpu(), marker='o', alpha=0.5, label=f't={tt:.2f}')

    ax.legend(prop={'size': legend_size}, loc='upper right', markerscale=markerscale)

    ###############################################
    # Graph Trajectories of Learned Marginal SDE  #
    ###############################################
    ax = axes[2]
    ax.set_title("Trajectories of Learned Marginal SDE", fontsize=20)
    ax.set_xlim(*x_bounds)
    ax.set_ylim(*y_bounds)
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot source and target
    imshow_density(density=path.p_simple, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Reds'))
    imshow_density(density=path.p_data, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Blues'))

    for traj_idx in range(num_samples // 10):
        ax.plot(xts[traj_idx,:,0].detach().cpu(), xts[traj_idx,:,1].detach().cpu(), alpha=0.5, color='black')

    ################################################
    # Graph Ground-Truth Marginal Probability Path #
    ################################################
    ax = axes[0]
    ax.set_title("Ground-Truth Marginal Probability Path", fontsize=20)
    ax.set_xlim(*x_bounds)
    ax.set_ylim(*y_bounds)
    ax.set_xticks([])
    ax.set_yticks([])

    for plot_idx in range(xts_every_n.shape[1]):
        tt = ts_every_n[plot_idx].unsqueeze(0).expand(num_samples, 1)
        marginal_samples = path.sample_marginal_path(tt)
        ax.scatter(marginal_samples[:,0].detach().cpu(), marginal_samples[:,1].detach().cpu(), marker='o', alpha=0.5, label=f't={tt[0,0].item():.2f}')

    # Plot source and target
    imshow_density(density=path.p_simple, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Reds'))
    imshow_density(density=path.p_data, x_bounds=x_bounds, y_bounds=y_bounds, bins=200, ax=ax, vmin=-10, alpha=0.25, cmap=plt.get_cmap('Blues'))

    ax.legend(prop={'size': legend_size}, loc='upper right', markerscale=markerscale)
        
    plt.savefig(output_file, bbox_inches="tight")
def wasserstein_distance(samples, target_samples):
    n = len(samples)
    # Uniform weights for empirical distributions
    a = torch.ones(n,device=device) / n
    b = torch.ones(n,device=device) / n

    # Pairwise cost matrix (Euclidean distances)
    M = ot.dist(samples, target_samples, metric='euclidean')  # shape (n, n)

    # --- Solve optimal transport ---
    res = ot.solve(M, a, b)
    return res.value
def simulate_flow(path,flow_model,num_samples,num_timesteps):
    # Construct integrator and plot trajectories
    ode = LearnedVectorFieldODE(flow_model)
    simulator = EulerSimulator(ode)
    x0 = path.p_simple.sample(num_samples) # (num_samples, 2)
    ts = torch.linspace(0.01, 1.0, num_timesteps).view(1,-1,1).expand(num_samples,-1,1).to(device) # (num_samples, nts, 1)
    xts = simulator.simulate_with_trajectory(x0, ts) # (bs, nts, dim)
    x1 = xts[:,-1,:]
    return x1
def simulate_score(path,flow_model,score_model,num_samples,num_timesteps):
    # Construct integrator and plot trajectories
    sigma = 2.0
    sde = LangevinFlowSDE(flow_model,score_model,sigma)
    simulator = EulerMaruyamaSimulator(sde)
    x0 = path.p_simple.sample(num_samples) # (num_samples, 2)
    ts = torch.linspace(0.01, 1.0, num_timesteps).view(1,-1,1).expand(num_samples,-1,1).to(device) # (num_samples, nts, 1)
    xts = simulator.simulate_with_trajectory(x0, ts) # (bs, nts, dim)
    x1 = xts[:,-1,:]
    return x1
def plot_results(results):
    # -------------------------------------------------------
    # Plot Wasserstein vs. timesteps (from `results` dict)
    # -------------------------------------------------------
    plt.figure(figsize=(6,5))

    # plot flow model curve
    if "flow_deterministic" in results:
        steps, Ws = results["flow_deterministic"]
        plt.plot(steps, Ws, marker="o", linewidth=2, label="Flow model (deterministic)")

    if "flow_stochastic" in results:
        steps, Ws = results["flow_stochastic"]
        plt.plot(steps, Ws, marker="o", linewidth=2, label="Flow model (stochastic)")

    # plot score model curve
    if "score_stochastic" in results:
        steps, Ws = results["score_stochastic"]
        plt.plot(steps, Ws, marker="s", linewidth=2, label="Score model (stochastic)")

    if "score_deterministic" in results:
        steps, Ws = results["score_deterministic"]
        plt.plot(steps, Ws, marker="s", linewidth=2, label="Score model (deterministic)")

    plt.ylabel("Error (Wasserstein-1 distance)", fontsize=14)
    plt.xlabel("Iterations (timesteps)", fontsize=14)
    plt.title("Convergence of Simulation Error vs Iterations", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    # -------------------------------------------------------
    # Save plot as PDF
    # -------------------------------------------------------
    plt.savefig("./convergence.pdf", bbox_inches="tight")