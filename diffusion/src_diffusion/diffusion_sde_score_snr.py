"""

Author: Yarong Chen
Date : 12 March 2026

Key concept:
------------
This file implements the core equations for a score-based diffusion model
under the Variance Preserving SDE (VP-SDE) formulation.

Forward SDE:
    dx = -1/2 * beta(t) * x dt + sqrt(beta(t)) dW

Marginal perturbation kernel:
    q(x_t | x_0) = N( mean(t) * x_0, std(t)^2 I )

For VP-SDE:
    log mean coeff = -1/4 (beta_max - beta_min) t^2 - 1/2 beta_min t
    mean(t)        = exp(log mean coeff)
    std(t)         = sqrt(1 - exp(2 * log mean coeff))

Training target (denoising score matching):
    score*(x_t, t | x_0) = \nabla_{x_t} log q(x_t | x_0)
                         = -(x_t - mean(t)x_0) / std(t)^2
                         = -z / std(t),  where x_t = mean(t)x_0 + std(t)z

Reverse-time SDE:
    dx = [ f(x,t) - g(t)^2 * score(x,t) ] dt + g(t) dW_bar

This file is the SDM/SDE counterpart of diffusion_gaussian.py in DDPM.

"""

import enum
import math

import numpy as np
import torch as th

from .nn import mean_flat


class SDEType(enum.Enum):
    VP = enum.auto()


class LossType(enum.Enum):
    SCORE_MSE = enum.auto()
    SCORE_MSE_LIKELIHOOD_WEIGHTED = enum.auto()
    SCORE_MSE_SNR_WEIGHTED = enum.auto() #snr_add


class VPSDE:
    """
    Variance Preserving SDE:
        dx = -1/2 beta(t) x dt + sqrt(beta(t)) dW

    This class plays a role similar to GaussianDiffusion in DDPM, but in
    continuous time. The most important functions are:
        - q_sample()         : forward perturbation x_0 -> x_t
        - training_losses()  : score matching loss
        - p_sample_loop()    : reverse-time sampling (Euler-Maruyama)
    """

    def __init__(
        self,
        *,
        beta_min=0.1,
        beta_max=20.0,
        loss_type=LossType.SCORE_MSE,
        rescale_timesteps=False,
        sampling_eps=1e-3,
        num_timesteps=1000,
        cfg_guidance_scale=1.5, #new_add
    ):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        self.sampling_eps = sampling_eps

        self.T = 1.0
        self.sde_type = SDEType.VP
        self.num_timesteps = num_timesteps
        self.cfg_guidance_scale = cfg_guidance_scale #new_add

    def beta(self, t):
        """
        Linear noise schedule:
            beta(t) = beta_min + t * (beta_max - beta_min)

        :param t: tensor of shape [N], continuous time in [0, 1]
        :return: tensor of shape [N]
        """
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def snr_weight(self, t): #snr_add
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        mean_coeff_sq = th.exp(2 * log_mean_coeff)
        var = th.clamp(1 - mean_coeff_sq, min=1e-8)
        snr = mean_coeff_sq / var
        weight = 1.0 / th.sqrt(th.clamp(snr, min=1e-3, max=1e3))
        return weight

    def marginal_log_mean_coeff(self, t):
        """
        For VP-SDE:
            log mean coeff(t)
            = -1/4 (beta_max - beta_min) t^2 - 1/2 beta_min t

        so that
            mean(t) = exp(log_mean_coeff)
        """
        return -0.25 * (self.beta_max - self.beta_min) * t ** 2 - 0.5 * self.beta_min * t

    def marginal_prob(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        q(x_t | x_0) = N(mean, std^2 I)
        mean = exp(log_mean_coeff(t)) * x_0
        std  = sqrt(1 - exp(2 * log_mean_coeff(t)))

        :param x_start: the [N x C x ...] tensor of clean inputs.
        :param t: a [N] tensor of continuous times in [0, 1].
        :return: a tuple (mean, std), where
                 - mean has shape x_start.shape
                 - std  has shape [N]
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        mean = _expand_tensor_shape(th.exp(log_mean_coeff), x_start.shape) * x_start
        std = th.sqrt(th.clamp(1.0 - th.exp(2.0 * log_mean_coeff), min=1e-12))
        return mean, std

    def sde(self, x, t):
        """
        Forward SDE coefficients:
            drift     = -1/2 beta(t) x
            diffusion = sqrt(beta(t))

        :param x: tensor [N x C x ...]
        :param t: tensor [N]
        :return: (drift, diffusion)
                 - drift shape     = x.shape
                 - diffusion shape = [N]
        """
        beta_t = self.beta(t)
        drift = -0.5 * _expand_tensor_shape(beta_t, x.shape) * x
        diffusion = th.sqrt(beta_t)
        return drift, diffusion

    def q_mean_variance(self, x_start, t):
        """
        Alias similar to DDPM style.

        :param x_start: clean tensor
        :param t: continuous time tensor [N]
        :return: (mean, variance, log_variance)
        """
        mean, std = self.marginal_prob(x_start, t)
        variance = _expand_tensor_shape(std ** 2, x_start.shape)
        log_variance = th.log(th.clamp(variance, min=1e-20))
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data to arbitrary continuous time t.

        x_t = mean(t) * x_0 + std(t) * z
        where z ~ N(0, I)

        :param x_start: the initial data batch.
        :param t: a [N] tensor of continuous times in [0, 1].
        :param noise: if specified, Gaussian noise of same shape as x_start.
        :return: a noisy version of x_start at time t.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape

        mean, std = self.marginal_prob(x_start, t)
        return mean + _expand_tensor_shape(std, x_start.shape) * noise

    def prior_sampling(self, shape, device):
        """
        Sample from the prior p_T(x) ~ N(0, I).
        """
        return th.randn(*shape, device=device)

    def prior_logp(self, z):
        """
        Compute log probability under the standard normal prior.

        :param z: [N x C x ...]
        :return: [N]
        """
        dims = list(range(1, len(z.shape)))
        N = np.prod(z.shape[1:])
        return -N / 2.0 * np.log(2 * np.pi) - th.sum(z ** 2, dim=dims) / 2.0

    def score_target(self, x_start, x_t, t, noise=None):
        """
        Compute the analytical score target:

            score*(x_t, t | x_0)
            = -(x_t - mean(t)x_0) / std(t)^2

        If x_t = mean(t)x_0 + std(t)z, then equivalently:
            score* = -z / std(t)

        :param x_start: clean sample x_0
        :param x_t: perturbed sample x_t
        :param t: continuous time tensor [N]
        :param noise: optional z used in x_t construction
        :return: true score tensor, same shape as x_start
        """
        mean, std = self.marginal_prob(x_start, t)

        if noise is not None:
            return -noise / _expand_tensor_shape(std, x_start.shape)

        return -(x_t - mean) / _expand_tensor_shape(std ** 2, x_start.shape)

    def _scale_timesteps(self, t):
        """
        Keep the same interface style as DDPM code.

        If rescale_timesteps=True, map continuous t in [0,1] to [0,1000].
        This is often convenient because the original U-Net timestep embedding
        was designed around DDPM-style 0..1000 ranges.
        """
        if self.rescale_timesteps:
            return t.float() * 1000.0
        return t

    def reverse_sde(self, x, t, score):
        """
        Reverse-time SDE coefficients:

            dx = [ f(x,t) - g(t)^2 score(x,t) ] dt + g(t) dW_bar

        where for VP-SDE:
            f(x,t) = -1/2 beta(t) x
            g(t)   = sqrt(beta(t))

        :param x: current state x_t
        :param t: current time [N]
        :param score: model-predicted score, same shape as x
        :return: (reverse_drift, diffusion)
        """
        drift, diffusion = self.sde(x, t)
        reverse_drift = drift - _expand_tensor_shape(diffusion ** 2, x.shape) * score
        return reverse_drift, diffusion

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        This is not a true Gaussian posterior as in DDPM, but a convenience
        wrapper so that the interface resembles diffusion_gaussian.py.

        Here we compute:
            - predicted score
            - predicted x_start from score
            - reverse drift / diffusion

        Using the VP-SDE relation:
            x_t = mean(t)x_0 + std(t)z
            score = -z / std(t)

        Therefore:
            z = - std(t) * score
            x_0 = (x_t - std(t) z) / mean(t)
                = (x_t + std(t)^2 score) / mean(t)

        :return: dict with keys:
                 - "score"
                 - "pred_xstart"
                 - "drift"
                 - "diffusion"
        """
        if model_kwargs is None:
            model_kwargs = {}
        
        cond = model_kwargs.get("cond", None) #new_add

        if cond is not None and self.cfg_guidance_scale != 1.0:
            zero_cond = th.zeros_like(cond)

            score_cond = model(
                x,
                self._scale_timesteps(t),
                cond=cond,
            )
            score_uncond = model(
                x,
                self._scale_timesteps(t),
                cond=zero_cond,
            )

            score = score_uncond + self.cfg_guidance_scale * (score_cond - score_uncond)
        else:
            score = model(x, self._scale_timesteps(t), **model_kwargs)
        
        assert score.shape == x.shape

        mean_coeff = th.exp(self.marginal_log_mean_coeff(t))
        std = th.sqrt(
            th.clamp(1.0 - th.exp(2.0 * self.marginal_log_mean_coeff(t)), min=1e-12)
        )


        # x0 = (x_t - std * eps) / mean_coeff
        pred_xstart = (
            x + _expand_tensor_shape(std ** 2, x.shape) * score
        ) / _expand_tensor_shape(mean_coeff, x.shape)

        if denoised_fn is not None:
            pred_xstart = denoised_fn(pred_xstart)
        if clip_denoised:
             pred_xstart = pred_xstart.clamp(-1, 1)

        drift, diffusion = self.reverse_sde(x, t, score)

        return {
            "score": score,
            "pred_xstart": pred_xstart,
            "drift": drift,
            "diffusion": diffusion,
        }

    def p_sample(
        self,
        model,
        x,
        t,
        dt,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
    ):
        """
        Take one reverse-time Euler-Maruyama step.

        Reverse-time SDE:
            dx = [f(x,t) - g(t)^2 score(x,t)] dt + g(t) dW_bar

        Since we sample backward in time, dt should be positive here and the
        actual numerical update uses:
            x_{t-dt} = x_t - reverse_drift * dt + diffusion * sqrt(dt) * noise

        :param model: score model
        :param x: current tensor at time t
        :param t: current time tensor [N]
        :param dt: positive scalar step size
        :return: dict with:
                 - "sample"
                 - "pred_xstart"
                 - "score"
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        noise = th.randn_like(x)
        diffusion = out["diffusion"]
        nonzero_mask = (
            (t > self.sampling_eps).float().view(-1, *([1] * (len(x.shape) - 1)))
        ) #stop add noise when the last step t=0
        sample = (
            x
            - out["drift"] * dt
            + nonzero_mask * _expand_tensor_shape(diffusion, x.shape) * math.sqrt(dt) * noise
        )

        return {
            "sample": sample,
            "pred_xstart": out["pred_xstart"],
            "score": out["score"],
        }

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        num_steps=None,
    ):
        """
        Generate samples from the model by numerically solving the reverse-time
        SDE with Euler-Maruyama.

        :param model: the score model.
        :param shape: sample shape (N, C, H, W)
        :param noise: optional initial latent at t=T
        :param num_steps: number of reverse integration steps
        :return: final sample tensor
        """
        final = None
        if num_steps is None:
            num_steps = self.num_timesteps
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            num_steps=num_steps,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        num_steps=None,
    ):
        """
        Yield intermediate samples during reverse-time SDE sampling.

        This is the SDM/SDE analogue of p_sample_loop_progressive() in DDPM.
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))

        if noise is not None:
            img = noise
        else:
            img = self.prior_sampling(shape, device)
        if num_steps is None:
           num_steps = self.num_timesteps
       
        times = th.linspace(self.T, self.sampling_eps, num_steps, device=device)
        dt = (self.T - self.sampling_eps) / (num_steps - 1)

        if progress:
            from tqdm.auto import tqdm
            times = tqdm(times)

        for time_scalar in times:
            t = th.ones(shape[0], device=device) * time_scalar
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    dt=dt,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute score matching losses at arbitrary continuous time t.

        Training procedure:
            1. sample z ~ N(0, I)
            2. construct x_t = mean(t)x_0 + std(t)z
            3. predict score s_theta(x_t, t)
            4. match analytical score:
                   score* = -z / std(t)

        Unweighted objective:
            L = || s_theta(x_t,t) - score*(x_t,t|x_0) ||^2
        Weighted objective:
            L = sigma(t)^2 * (|| s_theta(x_t,t) - score*(x_t,t|x_0) ||^2)    

        Likelihood-weighted variant:
            L = g(t)^2 || s_theta - score* ||^2
        where g(t)^2 = beta(t) for VP-SDE.

        :param model: score model
        :param x_start: clean input x_0
        :param t: continuous times [N], typically Uniform(eps, 1)
        :param model_kwargs: optional conditioning dict
        :param noise: optional Gaussian noise
        :return: dict with at least key "loss"
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        
        # continuous t in [0, 1] and DDPM-style discrete timesteps.
        if t.dtype in (th.int32, th.int64, th.long):
            t = t.float() / max(self.num_timesteps - 1, 1)
        else:
            t = t.float()

        t = th.clamp(t, min=self.sampling_eps, max=self.T)

        x_t = self.q_sample(x_start, t, noise=noise) #new_add

        # model predicts epsilon instead of score
        score_pred = model(x_t, self._scale_timesteps(t), **model_kwargs)
        score_true = self.score_target(x_start, x_t, t, noise=noise)

        assert score_pred.shape == score_true.shape == x_start.shape

        terms = {}
        sq_error = mean_flat((score_pred - score_true) ** 2)

        if self.loss_type == LossType.SCORE_MSE:
            terms["score_mse"] = sq_error
            terms["loss"] = sq_error
        elif self.loss_type == LossType.SCORE_MSE_LIKELIHOOD_WEIGHTED:
            weight = self.beta(t)
            terms["score_mse"] = sq_error
            terms["loss"] = weight * sq_error
        elif self.loss_type == LossType.SCORE_MSE_SNR_WEIGHTED:
            weight = self.snr_weight(t)
            terms["score_mse"] = sq_error
            terms["loss"] = weight * sq_error
        else:
            raise NotImplementedError(self.loss_type)

        # Diagnostics
        terms["x_t"] = x_t
        terms["eps_pred"] = score_pred
        terms["eps_true"] = score_true

        mean_coeff = th.exp(self.marginal_log_mean_coeff(t))
        std = th.sqrt(
            th.clamp(1.0 - th.exp(2.0 * self.marginal_log_mean_coeff(t)), min=1e-12)
        )

        pred_xstart = (
            x_t + _expand_tensor_shape(std ** 2, x_t.shape) * score_pred
        ) / _expand_tensor_shape(mean_coeff, x_t.shape)

        terms["pred_xstart"] = pred_xstart
        terms["xstart_mse"] = mean_flat((pred_xstart - x_start) ** 2)

        return terms


def _expand_tensor_shape(x, broadcast_shape):
    """
    Expand a [N] tensor into [N, 1, 1, ...] to match broadcast_shape.
    """
    while len(x.shape) < len(broadcast_shape):
        x = x[..., None]
    return x


def get_named_sde(sde_name, **kwargs):
    """
    Factory function, analogous to get_named_beta_schedule / GaussianDiffusion setup.
    """
    if sde_name.lower() == "vp":
        return VPSDE(**kwargs)
    else:
        raise NotImplementedError(f"unknown sde type: {sde_name}")
