# AgeWeaver Research Methodology

## 1. Problem Definition and Research Questions

Facial age progression (also known as age synthesis or face aging) is the task of synthetically altering a portrait of an individual to depict them at an older target age while strictly preserving their unique facial identity, bone structure, and photographic realism.

In recent literature, text-to-image and image-to-image diffusion models have emerged as state-of-the-art generative priors. However, standard diffusion conditioning faces a fundamental tension:
- **Over-editing**: High denoising strengths or generic age prompts corrupt the subject's unique biometric identity, transforming them into a different individual who merely matches the target age bracket.
- **Under-editing**: Low denoising strengths preserve identity but fail to manifest realistic age-related anatomical changes (e.g. skin thinning, wrinkling, silvering hair, changes in jawline elasticity).

We investigate the following core research questions:
- **RQ1**: Does face-attribute-aware structured prompt refinement improve target-age compliance compared to naive generic age prompts?
- **RQ2**: How does generation strength affect the trade-off between ArcFace identity cosine similarity and target-age progression?
- **RQ3**: Can an age-delta adaptive control policy dynamically modulate diffusion parameters to improve this trade-off?
- **RQ4**: Is cyclic round-trip reconstruction ($I_{\text{src}} \to I_{\text{tgt}} \to I_{\text{src}}$) an effective proxy for identity preservation in diffusion-based aging?

## 2. Theoretical Framework & FaceTT Foundation

The theoretical baseline for this work is inspired by **Face Time Traveller (FaceTT)** (Kar et al., 2026). FaceTT identifies key limitations in off-the-shelf diffusion models and introduces four conceptual mechanisms:

1. **Face-Attribute-Aware Prompt Refinement**: Instead of simple textual prompts, the input condition is enriched with decomposed anatomical descriptors corresponding to age brackets, skin characteristics, and hair appearance, while establishing identity invariant anchors.
2. **Angular Inversion**: A tuning-free spherical latent inversion mechanism designed to preserve geometric angles in the latent space during reconstruction.
3. **Adaptive Attention Control (AAC)**: Real-time injection and modulation of spatial self-attention and cross-attention maps during the reverse diffusion trajectory to anchor identity-bearing regions while permitting localized texture editing.
4. **Cyclic Identity Similarity**: A self-supervised evaluation protocol where an image is aged forward and then inverted backward to the source age, verifying whether the round-trip preserves identity in the embedding space of a deep face recognition network.

## 3. Mathematical Formulation

### 3.1 Denoising Diffusion Probabilistic Models (DDPM / DDIM)
Let $x_0$ be the source image. In latent diffusion, the image is encoded into latent space $z_0 = \mathcal{E}(x_0)$.
The forward process perturbs $z_0$ with Gaussian noise across continuous or discrete timesteps $t \in [0, T]$:
$$q(z_t | z_0) = \mathcal{N}\left(z_t; \sqrt{\bar{\alpha}_t} z_0, (1 - \bar{\alpha}_t) \mathbf{I}\right)$$

In image-to-image synthesis, instead of sampling from pure noise $z_T \sim \mathcal{N}(0, \mathbf{I})$, the process starts at an intermediate timestep $t_0 = \lfloor \sigma \cdot T \rfloor$, where $\sigma \in [0, 1]$ is the denoising strength:
$$z_{t_0} = \sqrt{\bar{\alpha}_{t_0}} z_0 + \sqrt{1 - \bar{\alpha}_{t_0}} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

The reverse process estimates the added noise via the trained UNet $\epsilon_\theta(z_t, t, c)$ conditioned on text embedding $c = \tau_\theta(y)$:
$$\tilde{\epsilon}_\theta(z_t, t, c) = \epsilon_\theta(z_t, t, \emptyset) + s \cdot \left(\epsilon_\theta(z_t, t, c) - \epsilon_\theta(z_t, t, \emptyset)\right)$$
where $s \ge 1.0$ is the Classifier-Free Guidance (CFG) scale.

### 3.2 AgeWeaver Adaptive Denoising Policy
Rather than holding $\sigma$ and $s$ constant across all age requests, AgeWeaver defines an age-difference metric:
$$\Delta_{\text{age}} = |\text{target\_age} - \text{source\_age}|$$

The adaptive controller assigns:
$$\sigma(\Delta) = \sigma_{\min} + (\sigma_{\max} - \sigma_{\min}) \cdot \min\left(1.0, \frac{\Delta_{\text{age}}}{\Delta_{\text{ref}}}\right)$$
$$s(\Delta) = s_{\min} + (s_{\max} - s_{\min}) \cdot \min\left(1.0, \frac{\Delta_{\text{age}}}{\Delta_{\text{ref}}}\right)$$
where $\Delta_{\text{ref}} = 50.0$, $\sigma_{\min} = 0.32$, $\sigma_{\max} = 0.58$, $s_{\min} = 7.0$, $s_{\max} = 8.5$.

### 3.3 Quantitative Evaluation Metrics

#### A. ArcFace Identity Cosine Similarity ($S_{\text{id}}$)
Using the pre-trained ArcFace ResNet-50 backbone ($\mathcal{F}_{\text{arc}}$):
$$\mathbf{e} = \frac{\mathcal{F}_{\text{arc}}(I)}{\|\mathcal{F}_{\text{arc}}(I)\|_2} \in \mathbb{S}^{511}$$
$$S_{\text{id}}(I_{\text{src}}, I_{\text{gen}}) = \mathbf{e}_{\text{src}}^\top \mathbf{e}_{\text{gen}}$$
Scores $> 0.60$ generally indicate strong identity preservation in the ArcFace metric space.

#### B. Target-Age Mean Absolute Error ($E_{\text{age}}$)
Using the pre-trained InsightFace `genderage` estimation model ($\mathcal{G}_{\text{age}}$):
$$\hat{y}_{\text{age}} = \mathcal{G}_{\text{age}}(I_{\text{gen}})$$
$$E_{\text{age}} = | \hat{y}_{\text{age}} - y_{\text{target}} |$$

#### C. Cyclic Identity Similarity ($S_{\text{cyc}}$)
Let $I_{\text{recon}}$ be the result of inverting $I_{\text{gen}}$ back to $y_{\text{source}}$:
$$S_{\text{cyc}} = \mathbf{e}_{\text{src}}^\top \mathbf{e}_{\text{recon}}$$
Measures whether the generation cycle is bijective and preserves the original identity trajectory.

