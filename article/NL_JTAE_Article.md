# A Nonlinear Joint Sparse-Core Tucker Autoencoder for Native Multi-Resolution Fusion of Sentinel-2 and Sentinel-3 Imagery

*Draft manuscript — target venues: IEEE Transactions on Geoscience and Remote Sensing / MDPI Remote Sensing.*

---

## Abstract

The Sentinel-2 (S2) and Sentinel-3 (S3) missions offer complementary observations of the land surface: S2 provides decametric spatial resolution with 13 spectral bands split across three native grids (10 m, 20 m, 60 m), whereas the S3 OLCI instrument provides 21 spectral bands at 300 m with quasi-daily revisit. We address the reconstruction of a super-resolved image carrying the 21 OLCI bands at 10 m resolution by fusing four observation streams of heterogeneous spatial and spectral dimensions. Our method, the Multi-Stream Nonlinear Joint Tucker Autoencoder (MS-NL-JTAE), rests on the hypothesis that all streams are degraded projections of a single latent scene admitting a shared *sparse Tucker core tensor* $\mathcal{G}$. Three design principles follow. First, the exact factorization of the resolution ratio, $30 = 2\times3\times5$, lets the encoder pyramid coincide with the native sensor grids, so that each stream is ingested at its own resolution without any prior resampling of the data. Second, the sparsity of the core is enforced *structurally* by channel-wise learned soft-thresholding — the proximal operator of the $\ell_1$ norm — rather than by a penalty alone. Third, the architecture routes all 21-band spectral information exclusively through the core, while skip connections carry only S2-derived geometry, making $\mathcal{G}$ the joint spatial–spectral code by construction. Training is fully self-supervised through differentiable sensor models (PSF and SRF); the ground truth is used for evaluation only. On the Pavia University benchmark under a Wald protocol with resolution ratio 30 and a strictly identical information regime, the proposed method outperforms a coupled sparse Tucker ALS baseline and a coupled Tucker autoencoder by +2.8 to +3.1 dB PSNR, with clear margins on ERGAS, SSIM and UIQI; a supervised oracle bound (37.7 dB, SAM 2.2°) shows the architecture's spectral ceiling, and we analyse the residual spectral-angle advantage of rigid multilinear decoders in the self-supervised regime as a prior, not an architectural, effect. We further show that the learned decomposition — permanent spatial factors versus a time-varying sparse core — provides a principled pathway to reconstructing 10 m time series from S3 acquisitions alone.

**Keywords:** image fusion; super-resolution; tensor decomposition; Tucker; sparse core; Sentinel-2; Sentinel-3; self-supervised learning; spatiotemporal fusion.

---

## 1. Introduction

Optical Earth-observation missions face a well-known three-way trade-off between spatial resolution, spectral richness and revisit frequency. Within the Copernicus programme, this trade-off is embodied by two complementary missions. Sentinel-2 (S2) [1] acquires 13 bands at decametric resolution but distributes them over three native grids — four bands at 10 m, six at 20 m, three at 60 m — with a 5-day nominal revisit that cloud cover frequently degrades in practice. Sentinel-3 (S3) OLCI [2] acquires 21 narrow bands, well suited to biophysical variable retrieval, at 300 m resolution with quasi-daily revisit. Fusing the two sensors to synthesize a *21-band image at 10 m* would combine the spectral capability of OLCI with the geometry of S2, and — once extended in time — its temporal cadence.

Two technological locks stand in the way. The first is **input heterogeneity**: the problem is natively *multi-input*, with four streams whose spatial grids span a factor of 30 and whose band counts range from 3 to 21. Most fusion pipelines evade this heterogeneity by interpolating all streams to the finest grid before processing; this injects non-observed pseudo-samples, imposes the computational cost of the fine grid on data that lack the corresponding frequency content, and places a fixed linear operator (the interpolator) at the very entrance of allegedly nonlinear models. The second lock is **supervision**: no 21-band, 10 m ground truth exists, so learning must be constrained by the physics of the sensors rather than by reference images.

This paper proposes the Multi-Stream Nonlinear Joint Tucker Autoencoder (MS-NL-JTAE), built on the scientific hypothesis that the latent scene admits an (approximately) low multilinear-rank, *sparse-core* Tucker structure shared by all four observation streams. Our contributions are:

1. **A natively multi-input architecture aligned with the sensor grids.** The resolution ratio factorizes exactly as $30 = 2 \times 3 \times 5$, matching the physical grid cascade 10 → 20 → 60 → 300 m. The encoder pyramid adopts strides (2, 3, 5) so that each stream enters the network at its own native level through a stream-specific stem and a multiplicative gated fusion. No observation is ever resampled outside the network.
2. **Structural core sparsity.** The joint core $\mathcal{G}$ lives on the S3 grid ($H/30 \times W/30 \times R_3$) and is produced by channel-wise *learned soft-thresholding*, the proximal operator of the $\ell_1$ norm, yielding exact zeros by construction and connecting the network to sparse-coding theory (ISTA/LISTA [20], FISTA [21]).
3. **Information routing as an architectural guarantee.** Skip connections originate only from S2-derived levels (at most 13 bands); the 21-band spectral content can reach the output only through $\mathcal{G}$. The core is therefore *forced* to be the joint spectral–spatial code, preventing the "skip bypass" failure mode of encoder–decoder fusion networks.
4. **A physics-constrained, leakage-free protocol.** All models are trained purely self-supervisedly from the four native observations through differentiable PSF/SRF sensor models; angular spectral fidelity (SAM) is imposed against the *S3 observations*, never against the evaluation reference. We compare against a coupled sparse Tucker ALS baseline and a coupled Tucker autoencoder under an identical information regime, and we report supervised results only as explicitly labelled oracle upper bounds.

We validate on the Pavia University hyperspectral benchmark under a Wald-type protocol [22] with resolution ratio 30, using PSNR, SAM, ERGAS, SSIM and UIQI, and we analyse the sparsity of the estimated core. Finally, we discuss how the learned factorization — permanent spatial factors, time-varying sparse core — extends to the reconstruction of dense 10 m time series from S3-only acquisitions, the long-term objective of this research programme.

---

## 2. Related Work

### 2.1 Model-based tensor and matrix fusion

Hyperspectral–multispectral (HS–MS) fusion has classically been posed as an inverse problem under linear degradation models. Matrix-factorization approaches include coupled non-negative matrix factorization (CNMF) [3] and subspace-regularized variational methods such as HySure [4]. Tensor approaches treat the scene as a third-order tensor: STEREO [5] couples canonical polyadic decompositions of the two observations and provides exact recovery conditions; SCOTT and CT-STAR [6] extend the guarantees to coupled *Tucker* decompositions solved via truncated HOSVD; CSTF [7] assumes a *sparse core* with mode dictionaries, solved by proximal optimization; NLSTF [8] exploits non-local spatial self-similarity. These methods are interpretable and enjoy identifiability results, but their expressiveness is bounded by multilinearity: intimate (nonlinear) spectral mixing, BRDF effects, residual atmospheric and co-registration errors all violate the multilinear observation model. Our baseline v0 belongs to this family — a coupled sparse-core Tucker solved by alternating dictionary updates and FISTA — and our proposed model is its nonlinear generalization.

### 2.2 Deep-learning fusion and unsupervised coupled autoencoders

CNN-based pansharpening and HS–MS fusion began with PNN [9] and evolved toward deeper designs (e.g., DHSIS [10]) and model-inspired unrolling such as MHF-Net [11], which unfolds an optimization scheme into a network. Closer to our setting are *unsupervised coupled autoencoders*: uSDN [12] and CUCaNet [13] couple per-sensor encoders through shared decoders and degradation constraints, learning from the test scene alone. This per-scene regime is an instance of internal learning, whose viability is established by the deep image prior [14] and zero-shot super-resolution [15]. Our model follows this self-supervised, per-scene paradigm but differs in two respects: the latent space is explicitly organized as a Tucker core with structural sparsity, and the inputs are ingested at native resolution rather than after interpolation.

### 2.3 Handling multi-resolution inputs

Three strategies dominate. (i) *Interpolate-then-fuse* (early fusion): all streams are upsampled to the finest grid and concatenated — simple, but statistically inconsistent (a 300 m pixel interpolated to 10 m contributes 900 samples for one observation) and linear at ingestion. (ii) *Per-stream encoders with feature alignment*: DSen2 [16] processes S2's 10/20/60 m groups natively for intra-S2 super-resolution; coupled autoencoders [12,13] use per-sensor branches but usually fuse at a single scale. (iii) *Unrolled model-based networks* [11] handle operators explicitly but inherit the linear model they unroll. Our contribution combines the strengths of (i)–(iii): a *progressive, grid-native fusion pyramid* whose levels coincide with the physical grids, with learned nonlinear rescaling and multiplicative gating at each level.

### 2.4 Spatiotemporal fusion

Reconstructing high-resolution time series from a fine-but-infrequent and a coarse-but-frequent sensor is the classical spatiotemporal fusion (STF) problem, formalized for Landsat/MODIS by STARFM [17], extended by ESTARFM [18] and FSDAF [19], and revisited with dictionary-pair learning (SPSTFM [23]) and deep networks. Section 6 shows that our factorization provides a latent-space alternative to pixel-space STF blending for the S2/S3 pair, with 21-band spectral output.

---

## 3. Problem Formulation and Proposed Method

### 3.1 Observation model

Let $\mathcal{S} \in \mathbb{R}^{H \times W \times C}$ denote the latent scene on the 10 m grid ($C = 21$ OLCI bands in the operational setting; $C = 103$ for the simulation benchmark of Section 4). The four observed streams are degraded projections of $\mathcal{S}$:

$$
\begin{aligned}
\mathcal{Y}_{10} &= \mathcal{S} \times_3 \mathbf{R}_{10} && \in \mathbb{R}^{H \times W \times 4},\\
\mathcal{Y}_{20} &= \mathcal{S} \times_1 \mathbf{P}_1^{(2)} \times_2 \mathbf{P}_2^{(2)} \times_3 \mathbf{R}_{20} && \in \mathbb{R}^{\frac{H}{2} \times \frac{W}{2} \times 6},\\
\mathcal{Y}_{60} &= \mathcal{S} \times_1 \mathbf{P}_1^{(6)} \times_2 \mathbf{P}_2^{(6)} \times_3 \mathbf{R}_{60} && \in \mathbb{R}^{\frac{H}{6} \times \frac{W}{6} \times 3},\\
\mathcal{Y}_{300} &= \mathcal{S} \times_1 \mathbf{P}_1^{(30)} \times_2 \mathbf{P}_2^{(30)} \times_3 \mathbf{R}_{300} && \in \mathbb{R}^{\frac{H}{30} \times \frac{W}{30} \times 21},
\end{aligned}
$$

where $\times_n$ is the mode-$n$ product, $\mathbf{P}^{(s)}$ are separable PSF-blur-and-decimation operators of factor $s$ (Gaussian PSF, $\sigma = 1$ px), and $\mathbf{R}$ are spectral response function (SRF) matrices. In the operational S2/S3 case $\mathbf{R}_{300} = \mathbf{I}$ (the target bands are OLCI's own), and the S2 SRFs couple only over the spectral overlap of OLCI (400–1020 nm), i.e., S2 bands B1–B8A; SWIR bands lie outside the OLCI support and are excluded from physical coupling.

### 3.2 Shared sparse Tucker core

Our central hypothesis is that $\mathcal{S}$ admits a Tucker representation with a sparse core shared by all streams:

$$
\mathcal{S} \;\approx\; \mathcal{G} \times_1 \mathbf{U} \times_2 \mathbf{V} \times_3 \mathbf{W}, \qquad \mathcal{G} \in \mathbb{R}^{R_1 \times R_2 \times R_3} \ \text{sparse},
$$

with spatial factors $\mathbf{U} \in \mathbb{R}^{H \times R_1}$, $\mathbf{V} \in \mathbb{R}^{W \times R_2}$ and spectral factor $\mathbf{W} \in \mathbb{R}^{C \times R_3}$. Substituting into the observation model yields, for each stream $i$,

$$
\mathcal{Y}_i \;\approx\; \mathcal{G} \times_1 (\mathbf{P}_1^{(i)}\mathbf{U}) \times_2 (\mathbf{P}_2^{(i)}\mathbf{V}) \times_3 (\mathbf{R}_i\mathbf{W}),
$$

i.e., *all four streams share the same core* $\mathcal{G}$; only the factors are degraded. This is the coupled Tucker model of [5–7]. An identifiability observation anchors our design: choosing the core's spatial support on the S3 grid, $R_1 = H/30$, $R_2 = W/30$, makes $\mathbf{P}_1^{(30)}\mathbf{U}$ square; if well conditioned, and if $R_3 \le 21$ with $\mathbf{R}_{300}\mathbf{W}$ of full column rank, the core is recoverable from $\mathcal{Y}_{300}$ alone in closed form given the factors. The information budget is likewise balanced: $\mathcal{Y}_{300}$ carries $\frac{HW}{900}\cdot 21$ measurements against $\frac{HW}{900}\cdot R_3$ core unknowns. The nonlinear model inherits this counting argument while replacing the multilinear maps.

### 3.3 Nonlinear generalization: a mode-wise correspondence

The generalization from multilinear to nonlinear is exact rather than metaphorical. In a convolutional network operating on a $(C, H, W)$ tensor: a $1\times1$ convolution acts on mode 3 (spectral) as a learned counterpart of $\mathbf{W}$; strided and transposed convolutions act on modes 1–2 (spatial) as learned counterparts of $\mathbf{U}, \mathbf{V}$; interleaved activations and gates compose these into nonlinear per-mode functions. The decoder
$\sigma \circ \mathrm{ConvT}_{\times2} \circ \sigma \circ \mathrm{ConvT}_{\times3} \circ \sigma \circ \mathrm{ConvT}_{\times5}$
is thus a nonlinear realization of $\mathcal{G} \times_1 \mathbf{U} \times_2 \mathbf{V} \times_3 \mathbf{W}$, retaining the Tucker structure — a compact latent core, mode-wise spatial upsampling, spectral synthesis — as an inductive bias while lifting the multilinearity restriction that physical mixing violates.

### 3.4 Architecture

**Grid-native encoder pyramid.** Each stream has a stem at its native resolution: the 10 m stem uses $3\times3$ convolutions (with an inception-style block for multi-scale receptive fields); the 20 m and 60 m stems use $3\times3$ convolutions; the 300 m stem is dominated by $1\times1$ convolutions, as OLCI's information at that grid is essentially spectral. The pyramid then descends 10 → 20 → 60 → 300 m via learned strided convolutions with strides (2, 3, 5) and kernel $s+2$, whose support provides anti-aliasing consistent with the PSF cascade (in the spirit of anti-aliased downsampling [24]). At each level $k$, descended features and the native stem output are merged by a **gated fusion** unit (GLU [25]):

$$
\mathbf{F}_k = \phi\!\left(\mathrm{Conv}[\mathbf{F}_{k-1}^{\downarrow};\, \mathbf{S}_k]\right) \odot \sigma\!\left(\mathrm{Conv}[\mathbf{F}_{k-1}^{\downarrow};\, \mathbf{S}_k]\right),
$$

a multiplicative — hence strongly nonlinear — and parameter-frugal mechanism suited to the per-scene training regime, where attention-based alternatives are data-hungry.

**Sparse core head.** At the 300 m level, a nonlinear mode-3 projection ($1\times1$ convolutions) produces a pre-core, to which channel-wise learned soft-thresholding is applied:

$$
\mathcal{G} = \mathrm{soft}_{\tau}\big(\psi(\mathbf{F}_3)\big), \qquad \mathrm{soft}_{\tau}(x) = \mathrm{sign}(x)\,\max(|x| - \tau, 0), \quad \tau = \mathrm{softplus}(\theta) \ \text{per channel}.
$$

Soft-thresholding is the proximal operator of the $\ell_1$ norm; sparsity of the core is therefore *structural* (exact zeros in the forward pass), with an auxiliary $\ell_1$ penalty steering the sparsity level. This ties the architecture to LISTA-style learned sparse coding [20].

A practical caveat proved decisive: the dead zone of soft-thresholding has zero gradient with respect to both its input and its threshold, so a coefficient that falls below $\tau$ can never recover — a ratchet that can silently kill the entire core while skip connections absorb the reconstruction. In particular, applying weight decay to the threshold parameter drives $\tau$ upward and *guarantees* core death. We therefore (i) exclude thresholds, biases and normalization parameters from weight decay, (ii) bound the threshold, $\tau = \tau_{\max}\,\sigma(\theta)$ with $\tau_{\max} = 0.05$, (iii) apply the $\ell_1$ penalty to the pre-threshold activations, where the gradient remains alive, and (iv) enable the shrinkage only after a 150-epoch warm-up. Channel dropout ($p = 0.1$) on the skip features during training further discourages the network from bypassing the core. We report core sparsity along training as a diagnostic: a monotone climb to 100 % signals pathway death, not successful compression.

**Information routing.** Skip connections feed the decoder from the fused 10/20/60 m levels — features derived exclusively from S2 (13 bands at most). The 21-band OLCI content enters the network only at the 300 m level and can reach the output only through $\mathcal{G}$. The network cannot shortcut the core: $\mathcal{G}$ is the sole conduit of the joint spectral code, which instantiates the shared-core hypothesis architecturally and is, additionally, the property that enables the temporal extension of Section 6.

**Decoder.** A mirror pyramid of transposed convolutions with strides (5, 3, 2), each followed by concatenation with the corresponding encoder skip and a convolutional block, ends in a $3\times3$ convolution and a sigmoid mapping to reflectances in $[0, 1]$. For Pavia University at $240\times240$, the model has $\approx 2.4$ M parameters and the core is $8 \times 8 \times R_3$ with $R_3 = 64$.

### 3.5 Self-supervised objective and protocol fairness

The prediction $\hat{\mathcal{S}}$ is re-degraded by the differentiable sensor models $\mathcal{D}_i$ (Section 3.1) and compared with each native observation:

$$
\mathcal{L} = \sum_{i \in \{10,20,60\}} \big\|\mathcal{D}_i(\hat{\mathcal{S}}) - \mathcal{Y}_i\big\|_2^2 \;+\; 2\,\big\|\mathcal{D}_{300}(\hat{\mathcal{S}}) - \mathcal{Y}_{300}\big\|_2^2 \;+\; \lambda_{\mathrm{sam}}\, \mathrm{SAM}\big(\mathcal{D}_{300}(\hat{\mathcal{S}}),\, \mathcal{Y}_{300}\big) \;+\; \lambda_1 \|\mathcal{G}\|_1 .
$$

Two methodological points deserve emphasis. First, the angular term applies the SAM *functional form* to the **S3 observations** — a data-fidelity term in spherical geometry, invariant to multiplicative illumination factors, motivated by the Lambertian decomposition of spectra into material direction and illumination magnitude. It never touches the evaluation reference; reporting SAM as a metric therefore involves no leakage, exactly as reporting PSNR while minimizing MSE (its monotone bijection) is standard practice [26]. Second, *all* compared methods run under the same information regime: only $\{\mathcal{Y}_i\}$ and the operators are visible during estimation; the reference $\mathcal{S}$ is used exclusively for scoring, and model selection during training uses the loss, never reference-based metrics. Supervised variants are reported only as explicitly labelled oracle upper bounds.

---

## 4. Experiments

### 4.1 Setup

**Data and Wald protocol.** We use the Pavia University hyperspectral image (ROSIS, 103 bands, 1.3 m GSD), centre-cropped to $240\times240$ and normalized to $[0,1]$ (99.9th percentile). Treating the image as the latent scene $\mathcal{S}$, we simulate the four streams with the operators of Section 3.1: 4 bands at scale 1, 6 bands at scale 2, 3 bands at scale 6 and 21 bands at scale 30 (Gaussian PSF, $\sigma = 1$ px; block-uniform SRFs). This reproduces the S2/S3 configuration — identical stream dimensions and resolution ratios — while providing a full reference for evaluation, per Wald's synthesis protocol [22].

**Compared methods.**
- **Tucker ALS (v0, linear baseline):** coupled sparse-core Tucker; dictionaries updated by Lipschitz gradient steps with column normalization, core by FISTA with soft-thresholding [21]; ranks $(48, 48, 12)$; initialization from the observations only (SVD of $\mathcal{Y}_{10}$ unfoldings for spatial modes; SRF-pseudo-inverse lifting of $\mathcal{Y}_{300}$'s spectral SVD).
- **CTAE (v1, hybrid):** coupled CNN encoders (interpolated 13-band S2 stack; interpolated 21-band S3) estimating a shared core decoded *multilinearly* through learned factors $\mathbf{A}, \mathbf{B}, \mathbf{C}$; ranks $(24, 24, 12)$.
- **NL-JTAE early-fusion (v2, v3 — legacy internal iterations):** fully nonlinear joint autoencoders ingesting a bilinearly upsampled 34-channel stack; v3 adds UNet-style skips, transposed-convolution upsampling and a supervised SAM loss. These iterations were trained with reference supervision and are reported for completeness under that label.
- **MS-NL-JTAE (v4, proposed):** the architecture of Section 3.4, self-supervised; and its supervised oracle variant.

**Implementation.** PyTorch; Adam ($lr = 2\times10^{-3}$ for the proposed model; weight decay $10^{-4}$ on convolutional weights only, per Section 3.4), ReduceLROnPlateau, gradient clipping at 1.0, 150-epoch shrinkage warm-up, early stopping on the self-supervised loss (patience 250), epoch budget 6000. Neural models trained on an NVIDIA A100 (OAR cluster); the ALS baseline runs on CPU. $\lambda_{\mathrm{sam}} = 0.1$, $\lambda_1 = 10^{-5}$.

**Metrics.** PSNR (radiometric accuracy), SAM in degrees (spectral shape), ERGAS with ratio 30 (band-wise relative error), SSIM and UIQI averaged over bands (spatial structure) [27, 28], plus the fraction of near-zero core coefficients ($|g| < 10^{-4}$) as the sparsity measure. The panel is deliberately redundant across error families so that no single optimized surrogate can dominate the assessment (Goodhart safeguard).

### 4.2 Quantitative results

**Table 1 — Fusion quality on Pavia University (Wald protocol, ratio 30).** Self-supervised rows share an identical information regime (observations and operators only; best state selected on the loss). Supervised rows are oracle bounds, labelled as such and not directly comparable. All neural models trained on one NVIDIA A100; runtimes include full per-scene optimization.

| Method | Synthesis | Input ingestion | Regime | PSNR (dB) ↑ | SAM (°) ↓ | ERGAS ↓ | SSIM ↑ | UIQI ↑ | Core sparsity | Time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| Tucker ALS (v0) | multilinear | native (operators) | self-sup. | 22.03 | 13.26 | 1.090 | 0.688 | 0.850 | 0.0 % | 9 |
| CTAE (v1) | multilinear dec. | interpolated | self-sup. | 22.26 | **9.85** | 1.034 | 0.549 | 0.798 | 1.0 % | 83 |
| **MS-NL-JTAE (v4, proposed)** | nonlinear | **native (learned)** | self-sup. | **25.10** | 11.76 | **0.759** | **0.875** | **0.937** | 3.1 % | 75 |
| NL-JTAE early fusion (v2, legacy) | nonlinear | interpolated | supervised | 25.90 | 5.37 | 5.30 | — | — | — | — |
| NL-JTAE v3 (legacy) | nonlinear | interpolated | supervised | 40.57 | 1.55 | 0.148 | 0.986 | — | — | — |
| MS-NL-JTAE (v4, oracle) | nonlinear | native (learned) | supervised | 37.73 | 2.20 | 0.206 | 0.974 | 0.994 | 14.1 % | 53 |

At equal information regime, the proposed model improves PSNR by **+3.1 dB** over the linear coupled-Tucker baseline and **+2.8 dB** over the CTAE, reduces ERGAS by 27–30 %, and dominates both structural metrics by a wide margin (SSIM 0.875 vs. 0.549–0.688; UIQI 0.937 vs. 0.798–0.850) — the signature of native ingestion: no interpolation footprint, learned scale changes, high-frequency structure recovered from the 10 m stream. One result runs against the sweep and deserves emphasis rather than concealment: the CTAE retains the best *spectral angle* (9.85° vs. 11.76°). Its rigid, global multilinear spectral factor — a rank-12 subspace lifted from the S3 observations — acts as a powerful spectral regularizer in the severely ill-posed self-supervised regime, at the cost of every spatial metric. The oracle bound (SAM 2.20°) shows that this is not an architectural ceiling: the proposed decoder can express near-perfect spectra when guided; what is missing in the self-supervised regime is a stronger global spectral prior. Hybridizing the two — a nonlinear decoder whose spectral head is softly constrained to an S3-lifted subspace — is the natural next iteration, and we note that the operational S2/S3 setting (where the 21 target bands are S3's own, $\mathbf{R}_{300} = \mathbf{I}$) is better posed in this respect than the 103-band simulation, since every target band is then directly anchored by the S3 data term.

### 4.3 Core sparsity analysis and the pathway-death ablation

The learned thresholds $\tau$ produce exact zeros in $\mathcal{G}$ during the forward pass; the retained configuration operates at 3.1 % (self-supervised) and 14.1 % (oracle) exact zeros with stable training curves. Sweeping $\lambda_1$ traces a sparsity–fidelity curve whose knee identifies the effective joint rank of the scene.

An ablation across training configurations doubles as a cautionary result. With naive settings (unbounded learned threshold subject to weight decay, post-threshold $\ell_1$), core sparsity climbs monotonically — 1 % → 74 % → 100 % — until $\mathcal{G}$ dies entirely, while the loss keeps decreasing: the skips absorb the reconstruction. Tellingly, the dead-core model still scores well on spatial metrics (25.9 dB PSNR self-supervised) but its spectral angle degrades below the CTAE's — direct evidence for the routing analysis of Section 3.4: when the core dies, the 21-band pathway is severed and only S2-derived information reaches the output. Conversely, forcing traffic through the core with heavy skip dropout keeps $\mathcal{G}$ alive but costs 3–5 dB. The guardrails of Section 3.4 (decay-free bounded thresholds, pre-threshold penalty, warm-up) resolve the dilemma: a living, sparse core at essentially no performance cost. We recommend monitoring core sparsity during training as a standard diagnostic for any sparse-bottleneck fusion network.

### 4.4 Visual and spectral analysis

Qualitative inspection (RGB composites, per-band greyscale panels, absolute-difference maps and per-pixel spectra; figures generated by `article/make_figures.py`) focuses on three signatures. (i) *Sharpness*: native ingestion avoids the low-pass footprint that bilinear pre-interpolation imprints on early-fusion variants; edges of buildings and road network in the Pavia scene are the sensitive structures. (ii) *Spectral consistency*: predicted spectra at randomly sampled pixels are compared with the reference across all 103 bands; the SAM-in-loss variant is expected to align spectral *shape* even where magnitude errors persist. (iii) *Grid artefacts*: the ×30 factor makes decimation artefacts conspicuous; the PSF-consistent strided encoder and transposed-convolution decoder should exhibit none.

### 4.5 Feasibility study on the standard two-stream protocol (HyperBench)

To situate the approach against the classical HSI–MSI fusion setting — and against the linear sparse coupled-Tucker solver in its home regime — we ran a two-stream variant of the proposed model through the HyperBench evaluation framework on Pavia University (full scene, 610×340×103): PSF families {Gaussian, Airy, sinc} × downsampling ratios {×4, ×8} × spatial SNR {35, 30 dB}, 4 MSI bands, six cases in total, identical seeds and degradations for all methods. The two-stream variant instantiates the same design: the ratio is factorized into a stride cascade (4 = 2·2, 8 = 2·2·2), the MSI is ingested natively at level 0, the HSI at the core level, with the sparse-core guardrails unchanged; training is per-case self-supervised (~1.5 min per case on one A100).

**Table 2 — HyperBench feasibility study (means over the six cases).**

| Method | PSNR (dB) ↑ | SAM (°) ↓ | ERGAS ↓ | SSIM ↑ |
|---|---|---|---|---|
| G-SCOTT (linear, sparse core) | 13.68 | 20.07 | 18.19 | 0.124 |
| Bicubic upsampling | 19.76 | 8.43 | 9.24 | 0.520 |
| **NL-JTAE (proposed, two-stream)** | **22.77** | **7.66** | **6.34** | **0.823** |

The proposed model leads on *all four metrics*, including the spectral angle — confirming that the SAM deficit observed in the four-stream ratio-30 experiment (Table 1) is a property of that extreme, severely ill-posed regime rather than of the architecture: at ratios 4–8, where the HSI anchors are dense enough, the nonlinear joint core recovers both geometry and spectra. Robustness across PSF families is consistent (best case: Airy, 24.2 dB at ×4; worst: sinc, 21.3 dB), and degradation from ×4 to ×8 is graceful (−0.5 dB) where bicubic loses 2 dB. The linear solver, tuned for its own protocol, is dominated throughout — including by bicubic — which illustrates the brittleness of multilinear fitting under PSF mismatch and noise.

### 4.6 Real-data feasibility (Andorra, S2/S3)

As a qualitative feasibility check on real acquisitions, the earlier fully-nonlinear iteration of the model was applied to a genuine S2/S3 pair over Andorra (July 2022; $480\times480$ at 10 m; 17 usable OLCI bands after excluding those outside the S2 spectral overlap and atmospheric absorption windows). Training was purely self-supervised — the operational regime, since no 10 m/21-band reference exists. Reconstructed bands exhibited inter-scale spectral consistency (Pearson correlation above 0.9 against the 300 m observations on bands Oa1–Oa10) and a sharpness gain of roughly ×3 over bilinear upsampling. The proposed multi-stream architecture applies to this setting unchanged, with $16\times16\times R_3$ core for the $480\times480$ tile.

---

## 5. Discussion

**Why native ingestion matters.** Beyond the empirical gains, early fusion is *conceptually* misaligned with the physics: interpolation manufactures samples that carry no information, and the network must then learn to discount them. Native ingestion keeps the information budget explicit — each stream contributes exactly its measurements, at its grid — and lets learned, nonlinear operators perform all scale changes. The stride factorization $30 = 2\times3\times5$ makes this alignment exact for the S2/S3 pair; sensor pairs without such factorization would require one learned resampling stage, a mild generalization.

**On metric-aligned losses.** Including an angular data-fidelity term while reporting SAM as a metric raises a legitimate fairness question. The protocol answers it on three levels: the loss consumes observations, never the reference; all compared methods share the information regime; and the metric panel spans error families (radiometric, angular, relative, structural) so that optimizing one surrogate cannot game the assessment. An ablation with $\lambda_{\mathrm{sam}} = 0$ additionally attributes gains between architecture and objective.

**Limitations.** The per-scene training regime, inherited from the unsupervised fusion literature [12–15], implies per-scene optimization cost (minutes on one GPU); amortized pre-training across scenes is an open extension. The PSF/SRF are assumed known; blind estimation (e.g., alternating operator refinement) is compatible with the framework. BatchNorm statistics under batch-of-one training warrant replacement with GroupNorm in future revisions. Finally, the simulated SRFs (block-uniform) are a simplification; plugging in tabulated S2/OLCI SRFs is straightforward.

---

## 6. Toward Temporal Generalization: Learned Spatial Dictionaries and S3-Only Reconstruction

The Tucker factorization separates precisely the quantities whose time scales differ. Writing $\mathcal{S}(t) \approx \mathcal{G}(t) \times_1 \mathbf{U} \times_2 \mathbf{V} \times_3 \mathbf{W}$, the spatial factors encode the quasi-permanent geometry of the landscape (field boundaries, roads, relief) while the sparse core carries the spectral–phenological state. In the trained network, the nonlinear counterpart of $(\mathbf{U}, \mathbf{V})$ is the pair {decoder weights, cached S2-derived skip features} — a *spatial memory* $\Phi$ learned at a cloud-free joint date $t_0$ — while $\mathcal{G}(t)$ is the only time-varying latent. Because the architecture routes all 21-band information through the core (Section 3.4), freezing $\Phi$ and re-estimating $\mathcal{G}(t_i)$ from each new S3 acquisition is exactly the update the design permits:

$$
\hat{\mathcal{G}}_i = \arg\min_{\mathcal{G}} \big\| \mathcal{D}_{300}\big(D_\theta(\mathcal{G}; \Phi)\big) - \mathcal{Y}_{300}(t_i) \big\|_2^2 + \lambda_1 \|\mathcal{G}\|_1 + \lambda_T \|\mathcal{G} - \hat{\mathcal{G}}_{i-1}\|_1,
$$

with a closed-form anchor in the linear regime (Section 3.2) and a *sparse-innovation* temporal prior ($\ell_1$ on core increments: surface changes between close dates are spatially localized). The optimization involves only $\frac{HW}{900} R_3$ variables — seconds per date — or can be amortized by the frozen S3 stem for real-time use. Validation avoids the missing-reference problem by a *leave-S2-out* design: at dates where both sensors are clear, reconstruct from S3 + $\Phi(t_0)$ only and score the SRF/PSF-degraded prediction against the withheld S2 image. The main theoretical locks — genuine ground changes, illumination/BRDF drift, inter-date co-registration — map to identified remedies (residual-driven skip gating, low-frequency radiometric normalization, sub-pixel shift estimation) and to the classical spatiotemporal-fusion evaluation methodology [17–19], against which the latent-space approach will be benchmarked.

---

## 7. Conclusion

We introduced MS-NL-JTAE, a fully nonlinear, natively multi-input Tucker autoencoder that fuses the four heterogeneous S2/S3 streams into a 21-band, 10 m product. Its three pillars — a sensor-grid-aligned encoder pyramid (strides 2·3·5), structural core sparsity via learned soft-thresholding, and exclusive routing of spectral information through the joint core — turn the shared sparse-core hypothesis from a regularizer into an architectural property, while differentiable PSF/SRF models enable leakage-free self-supervised training. Under a ratio-30 Wald protocol on Pavia University, the method is evaluated against linear and hybrid coupled-Tucker baselines at an identical information regime. The same factorization yields a concrete programme for temporal generalization: spatial dictionaries learned at a joint cloud-free date, and a sparse core updated from each quasi-daily S3 acquisition — a path toward dense decametric, hyperspectral-like time series from the Copernicus constellation.

---

## References

[1] M. Drusch et al., "Sentinel-2: ESA's optical high-resolution mission for GMES operational services," *Remote Sensing of Environment*, 2012.
[2] C. Donlon et al., "The Global Monitoring for Environment and Security (GMES) Sentinel-3 mission," *Remote Sensing of Environment*, 2012.
[3] N. Yokoya, T. Yairi, A. Iwasaki, "Coupled nonnegative matrix factorization unmixing for hyperspectral and multispectral data fusion," *IEEE TGRS*, 2012.
[4] M. Simões, J. Bioucas-Dias, L. Almeida, J. Chanussot, "A convex formulation for hyperspectral image superresolution via subspace-based regularization," *IEEE TGRS*, 2015.
[5] C. Kanatsoulis, X. Fu, N. Sidiropoulos, W.-K. Ma, "Hyperspectral super-resolution: A coupled tensor factorization approach," *IEEE Trans. Signal Processing*, 2018.
[6] C. Prévost, K. Usevich, P. Comon, D. Brie, "Hyperspectral super-resolution with coupled Tucker approximation: Recoverability and SVD-based algorithms," *IEEE Trans. Signal Processing*, 2020.
[7] S. Li, R. Dian, L. Fang, J. Bioucas-Dias, "Fusing hyperspectral and multispectral images via coupled sparse tensor factorization," *IEEE Trans. Image Processing*, 2018.
[8] R. Dian, L. Fang, S. Li, "Hyperspectral image super-resolution via non-local sparse tensor factorization," *CVPR*, 2017.
[9] G. Masi, D. Cozzolino, L. Verdoliva, G. Scarpa, "Pansharpening by convolutional neural networks," *Remote Sensing*, 2016.
[10] R. Dian, S. Li, A. Guo, L. Fang, "Deep hyperspectral image sharpening," *IEEE TNNLS*, 2018.
[11] Q. Xie, M. Zhou, Q. Zhao, D. Meng, W. Zuo, Z. Xu, "Multispectral and hyperspectral image fusion by MS/HS fusion net," *CVPR*, 2019.
[12] Y. Qu, H. Qi, C. Kwan, "Unsupervised sparse Dirichlet-net for hyperspectral image super-resolution," *CVPR*, 2018.
[13] J. Yao, D. Hong, J. Chanussot, D. Meng, X. Zhu, Z. Xu, "Cross-attention in coupled unmixing nets for unsupervised hyperspectral super-resolution," *ECCV*, 2020.
[14] D. Ulyanov, A. Vedaldi, V. Lempitsky, "Deep image prior," *CVPR*, 2018.
[15] A. Shocher, N. Cohen, M. Irani, "'Zero-shot' super-resolution using deep internal learning," *CVPR*, 2018.
[16] C. Lanaras, J. Bioucas-Dias, S. Galliani, E. Baltsavias, K. Schindler, "Super-resolution of Sentinel-2 images: Learning a globally applicable deep neural network," *ISPRS Journal of Photogrammetry and Remote Sensing*, 2018.
[17] F. Gao, J. Masek, M. Schwaller, F. Hall, "On the blending of the Landsat and MODIS surface reflectance: Predicting daily Landsat surface reflectance," *IEEE TGRS*, 2006.
[18] X. Zhu, J. Chen, F. Gao, X. Chen, J. Masek, "An enhanced spatial and temporal adaptive reflectance fusion model for complex heterogeneous regions," *Remote Sensing of Environment*, 2010.
[19] X. Zhu, E. Helmer, F. Gao, D. Liu, J. Chen, M. Lefsky, "A flexible spatiotemporal method for fusing satellite images with different resolutions," *Remote Sensing of Environment*, 2016.
[20] K. Gregor, Y. LeCun, "Learning fast approximations of sparse coding," *ICML*, 2010.
[21] A. Beck, M. Teboulle, "A fast iterative shrinkage-thresholding algorithm for linear inverse problems," *SIAM Journal on Imaging Sciences*, 2009.
[22] L. Wald, T. Ranchin, M. Mangolini, "Fusion of satellite images of different spatial resolutions: Assessing the quality of resulting images," *Photogrammetric Engineering and Remote Sensing*, 1997.
[23] B. Huang, H. Song, "Spatiotemporal reflectance fusion via sparse representation," *IEEE TGRS*, 2012.
[24] R. Zhang, "Making convolutional networks shift-invariant again," *ICML*, 2019.
[25] Y. Dauphin, A. Fan, M. Auli, D. Grangier, "Language modeling with gated convolutional networks," *ICML*, 2017.
[26] H. Zhao, O. Gallo, I. Frosio, J. Kautz, "Loss functions for image restoration with neural networks," *IEEE Transactions on Computational Imaging*, 2017.
[27] Z. Wang, A. Bovik, "A universal image quality index," *IEEE Signal Processing Letters*, 2002.
[28] Z. Wang, A. Bovik, H. Sheikh, E. Simoncelli, "Image quality assessment: From error visibility to structural similarity," *IEEE Trans. Image Processing*, 2004.
[29] R. Yuhas, A. Goetz, J. Boardman, "Discrimination among semi-arid landscape endmembers using the spectral angle mapper (SAM) algorithm," *JPL Airborne Geoscience Workshop*, 1992.
[30] C. Cremer, X. Li, D. Duvenaud, "Inference suboptimality in variational autoencoders," *ICML*, 2018.
