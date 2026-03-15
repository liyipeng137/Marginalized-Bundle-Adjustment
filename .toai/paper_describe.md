

# Marginalized Bundle Adjustment (MBA)

## Paper
- Title: Marginalized Bundle Adjustment: Multi-View Camera Pose from Monocular Depth Estimates
- Core topic: using monocular depth estimates as the main geometric prior for multi-view camera pose optimization
- Key novelty: introducing a bundle-adjustment objective tailored to dense, noisy monocular depth maps instead of sparse, high-precision triangulated points

---

## Executive Summary

MBA proposes a robust optimization framework for recovering multi-view camera poses from monocular depth estimates and dense image correspondences.

The main claim is:

- classical BA is well matched to sparse and relatively accurate geometric observations
- monocular depth maps are dense but noisy and high-variance
- therefore, directly plugging monocular depth into standard BA is suboptimal
- the optimization objective itself must be redesigned

MBA redesigns the BA objective by marginalizing over residual thresholds. Instead of evaluating inliers under a single threshold, it aggregates behavior across a range of thresholds. This yields a robust, differentiable objective based on the empirical residual distribution.

Practical consequence:

- the method is more tolerant to noisy monocular depth than standard reprojection-loss-driven BA
- the method can optimize camera intrinsics, extrinsics, and per-image affine depth corrections jointly
- the same principle can also be adapted to two-view robust estimation / RANSAC scoring

---

## Problem Setting

### Input
For each image `I_i`, the system assumes:

- a monocular depth prediction `D_i`
- dense correspondences `C_{i,j}` between selected image pairs `(i, j)`

These are produced by pretrained external models.

### Output
The system estimates:

- camera extrinsics `P`
- camera intrinsics `K`
- per-image affine depth correction parameters `A = {(alpha_i, beta_i)}`

The corrected depth map is:

```text
D'_i = alpha_i * D_i + beta_i
```

### Design Choice
The method does **not** optimize per-pixel depth values directly.
It only optimizes a lightweight affine correction per image.

Implication:

- preserves the global structural prior coming from the monocular depth model
- avoids turning the optimization into a full dense depth refinement problem
- keeps the system modular and compatible with different MDE backbones

---

## Core Insight

### Why classical BA is not ideal here
Classical BA assumes that the underlying geometric observations are reasonably accurate.
Typical examples:

- sparse keypoint correspondences
- triangulated 3D points
- low-noise reprojection constraints

Monocular depth maps violate this assumption:

- dense observations
- larger uncertainty
- scale / bias inconsistency across images
- local errors, hallucinations, edge bleeding, texture ambiguity

Therefore the issue is not only initialization quality.
The residual model itself is mismatched.

### What should replace standard robust losses
Instead of choosing one fixed inlier threshold `tau`, MBA asks:

- what if the objective rewarded solutions that perform well across many thresholds?
- what if threshold selection were marginalized out?

This leads to a robust objective based on the residual distribution rather than a single manually chosen cutoff.

---

## Residual Definition

For a matched pair `(p_{i,j,k}, q_{i,j,k})`:

- `p_{i,j,k}` is a pixel in image `i`
- `q_{i,j,k}` is its corresponding pixel in image `j`
- `p_{i,j,k}` is lifted into 3D using corrected depth `D'_i`
- that 3D point is projected into image `j`
- the reprojection discrepancy to `q_{i,j,k}` defines the projective residual `r_{i,j,k}`

Interpretation:

- if pose, intrinsics, and depth correction are correct, reprojection should align with the dense correspondence
- if they are wrong, residuals increase

This residual is the fundamental observation used by MBA.

---

## Objective: From Inlier Counting to Marginalized BA

### Step 1: binary thresholded score
Start from a RANSAC-like view:

```text
point is inlier if r < tau
```

The simplest score is the number of residuals below threshold `tau`.

Problem:

- non-differentiable
- highly sensitive to threshold choice
- small `tau` favors precision but hurts convergence
- large `tau` improves basin of attraction but reduces selectivity

### Step 2: interpret inlier counting statistically
Given many residuals, the number of residuals below threshold `tau` is proportional to the empirical CDF evaluated at `tau`:

```text
F(tau) = Pr[r < tau]
```

So thresholded inlier counting becomes equivalent to reading off the residual CDF at one threshold.

### Step 3: marginalize over thresholds
MBA replaces the single-threshold score with an integral over thresholds:

```text
S_m(X) = integral from 0 to tau_max of S_b(X, tau) d tau
```

Using the empirical residual distribution, this becomes proportional to:

```text
integral from 0 to tau_max of F(tau) d tau
```

Interpretation:

- maximize the area under the residual CDF up to `tau_max`
- reward solutions that improve residual quality across a range of tolerances
- remove dependence on one brittle threshold hyperparameter

This is the meaning of "marginalized" in MBA.

---

## Why the MBA Objective is Robust

The loss gradient depends on the empirical residual density `p(r)`.
This has an important effect:

- very large residuals usually lie in the tail of the distribution
- tail density is small
- therefore their gradient influence is naturally suppressed

Implication:

- outliers are downweighted automatically
- the system does not require a hand-designed fixed robust penalty to achieve this effect
- robustness emerges from the residual distribution itself

This is the most important algorithmic property of MBA.

---

## Relationship to MAGSAC

MBA is conceptually related to MAGSAC.

Shared idea:

- avoid relying on one fixed inlier threshold
- marginalize over thresholds instead

Difference:

- MAGSAC typically assumes a parametric residual model
- MBA uses the empirical residual distribution observed in dense multi-view optimization

A useful mental model:

- MAGSAC: threshold marginalization for robust sample consensus
- MBA: threshold marginalization adapted into a differentiable dense BA objective

---

## Pipeline

### 1. Precomputation
For each image and selected image pair:

- compute monocular depth
- compute dense correspondences
- build pairwise visibility / connectivity structure
- sample a fixed number of correspondences per graph edge

### 2. Pose graph construction
Create a graph where:

- node = image
- edge = sufficiently co-visible image pair

This limits optimization to meaningful pairwise relations and improves scalability.

### 3. Initialization
Initialize:

- intrinsics
- extrinsics
- affine depth parameters

The system uses geometric heuristics and graph traversal over the pose graph rather than relying on full global optimization from scratch.

### 4. Coarse optimization
The coarse stage operates on star-shaped local subgraphs.

Purpose:

- improve poor initial registrations locally
- avoid early failure from badly aligned images
- stabilize optimization before global refinement

Additional detail:

- residuals are transformed with `log(1 + r)` in the coarse stage for extra robustness

### 5. Fine optimization
After local stabilization, the method runs global optimization on the full pose graph using the MBA objective.

Result:

- refined camera parameters
- refined affine depth correction
- globally more consistent multi-view geometry

---

## Optimization Variables and Their Roles

### Camera extrinsics
Responsible for:

- image-to-world / relative camera motion
- final pose estimation quality

### Camera intrinsics
Responsible for:

- focal / principal point calibration behavior
- accurate projection geometry

### Per-image affine depth correction
Responsible for:

- compensating MDE scale and shift mismatch across frames
- aligning monocular depth predictions into a multi-view-consistent regime

This affine correction is critical because monocular depth often has inconsistent global scale and offset from frame to frame.

---

## Main Contributions

### Contribution 1
A BA objective specifically designed for dense, noisy monocular depth observations.

### Contribution 2
A formulation that converts thresholded inlier reasoning into a differentiable optimization objective using the empirical residual distribution.

### Contribution 3
A practical multi-view system that jointly optimizes camera parameters and affine depth corrections.

### Contribution 4
A unified principle that also extends to two-view robust estimation / RANSAC-style scoring.

---

## Conceptual Reframing

The paper implicitly reframes the classical SfM pipeline.

### Classical view
- correspondences -> structure and motion
- motion-from-correspondence
- structure emerges through triangulation and BA

### MBA view
- monocular depth provides approximate structure first
- motion is recovered by aligning views to that structure
- therefore: motion-from-structure

This inversion is one of the clearest ways to understand the paper.

---

## Practical Engineering Interpretation

If implementing or adapting this idea in another repository, the essential components are:

### Required modules
- monocular depth predictor
- dense correspondence predictor
- projective residual computation
- pose graph builder
- empirical residual histogram / KDE estimator
- optimizer over camera + affine depth parameters

### Optional substitutions
The framework is modular with respect to:

- which monocular depth model is used
- which correspondence / matching model is used
- which optimizer is used
- how pose graph edges are selected

### Most important abstraction
The key abstraction is **not** a specific backbone model.
The key abstraction is:

- dense geometric observations are noisy
- their residual distribution should define the robust objective

If this principle is preserved, the method can be re-instantiated with newer components.

---

## Why This Matters for Repository Refactoring

For an AI system reading this project, the most important takeaway is:

1. this is not just "depth-assisted SfM"
2. this is a new robust optimization primitive for using monocular depth in pose estimation
3. the project's main intellectual asset is the marginalized objective, not just the surrounding engineering

When refactoring, preserve the following conceptual boundaries:

### Boundary A: data providers
- monocular depth model
- correspondence model

These should remain replaceable.

### Boundary B: geometric core
- depth lifting
- reprojection
- residual computation
- pose graph organization

These define the geometry engine.

### Boundary C: robust objective
- empirical residual distribution
- CDF / histogram accumulation
- marginalized score / loss

This is the algorithmic identity of the project.

### Boundary D: optimization schedule
- initialization
- coarse local optimization
- fine global optimization

This is the convergence strategy.

A good refactor should isolate these four boundaries explicitly.

---

## Strengths

- principled use of monocular depth inside pose optimization
- robust to high-variance dense depth observations
- modular with respect to pretrained depth / matching backbones
- compatible with large multi-view pose graphs
- conceptually bridges robust estimation and bundle adjustment

---

## Limitations

- performance depends on the quality of monocular depth and dense correspondences
- affine depth correction may be insufficient when monocular depth errors are strongly nonlinear
- dense matching and dense optimization can be computationally expensive
- first-order optimization may be slower than highly optimized classical BA pipelines

---

## Minimal AI-Facing Summary

```text
MBA is a robust multi-view pose optimization framework built for monocular depth maps.

Standard BA assumes sparse and accurate geometry.
Monocular depth is dense but noisy, so standard BA is a mismatch.

MBA defines projective residuals using:
- monocular depth
- dense image correspondences
- camera intrinsics/extrinsics
- per-image affine depth correction

Instead of using one inlier threshold, MBA marginalizes over thresholds.
This is implemented by maximizing the area under the empirical residual CDF up to a cutoff.

Effect:
- reduced sensitivity to threshold choice
- automatic suppression of large-residual outliers
- robust joint optimization of camera parameters and depth correction

Core idea to preserve in code:
robustness is derived from the empirical residual distribution, not from a fixed hand-designed robust penalty alone.
```

---

## Refactor Notes for Future AI Agents

### If replacing the depth model
Check:
- output scale convention
- depth validity mask behavior
- consistency of affine correction assumptions

### If replacing the matching model
Check:
- coordinate convention
- confidence estimation
- density / sparsity tradeoff
- robustness on low-texture image pairs

### If replacing the optimizer
Preserve:
- marginalized objective semantics
- coarse-to-fine schedule
- residual distribution estimation step

### If extending to other tasks
Most natural extensions:
- two-view robust estimation
- relocalization
- hybrid systems combining feed-forward pose priors with MBA refinement

---

## Final Takeaway

The essential idea of the paper is:

- monocular depth should not be treated only as initialization
- it should be treated as a noisy dense geometric signal
- using that signal effectively requires a new BA objective
- MBA provides that objective by marginalizing over residual thresholds via the empirical residual distribution

In compact form:

```text
Classical BA fits sparse-accurate geometry.
MBA fits dense-noisy geometry.
```