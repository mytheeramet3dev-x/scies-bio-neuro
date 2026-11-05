//! Neural development, growth cone guidance, myelination biophysics, and structural plasticity.
//!
//! Provides:
//! - Chemotropic growth cone steering in gradient fields (e.g. Netrin, Semaphorin, Slit).
//! - Myelination biophysics, Rushton optimal $g$-ratio ($d_{\text{axon}} / D_{\text{fiber}} \approx 0.60 - 0.77$),
//!   and conduction velocity scaling ($v \propto \sqrt{d}$ unmyelinated vs $v \propto d$ myelinated).
//! - Activity-dependent synaptogenesis and developmental synaptic pruning.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Chemotropic guidance molecular cue types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum GuidanceCue {
    /// Netrin: typically chemoattractive to commissural axons.
    Netrin,
    /// Semaphorin: primarily chemorepulsive guidance cue.
    Semaphorin,
    /// Slit: chemorepulsive cue acting through Robo receptors preventing midline crossing.
    Slit,
    /// Ephrin: contact-mediated bidirectional attraction or repulsion.
    Ephrin,
}

/// An axonal growth cone navigating through an extracellular biochemical gradient field.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GrowthCone {
    /// 3D tip position in micrometers.
    pub position_um: [f64; 3],
    /// Unit velocity direction vector $(\hat{x}, \hat{y}, \hat{z})$.
    pub direction: [f64; 3],
    /// Elongation growth rate in $\mu\text{m/hour}$ (typically $20.0$ to $60.0$).
    pub growth_rate_um_h: f64,
}

impl GrowthCone {
    pub fn new(position_um: [f64; 3], direction: [f64; 3]) -> Self {
        let norm = (direction[0].powi(2) + direction[1].powi(2) + direction[2].powi(2))
            .sqrt()
            .max(1e-9);
        Self {
            position_um,
            direction: [
                direction[0] / norm,
                direction[1] / norm,
                direction[2] / norm,
            ],
            growth_rate_um_h: 30.0,
        }
    }

    /// Steers the growth cone along concentration gradient $\nabla C$ with cue sensitivity $\chi$.
    /// - $\chi > 0$: chemoattraction
    /// - $\chi < 0$: chemorepulsion
    pub fn steer_and_grow(&mut self, gradient: [f64; 3], sensitivity: f64, dt_hours: f64) {
        let mut new_dir = [
            self.direction[0] + sensitivity * gradient[0],
            self.direction[1] + sensitivity * gradient[1],
            self.direction[2] + sensitivity * gradient[2],
        ];

        let norm = (new_dir[0].powi(2) + new_dir[1].powi(2) + new_dir[2].powi(2))
            .sqrt()
            .max(1e-9);
        new_dir[0] /= norm;
        new_dir[1] /= norm;
        new_dir[2] /= norm;

        self.direction = new_dir;
        let step_dist = self.growth_rate_um_h * dt_hours;
        self.position_um[0] += self.direction[0] * step_dist;
        self.position_um[1] += self.direction[1] * step_dist;
        self.position_um[2] += self.direction[2] * step_dist;
    }
}

/// Biophysical parameters of an axonal fiber and its myelin sheath.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MyelinSheath {
    /// Inner axon diameter $d$ in micrometers ($\mu\text{m}$).
    pub axon_diameter_um: f64,
    /// Outer myelinated fiber diameter $D$ in micrometers ($\mu\text{m}$).
    pub fiber_diameter_um: f64,
    /// Number of wrapped myelin membrane lamellae.
    pub lamellae_count: usize,
}

impl MyelinSheath {
    /// Creates a myelinated axon with optimal Rushton $g$-ratio ($g \approx 0.65$).
    pub fn optimal_myelinated(axon_diameter_um: f64) -> Self {
        let g_ratio = 0.65;
        let fiber_diameter_um = axon_diameter_um / g_ratio;
        let lamellae_count = ((fiber_diameter_um - axon_diameter_um) / 0.035).round() as usize; // ~17 nm per bilayer
        Self {
            axon_diameter_um,
            fiber_diameter_um,
            lamellae_count: lamellae_count.max(2),
        }
    }

    /// Computes the biophysical $g$-ratio:
    /// $$g = \frac{d_{\text{axon}}}{D_{\text{fiber}}}$$
    pub fn g_ratio(&self) -> f64 {
        if self.fiber_diameter_um > 1e-4 {
            self.axon_diameter_um / self.fiber_diameter_um
        } else {
            1.0
        }
    }

    /// Computes the internodal length $L_{\text{internode}}$ between nodes of Ranvier (in $\mu\text{m}$):
    /// In mammalian peripheral and central nervous systems: $L \approx 100 \times d_{\text{axon}}$.
    pub fn internodal_length_um(&self) -> f64 {
        100.0 * self.axon_diameter_um
    }

    /// Calculates action potential conduction velocity in meters per second ($\text{m/s}$).
    /// - Unmyelinated: $v = 1.0 \times \sqrt{d_{\mu\text{m}}}$
    /// - Myelinated: $v = 6.0 \times d_{\mu\text{m}} \times f(g)$, where $f(g)$ peaks around $g \approx 0.65 - 0.77$.
    pub fn conduction_velocity_m_s(&self, is_myelinated: bool) -> f64 {
        let d = self.axon_diameter_um.max(0.1);
        if !is_myelinated || self.lamellae_count == 0 {
            1.0 * d.sqrt()
        } else {
            let g = self.g_ratio().clamp(0.2, 0.95);
            // Rushton velocity factor f(g) = -g * ln(g) normalized
            let g_factor = (-g * g.ln()) / (-0.65 * 0.65f64.ln());
            6.0 * d * g_factor
        }
    }
}

/// Synaptic pruning rule based on activity and critical threshold.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SynapticPruner {
    /// Synaptic weight threshold below which elimination occurs.
    pub elimination_threshold_ns: f64,
    /// Minimum average activation count required during developmental window.
    pub min_activity_count: usize,
}

impl SynapticPruner {
    pub fn new(elimination_threshold_ns: f64, min_activity_count: usize) -> Self {
        Self {
            elimination_threshold_ns,
            min_activity_count,
        }
    }

    /// Evaluates whether a synapse should be pruned.
    pub fn should_prune(&self, weight_ns: f64, activity_count: usize) -> bool {
        weight_ns < self.elimination_threshold_ns || activity_count < self.min_activity_count
    }
}
