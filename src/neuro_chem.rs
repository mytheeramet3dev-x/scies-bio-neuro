//! Neurotransmitter chemistry, enzymatic breakdown, transporter reuptake, and volume transmission.
//!
//! Models:
//! - Canonical neuromodulators and transmitters: Glutamate, GABA, Dopamine, Serotonin, Acetylcholine, Norepinephrine.
//! - Michaelis-Menten transporter clearance kinetics (e.g., DAT, SERT, EAAT):
//!   $$V = \frac{V_{\max} [S]}{K_m + [S]}$$
//! - 3D Volume Transmission diffusion in porous extracellular brain tissue (Nicholson 2001)
//!   incorporating volume fraction $\alpha \approx 0.2$ and tortuosity $\lambda \approx 1.6$:
//!   $$C(r, t) = \frac{Q}{\alpha (4\pi D^* t)^{3/2}} \exp\left(-\frac{r^2}{4 D^* t}\right)$$

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Canonical neurotransmitter chemical identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Neurotransmitter {
    /// L-Glutamate (primary excitatory transmitter).
    Glutamate,
    /// $\gamma$-Aminobutyric acid (primary inhibitory transmitter).
    GABA,
    /// Dopamine (reward prediction error, motor coordination).
    Dopamine,
    /// Serotonin / 5-Hydroxytryptamine (mood, affect, sensory gating).
    Serotonin,
    /// Acetylcholine (attention, memory consolidation, neuromuscular junction).
    Acetylcholine,
    /// Norepinephrine / Noradrenaline (arousal, gain control).
    Norepinephrine,
}

/// Transporter reuptake kinetics based on Michaelis-Menten dynamics:
/// $$v_{\text{reuptake}} = \frac{V_{\max} \cdot C}{K_m + C}$$
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TransporterKinetics {
    /// Maximum transport velocity $V_{\max}$ in $\mu\text{M/s}$.
    pub v_max_um_s: f64,
    /// Michaelis affinity constant $K_m$ in $\mu\text{M}$.
    pub k_m_um: f64,
}

impl TransporterKinetics {
    /// Canonical Dopamine Transporter (DAT): $V_{\max} \approx 4.0\ \mu\text{M/s}$, $K_m \approx 0.2\ \mu\text{M}$.
    pub fn dat() -> Self {
        Self {
            v_max_um_s: 4.0,
            k_m_um: 0.2,
        }
    }

    /// Canonical Serotonin Transporter (SERT): $V_{\max} \approx 1.5\ \mu\text{M/s}$, $K_m \approx 0.15\ \mu\text{M}$.
    pub fn sert() -> Self {
        Self {
            v_max_um_s: 1.5,
            k_m_um: 0.15,
        }
    }

    /// Canonical Excitatory Amino Acid Transporter (EAAT): $V_{\max} \approx 10.0\ \mu\text{M/s}$, $K_m \approx 2.0\ \mu\text{M}$.
    pub fn eaat() -> Self {
        Self {
            v_max_um_s: 10.0,
            k_m_um: 2.0,
        }
    }

    /// Computes clearance velocity in $\mu\text{M/s}$ given extracellular substrate concentration $C$ ($\mu\text{M}$).
    pub fn clearance_rate(&self, concentration_um: f64) -> f64 {
        let c = concentration_um.max(0.0);
        (self.v_max_um_s * c) / (self.k_m_um + c)
    }
}

/// Extracellular volume transmission diffusion solver (Nicholson 2001).
///
/// Models 3D non-synaptic paracrine diffusion of neurotransmitters through the brain extracellular space (ECS).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct VolumeTransmissionDiffusion {
    /// Free diffusion coefficient in water $D$ in $\mu\text{m}^2/\text{ms}$ (typically $\approx 0.5 - 0.8$).
    pub free_diffusion_d: f64,
    /// Tissue volume fraction $\alpha = V_{\text{ECS}} / V_{\text{total}}$ (typically $\approx 0.20$).
    pub volume_fraction_alpha: f64,
    /// Tissue tortuosity $\lambda$ (typically $\approx 1.6$, reducing apparent diffusion $D^* = D / \lambda^2$).
    pub tortuosity_lambda: f64,
    /// Non-specific clearance rate constant $k'$ in $\text{ms}^{-1}$.
    pub clearance_rate_k: f64,
}

impl Default for VolumeTransmissionDiffusion {
    fn default() -> Self {
        Self {
            free_diffusion_d: 0.76,      // Dopamine free diffusion in aqueous solution (~7.6e-6 cm²/s)
            volume_fraction_alpha: 0.20, // 20% extracellular volume fraction
            tortuosity_lambda: 1.60,     // Effective tortuosity of cortical neuropil
            clearance_rate_k: 0.005,     // ~200 ms clearance half-life
        }
    }
}

impl VolumeTransmissionDiffusion {
    /// Apparent diffusion coefficient $D^* = D / \lambda^2$.
    pub fn apparent_diffusion(&self) -> f64 {
        self.free_diffusion_d / self.tortuosity_lambda.powi(2)
    }

    /// Evaluates extracellular concentration $C(r, t)$ in nanomolar ($\text{nM}$) at distance $r$ ($\mu\text{m}$)
    /// and elapsed time $t$ ($\text{ms}$) following point-source burst release of $Q$ molecules:
    ///
    /// $$C(r, t) = \frac{Q}{\alpha (4\pi D^* t)^{3/2}} \exp\left(-\frac{r^2}{4 D^* t}\right) \exp(-k' t)$$
    pub fn concentration_at(&self, q_molecules: f64, distance_um: f64, time_ms: f64) -> f64 {
        if time_ms <= 1e-4 {
            return 0.0;
        }

        let d_star = self.apparent_diffusion();
        let four_pi_d_t = 4.0 * std::f64::consts::PI * d_star * time_ms;
        let denominator = self.volume_fraction_alpha * four_pi_d_t.powf(1.5);

        let spatial_decay = (-distance_um.powi(2) / (4.0 * d_star * time_ms)).exp();
        let elimination_decay = (-self.clearance_rate_k * time_ms).exp();

        // Convert molecules / µm³ to concentration (1 molecule / µm³ ≈ 1.66 nM)
        let density = (q_molecules / denominator) * spatial_decay * elimination_decay;
        density * 1.66
    }
}
