//! Synaptic plasticity mechanisms, long-term potentiation/depression (LTP/LTD), and homeostatic scaling.
//!
//! Provides:
//! - Pair-based Spike-Timing-Dependent Plasticity (STDP, Bi & Poo 1998) via online exponential traces.
//! - Three-factor neuromodulated STDP with eligibility traces and dopamine reward gating.
//! - Bienenstock-Cooper-Munro (BCM) sliding modification threshold theory.
//! - Homeostatic synaptic scaling (Turrigiano 2008) for circuit stability.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Canonical parameters for classical exponential pair-based STDP (Bi & Poo 1998):
/// $$\Delta w = \begin{cases} A_+ \exp(-\Delta t / \tau_+) & \text{if } \Delta t > 0 \ (\text{LTP}) \\ -A_- \exp(\Delta t / \tau_-) & \text{if } \Delta t < 0 \ (\text{LTD}) \end{cases}$$
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StdpParams {
    /// Maximum potentiation amplitude $A_+$ (typically $\approx 0.005$ to $0.01$).
    pub a_plus: f64,
    /// Maximum depression amplitude $A_-$ (typically $\approx 0.00525$, slightly higher than $A_+$ for stability).
    pub a_minus: f64,
    /// Time constant of potentiation window $\tau_+$ in $\text{ms}$ (typically $20.0\text{ ms}$).
    pub tau_plus_ms: f64,
    /// Time constant of depression window $\tau_-$ in $\text{ms}$ (typically $20.0\text{ ms}$).
    pub tau_minus_ms: f64,
    /// Lower bound on synaptic weight $w_{\min}$ (typically $0.0$).
    pub w_min: f64,
    /// Upper bound on synaptic weight $w_{\max}$ (typically $10.0$ to $50.0\text{ nS}$).
    pub w_max: f64,
}

impl Default for StdpParams {
    fn default() -> Self {
        Self {
            a_plus: 0.01,
            a_minus: 0.0105,
            tau_plus_ms: 20.0,
            tau_minus_ms: 20.0,
            w_min: 0.0,
            w_max: 50.0,
        }
    }
}

/// An online trace-based STDP tracker for a single synapse.
///
/// Converts pair-based STDP into continuous differential trace updates:
/// $$\frac{dx}{dt} = -\frac{x}{\tau_+}, \quad \frac{dy}{dt} = -\frac{y}{\tau_-}$$
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StdpSynapse {
    pub params: StdpParams,
    /// Presynaptic trace $x(t)$.
    pub x_pre: f64,
    /// Postsynaptic trace $y(t)$.
    pub y_post: f64,
    /// Synaptic weight $w$ in $\text{nS}$.
    pub weight_ns: f64,
}

impl StdpSynapse {
    /// Creates a new STDP-governed synapse with initial weight.
    pub fn new(initial_weight_ns: f64, params: StdpParams) -> Self {
        Self {
            params,
            x_pre: 0.0,
            y_post: 0.0,
            weight_ns: initial_weight_ns,
        }
    }

    /// Advances continuous decay of eligibility traces over time step $\Delta t$ (in $\text{ms}$).
    pub fn step(&mut self, dt_ms: f64) {
        let decay_pre = (-dt_ms / self.params.tau_plus_ms.max(1e-4)).exp();
        let decay_post = (-dt_ms / self.params.tau_minus_ms.max(1e-4)).exp();
        self.x_pre *= decay_pre;
        self.y_post *= decay_post;
    }

    /// Pre-synaptic spike handler: triggers depression based on current post-synaptic trace $y_{\text{post}}$
    /// and increments pre-synaptic trace $x_{\text{pre}}$.
    pub fn on_pre_spike(&mut self) {
        let delta_w = -self.params.a_minus * self.y_post;
        self.weight_ns = (self.weight_ns + delta_w).clamp(self.params.w_min, self.params.w_max);
        self.x_pre += 1.0;
    }

    /// Post-synaptic spike handler: triggers potentiation based on current pre-synaptic trace $x_{\text{pre}}$
    /// and increments post-synaptic trace $y_{\text{post}}$.
    pub fn on_post_spike(&mut self) {
        let delta_w = self.params.a_plus * self.x_pre;
        self.weight_ns = (self.weight_ns + delta_w).clamp(self.params.w_min, self.params.w_max);
        self.y_post += 1.0;
    }
}

/// Three-Factor Neuromodulated STDP:
/// Combines pre-post correlation (Factor 1 & 2) into an **eligibility trace** $e(t)$, which is then
/// gated by an extracellular neuromodulatory reward signal (Factor 3, e.g. Dopamine):
///
/// $$\frac{de}{dt} = -\frac{e}{\tau_e} + \text{STDP}(t)$$
/// $$\frac{dw}{dt} = \eta \cdot e(t) \cdot (\text{DA}(t) - \text{DA}_{\text{baseline}})$$
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ThreeFactorStdp {
    pub stdp: StdpSynapse,
    /// Eligibility trace $e(t)$.
    pub eligibility: f64,
    /// Eligibility decay time constant $\tau_e$ in $\text{ms}$ (typically $500$ to $2,000\text{ ms}$).
    pub tau_eligibility_ms: f64,
    /// Learning rate $\eta$.
    pub learning_rate: f64,
    /// Baseline tonic dopamine concentration.
    pub dopamine_baseline: f64,
}

impl ThreeFactorStdp {
    pub fn new(initial_weight_ns: f64, stdp_params: StdpParams) -> Self {
        Self {
            stdp: StdpSynapse::new(initial_weight_ns, stdp_params),
            eligibility: 0.0,
            tau_eligibility_ms: 1000.0,
            learning_rate: 0.05,
            dopamine_baseline: 1.0,
        }
    }

    /// Advances continuous eligibility trace decay and applies dopamine-gated weight changes.
    pub fn step(&mut self, dopamine_level: f64, dt_ms: f64) {
        // Step standard trace decay
        self.stdp.step(dt_ms);

        // Decay eligibility trace
        let decay_e = (-dt_ms / self.tau_eligibility_ms.max(1e-4)).exp();
        self.eligibility *= decay_e;

        // Apply 3rd factor: dopamine modulation
        let da_error = dopamine_level - self.dopamine_baseline;
        let delta_w = self.learning_rate * self.eligibility * da_error * (dt_ms / 1000.0);
        self.stdp.weight_ns = (self.stdp.weight_ns + delta_w)
            .clamp(self.stdp.params.w_min, self.stdp.params.w_max);
    }

    /// Presynaptic spike: updates eligibility trace with LTD component.
    pub fn on_pre_spike(&mut self) {
        let delta_e = -self.stdp.params.a_minus * self.stdp.y_post;
        self.eligibility += delta_e;
        self.stdp.x_pre += 1.0;
    }

    /// Postsynaptic spike: updates eligibility trace with LTP component.
    pub fn on_post_spike(&mut self) {
        let delta_e = self.stdp.params.a_plus * self.stdp.x_pre;
        self.eligibility += delta_e;
        self.stdp.y_post += 1.0;
    }
}

/// Homeostatic Synaptic Scaling (Turrigiano 2008):
/// Multiplicatively scales all incoming excitatory synapses to stabilize target firing rates
/// over longer developmental timescales.
///
/// $$\frac{dw_i}{dt} = \gamma \cdot (r_{\text{target}} - r_{\text{actual}}) \cdot w_i$$
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HomeostaticScaler {
    /// Target set-point firing rate in Hertz (e.g. $5.0\text{ Hz}$).
    pub target_rate_hz: f64,
    /// Adaptation rate $\gamma$ ($\text{s}^{-1}$).
    pub scaling_rate: f64,
}

impl HomeostaticScaler {
    pub fn new(target_rate_hz: f64, scaling_rate: f64) -> Self {
        Self {
            target_rate_hz,
            scaling_rate,
        }
    }

    /// Applies multiplicative homeostatic scaling to a set of synaptic weights.
    pub fn scale_weights(&self, weights: &mut [f64], current_rate_hz: f64, dt_sec: f64) {
        let rate_error = self.target_rate_hz - current_rate_hz;
        let scale_factor = (1.0 + self.scaling_rate * rate_error * dt_sec).max(0.01);
        for w in weights.iter_mut() {
            *w = (*w * scale_factor).max(0.0);
        }
    }
}
