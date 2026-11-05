//! Ion channels, Hodgkin-Huxley gating kinetics, and membrane biophysics.
//!
//! Employs the **Rush-Larsen exponential integrator** for unconditionally stable
//! simulation of stiff voltage-dependent gating variables even with larger time steps.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Classification of voltage- and ligand-gated ion channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ChannelType {
    /// Fast voltage-gated sodium channel ($I_{\text{Na}}$).
    NaV,
    /// Delayed-rectifier potassium channel ($I_{\text{K,DR}}$).
    KDR,
    /// Transient A-type potassium channel ($I_{\text{K,A}}$).
    KA,
    /// High-threshold L-type calcium channel ($I_{\text{Ca,L}}$).
    CaL,
    /// Low-threshold T-type calcium channel ($I_{\text{Ca,T}}$).
    CaT,
    /// Hyperpolarization-activated cyclic nucleotide-gated cation channel ($I_{\text{h}}$).
    HCN,
    /// Inward-rectifier potassium channel ($I_{\text{K,ir}}$).
    Kir,
    /// Non-specific passive leak conductance ($I_{\text{leak}}$).
    Leak,
}

/// A voltage-dependent gating variable $x \in [0, 1]$ satisfying:
/// $$\frac{dx}{dt} = \frac{x_\infty(V) - x}{\tau_x(V)} = \alpha_x(V)(1 - x) - \beta_x(V) x$$
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GatingVar {
    /// Current state value $x \in [0.0, 1.0]$.
    pub value: f64,
}

impl GatingVar {
    /// Initializes a gating variable at its steady-state value $x_\infty$.
    pub fn new(initial: f64) -> Self {
        Self {
            value: initial.clamp(0.0, 1.0),
        }
    }

    /// Advances the gating variable by $\Delta t$ (in milliseconds) using the **Rush-Larsen**
    /// exponential integration scheme:
    /// $$x(t + \Delta t) = x_\infty + (x(t) - x_\infty) \exp(-\Delta t / \tau_x)$$
    #[inline]
    pub fn update_rush_larsen(&mut self, x_inf: f64, tau_ms: f64, dt_ms: f64) {
        let tau = tau_ms.max(1e-6);
        let factor = (-dt_ms / tau).exp();
        self.value = x_inf + (self.value - x_inf) * factor;
    }

    /// Advances the gating variable using $\alpha(V)$ and $\beta(V)$ rate coefficients (in $\text{ms}^{-1}$):
    #[inline]
    pub fn update_from_rates(&mut self, alpha: f64, beta: f64, dt_ms: f64) {
        let sum = alpha + beta;
        if sum > 1e-12 {
            let x_inf = alpha / sum;
            let tau = 1.0 / sum;
            self.update_rush_larsen(x_inf, tau, dt_ms);
        }
    }
}

/// A Hodgkin-Huxley-style ion channel with activation and inactivation gates.
///
/// Current density:
/// $$I = g_{\max} \cdot m^p \cdot h^q \cdot (V - E_{\text{rev}}) \quad [\mu\text{A/cm}^2]$$
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HHChannel {
    pub channel_type: ChannelType,
    /// Maximum conductance density $g_{\max}$ in $\text{mS/cm}^2$.
    pub g_max: f64,
    /// Reversal potential $E_{\text{rev}}$ in $\text{mV}$.
    pub e_rev: f64,
    /// Activation gate variable ($m$).
    pub m: GatingVar,
    /// Activation gate power exponent $p \in \{1, 2, 3, 4\}$.
    pub m_power: usize,
    /// Inactivation gate variable ($h$).
    pub h: Option<GatingVar>,
    /// Inactivation gate power exponent $q \in \{1, 2\}$.
    pub h_power: usize,
}

impl HHChannel {
    /// Creates a classic Hodgkin-Huxley sodium channel ($I_{\text{Na}} = g_{\text{Na}} m^3 h (V - E_{\text{Na}})$).
    pub fn classic_sodium(v_init_mv: f64) -> Self {
        let (m_inf, _, h_inf, _) = Self::hh_rates_na(v_init_mv);
        Self {
            channel_type: ChannelType::NaV,
            g_max: 120.0, // mS/cm²
            e_rev: 50.0,  // mV
            m: GatingVar::new(m_inf),
            m_power: 3,
            h: Some(GatingVar::new(h_inf)),
            h_power: 1,
        }
    }

    /// Creates a classic Hodgkin-Huxley potassium channel ($I_{\text{K}} = g_{\text{K}} n^4 (V - E_{\text{K}})$).
    pub fn classic_potassium(v_init_mv: f64) -> Self {
        let (n_inf, _) = Self::hh_rates_k(v_init_mv);
        Self {
            channel_type: ChannelType::KDR,
            g_max: 36.0,  // mS/cm²
            e_rev: -77.0, // mV
            m: GatingVar::new(n_inf),
            m_power: 4,
            h: None,
            h_power: 0,
        }
    }

    /// Creates a classic passive leak channel ($I_{\text{leak}} = g_{\text{L}} (V - E_{\text{L}})$).
    pub fn classic_leak() -> Self {
        Self {
            channel_type: ChannelType::Leak,
            g_max: 0.3,     // mS/cm²
            e_rev: -54.387, // mV
            m: GatingVar::new(1.0),
            m_power: 1,
            h: None,
            h_power: 0,
        }
    }

    /// Computes instantaneous ionic current density $I$ in $\mu\text{A/cm}^2$ for membrane potential $V$ (in $\text{mV}$).
    pub fn current(&self, v_mv: f64) -> f64 {
        let mut g = self.g_max * self.m.value.powi(self.m_power as i32);
        if let Some(ref h_var) = self.h {
            g *= h_var.value.powi(self.h_power as i32);
        }
        g * (v_mv - self.e_rev)
    }

    /// Updates internal gating variables given membrane potential $V$ (in $\text{mV}$) over time step $\Delta t$ (in $\text{ms}$).
    pub fn step(&mut self, v_mv: f64, dt_ms: f64) {
        match self.channel_type {
            ChannelType::NaV => {
                let (m_inf, tau_m, h_inf, tau_h) = Self::hh_rates_na(v_mv);
                self.m.update_rush_larsen(m_inf, tau_m, dt_ms);
                if let Some(ref mut h_var) = self.h {
                    h_var.update_rush_larsen(h_inf, tau_h, dt_ms);
                }
            }
            ChannelType::KDR => {
                let (n_inf, tau_n) = Self::hh_rates_k(v_mv);
                self.m.update_rush_larsen(n_inf, tau_n, dt_ms);
            }
            ChannelType::KA => {
                // A-type transient K channel kinetics
                let a_inf = (1.0 / (1.0 + (-(v_mv + 31.4) / 23.5).exp())).powf(0.25);
                let tau_a = 0.5 + 1.0 / (1.0 + ((v_mv + 40.0) / 5.0).exp());
                self.m.update_rush_larsen(a_inf, tau_a, dt_ms);
                if let Some(ref mut h_var) = self.h {
                    let b_inf = 1.0 / (1.0 + ((v_mv + 70.0) / 6.0).exp());
                    let tau_b = 5.0 + 40.0 / (1.0 + ((v_mv + 40.0) / 5.0).exp());
                    h_var.update_rush_larsen(b_inf, tau_b, dt_ms);
                }
            }
            ChannelType::HCN => {
                // Hyperpolarization-activated Ih channel kinetics
                let h_inf = 1.0 / (1.0 + ((v_mv + 80.0) / 8.0).exp());
                let tau_h = 1000.0 / (((v_mv + 70.0) / 20.0).exp() + (-(v_mv + 70.0) / 20.0).exp());
                self.m.update_rush_larsen(h_inf, tau_h, dt_ms);
            }
            ChannelType::Leak => {}
            _ => {
                // Generic voltage dependence approximation
                let m_inf = 1.0 / (1.0 + (-(v_mv + 40.0) / 9.0).exp());
                self.m.update_rush_larsen(m_inf, 2.0, dt_ms);
            }
        }
    }

    /// Rate equations for classic Hodgkin-Huxley sodium channel:
    /// Returns `(m_inf, tau_m, h_inf, tau_h)`.
    fn hh_rates_na(v_mv: f64) -> (f64, f64, f64, f64) {
        let v = v_mv;
        let v_rel = v + 65.0; // Hodgkin-Huxley shifted to modern -65 mV baseline

        // alpha_m = 0.1 * (25 - v_rel) / (exp((25 - v_rel)/10) - 1)
        let alpha_m = if (25.0 - v_rel).abs() < 1e-6 {
            1.0
        } else {
            0.1 * (25.0 - v_rel) / (((25.0 - v_rel) / 10.0).exp() - 1.0)
        };
        let beta_m = 4.0 * (-v_rel / 18.0).exp();

        let sum_m = alpha_m + beta_m;
        let m_inf = alpha_m / sum_m;
        let tau_m = 1.0 / sum_m;

        let alpha_h = 0.07 * (-v_rel / 20.0).exp();
        let beta_h = 1.0 / (((30.0 - v_rel) / 10.0).exp() + 1.0);

        let sum_h = alpha_h + beta_h;
        let h_inf = alpha_h / sum_h;
        let tau_h = 1.0 / sum_h;

        (m_inf, tau_m, h_inf, tau_h)
    }

    /// Rate equations for classic Hodgkin-Huxley potassium channel:
    /// Returns `(n_inf, tau_n)`.
    fn hh_rates_k(v_mv: f64) -> (f64, f64) {
        let v_rel = v_mv + 65.0;
        let alpha_n = if (10.0 - v_rel).abs() < 1e-6 {
            0.1
        } else {
            0.01 * (10.0 - v_rel) / (((10.0 - v_rel) / 10.0).exp() - 1.0)
        };
        let beta_n = 0.125 * (-v_rel / 80.0).exp();

        let sum_n = alpha_n + beta_n;
        let n_inf = alpha_n / sum_n;
        let tau_n = 1.0 / sum_n;

        (n_inf, tau_n)
    }
}

/// Biophysical membrane patch combining capacitance, passive resting parameters, and ion channels.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Membrane {
    /// Specific membrane capacitance $C_m$ in $\mu\text{F/cm}^2$ (typically $\approx 1.0$).
    pub capacitance: f64,
    /// Membrane resting potential $V_{\text{rest}}$ in $\text{mV}$ (typically $-70.0$ to $-65.0$).
    pub rest_potential: f64,
    /// Active ion channels anchored within this membrane patch.
    pub channels: Vec<HHChannel>,
    /// Membrane surface area in $\text{cm}^2$.
    pub area_cm2: f64,
}

impl Membrane {
    /// Creates a default biophysical excitable membrane with classic Hodgkin-Huxley channels.
    pub fn classic_hh(v_rest_mv: f64, area_cm2: f64) -> Self {
        Self {
            capacitance: 1.0, // µF/cm²
            rest_potential: v_rest_mv,
            channels: vec![
                HHChannel::classic_sodium(v_rest_mv),
                HHChannel::classic_potassium(v_rest_mv),
                HHChannel::classic_leak(),
            ],
            area_cm2: area_cm2.max(1e-12),
        }
    }

    /// Computes total ionic current $I_{\text{total}}$ in nanoamperes ($\text{nA}$) for potential $V$ (in $\text{mV}$).
    pub fn total_ionic_current_na(&self, v_mv: f64) -> f64 {
        let current_density_ua_cm2: f64 = self.channels.iter().map(|ch| ch.current(v_mv)).sum();
        // 1 µA = 1,000 nA
        current_density_ua_cm2 * self.area_cm2 * 1_000.0
    }

    /// Advances all channel gating variables by $\Delta t$ milliseconds.
    pub fn step_channels(&mut self, v_mv: f64, dt_ms: f64) {
        for ch in &mut self.channels {
            ch.step(v_mv, dt_ms);
        }
    }
}
