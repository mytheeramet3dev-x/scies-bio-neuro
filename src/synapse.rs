//! Synaptic transmission, receptor kinetics, short-term plasticity, and axonal conduction delays.
//!
//! Includes support for:
//! - AMPA, NMDA (with voltage-dependent $\text{Mg}^{2+}$ block), $\text{GABA}_A$, and $\text{GABA}_B$ receptors.
//! - Electrical gap junctions (bidirectional ohmic coupling).
//! - Tsodyks-Markram (1998) short-term synaptic plasticity (depression & facilitation).
//! - Fixed-capacity circular ring buffer (`AxonalDelayBuffer`) for arbitrary axonal conduction distances
//!   (e.g., 5 cm conduction delays).

use crate::types::{NeuronId, SynapseId};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Receptor type and biophysical conductance model for chemical transmission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ReceptorType {
    /// Fast ionotropic glutamate receptor ($\alpha$-amino-3-hydroxy-5-methyl-4-isoxazolepropionic acid).
    AMPA,
    /// Slow ionotropic glutamate receptor with voltage-dependent magnesium unblock ($N$-methyl-D-aspartate).
    NMDA,
    /// Fast ionotropic $\gamma$-aminobutyric acid receptor ($\text{Cl}^-$ permeable, hyperpolarizing).
    GabaA,
    /// Slow metabotropic $\gamma$-aminobutyric acid receptor ($\text{K}^+$ permeable, hyperpolarizing).
    GabaB,
}

/// Kinetic time constants and reversal potential for a specific receptor type.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ReceptorKineticParams {
    /// Conductance rise time constant $\tau_{\text{rise}}$ in $\text{ms}$.
    pub tau_rise_ms: f64,
    /// Conductance decay time constant $\tau_{\text{decay}}$ in $\text{ms}$.
    pub tau_decay_ms: f64,
    /// Synaptic reversal potential $E_{\text{rev}}$ in $\text{mV}$.
    pub e_rev_mv: f64,
}

impl ReceptorKineticParams {
    /// Canonical AMPA kinetics: $\tau_{\text{rise}} = 0.5\text{ ms}$, $\tau_{\text{decay}} = 2.5\text{ ms}$, $E_{\text{rev}} = 0.0\text{ mV}$.
    pub fn ampa() -> Self {
        Self {
            tau_rise_ms: 0.5,
            tau_decay_ms: 2.5,
            e_rev_mv: 0.0,
        }
    }

    /// Canonical NMDA kinetics: $\tau_{\text{rise}} = 2.0\text{ ms}$, $\tau_{\text{decay}} = 80.0\text{ ms}$, $E_{\text{rev}} = 0.0\text{ mV}$.
    pub fn nmda() -> Self {
        Self {
            tau_rise_ms: 2.0,
            tau_decay_ms: 80.0,
            e_rev_mv: 0.0,
        }
    }

    /// Canonical $\text{GABA}_A$ kinetics: $\tau_{\text{rise}} = 0.5\text{ ms}$, $\tau_{\text{decay}} = 6.0\text{ ms}$, $E_{\text{rev}} = -70.0\text{ mV}$.
    pub fn gaba_a() -> Self {
        Self {
            tau_rise_ms: 0.5,
            tau_decay_ms: 6.0,
            e_rev_mv: -70.0,
        }
    }

    /// Canonical $\text{GABA}_B$ kinetics: $\tau_{\text{rise}} = 25.0\text{ ms}$, $\tau_{\text{decay}} = 150.0\text{ ms}$, $E_{\text{rev}} = -90.0\text{ mV}$.
    pub fn gaba_b() -> Self {
        Self {
            tau_rise_ms: 25.0,
            tau_decay_ms: 150.0,
            e_rev_mv: -90.0,
        }
    }
}

/// Tsodyks-Markram (1998) model for short-term synaptic plasticity (STP):
/// Simulates presynaptic vesicle depletion (depression) and calcium accumulation (facilitation).
///
/// Dynamic state:
/// - $u$: Fraction of available resources utilized per spike ($u \in [0, 1]$).
/// - $R$: Fraction of available vesicle pool ($R \in [0, 1]$).
///
/// Differential dynamics between spikes:
/// $$\frac{du}{dt} = \frac{U - u}{\tau_{\text{facil}}}, \quad \frac{dR}{dt} = \frac{1 - R}{\tau_{\text{rec}}}$$
///
/// Upon arrival of a pre-synaptic action potential:
/// $$u \leftarrow u + U(1 - u), \quad \Delta g \propto u \cdot R, \quad R \leftarrow R - u \cdot R$$
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TsodyksMarkram {
    /// Baseline release probability $U \in (0, 1]$.
    pub u_baseline: f64,
    /// Recovery time constant from depression $\tau_{\text{rec}}$ in $\text{ms}$.
    pub tau_rec_ms: f64,
    /// Facilitation time constant $\tau_{\text{facil}}$ in $\text{ms}$.
    pub tau_facil_ms: f64,
    /// Current utilization variable $u$.
    pub u: f64,
    /// Current available vesicle pool fraction $R$.
    pub r: f64,
}

impl TsodyksMarkram {
    /// Creates a neutral (non-depressing, non-facilitating) static synapse configuration.
    pub fn static_synapse() -> Self {
        Self {
            u_baseline: 1.0,
            tau_rec_ms: 1.0,
            tau_facil_ms: 0.0,
            u: 1.0,
            r: 1.0,
        }
    }

    /// Short-term depression (typical for cortical excitatory synapses between pyramidal cells):
    /// $U = 0.5$, $\tau_{\text{rec}} = 800\text{ ms}$, $\tau_{\text{facil}} = 20\text{ ms}$.
    pub fn depressing() -> Self {
        Self {
            u_baseline: 0.5,
            tau_rec_ms: 800.0,
            tau_facil_ms: 20.0,
            u: 0.5,
            r: 1.0,
        }
    }

    /// Short-term facilitation (typical for pyramidal-to-Martinotti interneuron connections):
    /// $U = 0.1$, $\tau_{\text{rec}} = 100\text{ ms}$, $\tau_{\text{facil}} = 1000\text{ ms}$.
    pub fn facilitating() -> Self {
        Self {
            u_baseline: 0.1,
            tau_rec_ms: 100.0,
            tau_facil_ms: 1000.0,
            u: 0.1,
            r: 1.0,
        }
    }

    /// Advances the recovery and facilitation states over time $\Delta t$ (in $\text{ms}$).
    pub fn step(&mut self, dt_ms: f64) {
        if self.tau_facil_ms > 1e-4 {
            let du = (self.u_baseline - self.u) * (dt_ms / self.tau_facil_ms);
            self.u = (self.u + du).clamp(0.0, 1.0);
        } else {
            self.u = self.u_baseline;
        }

        if self.tau_rec_ms > 1e-4 {
            let dr = (1.0 - self.r) * (dt_ms / self.tau_rec_ms);
            self.r = (self.r + dr).clamp(0.0, 1.0);
        } else {
            self.r = 1.0;
        }
    }

    /// Triggers a presynaptic spike and returns the transmission transmission efficiency factor $\in [0, 1]$.
    pub fn on_spike(&mut self) -> f64 {
        if self.tau_facil_ms > 1e-4 {
            self.u = (self.u + self.u_baseline * (1.0 - self.u)).clamp(0.0, 1.0);
        } else {
            self.u = self.u_baseline;
        }

        let transmission_weight = self.u * self.r;
        self.r = (self.r - transmission_weight).clamp(0.0, 1.0);
        transmission_weight
    }
}

/// A zero-allocation circular ring buffer for queuing spikes across arbitrary axonal conduction delays.
///
/// For a 5 cm ($50,000\ \mu\text{m}$) axon:
/// - Unmyelinated ($v \approx 1\text{ m/s}$): Delay $\approx 50\text{ ms}$. At $\Delta t = 0.1\text{ ms}$, $500$ ticks.
/// - Myelinated ($v \approx 25\text{ m/s}$): Delay $\approx 2.0\text{ ms}$. At $\Delta t = 0.1\text{ ms}$, $20$ ticks.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AxonalDelayBuffer {
    /// Ring buffer slots storing the accumulated transmission weights scheduled for arrival.
    buffer: Vec<f64>,
    /// Index corresponding to the current simulation time step.
    head: usize,
    /// Total delay in discrete simulation steps (ticks).
    delay_ticks: usize,
}

impl AxonalDelayBuffer {
    /// Creates a delay buffer given total delay in milliseconds and simulation time step $\Delta t$.
    pub fn new(delay_ms: f64, dt_ms: f64) -> Self {
        let dt = dt_ms.max(1e-4);
        let ticks = (delay_ms / dt).round().max(1.0) as usize;
        Self {
            buffer: vec![0.0; ticks],
            head: 0,
            delay_ticks: ticks,
        }
    }

    /// Pushes a presynaptic spike event with transmission efficiency $W$ into the future arrival slot.
    pub fn push_spike(&mut self, transmission_weight: f64) {
        let arrival_index = (self.head + self.delay_ticks - 1) % self.buffer.len();
        self.buffer[arrival_index] += transmission_weight;
    }

    /// Advances the buffer by one tick and returns the accumulated transmission weight arriving at the post-synapse.
    pub fn step(&mut self) -> f64 {
        let arriving_weight = self.buffer[self.head];
        self.buffer[self.head] = 0.0;
        self.head = (self.head + 1) % self.buffer.len();
        arriving_weight
    }

    /// Returns the programmed delay in discrete ticks.
    pub fn delay_ticks(&self) -> usize {
        self.delay_ticks
    }
}

/// Chemical synapse model with dual-exponential conductance kinetics and short-term plasticity.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ChemicalSynapse {
    pub id: SynapseId,
    pub pre_id: NeuronId,
    pub post_id: NeuronId,
    pub receptor: ReceptorType,
    pub kinetics: ReceptorKineticParams,
    /// Maximum synaptic peak conductance $g_{\max}$ in nanosiemens ($\text{nS}$).
    pub weight_ns: f64,
    /// Rising conductance component $g_1$ in $\text{nS}$.
    pub g_rise: f64,
    /// Decaying conductance component $g_2$ in $\text{nS}$.
    pub g_decay: f64,
    /// Conduction delay queue for action potential propagation.
    pub delay_buffer: AxonalDelayBuffer,
    /// Short-term plasticity model (Tsodyks-Markram).
    pub stp: TsodyksMarkram,
    /// Extracellular Magnesium concentration $[\text{Mg}^{2+}]_o$ in $\text{mM}$ (typically $1.0$ to $1.2$).
    pub mg_conc_mm: f64,
    /// Normalization factor ensuring peak of $(g_{\text{decay}} - g_{\text{rise}})$ equals $1.0$.
    pub normalization_factor: f64,
}

impl ChemicalSynapse {
    /// Creates a chemical synapse between two neurons.
    pub fn new(
        id: SynapseId,
        pre_id: NeuronId,
        post_id: NeuronId,
        receptor: ReceptorType,
        weight_ns: f64,
        delay_ms: f64,
        dt_ms: f64,
    ) -> Self {
        let kinetics = match receptor {
            ReceptorType::AMPA => ReceptorKineticParams::ampa(),
            ReceptorType::NMDA => ReceptorKineticParams::nmda(),
            ReceptorType::GabaA => ReceptorKineticParams::gaba_a(),
            ReceptorType::GabaB => ReceptorKineticParams::gaba_b(),
        };

        // Compute dual exponential peak normalization factor:
        let tr = kinetics.tau_rise_ms;
        let td = kinetics.tau_decay_ms;
        let norm = if (td - tr).abs() > 1e-4 {
            let t_peak = (tr * td / (td - tr)) * (td / tr).ln();
            let peak_val = (-t_peak / td).exp() - (-t_peak / tr).exp();
            if peak_val > 1e-6 {
                1.0 / peak_val
            } else {
                1.0
            }
        } else {
            1.0
        };

        Self {
            id,
            pre_id,
            post_id,
            receptor,
            kinetics,
            weight_ns,
            g_rise: 0.0,
            g_decay: 0.0,
            delay_buffer: AxonalDelayBuffer::new(delay_ms, dt_ms),
            stp: TsodyksMarkram::static_synapse(),
            mg_conc_mm: 1.0,
            normalization_factor: norm,
        }
    }

    /// Voltage-dependent magnesium unblock factor $B(V)$ for NMDA receptors:
    /// $$B(V) = \frac{1}{1 + \frac{[\text{Mg}^{2+}]_o}{3.57} \exp(-0.062 \cdot V)}$$
    #[inline]
    pub fn nmda_mg_block(v_post_mv: f64, mg_conc_mm: f64) -> f64 {
        1.0 / (1.0 + (mg_conc_mm / 3.57) * (-0.062 * v_post_mv).exp())
    }

    /// Notifies the synapse that the presynaptic neuron emitted an action potential.
    pub fn on_presynaptic_spike(&mut self) {
        let transmission_weight = self.stp.on_spike();
        self.delay_buffer.push_spike(transmission_weight);
    }

    /// Advances the synaptic state by $\Delta t$ (in $\text{ms}$) and returns post-synaptic current $I_{\text{syn}}$ in $\text{nA}$.
    ///
    /// Note:
    /// Current convention: Excitatory inward current depolarizes post-synaptic membrane.
    /// In neuronal membrane ODE: $C_m \frac{dV}{dt} = -I_{\text{ion}} + I_{\text{ext}} - I_{\text{syn}}$ where
    /// $I_{\text{syn}} = g(t)(V - E_{\text{rev}})$, or equivalently, input current $I = -g(t)(V - E_{\text{rev}})$.
    /// This method returns the current injected into the post-synaptic neuron in nanoamperes ($\text{nA}$).
    pub fn step(&mut self, v_post_mv: f64, dt_ms: f64) -> f64 {
        // Step short-term plasticity
        self.stp.step(dt_ms);

        // Step delay buffer and collect arriving spikes
        let arriving_weight = self.delay_buffer.step();
        if arriving_weight > 0.0 {
            let delta_g = self.weight_ns * arriving_weight * self.normalization_factor;
            self.g_rise += delta_g;
            self.g_decay += delta_g;
        }

        // Exponential decay of conductance components
        let decay_factor_rise = (-dt_ms / self.kinetics.tau_rise_ms.max(1e-4)).exp();
        let decay_factor_decay = (-dt_ms / self.kinetics.tau_decay_ms.max(1e-4)).exp();
        self.g_rise *= decay_factor_rise;
        self.g_decay *= decay_factor_decay;

        // Net conductance in nS
        let mut g_net = (self.g_decay - self.g_rise).max(0.0);

        // Apply NMDA voltage-dependent magnesium block if NMDA receptor
        if self.receptor == ReceptorType::NMDA {
            g_net *= Self::nmda_mg_block(v_post_mv, self.mg_conc_mm);
        }

        // I = -g * (V - E_rev) in nA
        // Conductance in nS (1e-9 S) * Voltage in mV (1e-3 V) = 1e-12 A = 1e-3 nA.
        // Wait: 1 nS * 1 mV = 1e-9 A/V * 1e-3 V = 1e-12 A = 0.001 nA = 1 pA.
        // Let's verify standard unit conversion:
        // If g = 10 nS, (V - Erev) = -70 mV -> g * dV = 10 * (-70) = -700 pA = -0.7 nA.
        // Depolarizing current injected = -g * (V - E_rev) * 1e-3 nA.
        let i_syn_na = -g_net * (v_post_mv - self.kinetics.e_rev_mv) * 1e-3;
        i_syn_na
    }
}

/// Electrical gap junction connecting two neurons with bidirectional ohmic coupling.
///
/// Current transmitted to post-synaptic neuron:
/// $$I_{\text{gap}} = g_{\text{gap}} \cdot (V_{\text{pre}} - V_{\text{post}})$$
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GapJunction {
    pub id: SynapseId,
    pub neuron_a: NeuronId,
    pub neuron_b: NeuronId,
    /// Coupling conductance $g_{\text{gap}}$ in nanosiemens ($\text{nS}$).
    pub conductance_ns: f64,
}

impl GapJunction {
    pub fn new(id: SynapseId, neuron_a: NeuronId, neuron_b: NeuronId, conductance_ns: f64) -> Self {
        Self {
            id,
            neuron_a,
            neuron_b,
            conductance_ns,
        }
    }

    /// Computes the ohmic current in nanoamperes ($\text{nA}$) injected into neuron A from neuron B.
    pub fn current_into_a(&self, v_a_mv: f64, v_b_mv: f64) -> f64 {
        self.conductance_ns * (v_b_mv - v_a_mv) * 1e-3
    }

    /// Computes the ohmic current in nanoamperes ($\text{nA}$) injected into neuron B from neuron A.
    pub fn current_into_b(&self, v_a_mv: f64, v_b_mv: f64) -> f64 {
        self.conductance_ns * (v_a_mv - v_b_mv) * 1e-3
    }
}

/// Unified synaptic connection within a neural circuit.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Synapse {
    Chemical(ChemicalSynapse),
    Electrical(GapJunction),
}

impl Synapse {
    /// Returns the unique synapse identifier.
    pub fn id(&self) -> SynapseId {
        match self {
            Synapse::Chemical(s) => s.id,
            Synapse::Electrical(g) => g.id,
        }
    }
}
