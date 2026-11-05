//! Single neuron computational models.
//!
//! Provides multiple levels of biological abstraction:
//! 1. **Izhikevich (2003)**: Hybrid quadratic integrate-and-fire model balancing numerical speed
//!    with the capacity to reproduce all 20 known biological cortical firing patterns.
//! 2. **Biophysical Hodgkin-Huxley**: Continuous multi-channel membrane action potential dynamics.
//! 3. **Adaptive Leaky Integrate-and-Fire (AdLIF)**: Fast linear subthreshold integration with
//!    spike-frequency adaptation (SFA) and absolute refractory periods.
//! 4. **Multi-Compartment Cable Neuron**: Dendritic, somatic, and axonal branching with axial
//!    resistance coupling ($R_a$).

use crate::ion::Membrane;
use crate::types::{CompartmentId, CompartmentKind, NeuronId, NeuronType};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Canonical parameters for the Izhikevich (2003) quadratic spiking model:
/// $$\frac{dv}{dt} = 0.04 v^2 + 5 v + 140 - u + I$$
/// $$\frac{du}{dt} = a(bv - u)$$
/// $$\text{if } v \ge 30\text{ mV, then } v \leftarrow c,\ u \leftarrow u + d$$
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct IzhikevichParams {
    /// Time scale of the recovery variable $u$ (typically $0.02$).
    pub a: f64,
    /// Sensitivity of the recovery variable $u$ to subthreshold fluctuations of $v$ (typically $0.2$).
    pub b: f64,
    /// After-spike reset value of the membrane potential $v$ (in $\text{mV}$, typically $-65.0$).
    pub c: f64,
    /// After-spike reset increment of the recovery variable $u$ (typically $8.0$ or $2.0$).
    pub d: f64,
}

impl IzhikevichParams {
    /// Regular Spiking (RS) excitatory cortical pyramidal neuron.
    pub fn regular_spiking() -> Self {
        Self {
            a: 0.02,
            b: 0.20,
            c: -65.0,
            d: 8.0,
        }
    }

    /// Fast Spiking (FS) inhibitory parvalbumin-positive interneuron.
    pub fn fast_spiking() -> Self {
        Self {
            a: 0.10,
            b: 0.20,
            c: -65.0,
            d: 2.0,
        }
    }

    /// Intrinsically Bursting (IB) cortical layer 5 pyramidal neuron.
    pub fn intrinsically_bursting() -> Self {
        Self {
            a: 0.02,
            b: 0.20,
            c: -55.0,
            d: 4.0,
        }
    }

    /// Chattering (CH) high-frequency rhythmic bursting neuron.
    pub fn chattering() -> Self {
        Self {
            a: 0.02,
            b: 0.20,
            c: -50.0,
            d: 2.0,
        }
    }

    /// Low-Threshold Spiking (LTS) somatostatin inhibitory interneuron.
    pub fn low_threshold_spiking() -> Self {
        Self {
            a: 0.02,
            b: 0.25,
            c: -65.0,
            d: 2.0,
        }
    }

    /// Thalamo-Cortical (TC) relay neuron (bistable tonic/burst).
    pub fn thalamo_cortical() -> Self {
        Self {
            a: 0.02,
            b: 0.25,
            c: -65.0,
            d: 0.05,
        }
    }
}

/// Adaptive Leaky Integrate-and-Fire (AdLIF) parameters with spike-frequency adaptation (SFA).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AdLifParams {
    /// Membrane time constant $\tau_m = R_m C_m$ (in $\text{ms}$, typically $15.0$ to $20.0$).
    pub tau_m: f64,
    /// Membrane resistance $R_m$ in $\text{M}\Omega$ (typically $10.0$ to $50.0$).
    pub r_m: f64,
    /// Resting membrane potential $V_{\text{rest}}$ in $\text{mV}$ (typically $-70.0$).
    pub v_rest: f64,
    /// Action potential firing threshold $V_{\text{thresh}}$ in $\text{mV}$ (typically $-50.0$).
    pub v_thresh: f64,
    /// Reset membrane potential $V_{\text{reset}}$ in $\text{mV}$ (typically $-65.0$).
    pub v_reset: f64,
    /// Absolute refractory period $\tau_{\text{ref}}$ in $\text{ms}$ (typically $2.0$).
    pub tau_ref: f64,
    /// Adaptation time constant $\tau_w$ in $\text{ms}$ (typically $100.0$).
    pub tau_w: f64,
    /// Adaptation current increment per spike $b$ in $\text{nA}$ (typically $0.05$).
    pub b_adapt: f64,
}

impl Default for AdLifParams {
    fn default() -> Self {
        Self {
            tau_m: 20.0,
            r_m: 10.0,
            v_rest: -70.0,
            v_thresh: -50.0,
            v_reset: -65.0,
            tau_ref: 2.0,
            tau_w: 120.0,
            b_adapt: 0.05,
        }
    }
}

/// A morphological compartment in a multi-compartment cable neuron.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Compartment {
    pub id: CompartmentId,
    pub kind: CompartmentKind,
    /// Length of the cylindrical cylinder in $\mu\text{m}$.
    pub length_um: f64,
    /// Diameter of the cylinder in $\mu\text{m}$.
    pub diameter_um: f64,
    /// Axial intracellular resistivity $R_a$ in $\Omega\cdot\text{cm}$ (typically $100.0$ to $250.0$).
    pub axial_resistance_ohm_cm: f64,
    /// Membrane potential $V$ in $\text{mV}$.
    pub v_mv: f64,
    /// Active ion channels and biophysical membrane patch.
    pub membrane: Membrane,
    /// Child connected compartments.
    pub children: Vec<CompartmentId>,
}

impl Compartment {
    /// Creates a cylindrical compartment with standard cable parameters.
    pub fn new(
        id: CompartmentId,
        kind: CompartmentKind,
        length_um: f64,
        diameter_um: f64,
        v_rest_mv: f64,
    ) -> Self {
        // Surface area of open cylinder A = pi * D * L (cm²)
        let l_cm = length_um * 1e-4;
        let d_cm = diameter_um * 1e-4;
        let area_cm2 = std::f64::consts::PI * d_cm * l_cm;

        Self {
            id,
            kind,
            length_um,
            diameter_um,
            axial_resistance_ohm_cm: 150.0,
            v_mv: v_rest_mv,
            membrane: Membrane::classic_hh(v_rest_mv, area_cm2),
            children: Vec::new(),
        }
    }

    /// Computes axial longitudinal resistance $R_{\text{axial}}$ to an adjacent compartment (in $\text{M}\Omega$):
    /// $$R_{\text{axial}} = \frac{4 R_a L}{\pi D^2} \times 10^{-4}$$
    pub fn axial_coupling_resistance_mohm(&self, other: &Compartment) -> f64 {
        let r1 = (4.0 * self.axial_resistance_ohm_cm * (self.length_um * 0.5 * 1e-4))
            / (std::f64::consts::PI * (self.diameter_um * 1e-4).powi(2));
        let r2 = (4.0 * other.axial_resistance_ohm_cm * (other.length_um * 0.5 * 1e-4))
            / (std::f64::consts::PI * (other.diameter_um * 1e-4).powi(2));
        // Convert Ohm to MOhm
        (r1 + r2) * 1e-6
    }
}

/// Dynamic state and formulation of a single neuron model.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum NeuronModel {
    /// Izhikevich (2003) phenomenological model: state `(v_mv, u_recovery)`.
    Izhikevich {
        params: IzhikevichParams,
        v: f64,
        u: f64,
    },
    /// Single-compartment biophysical Hodgkin-Huxley model.
    HodgkinHuxley {
        v: f64,
        membrane: Membrane,
    },
    /// Adaptive Leaky Integrate-and-Fire model.
    AdaptiveLif {
        params: AdLifParams,
        v: f64,
        w_adapt: f64,
        refractory_remaining_ms: f64,
    },
    /// Morphologically branched multi-compartment cable model.
    MultiCompartment {
        compartments: Vec<Compartment>,
        soma_index: usize,
    },
}

/// An individual neuron instance within a circuit or connectome.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Neuron {
    pub id: NeuronId,
    pub cell_type: NeuronType,
    pub model: NeuronModel,
    /// Spatial 3D anatomical position in micrometers $(x, y, z)$.
    pub position_um: [f64; 3],
    /// Accumulated external or synaptic current $I_{\text{ext}}$ in $\text{nA}$.
    pub current_injection_na: f64,
    /// Flag indicating whether an action potential occurred during the most recent step.
    pub fired_this_step: bool,
}

impl Neuron {
    /// Creates a regular-spiking pyramidal Izhikevich neuron.
    pub fn new_izhikevich(
        id: NeuronId,
        cell_type: NeuronType,
        params: IzhikevichParams,
        position_um: [f64; 3],
    ) -> Self {
        Self {
            id,
            cell_type,
            model: NeuronModel::Izhikevich {
                v: params.c,
                u: params.b * params.c,
                params,
            },
            position_um,
            current_injection_na: 0.0,
            fired_this_step: false,
        }
    }

    /// Creates a classic biophysical Hodgkin-Huxley neuron with soma area $A \approx 20,000\ \mu\text{m}^2$.
    pub fn new_hodgkin_huxley(id: NeuronId, cell_type: NeuronType, position_um: [f64; 3]) -> Self {
        let v_rest = -65.0;
        let area_cm2 = 2e-4; // 20,000 µm²
        Self {
            id,
            cell_type,
            model: NeuronModel::HodgkinHuxley {
                v: v_rest,
                membrane: Membrane::classic_hh(v_rest, area_cm2),
            },
            position_um,
            current_injection_na: 0.0,
            fired_this_step: false,
        }
    }

    /// Creates an Adaptive Leaky Integrate-and-Fire neuron.
    pub fn new_adlif(
        id: NeuronId,
        cell_type: NeuronType,
        params: AdLifParams,
        position_um: [f64; 3],
    ) -> Self {
        let v_rest = params.v_rest;
        Self {
            id,
            cell_type,
            model: NeuronModel::AdaptiveLif {
                params,
                v: v_rest,
                w_adapt: 0.0,
                refractory_remaining_ms: 0.0,
            },
            position_um,
            current_injection_na: 0.0,
            fired_this_step: false,
        }
    }

    /// Returns the current somatic membrane potential $V_m$ in $\text{mV}$.
    pub fn membrane_potential(&self) -> f64 {
        match &self.model {
            NeuronModel::Izhikevich { v, .. } => *v,
            NeuronModel::HodgkinHuxley { v, .. } => *v,
            NeuronModel::AdaptiveLif { v, .. } => *v,
            NeuronModel::MultiCompartment {
                compartments,
                soma_index,
            } => compartments[*soma_index].v_mv,
        }
    }

    /// Advances the neuronal state by time step $\Delta t$ (in $\text{ms}$) under total input current
    /// $I = I_{\text{inj}} + I_{\text{syn}}$ (in $\text{nA}$).
    ///
    /// Returns `true` if an action potential was emitted during this step.
    pub fn step(&mut self, dt_ms: f64) -> bool {
        self.fired_this_step = false;
        let i_ext = self.current_injection_na;

        match &mut self.model {
            NeuronModel::Izhikevich { params, v, u } => {
                // Check spike reset from previous peak
                if *v >= 30.0 {
                    *v = params.c;
                    *u += params.d;
                }

                // Two half-steps of numerical integration for numerical stability
                let dt2 = dt_ms * 0.5;
                for _ in 0..2 {
                    let dv = (0.04 * (*v).powi(2) + 5.0 * (*v) + 140.0 - *u + i_ext) * dt2;
                    let du = (params.a * (params.b * (*v) - *u)) * dt2;
                    *v += dv;
                    *u += du;
                }

                if *v >= 30.0 {
                    self.fired_this_step = true;
                }
            }
            NeuronModel::HodgkinHuxley { v, membrane } => {
                let v_old = *v;
                // Rush-Larsen update of gating variables
                membrane.step_channels(v_old, dt_ms);
                // Total ionic current in nA
                let i_ion = membrane.total_ionic_current_na(v_old);
                // Membrane capacitance C = Cm * Area (nF)
                // 1 µF/cm² * 1e-4 cm² = 1e-4 µF = 0.1 nF
                let c_nf = membrane.capacitance * membrane.area_cm2 * 1_000.0;
                // dV/dt = (I_ext - I_ion) / C_m [mV/ms]
                let dv = ((i_ext - i_ion) / c_nf) * dt_ms;
                *v += dv;

                // Spike detection: positive zero-crossing above +20 mV
                if v_old < 20.0 && *v >= 20.0 {
                    self.fired_this_step = true;
                }
            }
            NeuronModel::AdaptiveLif {
                params,
                v,
                w_adapt,
                refractory_remaining_ms,
            } => {
                if *refractory_remaining_ms > 0.0 {
                    *refractory_remaining_ms -= dt_ms;
                    *v = params.v_reset;
                } else {
                    // tau_m * dv/dt = -(v - v_rest) - R_m * w + R_m * I
                    let dv = (-(*v - params.v_rest) - params.r_m * (*w_adapt) + params.r_m * i_ext)
                        * (dt_ms / params.tau_m);
                    *v += dv;

                    if *v >= params.v_thresh {
                        self.fired_this_step = true;
                        *v = params.v_reset;
                        *w_adapt += params.b_adapt;
                        *refractory_remaining_ms = params.tau_ref;
                    }
                }
                // Adaptation current decay
                *w_adapt += (-*w_adapt / params.tau_w) * dt_ms;
            }
            NeuronModel::MultiCompartment {
                compartments,
                soma_index,
            } => {
                let num_comp = compartments.len();
                let mut axial_currents_na = vec![0.0; num_comp];

                // Compute inter-compartment longitudinal axial currents
                for i in 0..num_comp {
                    for &child_id in &compartments[i].children {
                        let j = child_id.0;
                        if j < num_comp {
                            let r_mohm = compartments[i].axial_coupling_resistance_mohm(&compartments[j]);
                            // I = (V_j - V_i) / R [nA]
                            let i_axial = (compartments[j].v_mv - compartments[i].v_mv) / r_mohm;
                            axial_currents_na[i] += i_axial;
                            axial_currents_na[j] -= i_axial;
                        }
                    }
                }

                // Step each compartment
                for (idx, comp) in compartments.iter_mut().enumerate() {
                    let v_old = comp.v_mv;
                    comp.membrane.step_channels(v_old, dt_ms);
                    let i_ion = comp.membrane.total_ionic_current_na(v_old);
                    let i_comp = if idx == *soma_index { i_ext } else { 0.0 } + axial_currents_na[idx];
                    let c_nf = comp.membrane.capacitance * comp.membrane.area_cm2 * 1_000.0;
                    let dv = ((i_comp - i_ion) / c_nf) * dt_ms;
                    comp.v_mv += dv;
                }

                let soma_v = compartments[*soma_index].v_mv;
                if soma_v >= 20.0 {
                    self.fired_this_step = true;
                }
            }
        }

        // Reset injection current for next integration step
        self.current_injection_na = 0.0;
        self.fired_this_step
    }
}
