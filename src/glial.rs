//! Glial cell support systems: Astrocytes, Microglia, and Oligodendrocytes.
//!
//! Models:
//! - **Astrocytes (`Astrocyte`)**: Tripartite synapse regulation, glutamate uptake (GLT-1/GLAST),
//!   $K^+$ spatial buffering (Kir4.1), and $\text{IP}_3$-mediated intracellular $\text{Ca}^{2+}$ signaling.
//! - **Microglia (`Microglia`)**: Dynamic immunological phenotypes (Ramified surveillance,
//!   M1 pro-inflammatory, M2 neuroprotective phagocytic state).
//! - **Oligodendrocytes**: Multi-axonal myelin wrapping and metabolic support.

use crate::types::GlialId;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Microglial immunological and morphological activation states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum MicrogliaState {
    /// Resting, ramified surveillance state with highly motile fine processes.
    RamifiedSurveillance,
    /// M1 pro-inflammatory phenotype: secretes TNF-$\alpha$, IL-$1\beta$, reactive oxygen species (ROS).
    M1ProInflammatory,
    /// M2 neuroprotective & phagocytic phenotype: promotes tissue repair and debris clearance.
    M2Neuroprotective,
}

/// An individual microglial cell monitoring local brain tissue.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Microglia {
    pub id: GlialId,
    pub state: MicrogliaState,
    pub position_um: [f64; 3],
    /// Surveillance radius in micrometers ($\mu\text{m}$, typically $20.0$ to $50.0$).
    pub surveillance_radius_um: f64,
    /// Cytokine secretion level in arbitrary units.
    pub cytokine_level: f64,
}

impl Microglia {
    pub fn new(id: GlialId, position_um: [f64; 3]) -> Self {
        Self {
            id,
            state: MicrogliaState::RamifiedSurveillance,
            position_um,
            surveillance_radius_um: 35.0,
            cytokine_level: 0.0,
        }
    }

    /// Evaluates activation transition in response to damage-associated molecular patterns (DAMPs) / ATP.
    pub fn step_activation(&mut self, damage_signal: f64, anti_inflammatory_signal: f64) {
        if damage_signal > 0.6 {
            self.state = MicrogliaState::M1ProInflammatory;
            self.cytokine_level = (self.cytokine_level + 0.1 * damage_signal).clamp(0.0, 1.0);
        } else if anti_inflammatory_signal > 0.5 {
            self.state = MicrogliaState::M2Neuroprotective;
            self.cytokine_level = (self.cytokine_level - 0.05).max(0.0);
        } else {
            self.state = MicrogliaState::RamifiedSurveillance;
            self.cytokine_level = (self.cytokine_level - 0.02).max(0.0);
        }
    }
}

/// An individual astrocyte maintaining tripartite synaptic homeostasis.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Astrocyte {
    pub id: GlialId,
    pub position_um: [f64; 3],
    /// Intracellular calcium concentration $[\text{Ca}^{2+}]_i$ in $\mu\text{M}$ (baseline $\approx 0.1\ \mu\text{M}$).
    pub ca_intracellular_um: f64,
    /// Extracellular potassium concentration $[K^+]_o$ in $\text{mM}$ (baseline $\approx 3.0\ \text{mM}$).
    pub k_extracellular_mm: f64,
    /// Clearance capacity of astrocytic glutamate transporters (GLT-1/EAAT2).
    pub glutamate_clearance_rate: f64,
}

impl Astrocyte {
    pub fn new(id: GlialId, position_um: [f64; 3]) -> Self {
        Self {
            id,
            position_um,
            ca_intracellular_um: 0.1,
            k_extracellular_mm: 3.0,
            glutamate_clearance_rate: 5.0,
        }
    }

    /// Advances the astrocyte state by time step $\Delta t$ (in $\text{ms}$).
    pub fn step(&mut self, extracellular_glutamate_um: f64, dt_ms: f64) {
        if extracellular_glutamate_um > 0.5 {
            let d_ca = 0.02 * extracellular_glutamate_um * (dt_ms / 10.0);
            self.ca_intracellular_um = (self.ca_intracellular_um + d_ca).min(2.5);
        } else {
            let decay = (-dt_ms / 500.0).exp();
            self.ca_intracellular_um = 0.1 + (self.ca_intracellular_um - 0.1) * decay;
        }

        let k_error = self.k_extracellular_mm - 3.0;
        let k_decay = (-dt_ms / 200.0).exp();
        self.k_extracellular_mm = 3.0 + k_error * k_decay;
    }
}
