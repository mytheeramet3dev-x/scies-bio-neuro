//! Neural network topology, circuit execution, and long-range brain-bridge interfaces.
//!
//! Provides the core container for neurons and synaptic interconnects, along with
//! the **Phoenix Operation** long-distance interface (`BrainBridgeInterface`) modeling
//! synthetic and biological transmission across 5 cm ($50,000\ \mu\text{m}$) tissue spans.

use crate::neuron::Neuron;
use crate::synapse::{ChemicalSynapse, GapJunction, ReceptorType, Synapse};
use crate::types::{NeuronId, SynapseId};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Transmission mechanism used by a long-distance brain bridge.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum BridgeMode {
    /// Pure biological axon tract.
    /// Conduction velocity scales according to axon diameter and myelination status:
    /// - Unmyelinated: $v \approx 1.0 \times \sqrt{d_{\mu\text{m}}}\ \text{m/s}$
    /// - Myelinated: $v \approx 6.0 \times d_{\mu\text{m}}\ \text{m/s}$
    BiologicalAxon {
        diameter_um: f64,
        myelinated: bool,
    },
    /// Optogenetic optical relay (Photodetector $\to$ solid-state pulse driver $\to$ blue laser/LED $\to$ ChR2 channel).
    OptogeneticRelay {
        /// Electronic and optical response latency in milliseconds ($\tau_{\text{opto}}$, typically $0.5$ to $2.0\text{ ms}$).
        instrument_latency_ms: f64,
        /// Optical irradiance delivered to target in $\text{mW/mm}^2$ (typically $5.0$ to $20.0$).
        irradiance_mw_mm2: f64,
    },
    /// Neuromorphic electrical bridge (Microelectrode recording array $\to$ low-latency DAC $\to$ micro-stimulator).
    ElectricalNeuromorphic {
        /// Hardware processing latency in milliseconds (typically $0.1$ to $0.4\text{ ms}$).
        dac_latency_ms: f64,
        /// Stimulus pulse amplitude in nanoamperes ($\text{nA}$).
        pulse_amplitude_na: f64,
        /// Pulse duration in milliseconds (typically $0.2\text{ ms}$).
        pulse_duration_ms: f64,
    },
}

impl BridgeMode {
    /// Computes the transmission latency in milliseconds across distance $L$ (in centimeters).
    pub fn calculate_latency_ms(&self, distance_cm: f64) -> f64 {
        let distance_m = distance_cm * 0.01;
        match self {
            BridgeMode::BiologicalAxon {
                diameter_um,
                myelinated,
            } => {
                let d = diameter_um.max(0.1);
                let velocity_m_s = if *myelinated {
                    6.0 * d
                } else {
                    1.0 * d.sqrt()
                };
                let conduction_time_sec = distance_m / velocity_m_s.max(0.1);
                conduction_time_sec * 1_000.0 // seconds to milliseconds
            }
            BridgeMode::OptogeneticRelay {
                instrument_latency_ms,
                ..
            } => {
                // Speed of light in optical fiber (~200,000 km/s) is negligible for 5 cm (~0.25 ns)
                *instrument_latency_ms
            }
            BridgeMode::ElectricalNeuromorphic { dac_latency_ms, .. } => *dac_latency_ms,
        }
    }
}

/// The **Phoenix Operation** Brain Bridge Interface:
/// Models a high-fidelity biological or synthetic link bridging two neurons or brain regions separated
/// by macroscopic biological distances (e.g., 5 cm).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BrainBridgeInterface {
    /// Upstream source neuron.
    pub source_id: NeuronId,
    /// Downstream target neuron.
    pub target_id: NeuronId,
    /// Physical bridge distance in centimeters (e.g. 5.0 cm).
    pub distance_cm: f64,
    /// Physical mechanism of the bridge.
    pub mode: BridgeMode,
    /// Effective transmission latency in milliseconds.
    pub latency_ms: f64,
    /// Ring buffer queueing transmitted signals.
    buffer: Vec<f64>,
    head: usize,
    delay_ticks: usize,
    /// Statistics: total spikes detected from source.
    pub total_transmitted_spikes: u64,
    /// Statistics: total pulses successfully delivered to target.
    pub total_delivered_pulses: u64,
    /// Active post-synaptic stimulus pulse remaining duration in milliseconds.
    active_pulse_remaining_ms: f64,
    /// Active post-synaptic stimulus pulse amplitude in nanoamperes.
    active_pulse_amplitude_na: f64,
}

impl BrainBridgeInterface {
    /// Creates a new long-distance brain bridge interface.
    pub fn new(
        source_id: NeuronId,
        target_id: NeuronId,
        distance_cm: f64,
        mode: BridgeMode,
        dt_ms: f64,
    ) -> Self {
        let latency_ms = mode.calculate_latency_ms(distance_cm);
        let dt = dt_ms.max(1e-4);
        let ticks = (latency_ms / dt).round().max(1.0) as usize;

        Self {
            source_id,
            target_id,
            distance_cm,
            mode,
            latency_ms,
            buffer: vec![0.0; ticks],
            head: 0,
            delay_ticks: ticks,
            total_transmitted_spikes: 0,
            total_delivered_pulses: 0,
            active_pulse_remaining_ms: 0.0,
            active_pulse_amplitude_na: 0.0,
        }
    }

    /// Triggers the bridge when the source neuron spikes.
    pub fn trigger(&mut self, signal_strength: f64) {
        let arrival_index = (self.head + self.delay_ticks - 1) % self.buffer.len();
        self.buffer[arrival_index] += signal_strength;
        self.total_transmitted_spikes += 1;
    }

    /// Advances the bridge clock by one simulation step $\Delta t$ and returns stimulus current injected into target (in $\text{nA}$).
    pub fn step(&mut self, dt_ms: f64) -> f64 {
        let signal = self.buffer[self.head];
        self.buffer[self.head] = 0.0;
        self.head = (self.head + 1) % self.buffer.len();

        if signal > 0.0 {
            self.total_delivered_pulses += 1;
            let (amp, duration) = match self.mode {
                BridgeMode::BiologicalAxon { .. } => (18.0 * signal, 2.0),
                BridgeMode::OptogeneticRelay {
                    irradiance_mw_mm2, ..
                } => {
                    let i_max = 20.0;
                    let e_half = 5.0;
                    let i_photo = i_max * (irradiance_mw_mm2 / (irradiance_mw_mm2 + e_half));
                    (i_photo * signal, 2.0)
                }
                BridgeMode::ElectricalNeuromorphic {
                    pulse_amplitude_na,
                    pulse_duration_ms,
                    ..
                } => (pulse_amplitude_na * signal, pulse_duration_ms.max(2.0)),
            };
            self.active_pulse_amplitude_na = amp;
            self.active_pulse_remaining_ms = duration;
        }

        if self.active_pulse_remaining_ms > 0.0 {
            self.active_pulse_remaining_ms -= dt_ms;
            self.active_pulse_amplitude_na
        } else {
            0.0
        }
    }
}

/// A neural network circuit composed of neurons, synaptic connections, and optional brain bridges.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NeuralNetwork {
    pub neurons: Vec<Neuron>,
    pub synapses: Vec<Synapse>,
    pub bridges: Vec<BrainBridgeInterface>,
}

impl Default for NeuralNetwork {
    fn default() -> Self {
        Self::new()
    }
}

impl NeuralNetwork {
    /// Creates an empty neural network.
    pub fn new() -> Self {
        Self {
            neurons: Vec::new(),
            synapses: Vec::new(),
            bridges: Vec::new(),
        }
    }

    /// Adds a neuron to the network and returns its assigned `NeuronId`.
    pub fn add_neuron(&mut self, neuron: Neuron) -> NeuronId {
        let id = neuron.id;
        self.neurons.push(neuron);
        id
    }

    /// Adds a chemical or electrical synapse to the network.
    pub fn add_synapse(&mut self, synapse: Synapse) {
        self.synapses.push(synapse);
    }

    /// Adds a chemical synapse directly.
    pub fn add_chemical_synapse(
        &mut self,
        pre_id: NeuronId,
        post_id: NeuronId,
        receptor: ReceptorType,
        weight_ns: f64,
        delay_ms: f64,
        dt_ms: f64,
    ) -> SynapseId {
        let id = SynapseId(self.synapses.len());
        let chem = ChemicalSynapse::new(id, pre_id, post_id, receptor, weight_ns, delay_ms, dt_ms);
        self.synapses.push(Synapse::Chemical(chem));
        id
    }

    /// Adds an electrical gap junction between two neurons.
    pub fn add_gap_junction(
        &mut self,
        neuron_a: NeuronId,
        neuron_b: NeuronId,
        conductance_ns: f64,
    ) -> SynapseId {
        let id = SynapseId(self.synapses.len());
        let gap = GapJunction::new(id, neuron_a, neuron_b, conductance_ns);
        self.synapses.push(Synapse::Electrical(gap));
        id
    }

    /// Registers a long-distance brain bridge interface.
    pub fn add_bridge(&mut self, bridge: BrainBridgeInterface) {
        self.bridges.push(bridge);
    }

    /// Retrieves an immutable reference to a neuron by `NeuronId`.
    pub fn get_neuron(&self, id: NeuronId) -> Option<&Neuron> {
        self.neurons.iter().find(|n| n.id == id)
    }

    /// Retrieves a mutable reference to a neuron by `NeuronId`.
    pub fn get_neuron_mut(&mut self, id: NeuronId) -> Option<&mut Neuron> {
        self.neurons.iter_mut().find(|n| n.id == id)
    }

    /// Advances the entire neural network simulation by time step $\Delta t$ (in $\text{ms}$).
    ///
    /// Execution order:
    /// 1. Collect which neurons fired in the previous step.
    /// 2. Inform chemical synapses and bridges of presynaptic spikes.
    /// 3. Step all synapses and bridges, accumulating post-synaptic currents into target neurons.
    /// 4. Step each neuron's membrane and channel dynamics.
    /// 5. Collect and return the list of `NeuronId`s that fired during this step.
    pub fn step(&mut self, dt_ms: f64) -> Vec<NeuronId> {
        // Step 1: Detect firing events from previous step
        let mut firing_neurons = Vec::new();
        for neuron in &self.neurons {
            if neuron.fired_this_step {
                firing_neurons.push(neuron.id);
            }
        }

        // Step 2: Trigger outgoing chemical synapses & bridges
        if !firing_neurons.is_empty() {
            for syn in &mut self.synapses {
                if let Synapse::Chemical(ref mut chem) = syn {
                    if firing_neurons.contains(&chem.pre_id) {
                        chem.on_presynaptic_spike();
                    }
                }
            }

            for bridge in &mut self.bridges {
                if firing_neurons.contains(&bridge.source_id) {
                    bridge.trigger(1.0);
                }
            }
        }

        // Step 3: Compute synaptic currents and collect into injections buffer
        // Map voltages
        let potentials: Vec<(NeuronId, f64)> = self
            .neurons
            .iter()
            .map(|n| (n.id, n.membrane_potential()))
            .collect();

        let mut injections: Vec<(NeuronId, f64)> = Vec::new();

        // Process chemical synapses
        for syn in &mut self.synapses {
            match syn {
                Synapse::Chemical(chem) => {
                    let v_post = potentials
                        .iter()
                        .find(|(id, _)| *id == chem.post_id)
                        .map(|(_, v)| *v)
                        .unwrap_or(-70.0);
                    let i_inj = chem.step(v_post, dt_ms);
                    if i_inj.abs() > 1e-12 {
                        injections.push((chem.post_id, i_inj));
                    }
                }
                Synapse::Electrical(gap) => {
                    let v_a = potentials
                        .iter()
                        .find(|(id, _)| *id == gap.neuron_a)
                        .map(|(_, v)| *v)
                        .unwrap_or(-70.0);
                    let v_b = potentials
                        .iter()
                        .find(|(id, _)| *id == gap.neuron_b)
                        .map(|(_, v)| *v)
                        .unwrap_or(-70.0);

                    let i_into_a = gap.current_into_a(v_a, v_b);
                    let i_into_b = gap.current_into_b(v_a, v_b);

                    if i_into_a.abs() > 1e-12 {
                        injections.push((gap.neuron_a, i_into_a));
                    }
                    if i_into_b.abs() > 1e-12 {
                        injections.push((gap.neuron_b, i_into_b));
                    }
                }
            }
        }

        // Process long-distance bridges
        for bridge in &mut self.bridges {
            let i_bridge = bridge.step(dt_ms);
            if i_bridge.abs() > 1e-12 {
                injections.push((bridge.target_id, i_bridge));
            }
        }

        // Apply all accumulated injections to neurons
        for (target_id, current_na) in injections {
            if let Some(target_neuron) = self.get_neuron_mut(target_id) {
                target_neuron.current_injection_na += current_na;
            }
        }

        // Step 4: Step each neuron model and collect newly emitted action potentials
        let mut newly_fired = Vec::new();
        for neuron in &mut self.neurons {
            if neuron.step(dt_ms) {
                newly_fired.push(neuron.id);
            }
        }

        newly_fired
    }
}
