//! Simulation engine, clock manager, stimulus scheduling, and event loop orchestration.
//!
//! Provides the primary runner for multi-neuron biophysical experiments, managing
//! precise numerical integration time-stepping, current clamp pulses, and event logging.

use crate::io::{SpikeEvent, VoltageRecorder};
use crate::network::NeuralNetwork;
use crate::types::NeuronId;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Discrete high-precision simulation clock.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SimulationClock {
    /// Current elapsed simulation time in milliseconds.
    pub current_time_ms: f64,
    /// Integration step size $\Delta t$ in milliseconds (e.g., $0.05$ or $0.1$).
    pub dt_ms: f64,
    /// Total discrete integration steps elapsed.
    pub step_count: u64,
}

impl SimulationClock {
    /// Creates a new clock starting at $t = 0.0$ ms.
    pub fn new(dt_ms: f64) -> Self {
        Self {
            current_time_ms: 0.0,
            dt_ms: dt_ms.max(1e-5),
            step_count: 0,
        }
    }

    /// Advances the clock by one $\Delta t$ step.
    pub fn tick(&mut self) {
        self.step_count += 1;
        self.current_time_ms = self.step_count as f64 * self.dt_ms;
    }
}

/// A scheduled current-clamp injection pulse targeting a specific neuron.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StimulusPulse {
    pub target_neuron: NeuronId,
    /// Onset time in milliseconds.
    pub start_ms: f64,
    /// Duration of pulse in milliseconds.
    pub duration_ms: f64,
    /// Injected current amplitude in nanoamperes ($\text{nA}$).
    pub amplitude_na: f64,
}

impl StimulusPulse {
    /// Checks whether the stimulus pulse is active at time $t$ (in $\text{ms}$).
    pub fn is_active_at(&self, time_ms: f64) -> bool {
        time_ms >= self.start_ms && time_ms < (self.start_ms + self.duration_ms)
    }
}

/// The main simulation engine orchestrating the network, stimuli, and data acquisition.
#[derive(Debug, Clone, PartialEq)]
pub struct SimulationEngine {
    pub network: NeuralNetwork,
    pub clock: SimulationClock,
    pub stimuli: Vec<StimulusPulse>,
    pub recorded_spikes: Vec<SpikeEvent>,
    pub recorder: Option<VoltageRecorder>,
    pub sampling_interval_ms: f64,
    last_sample_time_ms: f64,
}

impl SimulationEngine {
    /// Initializes a simulation engine for a given network and integration time step $\Delta t$.
    pub fn new(network: NeuralNetwork, dt_ms: f64) -> Self {
        Self {
            network,
            clock: SimulationClock::new(dt_ms),
            stimuli: Vec::new(),
            recorded_spikes: Vec::new(),
            recorder: None,
            sampling_interval_ms: dt_ms,
            last_sample_time_ms: -1.0,
        }
    }

    /// Schedules a current injection pulse.
    pub fn add_stimulus(&mut self, pulse: StimulusPulse) {
        self.stimuli.push(pulse);
    }

    /// Attaches a multi-channel continuous voltage recorder.
    pub fn attach_recorder(&mut self, channels: Vec<NeuronId>, sampling_interval_ms: f64) {
        self.sampling_interval_ms = sampling_interval_ms.max(self.clock.dt_ms);
        self.recorder = Some(VoltageRecorder::new(channels));
    }

    /// Advances the simulation by a single time step $\Delta t$.
    pub fn step(&mut self) -> Vec<NeuronId> {
        let t = self.clock.current_time_ms;
        let dt = self.clock.dt_ms;

        // Apply active stimulus pulses
        for stim in &self.stimuli {
            if stim.is_active_at(t) {
                if let Some(neuron) = self.network.get_neuron_mut(stim.target_neuron) {
                    neuron.current_injection_na += stim.amplitude_na;
                }
            }
        }

        // Record voltage traces if interval reached
        if let Some(ref mut rec) = self.recorder {
            if self.last_sample_time_ms < 0.0 || (t - self.last_sample_time_ms) >= (self.sampling_interval_ms - 1e-6) {
                let potentials: Vec<f64> = rec
                    .channels
                    .iter()
                    .map(|&ch| {
                        self.network
                            .get_neuron(ch)
                            .map(|n| n.membrane_potential())
                            .unwrap_or(-70.0)
                    })
                    .collect();
                rec.sample(t, &potentials);
                self.last_sample_time_ms = t;
            }
        }

        // Advance network
        let fired = self.network.step(dt);
        for &nid in &fired {
            self.recorded_spikes.push(SpikeEvent {
                time_ms: t,
                neuron_id: nid,
            });
        }

        self.clock.tick();
        fired
    }

    /// Runs the simulation until total elapsed duration (in milliseconds) is reached.
    pub fn run(&mut self, total_duration_ms: f64) {
        while self.clock.current_time_ms < total_duration_ms {
            self.step();
        }
    }
}
