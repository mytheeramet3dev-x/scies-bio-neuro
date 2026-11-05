//! # scies-bio-neuro
//!
//! A high-performance, biophysically grounded computational neuroscience simulation library for Rust.
//!
//! ## Core Features
//!
//! - **Ion Biophysics (`ion`)**: Rush-Larsen exponential integrator for stiff gating kinetics ($I_{\text{Na}}$, $I_{\text{K}}$, $I_{\text{A}}$, $I_{\text{h}}$, $I_{\text{Ca}}$).
//! - **Neuron Models (`neuron`)**:
//!   - Izhikevich (2003) phenomenological models (Regular Spiking, Fast Spiking, Intrinsically Bursting, etc.).
//!   - Classic Hodgkin-Huxley single-compartment excitable patches.
//!   - Adaptive Leaky Integrate-and-Fire (AdLIF) with spike-frequency adaptation (SFA).
//!   - Multi-compartment cable models with axial intracellular resistance coupling ($R_a$).
//! - **Synaptic Transmission (`synapse`)**:
//!   - Dual-exponential AMPA, NMDA (with voltage-dependent $\text{Mg}^{2+}$ block), $\text{GABA}_A$, $\text{GABA}_B$ receptors.
//!   - Bidirectional electrical gap junctions.
//!   - Tsodyks-Markram (1998) short-term depression and facilitation.
//!   - Fixed-capacity circular ring buffers (`AxonalDelayBuffer`) for conduction delays across arbitrary distances.
//! - **Network & Phoenix Brain Bridge (`network`)**:
//!   - Scalable circuit orchestrator.
//!   - **Phoenix Operation 5 cm Bridge (`BrainBridgeInterface`)**: Biological myelinated/unmyelinated axons,
//!     optogenetic optical relays, and neuromorphic DAC microelectrode bridges.
//! - **Signal Processing (`signal`)**: Spike train metrics (ISI, $CV_{\text{ISI}}$, Fano factor, PSTH), cross-correlation,
//!   coincidence synchrony, extracellular LFP recording, and oscillation frequency band decomposition (Delta through Gamma).
//! - **Plasticity (`plasticity`)**: Bi & Poo (1998) pair-based STDP, three-factor neuromodulated STDP, and Turrigiano homeostatic scaling.
//! - **Development (`develop`)**: Chemotropic growth cone guidance, myelin $g$-ratio optimization, and developmental pruning.
//! - **Neurochemistry (`neuro_chem`)**: Michaelis-Menten transporter clearance (DAT, SERT, EAAT) and Nicholson 3D volume transmission diffusion.
//! - **Electrophysiology I/O (`io`)**: CSV export, multi-unit voltage traces, and terminal ASCII spike rasters.
//! - **Simulation Engine (`engine`)**: Deterministic clock and scheduled stimulus pulse injections.

pub mod develop;
pub mod engine;
pub mod io;
pub mod ion;
pub mod network;
pub mod neuro_chem;
pub mod neuron;
pub mod plasticity;
pub mod signal;
pub mod synapse;
pub mod types;

// Convenient re-exports of primary types
pub use develop::{GrowthCone, GuidanceCue, MyelinSheath, SynapticPruner};
pub use engine::{SimulationClock, SimulationEngine, StimulusPulse};
pub use io::{SpikeEvent, VoltageRecorder};
pub use ion::{ChannelType, GatingVar, HHChannel, Membrane};
pub use network::{BrainBridgeInterface, BridgeMode, NeuralNetwork};
pub use neuro_chem::{Neurotransmitter, TransporterKinetics, VolumeTransmissionDiffusion};
pub use neuron::{AdLifParams, Compartment, IzhikevichParams, Neuron, NeuronModel};
pub use plasticity::{HomeostaticScaler, StdpParams, StdpSynapse, ThreeFactorStdp};
pub use signal::{OscillationBand, SpikeTrain};
pub use synapse::{AxonalDelayBuffer, ChemicalSynapse, GapJunction, ReceptorType, Synapse, TsodyksMarkram};
pub use types::{CompartmentId, CompartmentKind, NetworkId, NeuronId, NeuronType, SynapseId};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rush_larsen_stability() {
        let mut gate = GatingVar::new(0.0);
        // Step with large dt = 0.5 ms towards steady-state 1.0 with tau = 1.0 ms
        for _ in 0..10 {
            gate.update_rush_larsen(1.0, 1.0, 0.5);
        }
        assert!(gate.value > 0.99 && gate.value <= 1.0);
    }

    #[test]
    fn test_hodgkin_huxley_action_potential_generation() {
        let mut hh = Neuron::new_hodgkin_huxley(NeuronId(0), NeuronType::PyramidalL2_3, [0.0, 0.0, 0.0]);
        let dt = 0.02; // 0.02 ms integration step
        let mut fired = false;

        // Apply depolarizing current pulse 2.5 nA for 5 ms
        for step in 0..500 {
            let t = step as f64 * dt;
            if (1.0..6.0).contains(&t) {
                hh.current_injection_na = 2.5;
            }
            if hh.step(dt) {
                fired = true;
            }
        }
        assert!(fired, "Hodgkin-Huxley neuron should fire an action potential under depolarization!");
    }

    #[test]
    fn test_izhikevich_regular_spiking() {
        let mut izh = Neuron::new_izhikevich(
            NeuronId(1),
            NeuronType::PyramidalL2_3,
            IzhikevichParams::regular_spiking(),
            [0.0, 0.0, 0.0],
        );
        let dt = 0.1;
        let mut spike_count = 0;

        for _ in 0..1000 {
            // 100 ms with 10.0 injection
            izh.current_injection_na = 10.0;
            if izh.step(dt) {
                spike_count += 1;
            }
        }
        assert!(
            (2..=5).contains(&spike_count),
            "Regular spiking neuron should emit rhythmic spikes, got {}",
            spike_count
        );
    }

    #[test]
    fn test_axonal_conduction_delay_5cm() {
        let dt_ms = 0.1;
        // 5 cm unmyelinated axon: velocity = 1.0 m/s -> delay = 0.05 m / 1.0 m/s = 50 ms = 500 ticks
        let mut buffer = AxonalDelayBuffer::new(50.0, dt_ms);
        assert_eq!(buffer.delay_ticks(), 500);

        buffer.push_spike(1.0);

        // Before 500 ticks, output must be zero
        for _ in 0..499 {
            let out = buffer.step();
            assert_eq!(out, 0.0);
        }
        // At tick 500, the delayed action potential arrives!
        let arrived = buffer.step();
        assert!((arrived - 1.0).abs() < 1e-6, "Spike should arrive exactly at 50 ms!");
    }

    #[test]
    fn test_phoenix_brain_bridge_optogenetic_and_biological() {
        let bio_mode = BridgeMode::BiologicalAxon {
            diameter_um: 1.0,
            myelinated: true,
        };
        // 5 cm myelinated: v = 6 * 1.0 = 6 m/s -> 0.05 / 6 = 8.33 ms
        let bio_latency = bio_mode.calculate_latency_ms(5.0);
        assert!((bio_latency - 8.333).abs() < 0.01);

        let opto_mode = BridgeMode::OptogeneticRelay {
            instrument_latency_ms: 1.5,
            irradiance_mw_mm2: 10.0,
        };
        let opto_latency = opto_mode.calculate_latency_ms(5.0);
        assert_eq!(opto_latency, 1.5);

        let mut bridge = BrainBridgeInterface::new(NeuronId(0), NeuronId(1), 5.0, opto_mode, 0.1);
        bridge.trigger(1.0);
        assert_eq!(bridge.total_transmitted_spikes, 1);

        let mut delivered = false;
        for _ in 0..20 {
            let i = bridge.step(0.1);
            if i > 0.0 {
                delivered = true;
                break;
            }
        }
        assert!(delivered, "Optogenetic bridge must deliver photocurrent to target!");
        assert_eq!(bridge.total_delivered_pulses, 1);
    }

    #[test]
    fn test_chemical_synapse_nmda_magnesium_block() {
        let block_depol = ChemicalSynapse::nmda_mg_block(0.0, 1.0);
        let block_hyper = ChemicalSynapse::nmda_mg_block(-70.0, 1.0);
        // At depolarized potential (0 mV), block is lifted (higher conductance)
        // At hyperpolarized potential (-70 mV), Mg2+ blocks the pore
        assert!(block_depol > block_hyper * 5.0, "Depolarization must relieve NMDA Mg2+ block!");
    }

    #[test]
    fn test_spike_train_statistics() {
        let mut st = SpikeTrain::new();
        // Regular 10 ms interval spikes: 10, 20, 30, 40, 50
        for i in 1..=5 {
            st.push(i as f64 * 10.0);
        }
        assert_eq!(st.spike_count(), 5);
        assert_eq!(st.mean_isi_ms(), Some(10.0));
        assert!(st.isi_std_dev().unwrap() < 1e-6);
        assert!(st.coefficient_of_variation_isi().unwrap() < 1e-6); // Perfect clock CV = 0
    }

    #[test]
    fn test_stdp_potentiation_and_depression() {
        let mut stdp = StdpSynapse::new(5.0, StdpParams::default());
        let dt = 1.0;

        // Case 1: Pre before Post (LTP)
        stdp.on_pre_spike();
        // Advance 10 ms
        for _ in 0..10 {
            stdp.step(dt);
        }
        let w_before = stdp.weight_ns;
        stdp.on_post_spike();
        assert!(stdp.weight_ns > w_before, "Pre-before-post must induce LTP potentiation!");

        // Case 2: Post before Pre (LTD)
        let mut stdp2 = StdpSynapse::new(5.0, StdpParams::default());
        stdp2.on_post_spike();
        for _ in 0..10 {
            stdp2.step(dt);
        }
        let w_before2 = stdp2.weight_ns;
        stdp2.on_pre_spike();
        assert!(stdp2.weight_ns < w_before2, "Post-before-pre must induce LTD depression!");
    }

    #[test]
    fn test_myelin_rushton_g_ratio() {
        let myelin = MyelinSheath::optimal_myelinated(1.0);
        let g = myelin.g_ratio();
        assert!(g >= 0.60 && g <= 0.70, "Optimal g-ratio should be around 0.65, got {}", g);

        let v_myelin = myelin.conduction_velocity_m_s(true);
        let v_unmyelin = myelin.conduction_velocity_m_s(false);
        assert!(v_myelin > v_unmyelin * 4.0, "Myelinated conduction must be substantially faster!");
    }

    #[test]
    fn test_volume_transmission_diffusion() {
        let vt = VolumeTransmissionDiffusion::default();
        let c_near = vt.concentration_at(1e6, 5.0, 10.0);
        let c_far = vt.concentration_at(1e6, 50.0, 10.0);
        assert!(c_near > c_far, "Concentration near release site must exceed distant concentration!");
    }
}
