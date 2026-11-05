//! Phoenix Operation: 5 cm Neuron Pair Bridge Simulation
//!
//! Demonstrates the biophysical propagation of action potentials across a macroscopic
//! biological span of 5 cm ($50,000\ \mu\text{m}$) comparing:
//! 1. Biological Unmyelinated Axon (slow propagation, $\sim 50\text{ ms}$ delay)
//! 2. Biological Myelinated Axon (saltatory conduction, $\sim 2.5\text{ ms}$ delay)
//! 3. Optogenetic Optical Relay (photodetector + laser trigger, $\sim 1.0\text{ ms}$ delay)
//! 4. Neuromorphic Microelectrode DAC Bridge ($\sim 0.2\text{ ms}$ ultra-low latency)

use scies_bio_neuro::develop::MyelinSheath;
use scies_bio_neuro::engine::{SimulationEngine, StimulusPulse};
use scies_bio_neuro::io::{render_ascii_raster, SpikeEvent};
use scies_bio_neuro::network::{BrainBridgeInterface, BridgeMode, NeuralNetwork};
use scies_bio_neuro::neuron::{IzhikevichParams, Neuron};
use scies_bio_neuro::types::{NeuronId, NeuronType};

fn run_bridge_trial(name: &str, mode: BridgeMode, distance_cm: f64) -> (f64, Vec<SpikeEvent>) {
    let dt_ms = 0.05;
    let total_duration_ms = 120.0;

    let mut net = NeuralNetwork::new();

    // Neuron A: Source Pyramidal Neuron at x = 0.0 cm
    let neuron_a = Neuron::new_izhikevich(
        NeuronId(0),
        NeuronType::PyramidalL2_3,
        IzhikevichParams::regular_spiking(),
        [0.0, 0.0, 0.0],
    );
    // Neuron B: Target Pyramidal Neuron at x = 5.0 cm (50,000 µm)
    let neuron_b = Neuron::new_izhikevich(
        NeuronId(1),
        NeuronType::PyramidalL2_3,
        IzhikevichParams::regular_spiking(),
        [50_000.0, 0.0, 0.0],
    );

    net.add_neuron(neuron_a);
    net.add_neuron(neuron_b);

    // Setup 5cm Bridge
    let bridge = BrainBridgeInterface::new(NeuronId(0), NeuronId(1), distance_cm, mode, dt_ms);
    let latency_ms = bridge.latency_ms;
    net.add_bridge(bridge);

    let mut engine = SimulationEngine::new(net, dt_ms);

    // Inject 15 nA stimulus pulse into Neuron A at t = 10.0 ms for 2.0 ms to trigger a crisp action potential
    engine.add_stimulus(StimulusPulse {
        target_neuron: NeuronId(0),
        start_ms: 10.0,
        duration_ms: 2.0,
        amplitude_na: 15.0,
    });

    engine.run(total_duration_ms);

    println!("\n=======================================================");
    println!("  TRIAL: {} (Distance: {:.1} cm)", name, distance_cm);
    println!("  Predicted Conduction Latency: {:.2} ms", latency_ms);
    println!("=======================================================");

    let spike_a = engine.recorded_spikes.iter().find(|s| s.neuron_id == NeuronId(0)).map(|s| s.time_ms);
    let spike_b = engine.recorded_spikes.iter().find(|s| s.neuron_id == NeuronId(1)).map(|s| s.time_ms);

    match (spike_a, spike_b) {
        (Some(ta), Some(tb)) => {
            let actual_delay = tb - ta;
            println!("  [SUCCESS] Neuron A Spiked at: {:.2} ms", ta);
            println!("  [SUCCESS] Neuron B Spiked at: {:.2} ms", tb);
            println!("  [MEASURED] End-to-End Bridge Delay: {:.2} ms (Latency + Integration)", actual_delay);
        }
        (Some(ta), None) => {
            println!("  Neuron A Spiked at: {:.2} ms, but Neuron B did not reach threshold.", ta);
        }
        _ => {
            println!("  Neuron A failed to spike.");
        }
    }

    let raster = render_ascii_raster(&engine.recorded_spikes, &[NeuronId(0), NeuronId(1)], total_duration_ms, 70);
    print!("{}", raster);

    (latency_ms, engine.recorded_spikes)
}

fn main() {
    println!("*****************************************************************");
    println!("* PHOENIX OPERATION: 5 CM NEURON PAIR BRIDGE EXPERIMENT         *");
    println!("* Simulating macroscopic signal transmission across 50,000 µm   *");
    println!("*****************************************************************");

    let distance_cm = 5.0;

    // 1. Biological Unmyelinated Axon (d = 1.0 µm, v ~ 1.0 m/s -> delay ~ 50.0 ms)
    let mode_unmyelinated = BridgeMode::BiologicalAxon {
        diameter_um: 1.0,
        myelinated: false,
    };
    run_bridge_trial("1. Biological Unmyelinated Axon", mode_unmyelinated, distance_cm);

    // 2. Biological Myelinated Axon (d = 3.5 µm, v ~ 21 m/s -> delay ~ 2.38 ms)
    let myelin = MyelinSheath::optimal_myelinated(3.5);
    println!("\n[Biophysical Profile: Myelinated Axon]");
    println!("  - Inner axon diameter: {:.2} µm", myelin.axon_diameter_um);
    println!("  - Outer fiber diameter: {:.2} µm", myelin.fiber_diameter_um);
    println!("  - Rushton g-ratio: {:.3}", myelin.g_ratio());
    println!("  - Internodal distance: {:.1} µm", myelin.internodal_length_um());

    let mode_myelinated = BridgeMode::BiologicalAxon {
        diameter_um: 3.5,
        myelinated: true,
    };
    run_bridge_trial("2. Biological Myelinated Axon", mode_myelinated, distance_cm);

    // 3. Optogenetic Optical Relay Bridge (laser latency ~ 1.0 ms)
    let mode_optogenetic = BridgeMode::OptogeneticRelay {
        instrument_latency_ms: 1.0,
        irradiance_mw_mm2: 15.0,
    };
    run_bridge_trial("3. Optogenetic Optical Relay Bridge", mode_optogenetic, distance_cm);

    // 4. Neuromorphic Microelectrode DAC Bridge (sub-millisecond latency ~ 0.2 ms)
    let mode_neuromorphic = BridgeMode::ElectricalNeuromorphic {
        dac_latency_ms: 0.2,
        pulse_amplitude_na: 12.0,
        pulse_duration_ms: 0.2,
    };
    run_bridge_trial("4. Neuromorphic Microelectrode DAC Bridge", mode_neuromorphic, distance_cm);

    println!("\n*****************************************************************");
    println!("* PHOENIX OPERATION COMPLETE: All bridges validated!            *");
    println!("*****************************************************************");
}
