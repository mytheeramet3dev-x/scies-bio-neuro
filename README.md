# `scies-bio-neuro`

A high-performance, biophysically grounded computational neuroscience, cellular electrophysiology, and brain-bridge modeling toolkit written in 100% pure Rust with zero default dependencies.

---

## 🧠 Architecture Overview

`scies-bio-neuro` is organized into modular layers reflecting biological hierarchy:

```
scies_bio_neuro::
├── types       — NeuronId, SynapseId, CompartmentId, NeuronType classifications
├── ion         — Ion channels, Rush-Larsen exponential integrator, membrane biophysics
├── neuron      — Single neuron models (Izhikevich, Hodgkin-Huxley, AdLIF, multi-compartment cable)
├── synapse     — AMPA, NMDA (Mg²⁺ block), GABA_A/B, Gap junctions, Tsodyks-Markram STP, ring delay buffer
├── network     — Circuit orchestrator, connectivity patterns, and Phoenix BrainBridgeInterface
├── signal      — Spike train metrics (ISI, CV, Fano factor, PSTH), LFP forward model, spectral band power
├── plasticity  — Bi & Poo (1998) STDP, three-factor dopamine modulation, Turrigiano homeostatic scaling
├── develop     — Growth cone chemotropism, myelin Rushton g-ratio optimization, synaptogenesis/pruning
├── neuro_chem  — Michaelis-Menten transporters (DAT, SERT, EAAT), Nicholson 3D volume transmission
├── io          — Electrophysiology recording, voltage traces, CSV serialization, ASCII spike rasters
└── engine      — Discrete-time simulation engine, clock manager, and current clamp scheduling
```

---

## ⚡ Core Biophysical Features

### 1. Rush-Larsen Exponential Integration (`ion`)
Classical Hodgkin-Huxley gating equations ($\dot{x} = (x_\infty - x) / \tau_x$) are numerically stiff ($\tau_m \approx 0.1\text{ ms}$). `scies-bio-neuro` uses the exact Rush-Larsen analytic exponential update:
$$x(t + \Delta t) = x_\infty(V) + (x(t) - x_\infty(V)) \exp(-\Delta t / \tau_x(V))$$
ensuring unconditional numerical stability without explosion even at larger time steps.

### 2. Multi-Level Neuron Models (`neuron`)
- **Izhikevich (2003)**: Reproduces Regular Spiking (RS), Fast Spiking (FS), Intrinsically Bursting (IB), Chattering (CH), Low-Threshold Spiking (LTS), and Thalamo-Cortical (TC) firing patterns.
- **Hodgkin-Huxley Continuous Patch**: Voltage-gated $I_{\text{Na}}$, $I_{\text{K}}$, $I_{\text{A}}$, $I_{\text{h}}$, and leak currents.
- **Adaptive Leaky Integrate-and-Fire (AdLIF)**: Linear subthreshold integration with spike-frequency adaptation (SFA) and absolute refractory intervals.
- **Multi-Compartment Cable Model**: Dendrite, Soma, and Axon Initial Segment (AIS) coupled via longitudinal axial resistance $R_a$.

### 3. Receptor Kinetics & Short-Term Plasticity (`synapse`)
- **AMPA / GABA**: Dual-exponential rise and decay kinetics with exact peak conductance normalization.
- **NMDA Receptors**: Voltage-dependent magnesium unblock factor:
  $$B(V) = \frac{1}{1 + \frac{[\text{Mg}^{2+}]_o}{3.57} \exp(-0.062 \cdot V)}$$
- **Tsodyks-Markram (1998)**: Presynaptic vesicle depletion (depression) and calcium accumulation (facilitation).
- **`AxonalDelayBuffer`**: Zero-allocation circular ring buffer for delayed spike arrival across arbitrary conduction distances.

### 4. Phoenix Operation: 5 cm Neuron Pair Bridge (`network`)
Models macroscopic signal transmission across a 5 cm ($50,000\ \mu\text{m}$) tissue span under four modalities:
1. **Biological Unmyelinated Axon**: $v \approx 1.0 \times \sqrt{d}\ \text{m/s} \implies \sim 50.0\text{ ms}$ delay.
2. **Biological Myelinated Axon**: Saltatory conduction with optimal Rushton $g$-ratio ($g \approx 0.65$), $v \approx 6.0 \times d\ \text{m/s} \implies \sim 2.4\text{ ms}$ delay.
3. **Optogenetic Optical Relay**: Photodetector driving a solid-state blue laser/LED to stimulate Channelrhodopsin-2 (ChR2) $\implies \sim 1.0\text{ ms}$ latency.
4. **Neuromorphic Microelectrode DAC Bridge**: Sub-millisecond direct electrode stimulation $\implies \sim 0.2\text{ ms}$ latency.

---

## 🚀 Running the Example

Run the 5 cm neuron pair bridge simulation demo:

```bash
cargo run --example phoenix_5cm_bridge_demo
```

Sample output:
```
*****************************************************************
* PHOENIX OPERATION: 5 CM NEURON PAIR BRIDGE EXPERIMENT         *
* Simulating macroscopic signal transmission across 50,000 µm   *
*****************************************************************

=======================================================
  TRIAL: 1. Biological Unmyelinated Axon (Distance: 5.0 cm)
  Predicted Conduction Latency: 50.00 ms
=======================================================
  [SUCCESS] Neuron A Spiked at: 12.95 ms
  [SUCCESS] Neuron B Spiked at: 65.20 ms
  [MEASURED] End-to-End Bridge Delay: 52.25 ms (Latency + Integration)
--- SPIKE RASTER (0.0 ms -> 120.0 ms, 70 columns) ---
N00 [       |                                                              ]
N01 [                                      |                               ]
----------------------------------------------------------------------------
```

---

## 🧪 Testing

Run the test suite:

```bash
cargo test --all-features
```

---

## 📜 License

MIT License. Designed for advanced computational neuroscience, brain-computer interfaces, and regenerative neural engineering.
