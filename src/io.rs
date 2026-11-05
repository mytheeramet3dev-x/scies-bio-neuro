//! Electrophysiology recording, spike raster generation, and trace export formats.
//!
//! Provides:
//! - Multi-channel `VoltageRecorder` for tracking membrane trajectories across neurons.
//! - Spike event recording and CSV export.
//! - Terminal ASCII spike raster generator for rapid visual inspection of network synchronization.

use crate::types::NeuronId;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// An individual action potential event recorded during simulation.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SpikeEvent {
    pub time_ms: f64,
    pub neuron_id: NeuronId,
}

/// Continuous voltage recorder for multi-unit electrophysiology.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct VoltageRecorder {
    /// Monotonically increasing simulation timestamps in milliseconds.
    pub timestamps_ms: Vec<f64>,
    /// Tracked neuron identifiers.
    pub channels: Vec<NeuronId>,
    /// Recorded voltage traces: `traces[channel_idx][sample_idx]`.
    pub traces: Vec<Vec<f64>>,
}

impl VoltageRecorder {
    /// Creates a recorder configured to sample specified neurons.
    pub fn new(channels: Vec<NeuronId>) -> Self {
        let n = channels.len();
        Self {
            timestamps_ms: Vec::new(),
            channels,
            traces: vec![Vec::new(); n],
        }
    }

    /// Records instantaneous voltages for all tracked channels.
    pub fn sample(&mut self, time_ms: f64, potentials: &[f64]) {
        self.timestamps_ms.push(time_ms);
        for (idx, &v) in potentials.iter().enumerate() {
            if idx < self.traces.len() {
                self.traces[idx].push(v);
            }
        }
    }

    /// Serializes continuous voltage traces into comma-separated values (CSV) string.
    pub fn to_csv(&self) -> String {
        let mut out = String::from("time_ms");
        for ch in &self.channels {
            out.push_str(&format!(",neuron_{}", ch.0));
        }
        out.push('\n');

        let num_samples = self.timestamps_ms.len();
        for i in 0..num_samples {
            out.push_str(&format!("{:.3}", self.timestamps_ms[i]));
            for ch_idx in 0..self.channels.len() {
                out.push_str(&format!(",{:.3}", self.traces[ch_idx][i]));
            }
            out.push('\n');
        }

        out
    }
}

/// Formats a collection of spike events into CSV format (`time_ms,neuron_id`).
pub fn spikes_to_csv(spikes: &[SpikeEvent]) -> String {
    let mut out = String::from("time_ms,neuron_id\n");
    for s in spikes {
        out.push_str(&format!("{:.3},{}\n", s.time_ms, s.neuron_id.0));
    }
    out
}

/// Generates an ASCII spike raster diagram suitable for terminal output.
///
/// Parameters:
/// - `spikes`: Slice of recorded `SpikeEvent`s.
/// - `total_duration_ms`: Total duration of the simulation in milliseconds.
/// - `columns`: Display width in character columns (e.g., 60 to 100).
pub fn render_ascii_raster(
    spikes: &[SpikeEvent],
    neurons: &[NeuronId],
    total_duration_ms: f64,
    columns: usize,
) -> String {
    if total_duration_ms <= 0.0 || columns == 0 || neurons.is_empty() {
        return String::from("Empty raster plot.\n");
    }

    let mut out = String::new();
    out.push_str(&format!(
        "--- SPIKE RASTER (0.0 ms -> {:.1} ms, {} columns) ---\n",
        total_duration_ms, columns
    ));

    for &nid in neurons {
        let mut row = vec![' '; columns];
        for s in spikes {
            if s.neuron_id == nid {
                let col = ((s.time_ms / total_duration_ms) * columns as f64).floor() as usize;
                let safe_col = col.min(columns - 1);
                row[safe_col] = '|';
            }
        }
        let row_str: String = row.into_iter().collect();
        out.push_str(&format!("N{:02} [{}]\n", nid.0, row_str));
    }

    out.push_str(&format!("{:-<width$}\n", "", width = columns + 6));
    out
}
