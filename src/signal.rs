//! Neural signal processing, spike train statistics, and electrophysiological recording.
//!
//! Provides:
//! - `SpikeTrain` analysis: Inter-spike intervals (ISI), firing rate, $CV_{\text{ISI}}$, Fano factor, peristimulus time histograms (PSTH).
//! - Coincidence and synchrony indices between spike trains.
//! - Extracellular electrode local field potential (LFP) forward modeling based on volume conduction:
//!   $$V_{\text{LFP}}(\mathbf{r}_e) = \frac{1}{4\pi\sigma} \sum_{i} \frac{I_i}{\|\mathbf{r}_e - \mathbf{r}_i\|}$$
//! - Oscillation frequency band decomposition (Delta, Theta, Alpha, Beta, Gamma) via discrete spectral analysis.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Canonical neural oscillation frequency bands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum OscillationBand {
    /// Deep sleep, slow-wave activity ($0.5 - 4.0\text{ Hz}$).
    Delta,
    /// Hippocampal navigation, memory consolidation ($4.0 - 8.0\text{ Hz}$).
    Theta,
    /// Posterior resting rhythm, sensory gating ($8.0 - 13.0\text{ Hz}$).
    Alpha,
    /// Motor planning, active concentration, sensorimotor rhythm ($13.0 - 30.0\text{ Hz}$).
    Beta,
    /// Cross-modal binding, cognitive processing, local circuit gamma ($30.0 - 100.0\text{ Hz}$).
    Gamma,
}

impl OscillationBand {
    /// Returns the frequency bounds `(f_min_hz, f_max_hz)`.
    pub fn frequency_range_hz(&self) -> (f64, f64) {
        match self {
            OscillationBand::Delta => (0.5, 4.0),
            OscillationBand::Theta => (4.0, 8.0),
            OscillationBand::Alpha => (8.0, 13.0),
            OscillationBand::Beta => (13.0, 30.0),
            OscillationBand::Gamma => (30.0, 100.0),
        }
    }
}

/// A sequence of action potential timestamps recorded from a single neuron or unit.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SpikeTrain {
    /// Spike arrival timestamps in milliseconds (monotonically increasing).
    pub timestamps_ms: Vec<f64>,
}

impl SpikeTrain {
    /// Creates an empty spike train.
    pub fn new() -> Self {
        Self {
            timestamps_ms: Vec::new(),
        }
    }

    /// Appends a new action potential spike timestamp (in milliseconds).
    pub fn push(&mut self, timestamp_ms: f64) {
        self.timestamps_ms.push(timestamp_ms);
    }

    /// Total number of spikes recorded.
    pub fn spike_count(&self) -> usize {
        self.timestamps_ms.len()
    }

    /// Computes the mean firing rate in Hertz ($\text{spikes/sec}$):
    /// $$\bar{r} = \frac{N}{T} \times 1000$$
    pub fn mean_firing_rate_hz(&self, total_duration_ms: f64) -> f64 {
        if total_duration_ms <= 0.0 {
            0.0
        } else {
            (self.timestamps_ms.len() as f64 / total_duration_ms) * 1_000.0
        }
    }

    /// Computes all consecutive Inter-Spike Intervals (ISI) in milliseconds.
    pub fn inter_spike_intervals_ms(&self) -> Vec<f64> {
        if self.timestamps_ms.len() < 2 {
            return Vec::new();
        }
        self.timestamps_ms
            .windows(2)
            .map(|w| (w[1] - w[0]).max(0.0))
            .collect()
    }

    /// Computes the mean Inter-Spike Interval in milliseconds ($\mu_{\text{ISI}}$).
    pub fn mean_isi_ms(&self) -> Option<f64> {
        let isis = self.inter_spike_intervals_ms();
        if isis.is_empty() {
            None
        } else {
            let sum: f64 = isis.iter().sum();
            Some(sum / isis.len() as f64)
        }
    }

    /// Computes the sample standard deviation of ISIs ($\sigma_{\text{ISI}}$).
    pub fn isi_std_dev(&self) -> Option<f64> {
        let isis = self.inter_spike_intervals_ms();
        if isis.len() < 2 {
            return None;
        }
        let mean = isis.iter().sum::<f64>() / isis.len() as f64;
        let variance = isis
            .iter()
            .map(|&isi| (isi - mean).powi(2))
            .sum::<f64>()
            / (isis.len() - 1) as f64;
        Some(variance.sqrt())
    }

    /// Computes the Coefficient of Variation of Inter-Spike Intervals ($CV_{\text{ISI}} = \sigma / \mu$):
    /// - $CV = 0$: Perfectly regular pacemaker firing.
    /// - $CV = 1.0$: Homogeneous Poisson process.
    /// - $CV > 1.0$: Bursting or clustered firing dynamics.
    pub fn coefficient_of_variation_isi(&self) -> Option<f64> {
        let mean = self.mean_isi_ms()?;
        let std = self.isi_std_dev()?;
        if mean > 1e-6 {
            Some(std / mean)
        } else {
            None
        }
    }

    /// Computes the Fano factor of spike counts over sliding non-overlapping time windows:
    /// $$F = \frac{\operatorname{Var}(N_W)}{\operatorname{E}(N_W)}$$
    pub fn fano_factor(&self, window_size_ms: f64, total_duration_ms: f64) -> Option<f64> {
        if window_size_ms <= 0.0 || total_duration_ms < window_size_ms {
            return None;
        }

        let num_bins = (total_duration_ms / window_size_ms).floor() as usize;
        if num_bins < 2 {
            return None;
        }

        let mut counts = vec![0.0; num_bins];
        for &t in &self.timestamps_ms {
            let bin = (t / window_size_ms).floor() as usize;
            if bin < num_bins {
                counts[bin] += 1.0;
            }
        }

        let mean = counts.iter().sum::<f64>() / num_bins as f64;
        if mean < 1e-6 {
            return None;
        }

        let variance = counts
            .iter()
            .map(|&c| (c - mean).powi(2))
            .sum::<f64>()
            / (num_bins - 1) as f64;

        Some(variance / mean)
    }

    /// Computes a peristimulus time histogram (PSTH) / firing rate trace (time_center_ms, rate_hz).
    pub fn sliding_window_rate(
        &self,
        window_size_ms: f64,
        step_ms: f64,
        total_duration_ms: f64,
    ) -> Vec<(f64, f64)> {
        let mut result = Vec::new();
        let mut t_center = window_size_ms * 0.5;

        while t_center + window_size_ms * 0.5 <= total_duration_ms {
            let t_start = t_center - window_size_ms * 0.5;
            let t_end = t_center + window_size_ms * 0.5;

            let count = self
                .timestamps_ms
                .iter()
                .filter(|&&t| t >= t_start && t < t_end)
                .count();

            let rate_hz = (count as f64 / (window_size_ms * 1e-3)).max(0.0);
            result.push((t_center, rate_hz));
            t_center += step_ms;
        }

        result
    }
}

/// Computes the cross-correlation between two spike trains with binning $\Delta \tau$.
/// Returns `(lag_ms, normalized_correlation)` pairs centered around lag zero.
pub fn cross_correlation(
    train_a: &SpikeTrain,
    train_b: &SpikeTrain,
    max_lag_ms: f64,
    bin_size_ms: f64,
    total_duration_ms: f64,
) -> Vec<(f64, f64)> {
    let num_bins = (total_duration_ms / bin_size_ms).ceil() as usize;
    if num_bins == 0 {
        return Vec::new();
    }

    let mut binned_a = vec![0.0; num_bins];
    let mut binned_b = vec![0.0; num_bins];

    for &t in &train_a.timestamps_ms {
        let bin = (t / bin_size_ms).floor() as usize;
        if bin < num_bins {
            binned_a[bin] += 1.0;
        }
    }
    for &t in &train_b.timestamps_ms {
        let bin = (t / bin_size_ms).floor() as usize;
        if bin < num_bins {
            binned_b[bin] += 1.0;
        }
    }

    let max_lag_bins = (max_lag_ms / bin_size_ms).round() as isize;
    let mut results = Vec::new();

    let norm = ((binned_a.iter().map(|x| x * x).sum::<f64>())
        * (binned_b.iter().map(|x| x * x).sum::<f64>()))
        .sqrt()
        .max(1e-9);

    for lag in -max_lag_bins..=max_lag_bins {
        let mut sum = 0.0;
        for i in 0..num_bins as isize {
            let j = i + lag;
            if j >= 0 && j < num_bins as isize {
                sum += binned_a[i as usize] * binned_b[j as usize];
            }
        }
        let lag_ms = lag as f64 * bin_size_ms;
        results.push((lag_ms, sum / norm));
    }

    results
}

/// Computes the coincidence synchrony index between two spike trains within tolerance window $\Delta t_{\text{coinc}}$.
/// Normalized between $0.0$ (independent) and $1.0$ (perfect synchrony).
pub fn synchrony_index(
    train_a: &SpikeTrain,
    train_b: &SpikeTrain,
    coincidence_window_ms: f64,
) -> f64 {
    if train_a.timestamps_ms.is_empty() || train_b.timestamps_ms.is_empty() {
        return 0.0;
    }

    let mut coincidences = 0;
    for &ta in &train_a.timestamps_ms {
        for &tb in &train_b.timestamps_ms {
            if (ta - tb).abs() <= coincidence_window_ms {
                coincidences += 1;
                break;
            }
        }
    }

    let total = train_a.timestamps_ms.len().max(train_b.timestamps_ms.len()) as f64;
    (coincidences as f64 / total).clamp(0.0, 1.0)
}

/// Extracellular point electrode for recording Local Field Potentials (LFP).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ExtracellularElectrode {
    /// 3D coordinates $(x, y, z)$ in micrometers ($\mu\text{m}$).
    pub position_um: [f64; 3],
    /// Extracellular conductivity $\sigma_e$ in $\text{S/m}$ (typically $0.3\ \text{S/m}$).
    pub conductivity_s_m: f64,
}

impl ExtracellularElectrode {
    pub fn new(position_um: [f64; 3]) -> Self {
        Self {
            position_um,
            conductivity_s_m: 0.3,
        }
    }

    /// Computes instantaneous extracellular potential $V_{\text{LFP}}$ in microvolts ($\mu\text{V}$)
    /// given current sources: `&[(source_position_um, current_na)]`.
    ///
    /// $$V(\mathbf{r}) = \frac{1}{4 \pi \sigma_e} \sum_{i} \frac{I_i}{\|\mathbf{r} - \mathbf{r}_i\|}$$
    pub fn record_potential_uv(&self, sources: &[([f64; 3], f64)]) -> f64 {
        let mut v_sum_v = 0.0;
        let sigma = self.conductivity_s_m;

        for (pos, i_na) in sources {
            let dx = (self.position_um[0] - pos[0]) * 1e-6; // meters
            let dy = (self.position_um[1] - pos[1]) * 1e-6;
            let dz = (self.position_um[2] - pos[2]) * 1e-6;
            let dist_m = (dx * dx + dy * dy + dz * dz).sqrt().max(10e-6); // minimum 10 µm cutoff
            let i_amp = *i_na * 1e-9; // nanoamperes to amperes

            v_sum_v += i_amp / (4.0 * std::f64::consts::PI * sigma * dist_m);
        }

        v_sum_v * 1e6 // volts to microvolts
    }
}

/// Computes spectral power in a specific oscillation band from a discrete signal trace using
/// the Discrete Fourier Transform (DFT).
///
/// Returns the integrated power $(\mu\text{V}^2)$ within the band.
pub fn band_power(signal: &[f64], sampling_rate_hz: f64, band: OscillationBand) -> f64 {
    let n = signal.len();
    if n < 4 || sampling_rate_hz <= 0.0 {
        return 0.0;
    }

    let (f_min, f_max) = band.frequency_range_hz();
    let df = sampling_rate_hz / n as f64;
    let mut total_power = 0.0;

    for k in 0..(n / 2) {
        let f = k as f64 * df;
        if f >= f_min && f <= f_max {
            // Compute DFT component at bin k
            let mut real = 0.0;
            let mut imag = 0.0;
            for (t, &val) in signal.iter().enumerate() {
                let angle = 2.0 * std::f64::consts::PI * (k * t) as f64 / n as f64;
                real += val * angle.cos();
                imag -= val * angle.sin();
            }
            let power = (real * real + imag * imag) / (n as f64 * n as f64);
            total_power += power;
        }
    }

    total_power
}
