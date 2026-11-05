//! Common identifiers, biological units, and core definitions for neuroscience simulations.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Unique identifier for an individual neuron within a population or network.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NeuronId(pub usize);

/// Unique identifier for a synaptic connection between two neurons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SynapseId(pub usize);

/// Unique identifier for a dendritic or axonal compartment within a multi-compartment neuron.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CompartmentId(pub usize);

/// Unique identifier for a brain region or sub-network.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NetworkId(pub usize);

/// Anatomical compartment classification within a morphological neuron tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum CompartmentKind {
    /// Cell body / soma.
    Soma,
    /// Axon initial segment (AIS) - primary site of action potential initiation.
    AxonInitialSegment,
    /// Axon shaft or terminal branch.
    Axon,
    /// Basal dendritic tree.
    BasalDendrite,
    /// Apical dendritic trunk or tuft.
    ApicalDendrite,
}

/// Canonical classification of biological neuronal cell types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum NeuronType {
    // Excitatory Principal Neurons
    PyramidalL2_3,
    PyramidalL5,
    GranuleCell,
    PurkinjeCell,
    PlaceCell,
    GridCell,

    // Inhibitory Interneurons
    ParvalbuminIN, // Fast-spiking basket cells
    SomatostatinIN, // Dendrite-targeting Martinotti cells
    VIPPositiveIN, // Disinhibitory interneurons

    // Neuromodulatory Neurons
    DopaminergicDA,
    SerotoninergicRN,
    CholinergicBF,
}
