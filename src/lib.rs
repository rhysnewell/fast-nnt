//! Fast Rust implementation of the NeighbourNet algorithm (Bryant & Moulton,
//! 2004) for phylogenetic analysis.
//!
//! From a square, symmetric distance matrix, fast-nnt constructs an *implicit*
//! split network — a planar diagram that summarises conflicting signal in the
//! data for exploratory analysis. Unlike *explicit* networks, the parallelogram
//! "boxes" do not model specific reticulation events such as hybridisation or
//! introgression. A 2-D layout for the network is computed using the
//! equal-angle algorithm and returned alongside the splits in a [`Nexus`].
//!
//! [`run_fast_nnt_from_memory`] is the single entry point shared by the CLI and
//! the Python and R bindings.

use std::{env, time::Instant};

use anyhow::{Context, Result};
use env_logger::Builder;
use log::{LevelFilter, info};
use ndarray::Array2;

use crate::{
    cli::{NeighbourNetArgs, ProgramArgs},
    neighbour_net::neighbour_net::NeighbourNet,
    nexus::nexus::Nexus,
};

pub mod algorithms;
pub mod cli;
pub mod data;
pub mod neighbour_net;
pub mod nexus;
pub mod ordering;
pub mod phylo;
pub mod splits;
pub mod utils;
pub mod weights;

#[cfg(test)]
pub(crate) mod test_helpers;

#[macro_use]
extern crate log;

pub fn set_log_level(matches: &ProgramArgs, is_last: bool, program_name: &str, version: &str) {
    let mut log_level = LevelFilter::Info;
    let mut specified = false;
    if matches.verbose {
        specified = true;
        log_level = LevelFilter::Debug;
    }
    if matches.quiet {
        specified = true;
        log_level = LevelFilter::Error;
    }
    if specified || is_last {
        let mut builder = Builder::new();
        builder.filter_level(log_level);
        if let Ok(rust_log) = env::var("RUST_LOG") {
            builder.parse_filters(&rust_log);
        }
        let _ = builder.try_init();
    }
    if is_last {
        info!("{} version {}", program_name, version);
    }
}

/// Build a NeighbourNet split network from a distance matrix.
///
/// The single entry point shared by the CLI and the Python and R bindings. The
/// returned [`Nexus`] holds the weighted splits and a 2-D equal-angle layout of
/// the implicit split network.
///
/// - `dist`: square, symmetric distance matrix (n x n)
/// - `labels`: taxon labels, length n
/// - `args`: ordering, weight-inference, and NNLS parameters
pub fn run_fast_nnt_from_memory(
    dist: Array2<f64>,
    labels: Vec<String>,
    args: NeighbourNetArgs,
) -> Result<Nexus> {
    let t0 = Instant::now();

    // Validate
    let neighbour_net = NeighbourNet::from_distance_matrix(dist, labels, args)
        .context("Validating distance matrix")?;
    let nexus = neighbour_net
        .into_nexus()
        .context("Performing neighbour net analysis")?;
    info!("Finished NeighbourNet in {:?}", t0.elapsed());
    Ok(nexus)
}

/// Initialize the global Rayon thread pool (call once per process).
pub enum RayonInitStatus {
    Initialized,
    AlreadyInitializedSame,
    AlreadyInitializedDifferent { current: usize },
}

pub fn init_rayon_threads(threads: usize) -> Result<RayonInitStatus> {
    if threads == 0 {
        return Ok(RayonInitStatus::Initialized);
    }

    match rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
    {
        Ok(()) => Ok(RayonInitStatus::Initialized),
        Err(e) => {
            let msg = e.to_string();
            if msg.contains("global thread pool has already been initialized") {
                let current = rayon::current_num_threads();
                if current == threads {
                    Ok(RayonInitStatus::AlreadyInitializedSame)
                } else {
                    Ok(RayonInitStatus::AlreadyInitializedDifferent { current })
                }
            } else {
                Err(e.into())
            }
        }
    }
}
