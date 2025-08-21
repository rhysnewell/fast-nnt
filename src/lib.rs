use std::{env, time::Instant};

use anyhow::{Result, Context};
use env_logger::Builder;
use log::{LevelFilter, info};
use ndarray::Array2;

use crate::{cli::{NeighbourNetArgs, ProgramArgs}, neighbour_net::neighbour_net::NeighbourNet, nexus::nexus::Nexus};

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
        if env::var("RUST_LOG").is_ok() {
            builder.parse_filters(&env::var("RUST_LOG").unwrap());
        }
        if builder.try_init().is_err() {
            panic!("Failed to set log level - has it been specified multiple times?")
        }
    }
    if is_last {
        info!("{} version {}", program_name, version);
    }
}


/// The single entry point for bindings.
///
/// - `dist`: square distance matrix (n x n)
/// - `labels`: length n
/// - `args`: algorithm params (bindings can construct this or you add a smaller ArgsLite)
pub fn run_fast_nnt_from_memory(
    dist: Array2<f64>,
    labels: Vec<String>,
    args: NeighbourNetArgs
) -> Result<Nexus> {
    let t0 = Instant::now();

    // Validate
    let neighbour_net = NeighbourNet::from_distance_matrix(dist, labels, args);
    let nexus = neighbour_net.generate_nexus().context("Performing neighbour net analysis")?;
    info!("Finished NeighbourNet in {:?}", t0.elapsed());
    Ok(nexus)
}