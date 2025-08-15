use std::env;

use env_logger::Builder;
use log::LevelFilter;

use crate::cli::ProgramArgs;

pub mod cli;
pub mod neighbour_net;
pub mod ordering;
pub mod splits;
pub mod weights;
pub mod utils;


#[macro_use]
extern crate log;
#[macro_use]
extern crate anyhow;


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