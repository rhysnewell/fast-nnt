use clap::{Args, Parser, Subcommand};

use crate::{
    ordering::OrderingMethod,
    weights::{InferenceMethod, active_set_weights::NNLSParams},
};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct ProgramArgs {
    #[command(subcommand)]
    pub subcommand: ProgramSubcommand,
    #[arg(
        short,
        long,
        default_value = "1",
        global = true,
        help = "Number of threads to use."
    )]
    pub threads: usize,
    #[arg(
        short,
        long,
        default_value = "false",
        conflicts_with = "quiet",
        global = true
    )]
    pub verbose: bool,
    #[arg(
        short,
        long,
        default_value = "false",
        conflicts_with = "verbose",
        global = true
    )]
    pub quiet: bool,
    #[arg(
        short = 'd',
        long,
        default_value = "output",
        global = true,
        help = "Output directory"
    )]
    pub output_directory: String,
}

#[derive(Subcommand, Debug)]
pub enum ProgramSubcommand {
    #[clap(
        name = "neighbour_net",
        about = "Run the SplitsTree NeighborNet algorithm"
    )]
    NeighborNet(NeighbourNetArgs),
}

#[derive(Args, Debug)]
pub struct NeighbourNetArgs {
    /// Input distance matrix file path
    #[arg(short, long, help = "Input distance matrix file path", required = true)]
    pub input: String,
    #[arg(
        short,
        long,
        help = "Output prefix for result files",
        default_value = "output"
    )]
    pub output_prefix: String,
    #[arg(
        short = 'O',
        long,
        help = "The ordering algorithm to use to get splits cycle",
        default_value = "splits-tree4"
    )]
    pub ordering: OrderingMethod,
    #[arg(
        long,
        help = "Split-weight inference algorithm",
        default_value = "active-set"
    )]
    pub inference: InferenceMethod,
    #[clap(flatten)]
    pub nnls_params: NNLSParams,
}

impl NeighbourNetArgs {
    pub fn default() -> Self {
        Self {
            input: String::new(),
            output_prefix: String::from("output"),
            ordering: OrderingMethod::SplitsTree4,
            inference: InferenceMethod::ActiveSet,
            nnls_params: NNLSParams::default(),
        }
    }
}
