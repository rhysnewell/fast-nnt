use clap::ValueEnum;

pub mod ordering_huson2023;
pub mod ordering_splitstree4;
pub mod ordering_matrix;

// make clap enum
#[derive(ValueEnum, Clone, Debug)]
pub enum OrderingMethod {
    Huson2023,
    SplitsTree4
}