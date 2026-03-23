pub mod active_set_weights;
pub mod band;
pub mod splitstree4_weights;

pub use splitstree4_weights::CGSolveStats;

use clap::ValueEnum;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
#[clap(rename_all = "kebab-case")]
pub enum InferenceMethod {
    #[clap(alias = "activeset")]
    ActiveSet,
    #[clap(alias = "splitstree4", alias = "splits-tree4", alias = "conjugate-gradient")]
    CG,
}

impl InferenceMethod {
    pub fn as_str(&self) -> &'static str {
        match self {
            InferenceMethod::ActiveSet => "active-set",
            InferenceMethod::CG => "cg",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "active-set" | "active_set" | "activeset" | "ActiveSet" => InferenceMethod::ActiveSet,
            "cg" | "CG" | "conjugate-gradient" | "conjugate_gradient"
            | "splitstree4" | "splits-tree4" | "SplitsTree4" => InferenceMethod::CG,
            _ => InferenceMethod::default(),
        }
    }

    pub fn from_option(opt: Option<&str>) -> Self {
        opt.map_or_else(InferenceMethod::default, InferenceMethod::from_str)
    }
}

impl Default for InferenceMethod {
    fn default() -> Self {
        InferenceMethod::ActiveSet
    }
}
