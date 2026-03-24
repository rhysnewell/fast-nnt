pub mod active_set_weights;
pub mod band;
pub mod splitstree4_weights;

pub use splitstree4_weights::CGSolveStats;

use clap::ValueEnum;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
#[clap(rename_all = "kebab-case")]
pub enum InferenceMethod {
    #[clap(alias = "activeset", alias = "active_set")]
    ActiveSet,
    #[clap(alias = "splitstree4", alias = "splits-tree4", alias = "splits_tree4", alias = "conjugate-gradient", alias = "conjugate_gradient")]
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
        let normalized: String = s.to_lowercase().replace(['-', '_'], "");
        match normalized.as_str() {
            "activeset" => InferenceMethod::ActiveSet,
            "cg" | "conjugategradient" | "splitstree4" => InferenceMethod::CG,
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
