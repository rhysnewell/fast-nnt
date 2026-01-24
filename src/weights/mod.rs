pub mod active_set_weights;
pub mod splitstree4_weights;

use clap::ValueEnum;

#[derive(Clone, Copy, Debug, ValueEnum)]
#[clap(rename_all = "kebab-case")]
pub enum InferenceMethod {
    #[clap(alias = "activeset")]
    ActiveSet,
    #[clap(alias = "splitstree4")]
    SplitsTree4,
}

impl InferenceMethod {
    pub fn as_str(&self) -> &'static str {
        match self {
            InferenceMethod::ActiveSet => "active-set",
            InferenceMethod::SplitsTree4 => "splitstree4",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "active-set" | "active_set" | "activeset" | "ActiveSet" => InferenceMethod::ActiveSet,
            "splitstree4" | "splits-tree4" | "SplitsTree4" => InferenceMethod::SplitsTree4,
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
