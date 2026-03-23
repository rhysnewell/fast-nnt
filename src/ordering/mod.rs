use clap::ValueEnum;

pub mod ordering_huson2023;
pub mod ordering_matrix;
pub mod ordering_splitstree4;

// make clap enum
#[derive(ValueEnum, Clone, Debug)]
#[clap(rename_all = "kebab-case")]
pub enum OrderingMethod {
    #[clap(alias = "huson2023")]
    ClosestPair,
    #[clap(alias = "splitstree4", alias = "splits-tree4")]
    Multiway,
}

impl OrderingMethod {
    pub fn as_str(&self) -> &str {
        match self {
            OrderingMethod::ClosestPair => "closest-pair",
            OrderingMethod::Multiway => "multiway",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "closest-pair" | "closest_pair" | "ClosestPair"
            | "Huson2023" | "huson2023" => OrderingMethod::ClosestPair,
            "multiway" | "Multiway"
            | "SplitsTree4" | "splitstree4" | "splits-tree4" => OrderingMethod::Multiway,
            _ => OrderingMethod::default(),
        }
    }

    pub fn from_option(opt: Option<&str>) -> Self {
        opt.map_or_else(OrderingMethod::default, OrderingMethod::from_str)
    }
}

impl Default for OrderingMethod {
    fn default() -> Self {
        OrderingMethod::Multiway
    }
}
