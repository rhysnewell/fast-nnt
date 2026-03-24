use clap::ValueEnum;

pub mod ordering_huson2023;
pub mod ordering_matrix;
pub mod ordering_splitstree4;

// make clap enum
#[derive(ValueEnum, Clone, Debug)]
#[clap(rename_all = "kebab-case")]
pub enum OrderingMethod {
    #[clap(alias = "huson2023", alias = "closest_pair")]
    ClosestPair,
    #[clap(alias = "splitstree4", alias = "splits-tree4", alias = "splits_tree4")]
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
        let normalized: String = s.to_lowercase().replace(['-', '_'], "");
        match normalized.as_str() {
            "closestpair" | "huson2023" => OrderingMethod::ClosestPair,
            "multiway" | "splitstree4" => OrderingMethod::Multiway,
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
