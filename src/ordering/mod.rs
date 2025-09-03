use clap::ValueEnum;

pub mod ordering_huson2023;
pub mod ordering_matrix;
pub mod ordering_splitstree4;

// make clap enum
#[derive(ValueEnum, Clone, Debug)]
pub enum OrderingMethod {
    Huson2023,
    SplitsTree4,
}

impl OrderingMethod {
    pub fn as_str(&self) -> &str {
        match self {
            OrderingMethod::Huson2023 => "Huson2023",
            OrderingMethod::SplitsTree4 => "SplitsTree4",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "Huson2023" | "huson2023" => OrderingMethod::Huson2023,
            "SplitsTree4" | "splitstree4" | "splits-tree4" => OrderingMethod::SplitsTree4,
            _ => OrderingMethod::default(),
        }
    }

    pub fn from_option(opt: Option<&str>) -> Self {
        match opt {
            Some("Huson2023") | Some("huson2023") => OrderingMethod::Huson2023,
            Some("SplitsTree4") | Some("splitstree4") | Some("splits-tree4") => {
                OrderingMethod::SplitsTree4
            }
            _ => OrderingMethod::default(),
        }
    }
}

impl Default for OrderingMethod {
    fn default() -> Self {
        OrderingMethod::SplitsTree4
    }
}
