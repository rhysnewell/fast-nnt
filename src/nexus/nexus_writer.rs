use std::fs::File;
use std::io::{BufWriter, Write};
use std::fmt::{self, Write as FmtWrite};
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use ndarray::Array2;

use crate::data::splits_blocks::{Compatibility, SplitsBlock};
use crate::phylo::phylo_splits_graph::PhyloSplitsGraph;
use crate::algorithms::equal_angle::assign_coordinates_to_nodes;
use crate::nexus::{
    fmt_f,
    trim_float,
    bitset_to_string,
    escape_label,
    write_title_and_link,
    network_writer::write_network_block_splits,
};

/// A small container for optional pieces you might want in blocks.
#[derive(Default, Clone)]
pub struct NexusProperties {
    /// e.g., "fit=99.92 leastsquares cyclic"
    pub splits_properties: Option<String>,
    /// optional total angle for equal-angle placement (default 360)
    pub total_angle_deg: Option<f64>,
    /// root split id to small-weight edges for the quick coordinate helper (see assign_coordinates_to_nodes)
    pub network_root_split: Option<i32>,
    /// if true, write Distances FORMAT labels=no diagonal triangle=both (SplitsTree-ish)
    pub distances_triangle_both: bool,
}


/// Adapter: let `fmt::Write` target any `io::Write`.
struct IoFmt<W: Write>(W);
impl<W: Write> FmtWrite for IoFmt<W> {
    fn write_str(&mut self, s: &str) -> Result<(), fmt::Error> {
        self.0.write_all(s.as_bytes()).map_err(|_| fmt::Error)
    }
}

/// High-level: write a complete NEXUS file directly to a file path.
pub fn write_nexus_all_to_path<P: AsRef<Path>>(
    path: P,
    taxa_labels: &[String],
    distances: Option<&Array2<f64>>,
    splits_block: Option<&SplitsBlock>,
    cycle_1_based: Option<&[usize]>,      // [0, t1..tn], 1-based
    graph: Option<&PhyloSplitsGraph>,
    fit_percent: Option<f64>,             // for Splits PROPERTIES fit=...
    props: NexusProperties,
) -> anyhow::Result<()> {
    let file = File::create(&path)
        .with_context(|| format!("failed to create NEXUS file at {}", path.as_ref().display()))?;
    let buf = BufWriter::new(file);
    write_nexus_all_to_writer(buf, taxa_labels, distances, splits_block, cycle_1_based, graph, fit_percent, props)
        .with_context(|| format!("failed while writing NEXUS to {}", path.as_ref().display()))?;
    Ok(())
}

/// Lower-level: write a complete NEXUS file to any `io::Write` sink.
/// This streams to disk (or socket) without building a giant String.
pub fn write_nexus_all_to_writer<W: Write>(
    mut sink: W,
    taxa_labels: &[String],
    distances: Option<&Array2<f64>>,
    splits_block: Option<&SplitsBlock>,
    cycle_1_based: Option<&[usize]>,      // [0, t1..tn], 1-based
    graph: Option<&PhyloSplitsGraph>,
    fit_percent: Option<f64>,             // for Splits PROPERTIES fit=...
    props: NexusProperties,
) -> anyhow::Result<()> {
    let mut out = IoFmt(&mut sink);


    debug!("Writing NEXUS with {} taxa, {} splits, distances: {:?}, cycle: {:?}, graph: {:?}",
           taxa_labels.len(), splits_block.map_or(0, |s| s.nsplits()), distances.is_some(), cycle_1_based, graph.is_some());
    debug!("Writing header.");
    write_header(&mut out)?;

    // TAXA
    debug!("Writing taxa block.");
    write_taxa_block(&mut out, taxa_labels)?;

    // Distances (optional)
    if let Some(dm) = distances {
        debug!("Writing distances block.");
        write_distances_block(&mut out, dm, props.distances_triangle_both)?;
    }

    // SPLITS (optional)
    if let (Some(sp), Some(cycle)) = (splits_block, cycle_1_based) {
        // let cycle_norm = normalize_cycle(cycle);
        debug!("Writing splits block with {} splits.", sp.nsplits());
        write_splits_block(
            &mut out,
            taxa_labels.len(),
            sp,
            props.splits_properties.as_deref(),
            None, // link is not used in this context
        )?;
    }

    // NETWORK (optional): requires a PhyloSplitsGraph + cycle for angles & coordinates
    if let (Some(g), Some(sp), Some(cycle)) = (graph, splits_block, cycle_1_based) {
        // angles & coords
        // let total = props.total_angle_deg.unwrap_or(360.0);
        // let ntax = taxa_labels.len(); // expecting 1-based input usage elsewhere
        // let cycle_norm = normalize_cycle(cycle);

        // // assign per-split angle & push to edges (skip forbiddens — if you need that, pass a bitset and modify here)
        // assign_angles_to_edges(ntax, sp, &cycle_norm, &mut g.clone_graph(), None, total);
        // NOTE: we call assign_angles_to_edges onto a clone if you don't want to mutate the input `graph`.
        // If you do want to mutate original, call on &mut graph before passing here.

        // quick coordinates (use_weights=true is typical for to-scale)
        debug!("Assigning coordinates to nodes.");
        let coords = assign_coordinates_to_nodes(true, g, 1, props.network_root_split.unwrap_or(0));

        debug!("Writing network block.");
        write_network_block_splits(&mut out, g, &coords, taxa_labels)?;
    }

    write_st_assumptions_block(&mut out)?;

    Ok(())
}

/* ---------------- Individual blocks ---------------- */

pub fn write_header<W: FmtWrite>(mut w: W) -> Result<()> {
    writeln!(w, "#nexus\n")?;
    Ok(())
}

pub fn write_taxa_block<W: FmtWrite>(mut w: W, taxa_labels: &[String]) -> Result<()> {
    let ntax = taxa_labels.len();
    writeln!(w, "BEGIN Taxa;")?;
    writeln!(w, "DIMENSIONS ntax={};", ntax)?;
    writeln!(w, "TAXLABELS")?;
    for (i, name) in taxa_labels.iter().enumerate() {
        // 1-based print (optional) — SplitsTree often shows [index] 'name'
        writeln!(w, "[{}] '{}'", i + 1, escape_label(name))?;
    }
    writeln!(w, "\n;")?;
    writeln!(w, "END; [Taxa]\n")?;
    Ok(())
}

/// Distances block similar to SplitsTree prints:
/// FORMAT labels=no diagonal triangle=both;
/// MATRIX: full square or triangular acceptable. We’ll output full square for clarity.
pub fn write_distances_block<W: FmtWrite>(
    mut w: W,
    matrix: &ndarray::Array2<f64>,
    triangle_both_format: bool,
) -> Result<()> {
    let (n, m) = (matrix.nrows(), matrix.ncols());
    if n != m {
        return Err(anyhow!("Distances matrix must be square"));
    }

    writeln!(w, "BEGIN Distances;")?;
    writeln!(w, "DIMENSIONS ntax={};", n)?;
    if triangle_both_format {
        writeln!(w, "FORMAT labels=no diagonal triangle=both;")?;
    } else {
        writeln!(w, "FORMAT labels=no diagonal;")?;
    }
    writeln!(w, "MATRIX # The square distance matrix")?;

    for i in 0..n {
        // print row i (full square)
        let mut first = true;
        for j in 0..m {
            if !first { write!(w, " ")?; }
            first = false;
            write!(w, "{}", fmt_f(matrix[[i, j]]))?;
        }
        writeln!(w)?;
    }

    writeln!(w, ";")?;
    writeln!(w, "END; [Distances]\n")?;
    Ok(())
}


/// Public entry: write the SPLITS block as per SplitsTree Java writer.
/// `ntax` should come from your TaxaBlock.
pub fn write_splits_block<W: FmtWrite>(
    mut w: W,
    ntax: usize,
    splits_block: &SplitsBlock,
    title: Option<&str>,   // optional; Java calls writeTitleAndLink()
    link: Option<&str>,    // optional
) -> Result<()> {
    let nsplits = splits_block.nsplits();
    let format = splits_block.format();
    let write_confidences = splits_block.has_confidence_values();

    // Header
    writeln!(w, "\nBEGIN SPLITS;")?;
    write_title_and_link(&mut w, title, link)?;
    writeln!(w, "DIMENSIONS ntax={} nsplits={};", ntax, nsplits)?;

    // FORMAT
    write!(w, "FORMAT")?;
    if format.labels {
        write!(w, " labels=left")?;
    } else {
        write!(w, " labels=no")?;
    }
    if format.weights {
        write!(w, " weights=yes")?;
    } else {
        write!(w, " weights=no")?;
    }
    if write_confidences {
        write!(w, " confidences=yes")?;
    } else {
        write!(w, " confidences=no")?;
    }
    if format.show_both_sides {
        write!(w, " showBothSides=yes")?;
    } else {
        write!(w, " showBothSides=no")?;
    }
    writeln!(w, ";")?;

    // THRESHOLD (optional)
    if splits_block.threshold() != 0.0 {
        writeln!(w, "THRESHOLD={};", trim_float(splits_block.threshold() as f64, 8))?;
    }

    // PROPERTIES
    write!(w, "PROPERTIES fit={}", trim_float(splits_block.fit() as f64, 2))?;
    match splits_block.compatibility() {
        Compatibility::Compatible      => write!(w, " compatible")?,
        Compatibility::Cyclic          => write!(w, " cyclic")?,
        Compatibility::WeaklyCompatible=> write!(w, " weakly compatible")?,
        Compatibility::Incompatible    => write!(w, " non compatible")?,
        Compatibility::Unknown         => { /* nothing */ }
    }
    writeln!(w, ";")?;

    // CYCLE (optional) — 1-based, skip index 0
    if let Some(cycle) = splits_block.cycle() {
        write!(w, "CYCLE")?;
        for i in 1..cycle.len() {
            write!(w, " {}", cycle[i])?;
        }
        writeln!(w, ";")?;
    }

    // SPLITSLABELS (optional) — in key order
    if !splits_block.split_labels().is_empty() {
        write!(w, "SPLITSLABELS")?;
        for (_sid, label) in splits_block.split_labels().iter() {
            write!(w, " '{}'", escape_label(label))?;
        }
        writeln!(w, ";")?;
    }

    // MATRIX
    writeln!(w, "MATRIX")?;
    {
        let mut t = 1usize; // 1-based split id in output order
        for split in splits_block.splits() {
            // prefix
            write!(w, "[{}, size={}] \t", t, split.size())?;

            // optional label
            if format.labels {
                let lab = split.get_label().unwrap_or("");
                write!(w, " '{}' \t", escape_label(lab))?;
            }

            // optional weight
            if format.weights {
                write!(w, " {} \t", trim_float(split.weight, 8))?;
            }

            // optional confidence (only if write_confidences)
            if write_confidences {
                write!(w, " {} \t", trim_float(split.get_confidence(), 8))?;
            }

            // A side
            write!(w, " {}", bitset_to_string(split.get_a()))?;
            // optional " | B side"
            if format.show_both_sides {
                write!(w, " | {}", bitset_to_string(split.get_b()))?;
            }

            writeln!(w, ",")?;
            t += 1;
        }
    }
    writeln!(w, ";")?;
    writeln!(w, "END; [SPLITS]")?;

    Ok(())
}




pub fn write_st_assumptions_block<W: FmtWrite>(mut w: W) -> Result<()> {
    writeln!(w, "BEGIN st_Assumptions;")?;
    writeln!(w, "uptodate;")?;
    writeln!(w, "disttransform=NeighborNet;")?;
    writeln!(w, "splitstransform=EqualAngle;")?;
    writeln!(w, "SplitsPostProcess filter=dimension value=4;")?;
    writeln!(w, " exclude  no missing;")?;
    writeln!(w, "autolayoutnodelabels;")?;
    writeln!(w, "END; [st_Assumptions]\n")?;
    Ok(())
}




