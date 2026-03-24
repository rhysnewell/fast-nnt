use std::fmt::{self, Write as FmtWrite};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use ndarray::Array2;

use crate::algorithms::equal_angle::assign_coordinates_to_nodes;
use crate::data::splits_blocks::{Compatibility, SplitsBlock};
use crate::nexus::{
    bitset_to_string, escape_label, fmt_f, network_writer::write_network_block_splits, trim_float,
    write_title_and_link,
};
use crate::phylo::phylo_splits_graph::PhyloSplitsGraph;

/// A small container for optional pieces
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
    cycle_1_based: Option<&[usize]>, // [0, t1..tn], 1-based
    graph: Option<&PhyloSplitsGraph>,
    fit_percent: Option<f64>, // for Splits PROPERTIES fit=...
    props: NexusProperties,
) -> anyhow::Result<()> {
    let file = File::create(&path)
        .with_context(|| format!("failed to create NEXUS file at {}", path.as_ref().display()))?;
    let buf = BufWriter::new(file);
    write_nexus_all_to_writer(
        buf,
        taxa_labels,
        distances,
        splits_block,
        cycle_1_based,
        graph,
        fit_percent,
        props,
    )
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
    cycle_1_based: Option<&[usize]>, // [0, t1..tn], 1-based
    graph: Option<&PhyloSplitsGraph>,
    _fit_percent: Option<f64>, // for Splits PROPERTIES fit=...
    props: NexusProperties,
) -> anyhow::Result<()> {
    let mut out = IoFmt(&mut sink);

    debug!(
        "Writing NEXUS with {} taxa, {} splits, distances: {:?}, cycle: {:?}, graph: {:?}",
        taxa_labels.len(),
        splits_block.map_or(0, |s| s.nsplits()),
        distances.is_some(),
        cycle_1_based,
        graph.is_some()
    );
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
    if let Some(sp) = splits_block {
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
    if let Some(g) = graph {
        // angles & coords
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
    writeln!(w, "MATRIX")?;

    for i in 0..n {
        // print row i (full square)
        let mut first = true;
        for j in 0..m {
            if !first {
                write!(w, " ")?;
            }
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
pub fn write_splits_block<W: FmtWrite>(
    mut w: W,
    ntax: usize,
    splits_block: &SplitsBlock,
    title: Option<&str>, // optional; Java calls writeTitleAndLink()
    link: Option<&str>,  // optional
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
        writeln!(
            w,
            "THRESHOLD={};",
            trim_float(splits_block.threshold() as f64, 8)
        )?;
    }

    // PROPERTIES
    write!(
        w,
        "PROPERTIES fit={}",
        trim_float(splits_block.fit() as f64, 2)
    )?;
    match splits_block.compatibility() {
        Compatibility::Compatible => write!(w, " compatible")?,
        Compatibility::Cyclic => write!(w, " cyclic")?,
        Compatibility::WeaklyCompatible => write!(w, " weakly compatible")?,
        Compatibility::Incompatible => write!(w, " non compatible")?,
        Compatibility::Unknown => { /* nothing */ }
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
                let lab = split.label().unwrap_or("");
                write!(w, " '{}' \t", escape_label(lab))?;
            }

            // optional weight
            if format.weights {
                write!(w, " {} \t", trim_float(split.weight, 8))?;
            }

            // optional confidence (only if write_confidences)
            if write_confidences {
                write!(w, " {} \t", trim_float(split.confidence, 8))?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::splits_blocks::{Compatibility, SplitsBlock};
    use crate::splits::asplit::ASplit;
    use fixedbitset::FixedBitSet;
    use ndarray::Array2;

    fn mk_bitset(ntax: usize, elems: &[usize]) -> FixedBitSet {
        let mut bs = FixedBitSet::with_capacity(ntax + 1);
        bs.grow(ntax + 1);
        for &t in elems {
            bs.insert(t);
        }
        bs
    }

    fn mk_split(a: &[usize], ntax: usize, w: f64) -> ASplit {
        let a_bs = mk_bitset(ntax, a);
        ASplit::from_a_ntax_with_weight_conf(a_bs, ntax, w, -1.0)
    }

    fn sample_labels() -> Vec<String> {
        vec![
            "Alpha".to_string(),
            "Beta".to_string(),
            "Gamma".to_string(),
            "Delta".to_string(),
        ]
    }

    fn sample_splits_block() -> SplitsBlock {
        let ntax = 4;
        let mut sb = SplitsBlock::new();
        // split {1} | {2,3,4} weight 1.5
        sb.push(mk_split(&[1], ntax, 1.5));
        // split {1,2} | {3,4} weight 0.75
        sb.push(mk_split(&[1, 2], ntax, 0.75));
        // split {3} | {1,2,4} weight 2.0
        sb.push(mk_split(&[3], ntax, 2.0));
        sb.set_fit(99.5);
        sb.set_compatibility(Compatibility::Cyclic);
        sb.set_cycle(vec![0, 1, 2, 3, 4], true).unwrap();
        sb
    }

    fn sample_distance_matrix() -> Array2<f64> {
        Array2::from_shape_vec(
            (4, 4),
            vec![
                0.0, 1.0, 2.0, 3.0, 1.0, 0.0, 1.5, 2.5, 2.0, 1.5, 0.0, 1.0, 3.0, 2.5, 1.0, 0.0,
            ],
        )
        .unwrap()
    }

    // ---------- write_header ----------

    #[test]
    fn header_outputs_nexus_marker() {
        let mut buf = String::new();
        write_header(&mut buf).unwrap();
        assert!(
            buf.contains("#nexus"),
            "Header must contain '#nexus' marker"
        );
    }

    // ---------- write_taxa_block ----------

    #[test]
    fn taxa_block_contains_ntax_and_labels() {
        let labels = sample_labels();
        let mut buf = String::new();
        write_taxa_block(&mut buf, &labels).unwrap();

        assert!(buf.contains("BEGIN Taxa;"), "Must open Taxa block");
        assert!(buf.contains("DIMENSIONS ntax=4;"), "Must have ntax=4");
        assert!(buf.contains("TAXLABELS"), "Must have TAXLABELS keyword");
        assert!(buf.contains("[1] 'Alpha'"), "Must list first taxon");
        assert!(buf.contains("[2] 'Beta'"), "Must list second taxon");
        assert!(buf.contains("[3] 'Gamma'"), "Must list third taxon");
        assert!(buf.contains("[4] 'Delta'"), "Must list fourth taxon");
        assert!(buf.contains("END; [Taxa]"), "Must close Taxa block");
    }

    #[test]
    fn taxa_block_escapes_single_quotes() {
        let labels = vec!["O'Brien".to_string(), "Normal".to_string()];
        let mut buf = String::new();
        write_taxa_block(&mut buf, &labels).unwrap();

        assert!(
            buf.contains("'O''Brien'"),
            "Single quotes in labels must be escaped by doubling"
        );
    }

    #[test]
    fn taxa_block_empty_labels() {
        let labels: Vec<String> = vec![];
        let mut buf = String::new();
        write_taxa_block(&mut buf, &labels).unwrap();

        assert!(buf.contains("DIMENSIONS ntax=0;"));
        assert!(buf.contains("BEGIN Taxa;"));
        assert!(buf.contains("END; [Taxa]"));
    }

    // ---------- write_distances_block ----------

    #[test]
    fn distances_block_contains_matrix() {
        let dm = sample_distance_matrix();
        let mut buf = String::new();
        write_distances_block(&mut buf, &dm, false).unwrap();

        assert!(
            buf.contains("BEGIN Distances;"),
            "Must open Distances block"
        );
        assert!(
            buf.contains("DIMENSIONS ntax=4;"),
            "Must have ntax dimension"
        );
        assert!(
            buf.contains("FORMAT labels=no diagonal;"),
            "Must have FORMAT line (non-triangle-both)"
        );
        assert!(buf.contains("MATRIX"), "Must have MATRIX keyword");
        assert!(buf.contains("END; [Distances]"), "Must close block");
        // Check a specific value from the matrix
        assert!(
            buf.contains("0 1 2 3"),
            "First row should contain '0 1 2 3'"
        );
    }

    #[test]
    fn distances_block_triangle_both_format() {
        let dm = sample_distance_matrix();
        let mut buf = String::new();
        write_distances_block(&mut buf, &dm, true).unwrap();

        assert!(
            buf.contains("FORMAT labels=no diagonal triangle=both;"),
            "Must have triangle=both format"
        );
    }

    #[test]
    fn distances_block_non_square_matrix_fails() {
        let dm = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let mut buf = String::new();
        let result = write_distances_block(&mut buf, &dm, false);
        assert!(result.is_err(), "Non-square matrix should produce an error");
    }

    #[test]
    fn distances_block_1x1_matrix() {
        let dm = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();
        let mut buf = String::new();
        write_distances_block(&mut buf, &dm, false).unwrap();

        assert!(buf.contains("DIMENSIONS ntax=1;"));
        assert!(buf.contains("0"));
    }

    // ---------- write_splits_block ----------

    #[test]
    fn splits_block_format_and_content() {
        let sb = sample_splits_block();
        let ntax = 4;
        let mut buf = String::new();
        write_splits_block(&mut buf, ntax, &sb, None, None).unwrap();

        assert!(buf.contains("BEGIN SPLITS;"), "Must open SPLITS block");
        assert!(
            buf.contains("DIMENSIONS ntax=4 nsplits=3;"),
            "Must have correct dimensions"
        );
        assert!(buf.contains("FORMAT"), "Must have FORMAT keyword");
        assert!(
            buf.contains("weights=yes"),
            "Default format has weights=yes"
        );
        assert!(
            buf.contains("confidences=no"),
            "No confidence values set (-1.0)"
        );
        assert!(
            buf.contains("PROPERTIES fit=99.5"),
            "Must have fit property"
        );
        assert!(buf.contains("cyclic"), "Must show cyclic compatibility");
        assert!(buf.contains("CYCLE"), "Must have CYCLE line");
        assert!(buf.contains("MATRIX"), "Must have MATRIX section");
        assert!(buf.contains("END; [SPLITS]"), "Must close SPLITS block");
    }

    #[test]
    fn splits_block_with_title_and_link() {
        let sb = sample_splits_block();
        let mut buf = String::new();
        write_splits_block(&mut buf, 4, &sb, Some("My Splits"), Some("Taxa:default")).unwrap();

        assert!(buf.contains("TITLE 'My Splits';"), "Must have title");
        assert!(buf.contains("LINK 'Taxa:default';"), "Must have link");
    }

    #[test]
    fn splits_block_with_labels() {
        let ntax = 3;
        let mut sb = SplitsBlock::new();
        sb.push(mk_split(&[1], ntax, 1.0));
        sb.push(mk_split(&[2], ntax, 0.5));
        sb.format_mut().labels = true;
        sb.set_fit(50.0);

        let mut buf = String::new();
        write_splits_block(&mut buf, ntax, &sb, None, None).unwrap();

        assert!(
            buf.contains("labels=left"),
            "Format should show labels=left"
        );
    }

    #[test]
    fn splits_block_with_confidence() {
        let ntax = 3;
        let mut sb = SplitsBlock::new();
        let a_bs = mk_bitset(ntax, &[1]);
        let s = ASplit::from_a_ntax_with_weight_conf(a_bs, ntax, 1.0, 0.95);
        sb.push(s);
        sb.set_fit(80.0);

        let mut buf = String::new();
        write_splits_block(&mut buf, ntax, &sb, None, None).unwrap();

        assert!(
            buf.contains("confidences=yes"),
            "Must show confidences=yes when confidence values exist"
        );
        assert!(buf.contains("0.95"), "Must include the confidence value");
    }

    #[test]
    fn splits_block_with_split_labels() {
        let ntax = 3;
        let mut sb = SplitsBlock::new();
        sb.push(mk_split(&[1], ntax, 1.0));
        sb.push(mk_split(&[2], ntax, 0.5));
        sb.set_split_label(1, "split-A");
        sb.set_split_label(2, "split-B");
        sb.set_fit(60.0);

        let mut buf = String::new();
        write_splits_block(&mut buf, ntax, &sb, None, None).unwrap();

        assert!(
            buf.contains("SPLITSLABELS"),
            "Must have SPLITSLABELS section"
        );
        assert!(buf.contains("'split-A'"), "Must include split label A");
        assert!(buf.contains("'split-B'"), "Must include split label B");
    }

    #[test]
    fn splits_block_show_both_sides() {
        let ntax = 3;
        let mut sb = SplitsBlock::new();
        sb.push(mk_split(&[1], ntax, 1.0));
        sb.format_mut().show_both_sides = true;
        sb.set_fit(70.0);

        let mut buf = String::new();
        write_splits_block(&mut buf, ntax, &sb, None, None).unwrap();

        assert!(
            buf.contains("showBothSides=yes"),
            "Must show showBothSides=yes"
        );
        assert!(
            buf.contains(" | "),
            "Matrix rows must contain '|' separator for both sides"
        );
    }

    #[test]
    fn splits_block_with_threshold() {
        let ntax = 3;
        let mut sb = SplitsBlock::new();
        sb.push(mk_split(&[1], ntax, 1.0));
        sb.set_threshold(0.01);
        sb.set_fit(90.0);

        let mut buf = String::new();
        write_splits_block(&mut buf, ntax, &sb, None, None).unwrap();

        assert!(
            buf.contains("THRESHOLD=0.01;"),
            "Must include THRESHOLD line when non-zero"
        );
    }

    #[test]
    fn splits_block_compatibility_variants() {
        let ntax = 3;

        for (compat, expected_str) in &[
            (Compatibility::Compatible, "compatible"),
            (Compatibility::Cyclic, "cyclic"),
            (Compatibility::WeaklyCompatible, "weakly compatible"),
            (Compatibility::Incompatible, "non compatible"),
        ] {
            let mut sb = SplitsBlock::new();
            sb.push(mk_split(&[1], ntax, 1.0));
            sb.set_fit(50.0);
            sb.set_compatibility(*compat);

            let mut buf = String::new();
            write_splits_block(&mut buf, ntax, &sb, None, None).unwrap();
            assert!(
                buf.contains(expected_str),
                "Compatibility {:?} should produce '{}' in output, got:\n{}",
                compat,
                expected_str,
                buf
            );
        }
    }

    // ---------- write_st_assumptions_block ----------

    #[test]
    fn st_assumptions_block_content() {
        let mut buf = String::new();
        write_st_assumptions_block(&mut buf).unwrap();

        assert!(buf.contains("BEGIN st_Assumptions;"));
        assert!(buf.contains("uptodate;"));
        assert!(buf.contains("disttransform=NeighborNet;"));
        assert!(buf.contains("splitstransform=EqualAngle;"));
        assert!(buf.contains("END; [st_Assumptions]"));
    }

    // ---------- full write_nexus_all_to_writer ----------

    #[test]
    fn full_nexus_to_writer_minimal() {
        let labels = sample_labels();
        let mut output = Vec::<u8>::new();

        write_nexus_all_to_writer(
            &mut output,
            &labels,
            None, // no distances
            None, // no splits
            None, // no cycle
            None, // no graph
            None, // no fit
            NexusProperties::default(),
        )
        .unwrap();

        let text = String::from_utf8(output).unwrap();
        assert!(text.contains("#nexus"), "Must start with nexus header");
        assert!(text.contains("BEGIN Taxa;"), "Must have taxa block");
        assert!(text.contains("ntax=4"), "Must have correct ntax");
        assert!(
            text.contains("BEGIN st_Assumptions;"),
            "Must have assumptions block"
        );
        // Should NOT contain optional blocks
        assert!(
            !text.contains("BEGIN Distances;"),
            "No distances block when None"
        );
        assert!(!text.contains("BEGIN SPLITS;"), "No splits block when None");
        assert!(
            !text.contains("BEGIN Network;"),
            "No network block when None"
        );
    }

    #[test]
    fn full_nexus_with_distances() {
        let labels = sample_labels();
        let dm = sample_distance_matrix();
        let mut output = Vec::<u8>::new();

        write_nexus_all_to_writer(
            &mut output,
            &labels,
            Some(&dm),
            None,
            None,
            None,
            None,
            NexusProperties::default(),
        )
        .unwrap();

        let text = String::from_utf8(output).unwrap();
        assert!(text.contains("#nexus"));
        assert!(text.contains("BEGIN Taxa;"));
        assert!(text.contains("BEGIN Distances;"));
        assert!(text.contains("END; [Distances]"));
    }

    #[test]
    fn full_nexus_with_splits() {
        let labels = sample_labels();
        let sb = sample_splits_block();
        let mut output = Vec::<u8>::new();

        write_nexus_all_to_writer(
            &mut output,
            &labels,
            None,
            Some(&sb),
            None,
            None,
            None,
            NexusProperties::default(),
        )
        .unwrap();

        let text = String::from_utf8(output).unwrap();
        assert!(text.contains("#nexus"));
        assert!(text.contains("BEGIN Taxa;"));
        assert!(text.contains("BEGIN SPLITS;"));
        assert!(text.contains("nsplits=3"));
        assert!(text.contains("END; [SPLITS]"));
    }

    #[test]
    fn full_nexus_with_distances_and_splits() {
        let labels = sample_labels();
        let dm = sample_distance_matrix();
        let sb = sample_splits_block();
        let mut output = Vec::<u8>::new();

        write_nexus_all_to_writer(
            &mut output,
            &labels,
            Some(&dm),
            Some(&sb),
            None,
            None,
            None,
            NexusProperties::default(),
        )
        .unwrap();

        let text = String::from_utf8(output).unwrap();
        // Verify ordering: header, Taxa, Distances, Splits, Assumptions
        let nexus_pos = text.find("#nexus").unwrap();
        let taxa_pos = text.find("BEGIN Taxa;").unwrap();
        let dist_pos = text.find("BEGIN Distances;").unwrap();
        let splits_pos = text.find("BEGIN SPLITS;").unwrap();
        let assume_pos = text.find("BEGIN st_Assumptions;").unwrap();

        assert!(nexus_pos < taxa_pos, "Header before Taxa");
        assert!(taxa_pos < dist_pos, "Taxa before Distances");
        assert!(dist_pos < splits_pos, "Distances before Splits");
        assert!(splits_pos < assume_pos, "Splits before Assumptions");
    }

    #[test]
    fn full_nexus_triangle_both_property() {
        let labels = sample_labels();
        let dm = sample_distance_matrix();
        let mut output = Vec::<u8>::new();

        let props = NexusProperties {
            distances_triangle_both: true,
            ..Default::default()
        };

        write_nexus_all_to_writer(
            &mut output,
            &labels,
            Some(&dm),
            None,
            None,
            None,
            None,
            props,
        )
        .unwrap();

        let text = String::from_utf8(output).unwrap();
        assert!(text.contains("triangle=both"));
    }
}
