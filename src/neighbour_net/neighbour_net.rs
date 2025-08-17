use anyhow::{Context, Result, anyhow};
use fixedbitset::FixedBitSet;
use log::{debug, info, warn};
use ndarray::Array2;
use rayon::prelude::*;
use serde::Serialize;
use std::{fs, path::Path, time::Instant};

use crate::algorithms::equal_angle::{equal_angle_apply, EqualAngleOpts};
use crate::cli::NeighborNetArgs;
use crate::data::splits_blocks::{self, SplitsBlock};
use crate::nexus::nexus_writer::{write_nexus_all_to_path, NexusProperties};
use crate::ordering::ordering_graph::compute_ordering;
use crate::phylo::phylo_splits_graph::PhyloSplitsGraph;
use crate::utils::compute_least_squares_fit;
use crate::weights::active_set_weights::{NNLSParams, compute_asplits};

pub struct NeighbourNet {
    out_dir: String,
    args: NeighborNetArgs,
}

impl NeighbourNet {
    pub fn new(out_dir: String, args: NeighborNetArgs) -> Self {
        NeighbourNet { out_dir, args }
    }

    pub fn run(&self) -> Result<()> {
        let t0 = Instant::now();

        // 1) Load distance matrix (+ labels + parse meta)
        let (distance_matrix, labels, parse_meta) = self
            .load_distance_matrix(&self.args.input)
            .context("loading distance matrix")?;
        let n = distance_matrix.nrows();
        info!("Loaded distance matrix: {}x{}", n, n);

        // 2) Compute NeighborNet cycle (1-based with leading 0)
        let t_cycle = Instant::now();
        let mut cycle = compute_ordering(&distance_matrix);
        if cycle.first().copied() != Some(0) {
            cycle = std::iter::once(0usize)
                .chain(cycle.into_iter().map(|i| i + 1))
                .collect();
        }
        let cycle_sec = t_cycle.elapsed().as_secs_f64();
        info!("Computed cycle in {:.3}s", cycle_sec);
        debug!("Cycle (1-based): {:?}", &cycle[1..]);

        // 3) Active-Set NNLS
        let t_nnls = Instant::now();
        let mut params = self.args.nnls_params.clone();
        let splits =
            compute_asplits(&cycle, &distance_matrix, &mut params, None).context("ASplits solved")?;
        let nnls_sec = t_nnls.elapsed().as_secs_f64();
        info!(
            "Estimated {} splits (cutoff = {}) in {:.3}s",
            splits.len(),
            params.cutoff,
            nnls_sec
        );

        
        // 4) Least-squares fit
        let t_fit = Instant::now();
        let fit = compute_least_squares_fit(&distance_matrix, &splits);
        let fit_sec = t_fit.elapsed().as_secs_f64();
        info!(
            "Least-squares fit: {:.4} % (computed in {:.3}s)",
            fit, fit_sec
        );

        // 5) Create splits blocks
        let t_spl = Instant::now();
        let mut splits_blocks = splits_blocks::SplitsBlock::new();
        splits_blocks.set_splits(splits);
        splits_blocks.set_fit(fit);
        // splits_blocks.set_threshold(params.threshold);
        // splits_blocks.set_partial(params.partial);
        splits_blocks.set_cycle(cycle, true)?;
        let splits_sec = t_spl.elapsed().as_secs_f64();
        info!("Created splits block with {} splits in {:.3}s", splits_blocks.nsplits(), splits_sec);

        // 6) Create phylogenetic splits graph
        let t_graph = Instant::now();
        let mut graph = PhyloSplitsGraph::new();
        let n_taxa = distance_matrix.nrows();
        // then:
        let cycle = splits_blocks.cycle().expect("Cycle not yet set."); // ensure cycle[1]==1
        let mut used_splits = FixedBitSet::with_capacity(splits_blocks.nsplits() + 1);
        equal_angle_apply(
            EqualAngleOpts::default(),
            &labels,
            &splits_blocks,
            &mut graph,
            None,
            &mut used_splits,
        )?;
        let graph_sec = t_graph.elapsed().as_secs_f64();
        info!("Created phylogenetic splits graph in {:.3}s", graph_sec);

        // 7) Outputs
        let t_out = Instant::now();
        self.output_results(&cycle, &labels, &splits_blocks, &distance_matrix, &graph, fit)
            .context("writing outputs")?;
        let out_sec = t_out.elapsed().as_secs_f64();
        info!("Wrote outputs in {:.3}s", out_sec);

        // 8) Build + write run log
        let run_log_path = Path::new(&self.out_dir).join("run_log.json");

        let stats = self.build_run_stats(
            &parse_meta,
            &labels,
            &splits_blocks,
            &params,
            fit,
            RunTimings {
                load_sec: parse_meta.load_sec,
                cycle_sec,
                nnls_sec,
                fit_sec,
                splits_sec,
                graph_sec,
                output_sec: out_sec,
                total_sec: t0.elapsed().as_secs_f64(),
            },
        );
        fs::write(&run_log_path, serde_json::to_string_pretty(&stats)?)?;
        info!("Run log written: {}", run_log_path.display());

        info!("Done in {:.3}s total.", t0.elapsed().as_secs_f64());
        Ok(())
    }

    /* ───────────── I/O ───────────── */

    /// Parse CSV/TSV/; / | / space; header/index row optional.
    /// Returns (n×n distances, labels[1..=n], parse meta incl. load time & symmetry fixes).
    fn load_distance_matrix(&self, path: &str) -> Result<(Array2<f64>, Vec<String>, ParseMeta)> {
        let t_load = Instant::now();
        let text = fs::read_to_string(path).with_context(|| format!("reading '{}'", path))?;

        // Detect delimiter
        let lines = text.lines().filter(|l| !l.trim().is_empty());
        let first_line = lines
            .clone()
            .find(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
            .ok_or_else(|| anyhow!("no data lines found"))?
            .to_string();
        let delim = detect_delim(&first_line);
        info!("Detected delimiter: {:?}", delim);

        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(false)
            .delimiter(delim as u8)
            .from_reader(text.as_bytes());

        let mut rows: Vec<Vec<String>> = Vec::new();
        for rec in rdr.records() {
            let rec = rec?;
            let row: Vec<String> = rec.iter().map(|s| s.trim().to_string()).collect();
            if !row.is_empty() && row.iter().any(|t| !t.is_empty()) {
                rows.push(row);
            }
        }
        if rows.is_empty() {
            return Err(anyhow!("empty table"));
        }

        let (has_header, has_index) = sniff_header_index(&rows)?;
        info!("Header: {}, Index column: {}", has_header, has_index);

        // Derive labels + numeric region
        let (labels, start_row, start_col) = if has_header && has_index {
            let header = &rows[0];
            let labels = header[1..].iter().map(|s| s.to_string()).collect();
            (labels, 1usize, 1usize)
        } else if has_header && !has_index {
            let header = &rows[0];
            let labels = header.iter().map(|s| s.to_string()).collect();
            (labels, 1usize, 0usize)
        } else if !has_header && has_index {
            let labels = rows.iter().map(|r| r[0].to_string()).collect::<Vec<_>>();
            (labels, 0usize, 1usize)
        } else {
            let n = rows.len();
            let labels = (1..=n).map(|i| format!("t{}", i)).collect::<Vec<_>>();
            (labels, 0usize, 0usize)
        };

        let n = rows.len() - start_row;
        let m = rows[start_row].len() - start_col;
        if n != m {
            return Err(anyhow!(
                "parsed table is not square: rows={}, cols={}",
                n,
                m
            ));
        }

        // Parse numbers
        let mut mat = Array2::<f64>::zeros((n, n));
        for (ri, row) in rows[start_row..].iter().enumerate() {
            for (ci, tok) in row[start_col..start_col + n].iter().enumerate() {
                let val: f64 = tok.parse().with_context(|| {
                    format!(
                        "parsing number at row {}, col {}",
                        ri + start_row + 1,
                        ci + start_col + 1
                    )
                })?;
                mat[[ri, ci]] = val;
            }
        }

        // Symmetrize (count repairs)
        let mut symmetry_pairs_fixed = 0usize;
        for i in 0..n {
            mat[[i, i]] = 0.0;
            for j in (i + 1)..n {
                let a = mat[[i, j]];
                let b = mat[[j, i]];
                if (a - b).abs() > 1e-12 {
                    let avg = 0.5 * (a + b);
                    mat[[i, j]] = avg;
                    mat[[j, i]] = avg;
                    symmetry_pairs_fixed += 1;
                }
            }
        }
        if symmetry_pairs_fixed > 0 {
            warn!(
                "Distance matrix not perfectly symmetric; averaged {} off-diagonal pairs",
                symmetry_pairs_fixed
            );
        }

        // Labels sanity
        let labels = if labels.len() == n {
            labels
        } else {
            warn!(
                "Label count ({}) != n ({}). Synthesizing t1..tn labels.",
                labels.len(),
                n
            );
            (1..=n).map(|i| format!("t{}", i)).collect()
        };

        let meta = ParseMeta {
            delimiter: delim,
            has_header,
            has_index,
            symmetry_pairs_fixed,
            load_sec: t_load.elapsed().as_secs_f64(),
        };

        Ok((mat, labels, meta))
    }

    fn output_results(
        &self,
        cycle: &[usize],
        labels: &[String],
        splits: &SplitsBlock,
        distances: &Array2<f64>,
        graph: &PhyloSplitsGraph,
        fit: f32,
    ) -> Result<()> {
        let out_dir = self.out_dir.clone();
        fs::create_dir_all(&out_dir).with_context(|| format!("creating {}", out_dir))?;

        // meta.json (kept)
        let meta_path = Path::new(&out_dir).join("meta.json");
        let meta = serde_json::json!({
            "fit_percent": fit,
            "ntax": labels.len(),
            "cutoff": self.args.nnls_params.cutoff,
            "input": self.args.input,
        });
        fs::write(&meta_path, serde_json::to_string_pretty(&meta)?)?;

        // Writer properties
        let props = NexusProperties {
            splits_properties: Some("leastsquares cyclic".to_string()),
            total_angle_deg: Some(360.0),
            network_root_split: None, // or Some(split_id)
            distances_triangle_both: true,
        };


        // NEXUS
        let nexus_path = Path::new(&out_dir).join(format!("{}.nex", self.args.output_prefix));
        write_nexus_all_to_path(
            &nexus_path,
            labels,
            Some(distances),
            Some(splits),
            Some(cycle),
            Some(graph),
            Some(fit as f64),
            props,
        )?;

        info!("Outputs:");
        info!("  {}", meta_path.display());
        info!("  {}", nexus_path.display());
        // info!("R quickstart:");
        // info!("  library(igraph)");
        // info!("  nodes <- read.csv('{}/nodes.csv')", out_dir);
        // info!("  edges <- read.csv('{}/edges_cycle.csv')", out_dir);
        // info!("  g <- graph_from_data_frame(edges, directed=FALSE, vertices=nodes)");
        // info!("  plot(g, layout=layout_in_circle(g))");
        // info!("Python/NetworkX quickstart:");
        // info!("  import pandas as pd, networkx as nx");
        // info!("  edges = pd.read_csv('{}/edges_cycle.csv')", out_dir);
        // info!("  G = nx.from_pandas_edgelist(edges, 'source', 'target')");
        // info!("  nx.draw_circular(G)");

        Ok(())
    }

    /* ───────────── run_log helpers ───────────── */

    fn build_run_stats(
        &self,
        parse: &ParseMeta,
        labels: &[String],
        splits: &SplitsBlock,
        nnls: &NNLSParams,
        fit_percent: f32,
        timings: RunTimings,
    ) -> RunLog {
        let n = labels.len();
        let npairs = n * (n - 1) / 2;

        // split stats
        let (num_trivial, num_nontrivial, sum_weights) = splits.splits()
            .par_bridge()
            .fold(
                || (0usize, 0usize, 0.0f64),
                |mut acc, s| {
                    if s.is_trivial() {
                        acc.0 += 1;
                    } else {
                        acc.1 += 1;
                    }
                    acc.2 += s.get_weight();
                    acc
                },
            )
            .reduce(|| (0, 0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));

        // system stats
        let sys = system_stats();

        RunLog {
            input: self.args.input.clone(),
            out_dir: self.out_dir.clone(),
            matrix: MatrixMeta {
                n,
                npairs,
                delimiter: parse.delimiter.to_string(),
                has_header: parse.has_header,
                has_index: parse.has_index,
                symmetry_pairs_fixed: parse.symmetry_pairs_fixed,
            },
            nnls: NNLSMeta {
                cutoff: nnls.cutoff,
                proj_grad_bound: nnls.proj_grad_bound,
                cgnr_iterations: nnls.cgnr_iterations,
                cgnr_tolerance: nnls.cgnr_tolerance,
                active_set_rho: nnls.active_set_rho,
                num_splits: splits.nsplits(),
                num_trivial_splits: num_trivial,
                num_nontrivial_splits: num_nontrivial,
                sum_weights,
            },
            fit_percent,
            timings,
            system: sys,
        }
    }
}

/* ───────────── metadata + logging structs ───────────── */

#[derive(Serialize, Clone)]
struct ParseMeta {
    delimiter: char,
    has_header: bool,
    has_index: bool,
    symmetry_pairs_fixed: usize,
    load_sec: f64,
}

#[derive(Serialize)]
struct MatrixMeta {
    n: usize,
    npairs: usize,
    delimiter: String,
    has_header: bool,
    has_index: bool,
    symmetry_pairs_fixed: usize,
}

#[derive(Serialize)]
struct NNLSMeta {
    cutoff: f64,
    proj_grad_bound: f64,
    cgnr_iterations: usize,
    cgnr_tolerance: f64,
    active_set_rho: f64,
    num_splits: usize,
    num_trivial_splits: usize,
    num_nontrivial_splits: usize,
    sum_weights: f64,
}

#[derive(Serialize)]
struct RunTimings {
    load_sec: f64,
    cycle_sec: f64,
    nnls_sec: f64,
    fit_sec: f64,
    output_sec: f64,
    splits_sec: f64,
    graph_sec: f64,
    total_sec: f64,
}

#[derive(Serialize)]
struct SystemStats {
    os: String,
    arch: String,
    num_cpus: usize,
    rayon_threads: usize,
    peak_rss_bytes: Option<u64>,
    current_rss_bytes: u64,
}

#[derive(Serialize)]
struct RunLog {
    input: String,
    out_dir: String,
    matrix: MatrixMeta,
    nnls: NNLSMeta,
    fit_percent: f32,
    timings: RunTimings,
    system: SystemStats,
}

/* ───────────── system / memory ───────────── */

fn system_stats() -> SystemStats {
    let os = std::env::consts::OS.to_string();
    let arch = std::env::consts::ARCH.to_string();
    let num_cpus = num_cpus::get();
    let rayon_threads = rayon::current_num_threads();
    let peak = peak_rss_bytes();
    let current = current_rss_bytes();
    SystemStats {
        os,
        arch,
        num_cpus,
        rayon_threads,
        peak_rss_bytes: peak,
        current_rss_bytes: current,
    }
}

fn current_rss_bytes() -> u64 {
    use sysinfo::{Pid, Process, System};
    let mut sys = System::new();
    sys.refresh_processes(sysinfo::ProcessesToUpdate::All, true);
    if let Ok(pid) = sysinfo::get_current_pid() {
        if let Some(p) = sys.process(pid) {
            // KiB -> bytes
            return p.memory() as u64 * 1024;
        }
    }
    0
}

#[cfg(target_os = "linux")]
fn peak_rss_bytes() -> Option<u64> {
    // Parse /proc/self/status VmHWM: "<num> kB"
    let s = fs::read_to_string("/proc/self/status").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb = rest.split_whitespace().nth(0)?.parse::<u64>().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

#[cfg(target_os = "macos")]
fn peak_rss_bytes() -> Option<u64> {
    // ru_maxrss is bytes on macOS
    unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut ru) == 0 {
            return Some(ru.ru_maxrss as u64);
        }
    }
    None
}

#[cfg(any(target_os = "freebsd", target_os = "openbsd", target_os = "netbsd"))]
fn peak_rss_bytes() -> Option<u64> {
    // ru_maxrss is kilobytes on *BSD
    unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut ru) == 0 {
            return Some((ru.ru_maxrss as u64) * 1024);
        }
    }
    None
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "macos",
    target_os = "freebsd",
    target_os = "openbsd",
    target_os = "netbsd"
)))]
fn peak_rss_bytes() -> Option<u64> {
    None // fallback: not supported on this platform
}

/* ───────────── misc helpers ───────────── */

fn default_out_dir(input: &str) -> String {
    let p = Path::new(input);
    if let Some(stem) = p.file_stem().and_then(|s| s.to_str()) {
        format!("{}_nn", stem)
    } else {
        "nn_out".to_string()
    }
}

/// Pick the delimiter with the most hits among common choices.
fn detect_delim(line: &str) -> char {
    let cands = [',', '\t', ';', '|', ' '];
    let mut best = (0usize, ',');
    for &c in &cands {
        let count = line.matches(c).count();
        if count > best.0 {
            best = (count, c);
        }
    }
    best.1
}

fn sniff_header_index(rows: &[Vec<String>]) -> anyhow::Result<(bool, bool)> {
    let is_num = |s: &str| s.parse::<f64>().is_ok();

    if rows.is_empty() {
        return Ok((false, false));
    }

    // Decide if there is an index column:
    // - require non-numeric in col 0 for at least one data row (i > 0), AND
    // - require at least two non-numeric first-column entries across first ~10 rows
    let sample_n = rows.len().min(10);
    let mut nonnum_first_col_total = 0usize;
    let mut nonnum_first_col_after_first = 0usize;
    for (i, r) in rows.iter().take(sample_n).enumerate() {
        if r.is_empty() {
            continue;
        }
        if !is_num(&r[0]) {
            nonnum_first_col_total += 1;
            if i > 0 {
                nonnum_first_col_after_first += 1;
            }
        }
    }
    let has_index = nonnum_first_col_after_first >= 1 && nonnum_first_col_total >= 2;

    // When checking header, ignore the index column if present
    let skip = if has_index { 1 } else { 0 };

    let first = &rows[0];
    let first_after = if first.len() > skip {
        &first[skip..]
    } else {
        &[][..]
    };
    let nonnum_in_first_after = first_after.iter().any(|s| !is_num(s));
    let first_num_count = first_after.iter().filter(|s| is_num(s)).count();

    let second_after = rows
        .get(1)
        .map(|r| if r.len() > skip { &r[skip..] } else { &[][..] });
    let second_num_count = second_after
        .map(|r| r.iter().filter(|s| is_num(s)).count())
        .unwrap_or(first_num_count);

    // Header if first (non-index) row has non-numeric tokens,
    // OR it has fewer numeric tokens than the second row
    let has_header = nonnum_in_first_after || first_num_count < second_num_count;

    Ok((has_header, has_index))
}

fn ones_to_string(bs: &fixedbitset::FixedBitSet) -> String {
    let mut v: Vec<usize> = bs.ones().filter(|&t| t != 0).collect();
    v.sort_unstable();
    v.iter()
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(";")
}
/* ───────────── tests ───────────── */

#[cfg(test)]
mod read_matrix_tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// helper: write content to a temp file and invoke f(&Path)
    fn with_temp(content: &str, f: impl FnOnce(&std::path::Path)) {
        let mut tf = NamedTempFile::new().expect("tmp");
        tf.write_all(content.as_bytes()).expect("write");
        f(tf.path());
        // file removed when tf drops
    }

    fn approx(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn csv_header_and_index() {
        // first cell empty means header + index
        let content = r#"
,A,B,C
A,0,1,2
B,1,0,3
C,2,3,0
"#;
        with_temp(content, |p| {
            let args = NeighborNetArgs {
                input: p.to_string_lossy().into_owned(),
                output_prefix: "output".into(),
                nnls_params: NNLSParams::default(),
            };
            let nn = NeighbourNet::new("/tmp".to_string(), args);
            let (mat, labels, meta) = nn.load_distance_matrix(&nn.args.input).unwrap();

            assert_eq!(labels, vec!["A", "B", "C"]);
            assert!(meta.has_header);
            assert!(meta.has_index);
            assert_eq!(meta.symmetry_pairs_fixed, 0);

            assert_eq!(mat.shape(), &[3, 3]);
            assert!(approx(mat[[0, 1]], 1.0, 1e-12));
            assert!(approx(mat[[1, 2]], 3.0, 1e-12));
            assert!(approx(mat[[2, 0]], 2.0, 1e-12));
        });
    }

    #[test]
    fn csv_header_no_index() {
        let content = r#"A,B,C
0,1,2
1,0,3
2,3,0
"#;
        with_temp(content, |p| {
            let args = NeighborNetArgs {
                input: p.to_string_lossy().into_owned(),
                output_prefix: "output".into(),
                nnls_params: NNLSParams::default(),
            };
            let nn = NeighbourNet::new("/tmp".to_string(), args);
            let (mat, labels, meta) = nn.load_distance_matrix(&nn.args.input).unwrap();

            assert_eq!(labels, vec!["A", "B", "C"]);
            assert!(meta.has_header);
            assert!(!meta.has_index);
            assert_eq!(mat.shape(), &[3, 3]);
        });
    }

    #[test]
    fn tsv_index_no_header() {
        let content = "A\t0\t1\t2\nB\t1\t0\t3\nC\t2\t3\t0\n";
        with_temp(content, |p| {
            let args = NeighborNetArgs {
                input: p.to_string_lossy().into_owned(),
                output_prefix: "output".into(),
                nnls_params: NNLSParams::default(),
            };
            let nn = NeighbourNet::new("/tmp".to_string(), args);
            let (mat, labels, meta) = nn.load_distance_matrix(&nn.args.input).unwrap();

            assert_eq!(labels, vec!["A", "B", "C"]);
            assert!(!meta.has_header);
            assert!(meta.has_index);
            assert_eq!(mat.shape(), &[3, 3]);
            assert!(approx(mat[[0, 2]], 2.0, 1e-12));
        });
    }

    #[test]
    fn space_delimited_no_header_no_index() {
        let content = "0 1 2\n1 0 3\n2 3 0\n";
        with_temp(content, |p| {
            let args = NeighborNetArgs {
                input: p.to_string_lossy().into_owned(),
                output_prefix: "output".into(),
                nnls_params: NNLSParams::default(),
            };
            let nn = NeighbourNet::new("/tmp".to_string(), args);
            let (mat, labels, meta) = nn.load_distance_matrix(&nn.args.input).unwrap();

            assert_eq!(labels, vec!["t1", "t2", "t3"]); // synthesized
            assert!(!meta.has_header);
            assert!(!meta.has_index);
            assert_eq!(mat.shape(), &[3, 3]);
        });
    }

    #[test]
    fn semicolon_header_no_index() {
        let content = "A;B;C\n0;1;2\n1;0;3\n2;3;0\n";
        with_temp(content, |p| {
            let args = NeighborNetArgs {
                input: p.to_string_lossy().into_owned(),
                output_prefix: "output".into(),
                nnls_params: NNLSParams::default(),
            };
            let nn = NeighbourNet::new("/tmp".to_string(), args);
            let (mat, labels, meta) = nn.load_distance_matrix(&nn.args.input).unwrap();

            assert_eq!(labels, vec!["A", "B", "C"]);
            assert!(meta.has_header);
            assert!(!meta.has_index);
            assert_eq!(mat.shape(), &[3, 3]);
        });
    }

    #[test]
    fn asymmetry_is_averaged_and_counted() {
        // m[0,1]=1.0, m[1,0]=1.1 -> expect 1.05 and 1 fix counted
        let content = r#"A,B,C
0,1.0,2.0
1.1,0,3.0
2.0,3.0,0
"#;
        with_temp(content, |p| {
            let args = NeighborNetArgs {
                input: p.to_string_lossy().into_owned(),
                output_prefix: "output".into(),
                nnls_params: NNLSParams::default(),
            };
            let nn = NeighbourNet::new("/tmp".to_string(), args);
            let (mat, labels, meta) = nn.load_distance_matrix(&nn.args.input).unwrap();

            assert!(meta.symmetry_pairs_fixed >= 1);
            assert!(approx(mat[[0, 1]], 1.05, 1e-12));
            assert!(approx(mat[[1, 0]], 1.05, 1e-12));
            assert_eq!(labels, vec!["A", "B", "C"]);
        });
    }

    #[test]
    fn non_square_rejected() {
        // 3x2 data after parsing ⇒ error
        let content = "A,B\n0,1\n1,0\n2,3\n";
        with_temp(content, |p| {
            let args = NeighborNetArgs {
                input: p.to_string_lossy().into_owned(),
                output_prefix: "output".into(),
                nnls_params: NNLSParams::default(),
            };
            let nn = NeighbourNet::new("/tmp".to_string(), args);
            let err = nn.load_distance_matrix(&nn.args.input);
            assert!(err.is_err());
        });
    }
}
