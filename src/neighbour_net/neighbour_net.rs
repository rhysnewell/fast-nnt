use anyhow::{Context, Result, anyhow};
use fixedbitset::FixedBitSet;
use log::{debug, info, warn};
use ndarray::Array2;
use rayon::prelude::*;
use serde::Serialize;
use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use crate::algorithms::equal_angle::{EqualAngleOpts, equal_angle_apply};
use crate::cli::NeighbourNetArgs;
use crate::data::splits_blocks::SplitsBlock;
use crate::nexus::nexus::Nexus;
use crate::nexus::nexus_writer::{NexusProperties, write_nexus_all_to_path};
use crate::ordering::OrderingMethod;
use crate::ordering::ordering_huson2023::compute_order_huson_2023;
use crate::ordering::ordering_splitstree4::compute_order_splits_tree4;
use crate::phylo::phylo_splits_graph::PhyloSplitsGraph;
use crate::splits::asplit::ASplit;
use crate::utils::compute_least_squares_fit;
use crate::weights::InferenceMethod;
use crate::weights::active_set_weights::{NNLSParams, compute_asplits};
use crate::weights::splitstree4_weights::{
    CGSolveStats, DEFAULT_CUTOFF, compute_splits as compute_cg_splits,
};

struct PipelineResult {
    params: NNLSParams,
    splits_block: SplitsBlock,
    graph: PhyloSplitsGraph,
    fit: f32,
    st4_stats: Option<CGSolveStats>,
    cycle_sec: f64,
    nnls_sec: f64,
    fit_sec: f64,
    splits_sec: f64,
    graph_sec: f64,
}

pub struct NeighbourNet {
    out_dir: String,
    args: NeighbourNetArgs,
    distance_matrix: Array2<f64>,
    labels: Vec<String>,
    parse_meta: ParseMeta,
}

impl NeighbourNet {
    pub fn new(out_dir: String, args: NeighbourNetArgs) -> Result<Self> {
        let (distance_matrix, labels, parse_meta) =
            Self::load_distance_matrix(&args.input).context("loading distance matrix")?;
        Ok(NeighbourNet {
            out_dir,
            args,
            distance_matrix,
            labels,
            parse_meta,
        })
    }

    pub fn from_distance_matrix(
        dist: Array2<f64>,
        labels: Vec<String>,
        args: NeighbourNetArgs,
    ) -> Result<Self> {
        let n = dist.nrows();
        if n != dist.ncols() {
            anyhow::bail!("Distance matrix must be square ({}x{})", n, dist.ncols());
        }
        Ok(NeighbourNet {
            out_dir: "output".to_string(),
            args,
            distance_matrix: dist,
            labels,
            parse_meta: ParseMeta {
                delimiter: ',',
                has_header: false,
                has_index: false,
                symmetry_pairs_fixed: 0,
                load_sec: 0.0,
            },
        })
    }

    pub fn run(&self) -> Result<()> {
        let t0 = Instant::now();
        let pipeline = self.compute_pipeline()?;

        // Outputs
        let t_out = Instant::now();
        self.output_results(
            pipeline
                .splits_block
                .cycle()
                .ok_or_else(|| anyhow!("cycle should be set after pipeline"))?,
            &self.labels,
            &pipeline.splits_block,
            &self.distance_matrix,
            &pipeline.graph,
            pipeline.fit,
        )
        .context("writing outputs")?;
        let out_sec = t_out.elapsed().as_secs_f64();
        info!("Wrote outputs in {:.3}s", out_sec);

        // Build + write run log
        let run_log_path = Path::new(&self.out_dir).join("run_log.json");
        let stats = self.build_run_stats(
            &self.parse_meta,
            &self.labels,
            &pipeline.splits_block,
            &pipeline.params,
            pipeline.fit,
            pipeline.st4_stats,
            RunTimings {
                load_sec: self.parse_meta.load_sec,
                cycle_sec: pipeline.cycle_sec,
                nnls_sec: pipeline.nnls_sec,
                fit_sec: pipeline.fit_sec,
                splits_sec: pipeline.splits_sec,
                graph_sec: pipeline.graph_sec,
                output_sec: out_sec,
                total_sec: t0.elapsed().as_secs_f64(),
            },
        );
        fs::write(&run_log_path, serde_json::to_string_pretty(&stats)?)?;
        info!("Run log written: {}", run_log_path.display());

        info!("Done in {:.3}s total.", t0.elapsed().as_secs_f64());
        Ok(())
    }

    /// Run the NeighbourNet pipeline and return a Nexus result.
    /// Consumes self to move labels and distance matrix into the Nexus without cloning.
    pub fn into_nexus(self) -> Result<Nexus> {
        let pipeline = self.compute_pipeline()?;
        Ok(Nexus::new(
            self.labels,
            self.distance_matrix,
            pipeline.splits_block,
            pipeline.graph,
        ))
    }

    /// Convenience alias that borrows self (clones labels and distance matrix).
    pub fn generate_nexus(&self) -> Result<Nexus> {
        let pipeline = self.compute_pipeline()?;
        Ok(Nexus::new(
            self.labels.clone(),
            self.distance_matrix.clone(),
            pipeline.splits_block,
            pipeline.graph,
        ))
    }

    fn compute_pipeline(&self) -> Result<PipelineResult> {
        let n = self.distance_matrix.nrows();
        info!("Loaded distance matrix: {}x{}", n, n);

        // 1) Compute NeighborNet cycle (1-based with leading 0)
        let t_cycle = Instant::now();
        let cycle = self.get_ordering().context("computing cycle")?;
        let cycle_sec = t_cycle.elapsed().as_secs_f64();
        info!("Computed cycle in {:.3}s", cycle_sec);
        debug!("Cycle (1-based): {:?}", &cycle[1..]);

        // 2) Split weight inference
        let t_nnls = Instant::now();
        let (params, splits, st4_stats) = self.compute_asplits(&cycle).context("ASplits solved")?;
        let nnls_sec = t_nnls.elapsed().as_secs_f64();
        info!(
            "Estimated {} splits (cutoff = {}) in {:.3}s",
            splits.len(),
            self.effective_cutoff(),
            nnls_sec
        );

        // 3) Least-squares fit
        let t_fit = Instant::now();
        let fit = compute_least_squares_fit(&self.distance_matrix, &splits);
        let fit_sec = t_fit.elapsed().as_secs_f64();
        info!(
            "Least-squares fit: {:.4} % (computed in {:.3}s)",
            fit, fit_sec
        );

        // 4) Create splits block
        let t_spl = Instant::now();
        let splits_block = self.create_splits_block(splits, fit, cycle)?;
        let splits_sec = t_spl.elapsed().as_secs_f64();
        info!(
            "Created splits block with {} splits in {:.3}s",
            splits_block.nsplits(),
            splits_sec
        );

        // 5) Create phylogenetic splits graph
        let t_graph = Instant::now();
        let graph = self.create_graph(&splits_block)?;
        let graph_sec = t_graph.elapsed().as_secs_f64();
        info!("Created phylogenetic splits graph in {:.3}s", graph_sec);

        Ok(PipelineResult {
            params,
            splits_block,
            graph,
            fit,
            st4_stats,
            cycle_sec,
            nnls_sec,
            fit_sec,
            splits_sec,
            graph_sec,
        })
    }

    pub fn get_ordering(&self) -> Result<Vec<usize>> {
        let mut cycle = match self.args.ordering {
            OrderingMethod::ClosestPair => {
                compute_order_huson_2023(&self.distance_matrix).context("computing cycle")?
            }
            OrderingMethod::Multiway => {
                compute_order_splits_tree4(&self.distance_matrix).context("computing cycle")?
            }
        };
        if cycle.first().copied() != Some(0) {
            cycle = std::iter::once(0usize)
                .chain(cycle.into_iter().map(|i| i + 1))
                .collect();
        }
        Ok(cycle)
    }

    pub fn compute_asplits(
        &self,
        cycle: &[usize],
    ) -> Result<(NNLSParams, Vec<ASplit>, Option<CGSolveStats>)> {
        match self.args.inference {
            InferenceMethod::ActiveSet => {
                let mut params = self.args.nnls_params.clone();
                let splits = compute_asplits(&cycle, &self.distance_matrix, &mut params, None)
                    .context("ASplits solved")?;
                Ok((params, splits, None))
            }
            InferenceMethod::CG => {
                let (splits, solve_stats) =
                    compute_cg_splits(cycle, &self.distance_matrix).context("CG weights solved")?;
                let mut params = self.args.nnls_params.clone();
                params.cutoff = DEFAULT_CUTOFF;
                Ok((params, splits, Some(solve_stats)))
            }
        }
    }

    fn effective_cutoff(&self) -> f64 {
        match self.args.inference {
            InferenceMethod::ActiveSet => self.args.nnls_params.cutoff,
            InferenceMethod::CG => DEFAULT_CUTOFF,
        }
    }

    pub fn create_splits_block(
        &self,
        splits: Vec<ASplit>,
        fit: f32,
        cycle: Vec<usize>,
    ) -> Result<SplitsBlock> {
        let mut splits_blocks = SplitsBlock::new();
        splits_blocks.set_splits(splits);
        splits_blocks.set_fit(fit);
        splits_blocks.set_cycle(cycle, true)?;
        Ok(splits_blocks)
    }

    pub fn create_graph(&self, splits_blocks: &SplitsBlock) -> Result<PhyloSplitsGraph> {
        let mut graph = PhyloSplitsGraph::new();
        let mut used_splits = FixedBitSet::with_capacity(splits_blocks.nsplits() + 1);
        equal_angle_apply(
            EqualAngleOpts::default(),
            &self.labels,
            &splits_blocks,
            &mut graph,
            None,
            &mut used_splits,
        )?;
        graph.create_node_ids();
        Ok(graph)
    }

    /* ───────────── I/O ───────────── */

    /// Parse CSV/TSV/; / | / space; header/index row optional.
    /// Returns (n×n distances, labels[1..=n], parse meta incl. load time & symmetry fixes).
    fn load_distance_matrix(path: &str) -> Result<(Array2<f64>, Vec<String>, ParseMeta)> {
        let t_load = Instant::now();
        let text = fs::read_to_string(path).with_context(|| format!("reading '{}'", path))?;

        let (delim, rows) = Self::parse_delimited_text(&text)?;

        let (has_header, has_index, misaligned_header) = Self::detect_header_index(&rows)?;
        info!("Header: {}, Index column: {}", has_header, has_index);

        let (labels, start_row, start_col) =
            Self::extract_labels(&rows, has_header, has_index, misaligned_header);

        let n = rows.len() - start_row;
        let m = rows[start_row].len() - start_col;
        if n != m {
            return Err(anyhow!(
                "parsed table is not square: rows={}, cols={}",
                n,
                m
            ));
        }

        let mut mat = Self::parse_numeric_region(&rows, start_row, start_col, n)?;

        let symmetry_pairs_fixed = Self::symmetrize(&mut mat, n);

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

    /// Detect delimiter and parse text into rows of string tokens.
    fn parse_delimited_text(text: &str) -> Result<(char, Vec<Vec<String>>)> {
        let first_line = text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .find(|l| !l.trim_start().starts_with('#'))
            .ok_or_else(|| anyhow!("no data lines found"))?;
        let delim = detect_delim(first_line);
        info!("Detected delimiter: {:?}", delim);

        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(false)
            .flexible(true)
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
        Ok((delim, rows))
    }

    /// Detect whether the table has a header row and/or index column.
    fn detect_header_index(rows: &[Vec<String>]) -> Result<(bool, bool, bool)> {
        let misaligned_header = rows.len() >= 2 && rows[0].len() + 1 == rows[1].len();
        let (has_header, has_index) = if misaligned_header {
            (true, true)
        } else {
            sniff_header_index(rows)?
        };
        Ok((has_header, has_index, misaligned_header))
    }

    /// Extract labels and determine the start offsets for the numeric region.
    fn extract_labels(
        rows: &[Vec<String>],
        has_header: bool,
        has_index: bool,
        misaligned_header: bool,
    ) -> (Vec<String>, usize, usize) {
        if misaligned_header {
            let labels = rows[0].iter().map(|s| s.to_string()).collect();
            (labels, 1, 1)
        } else if has_header && has_index {
            let labels = rows[0][1..].iter().map(|s| s.to_string()).collect();
            (labels, 1, 1)
        } else if has_header {
            let labels = rows[0].iter().map(|s| s.to_string()).collect();
            (labels, 1, 0)
        } else if has_index {
            let labels = rows.iter().map(|r| r[0].to_string()).collect();
            (labels, 0, 1)
        } else {
            let n = rows.len();
            let labels = (1..=n).map(|i| format!("t{}", i)).collect();
            (labels, 0, 0)
        }
    }

    /// Parse the numeric sub-region of the table into an n×n matrix.
    fn parse_numeric_region(
        rows: &[Vec<String>],
        start_row: usize,
        start_col: usize,
        n: usize,
    ) -> Result<Array2<f64>> {
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
        Ok(mat)
    }

    /// Symmetrize the matrix by averaging off-diagonal pairs. Returns the number of pairs fixed.
    fn symmetrize(mat: &mut Array2<f64>, n: usize) -> usize {
        let mut fixed = 0usize;
        for i in 0..n {
            mat[[i, i]] = 0.0;
            for j in (i + 1)..n {
                let a = mat[[i, j]];
                let b = mat[[j, i]];
                if (a - b).abs() > 1e-12 {
                    let avg = 0.5 * (a + b);
                    mat[[i, j]] = avg;
                    mat[[j, i]] = avg;
                    fixed += 1;
                }
            }
        }
        if fixed > 0 {
            warn!(
                "Distance matrix not perfectly symmetric; averaged {} off-diagonal pairs",
                fixed
            );
        }
        fixed
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
            "cutoff": self.effective_cutoff(),
            "input": self.args.input,
            "inference": self.args.inference.as_str(),
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
        let nexus_path = self.resolve_nexus_path();
        if let Some(parent) = nexus_path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)
                    .with_context(|| format!("creating {}", parent.display()))?;
            }
        }
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

        Ok(())
    }

    /// Resolve output `.nex` path from `-o/--output-prefix`.
    ///
    /// - If `output_prefix` is a plain name (no directory components), write into `out_dir`.
    /// - If it contains a path (relative or absolute), treat it as explicit destination stem/path.
    fn resolve_nexus_path(&self) -> PathBuf {
        let prefix_path = Path::new(&self.args.output_prefix);
        let has_explicit_path = prefix_path.is_absolute()
            || prefix_path
                .parent()
                .is_some_and(|p| !p.as_os_str().is_empty());

        let mut nexus_path = if has_explicit_path {
            prefix_path.to_path_buf()
        } else {
            Path::new(&self.out_dir).join(prefix_path)
        };
        if nexus_path.extension().is_none() {
            nexus_path.set_extension("nex");
        }
        nexus_path
    }

    /* ───────────── run_log helpers ───────────── */

    fn build_run_stats(
        &self,
        parse: &ParseMeta,
        labels: &[String],
        splits: &SplitsBlock,
        nnls: &NNLSParams,
        fit_percent: f32,
        cg_solve_stats: Option<CGSolveStats>,
        timings: RunTimings,
    ) -> RunLog {
        let n = labels.len();
        let npairs = n * (n - 1) / 2;

        // split stats
        let (num_trivial, num_nontrivial, sum_weights) = splits
            .splits()
            .par_bridge()
            .fold(
                || (0usize, 0usize, 0.0f64),
                |mut acc, s| {
                    if s.is_trivial() {
                        acc.0 += 1;
                    } else {
                        acc.1 += 1;
                    }
                    acc.2 += s.weight;
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
                cutoff: self.effective_cutoff(),
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
            cg_solve_stats,
            timings,
            system: sys,
            inference: self.args.inference.as_str().to_string(),
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
    inference: String,
    fit_percent: f32,
    cg_solve_stats: Option<CGSolveStats>,
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
    use sysinfo::System;
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
    // Safe fallback: peak RSS is not available without platform-specific syscalls.
    None
}

#[cfg(any(target_os = "freebsd", target_os = "openbsd", target_os = "netbsd"))]
fn peak_rss_bytes() -> Option<u64> {
    // Safe fallback: peak RSS is not available without platform-specific syscalls.
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

/* ───────────── tests ───────────── */

#[cfg(test)]
mod read_matrix_tests {
    use crate::ordering::OrderingMethod;
    use crate::weights::InferenceMethod;

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
            let args = NeighbourNetArgs {
                input: p.to_string_lossy().into_owned(),
                output_prefix: "output".into(),
                ordering: OrderingMethod::ClosestPair,
                inference: InferenceMethod::ActiveSet,
                nnls_params: NNLSParams::default(),
            };
            let (mat, labels, meta) = NeighbourNet::load_distance_matrix(&args.input).unwrap();

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
            let args = NeighbourNetArgs {
                input: p.to_string_lossy().into_owned(),
                output_prefix: "output".into(),
                ordering: OrderingMethod::ClosestPair,
                inference: InferenceMethod::ActiveSet,
                nnls_params: NNLSParams::default(),
            };
            let (mat, labels, meta) = NeighbourNet::load_distance_matrix(&args.input).unwrap();

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
            let args = NeighbourNetArgs {
                input: p.to_string_lossy().into_owned(),
                output_prefix: "output".into(),
                ordering: OrderingMethod::ClosestPair,
                inference: InferenceMethod::ActiveSet,
                nnls_params: NNLSParams::default(),
            };
            let (mat, labels, meta) = NeighbourNet::load_distance_matrix(&args.input).unwrap();

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
            let args = NeighbourNetArgs {
                input: p.to_string_lossy().into_owned(),
                output_prefix: "output".into(),
                ordering: OrderingMethod::ClosestPair,
                inference: InferenceMethod::ActiveSet,
                nnls_params: NNLSParams::default(),
            };
            let (mat, labels, meta) = NeighbourNet::load_distance_matrix(&args.input).unwrap();

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
            let args = NeighbourNetArgs {
                input: p.to_string_lossy().into_owned(),
                output_prefix: "output".into(),
                ordering: OrderingMethod::ClosestPair,
                inference: InferenceMethod::ActiveSet,
                nnls_params: NNLSParams::default(),
            };
            let (mat, labels, meta) = NeighbourNet::load_distance_matrix(&args.input).unwrap();

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
            let args = NeighbourNetArgs {
                input: p.to_string_lossy().into_owned(),
                output_prefix: "output".into(),
                ordering: OrderingMethod::ClosestPair,
                inference: InferenceMethod::ActiveSet,
                nnls_params: NNLSParams::default(),
            };
            let (mat, labels, meta) = NeighbourNet::load_distance_matrix(&args.input).unwrap();

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
            let args = NeighbourNetArgs {
                input: p.to_string_lossy().into_owned(),
                output_prefix: "output".into(),
                ordering: OrderingMethod::ClosestPair,
                inference: InferenceMethod::ActiveSet,
                nnls_params: NNLSParams::default(),
            };
            let err = NeighbourNet::load_distance_matrix(&args.input);
            assert!(err.is_err());
        });
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::cli::NeighbourNetArgs;
    use crate::ordering::OrderingMethod;
    use crate::weights::InferenceMethod;
    use crate::weights::active_set_weights::NNLSParams;
    use ndarray::{Array2, arr2};
    use std::collections::HashSet;

    fn default_args() -> NeighbourNetArgs {
        NeighbourNetArgs {
            input: String::new(),
            output_prefix: "output".into(),
            ordering: OrderingMethod::Multiway,
            inference: InferenceMethod::ActiveSet,
            nnls_params: NNLSParams::default(),
        }
    }

    /// Helper: run full pipeline via generate_nexus and return the Nexus result.
    fn run_pipeline(dist: Array2<f64>, labels: Vec<String>) -> Nexus {
        let args = default_args();
        let nn = NeighbourNet::from_distance_matrix(dist, labels, args)
            .expect("from_distance_matrix should succeed");
        nn.generate_nexus().expect("generate_nexus should succeed")
    }

    /// Helper: extract cycle from splits block (more reliable than graph.get_cycle).
    fn get_cycle_from_nexus(nexus: &Nexus) -> Vec<usize> {
        nexus
            .splits()
            .cycle()
            .expect("splits block should have a cycle")
            .to_vec()
    }

    /// Helper: validate that a cycle is a valid permutation of 1..=n with leading 0.
    fn assert_valid_cycle(cycle: &[usize], n: usize) {
        assert_eq!(
            cycle.len(),
            n + 1,
            "cycle should have n+1 elements (leading 0), got {}",
            cycle.len()
        );
        assert_eq!(cycle[0], 0, "cycle[0] should be 0 sentinel");
        let mut taxa: Vec<usize> = cycle[1..].to_vec();
        taxa.sort();
        assert_eq!(
            taxa,
            (1..=n).collect::<Vec<_>>(),
            "cycle should be a permutation of 1..={}",
            n
        );
    }

    /// End-to-end: 5-taxon distance matrix (the small_5_1 test data).
    /// Verifies cycle is a permutation, splits have non-negative weights,
    /// and graph has expected structure.
    #[test]
    fn e2e_5_taxa_multiway() {
        let dist = arr2(&[
            [0.0, 5.0, 9.0, 9.0, 8.0],
            [5.0, 0.0, 10.0, 10.0, 9.0],
            [9.0, 10.0, 0.0, 8.0, 7.0],
            [9.0, 10.0, 8.0, 0.0, 3.0],
            [8.0, 9.0, 7.0, 3.0, 0.0],
        ]);
        let labels: Vec<String> = vec!["a", "b", "c", "d", "e"]
            .into_iter()
            .map(String::from)
            .collect();

        let nexus = run_pipeline(dist.clone(), labels.clone());

        // Labels preserved
        assert_eq!(nexus.get_labels(), &labels);
        assert_eq!(nexus.num_labels(), 5);

        // Cycle is a valid permutation of 1..=5 (with leading 0)
        let cycle = get_cycle_from_nexus(&nexus);
        assert_valid_cycle(&cycle, 5);

        // Splits: all weights >= 0, at least n trivial splits
        let splits = nexus.splits();
        assert!(
            splits.nsplits() >= 5,
            "should have at least n trivial splits"
        );
        for split in splits.get_splits() {
            assert!(
                split.weight >= 0.0,
                "split weight should be non-negative, got {}",
                split.weight
            );
        }

        // Graph should have nodes and edges
        let graph = nexus.graph();
        assert!(
            graph.count_nodes() >= 5,
            "graph should have at least n nodes (one per taxon)"
        );
        assert!(
            graph.count_edges() >= 5,
            "graph should have at least n edges"
        );

        // Distance matrix preserved
        assert_eq!(nexus.distance_matrix(), &dist);
    }

    /// End-to-end with closest-pair ordering method.
    #[test]
    fn e2e_5_taxa_closest_pair() {
        let dist = arr2(&[
            [0.0, 5.0, 9.0, 9.0, 8.0],
            [5.0, 0.0, 10.0, 10.0, 9.0],
            [9.0, 10.0, 0.0, 8.0, 7.0],
            [9.0, 10.0, 8.0, 0.0, 3.0],
            [8.0, 9.0, 7.0, 3.0, 0.0],
        ]);
        let labels: Vec<String> = vec!["a", "b", "c", "d", "e"]
            .into_iter()
            .map(String::from)
            .collect();

        let args = NeighbourNetArgs {
            ordering: OrderingMethod::ClosestPair,
            ..default_args()
        };
        let nn = NeighbourNet::from_distance_matrix(dist, labels.clone(), args)
            .expect("from_distance_matrix should succeed");
        let nexus = nn.generate_nexus().expect("generate_nexus should succeed");

        let cycle = get_cycle_from_nexus(&nexus);
        assert_valid_cycle(&cycle, 5);

        assert!(nexus.splits().nsplits() >= 5);
        assert!(nexus.graph().count_nodes() >= 5);
    }

    /// End-to-end: 10-taxon distance matrix.
    /// Tests a slightly larger case to verify scalability of the pipeline.
    #[test]
    fn e2e_10_taxa() {
        let dist = arr2(&[
            [0.0, 5.0, 12.0, 7.0, 3.0, 9.0, 11.0, 6.0, 4.0, 10.0],
            [5.0, 0.0, 8.0, 2.0, 14.0, 5.0, 13.0, 7.0, 12.0, 1.0],
            [12.0, 8.0, 0.0, 4.0, 9.0, 3.0, 8.0, 2.0, 5.0, 6.0],
            [7.0, 2.0, 4.0, 0.0, 11.0, 7.0, 10.0, 4.0, 6.0, 9.0],
            [3.0, 14.0, 9.0, 11.0, 0.0, 8.0, 1.0, 13.0, 2.0, 7.0],
            [9.0, 5.0, 3.0, 7.0, 8.0, 0.0, 12.0, 5.0, 3.0, 4.0],
            [11.0, 13.0, 8.0, 10.0, 1.0, 12.0, 0.0, 6.0, 2.0, 8.0],
            [6.0, 7.0, 2.0, 4.0, 13.0, 5.0, 6.0, 0.0, 9.0, 7.0],
            [4.0, 12.0, 5.0, 6.0, 2.0, 3.0, 2.0, 9.0, 0.0, 5.0],
            [10.0, 1.0, 6.0, 9.0, 7.0, 4.0, 8.0, 7.0, 5.0, 0.0],
        ]);
        let labels: Vec<String> = (1..=10).map(|i| format!("t{}", i)).collect();

        let nexus = run_pipeline(dist, labels.clone());

        // Cycle
        let cycle = get_cycle_from_nexus(&nexus);
        assert_valid_cycle(&cycle, 10);

        // Splits
        let splits = nexus.splits();
        assert!(
            splits.nsplits() >= 10,
            "should have at least 10 trivial splits"
        );
        for split in splits.get_splits() {
            assert!(split.weight >= 0.0);
        }

        // Graph
        let graph = nexus.graph();
        assert!(graph.count_nodes() >= 10);
        assert!(graph.count_edges() >= 10);

        // Split records should be consistent
        let records = nexus.get_splits_records();
        assert_eq!(records.len(), splits.nsplits());
        for (_, weight, _, a_side, b_side) in &records {
            assert!(*weight >= 0.0);
            // Both sides should be non-empty
            assert!(!a_side.is_empty(), "A-side of split should not be empty");
            assert!(!b_side.is_empty(), "B-side of split should not be empty");
            // Union should cover all taxa
            let mut all: HashSet<usize> = a_side.iter().copied().collect();
            all.extend(b_side.iter());
            // The splits use 1-based indices; all taxa 1..=10 should appear
            for t in 1..=10 {
                assert!(
                    all.contains(&t),
                    "taxon {} missing from split bipartition",
                    t
                );
            }
        }
    }

    /// End-to-end: via the top-level run_fast_nnt_from_memory entry point
    /// (the same function Python/R bindings call).
    #[test]
    fn e2e_via_run_fast_nnt_from_memory() {
        let dist = arr2(&[
            [0.0, 5.0, 9.0, 9.0, 8.0],
            [5.0, 0.0, 10.0, 10.0, 9.0],
            [9.0, 10.0, 0.0, 8.0, 7.0],
            [9.0, 10.0, 8.0, 0.0, 3.0],
            [8.0, 9.0, 7.0, 3.0, 0.0],
        ]);
        let labels: Vec<String> = vec!["a", "b", "c", "d", "e"]
            .into_iter()
            .map(String::from)
            .collect();
        let args = default_args();

        let nexus = crate::run_fast_nnt_from_memory(dist, labels.clone(), args)
            .expect("run_fast_nnt_from_memory should succeed");

        assert_eq!(nexus.num_labels(), 5);
        assert!(nexus.num_splits() >= 5);
        let cycle = get_cycle_from_nexus(&nexus);
        assert_valid_cycle(&cycle, 5);
    }

    /// End-to-end: non-square matrix should be rejected.
    #[test]
    fn e2e_non_square_rejected() {
        let dist = Array2::<f64>::zeros((3, 4));
        let labels = vec!["a".into(), "b".into(), "c".into()];
        let result = NeighbourNet::from_distance_matrix(dist, labels, default_args());
        assert!(result.is_err());
    }

    /// End-to-end: 2-taxon (minimal) case produces valid output.
    #[test]
    fn e2e_2_taxa() {
        let dist = arr2(&[[0.0, 3.5], [3.5, 0.0]]);
        let labels = vec!["X".into(), "Y".into()];
        let nexus = run_pipeline(dist, labels);

        assert_eq!(nexus.num_labels(), 2);
        let cycle = get_cycle_from_nexus(&nexus);
        assert_valid_cycle(&cycle, 2);

        // Should have at least 1 split (the trivial one for each taxon)
        assert!(nexus.num_splits() >= 1);

        // All split weights non-negative
        for split in nexus.splits().get_splits() {
            assert!(split.weight >= 0.0);
        }
    }

    /// End-to-end: 3-taxon (minimal non-trivial) case.
    #[test]
    fn e2e_3_taxa() {
        let dist = arr2(&[[0.0, 4.0, 6.0], [4.0, 0.0, 5.0], [6.0, 5.0, 0.0]]);
        let labels = vec!["A".into(), "B".into(), "C".into()];
        let nexus = run_pipeline(dist, labels);

        assert_eq!(nexus.num_labels(), 3);
        let cycle = get_cycle_from_nexus(&nexus);
        assert_valid_cycle(&cycle, 3);

        // Should have exactly 3 trivial splits for a 3-taxon tree
        assert!(nexus.num_splits() >= 3);

        // Graph structure
        let graph = nexus.graph();
        assert!(graph.count_nodes() >= 3);
        assert!(graph.count_edges() >= 3);
    }

    /// End-to-end: all-zero distance matrix.
    /// All taxa are identical, so no non-trivial splits should have positive weight.
    #[test]
    fn e2e_all_zero_distances() {
        let n = 5;
        let dist = Array2::<f64>::zeros((n, n));
        let labels: Vec<String> = (1..=n).map(|i| format!("t{}", i)).collect();
        let nexus = run_pipeline(dist, labels);

        assert_eq!(nexus.num_labels(), n);

        // All split weights should be zero or very close to zero
        for split in nexus.splits().get_splits() {
            assert!(
                split.weight.abs() < 1e-8,
                "split weight should be ~0 for zero-distance matrix, got {}",
                split.weight
            );
        }
    }

    /// End-to-end: verify both ordering methods produce valid (but possibly different) results.
    #[test]
    fn e2e_ordering_methods_both_valid() {
        let dist = arr2(&[
            [0.0, 5.0, 9.0, 9.0, 8.0],
            [5.0, 0.0, 10.0, 10.0, 9.0],
            [9.0, 10.0, 0.0, 8.0, 7.0],
            [9.0, 10.0, 8.0, 0.0, 3.0],
            [8.0, 9.0, 7.0, 3.0, 0.0],
        ]);
        let labels: Vec<String> = vec!["a", "b", "c", "d", "e"]
            .into_iter()
            .map(String::from)
            .collect();

        for ordering in [OrderingMethod::Multiway, OrderingMethod::ClosestPair] {
            let ordering_name = format!("{:?}", ordering);
            let args = NeighbourNetArgs {
                ordering,
                ..default_args()
            };
            let nn = NeighbourNet::from_distance_matrix(dist.clone(), labels.clone(), args)
                .expect("from_distance_matrix should succeed");
            let nexus = nn
                .generate_nexus()
                .unwrap_or_else(|e| panic!("generate_nexus failed for {}: {}", ordering_name, e));

            let cycle = get_cycle_from_nexus(&nexus);
            assert_valid_cycle(&cycle, 5);
            assert!(
                nexus.num_splits() >= 5,
                "too few splits for {ordering_name}"
            );
        }
    }
}
