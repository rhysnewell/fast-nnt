use std::collections::{BTreeMap, HashMap};
use std::fmt::{self, Write as FmtWrite};

use petgraph::prelude::NodeIndex;

use crate::algorithms::equal_angle::Pt;
use crate::nexus::*;
use crate::phylo::phylo_graph::PhyloGraph;
use crate::phylo::phylo_splits_graph::PhyloSplitsGraph;

/// Write a SplitsTree-style NETWORK block:
/// BEGIN Network;
/// DIMENSIONS ntax=.. nvertices=.. nedges=..;
/// DRAW to_scale;
/// TRANSLATE
///   <leaf_id> 'TaxonLabel',
///   ...
/// ;
/// VERTICES
///   <id> <x> <y> s=n,
///   ...
/// ;
/// EDGES
///   <eid> <u_id> <v_id> s=<splitid> w=<weight>,
///   ...
/// ;
/// END; [Network]
pub fn write_network_block_splits<W: FmtWrite>(
    mut w: W,
    g: &PhyloSplitsGraph,
    coords: &HashMap<NodeIndex, Pt>,
    taxa_labels_1based: &[String],
) -> fmt::Result {
    let base: &PhyloGraph = &g.base;

    let ntax = taxa_labels_1based.len();
    let nverts = base.graph.node_count();
    let nedges = base.graph.edge_count();

    // Stable 1..N ids for nodes in iteration order
    let mut node_id = BTreeMap::<NodeIndex, usize>::new();
    {
        let mut next = 1usize;
        for v in base.graph.node_indices() {
            node_id.insert(v, next);
            next += 1;
        }
    }

    writeln!(w, "BEGIN Network;")?;
    writeln!(
        w,
        "DIMENSIONS ntax={} nvertices={} nedges={};",
        ntax, nverts, nedges
    )?;
    writeln!(w, "DRAW to_scale;")?;

    // TRANSLATE: leaf labels only (degree==1). Prefer taxon label if node maps
    // to exactly one taxon id; else fall back to NodeData.label if present.
    writeln!(w, "TRANSLATE")?;
    {
        let mut any = false;
        for v in base.graph.node_indices() {
            if node_degree(base, v) == 1 {
                if let Some(lbl) = leaf_label(base, v, taxa_labels_1based) {
                    if any {
                        writeln!(w, ",")?;
                    }
                    write!(w, "{} '{}'", node_id[&v], escape_label(&lbl))?;
                    any = true;
                }
            }
        }
        writeln!(w, ",")?;
    }
    writeln!(w, ";")?;

    // VERTICES: "<id> <x> <y> s=n,"
    writeln!(w, "VERTICES")?;
    for v in base.graph.node_indices() {
        let id = node_id[&v];
        let pt = coords.get(&v).copied().unwrap_or(Pt(0.0, 0.0));
        writeln!(w, "{} {} {} s=n,", id, fmt_f(pt.0), fmt_f(pt.1))?;
    }
    writeln!(w, ";")?;

    // TODO: Actual label layout calcs
    // write_vlabels_section(
    //     &mut w,
    //     base,
    //     &node_id,
    //     taxa_labels_1based,
    //     12,    // x
    //     0,     // y
    //     Some("Dialog-PLAIN-6"),
    // )?;

    // EDGES: "<eid> <u_id> <v_id> s=<splitid> w=<weight>,"
    writeln!(w, "EDGES")?;
    {
        let mut eid = 1usize;
        for e in base.graph.edge_indices() {
            let (u, v) = base.graph.edge_endpoints(e).expect("valid endpoints");
            let su = node_id[&u];
            let sv = node_id[&v];
            let sid = g.get_split(e);
            let wgt = base.weight(e);
            writeln!(w, "{} {} {} s={} w={},", eid, su, sv, sid, fmt_f(wgt))?;
            eid += 1;
        }
    }
    writeln!(w, "END; [Network]")?;
    Ok(())
}

/// Write a VLABELS section for nodes that have taxa (i.e., the leaves).
/// - `x_off`/`y_off` are the same fixed offsets for every label.
/// - `font` is optional; when `Some`, we append `f='<font>'`.
pub fn write_vlabels_section<W: FmtWrite>(
    mut w: W,
    base: &PhyloGraph,
    node_id: &BTreeMap<NodeIndex, usize>,
    taxa_labels_1based: &[String],
    x_off: i32,
    y_off: i32,
    mut font: Option<&str>,
) -> fmt::Result {
    writeln!(w, "VLABELS")?;

    // We print labels for nodes that correspond to exactly one taxon
    // (these are the leaves in our construction).
    for v in base.graph.node_indices() {
        // Look up the taxon-mapped label if this node has exactly one taxon
        if let Some(n2t) = base.node2taxa() {
            if let Some(list) = n2t.get(&v) {
                if list.len() == 1 {
                    let t = list[0];
                    if t >= 1 && t <= taxa_labels_1based.len() {
                        let label = &taxa_labels_1based[t - 1];
                        let id = node_id[&v];
                        write!(
                            w,
                            "{} '{}' x={} y={}",
                            id,
                            escape_label(label),
                            x_off,
                            y_off
                        )?;
                        if let Some(f) = font {
                            write!(w, " f='{}'", f)?;
                            font = None;
                        }
                        writeln!(w, ",")?;
                    }
                }
            }
        }
    }
    writeln!(w, ";")?;
    Ok(())
}

/* ---------- helpers ---------- */

pub fn node_degree(g: &PhyloGraph, v: NodeIndex) -> usize {
    g.graph.neighbors(v).count()
}

/// Prefer taxon label if node maps to exactly one taxon; else use node label
pub fn leaf_label(base: &PhyloGraph, v: NodeIndex, taxa_labels_1based: &[String]) -> Option<String> {
    // One mapped taxon?
    if let Some(n2t) = base.node2taxa() {
        if let Some(list) = n2t.get(&v) {
            if list.len() == 1 {
                let t = list[0];
                if t >= 1 && t <= taxa_labels_1based.len() {
                    return Some(taxa_labels_1based[t - 1].clone());
                }
            }
        }
    }
    // Else use node label (if any)
    base.node_label(v).map(|s| s.to_string())
}
