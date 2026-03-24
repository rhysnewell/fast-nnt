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
pub fn leaf_label(
    base: &PhyloGraph,
    v: NodeIndex,
    taxa_labels_1based: &[String],
) -> Option<String> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::equal_angle::Pt;
    use crate::phylo::phylo_graph::PhyloGraph;
    use crate::phylo::phylo_splits_graph::PhyloSplitsGraph;
    use std::collections::HashMap;

    /// Build a minimal PhyloSplitsGraph with 3 leaf nodes (taxa 1..3) connected
    /// through an internal node, forming a star topology:
    ///
    ///   leaf1 --e1-- internal --e2-- leaf2
    ///                   |
    ///                  e3
    ///                   |
    ///                 leaf3
    fn make_star_graph() -> (PhyloSplitsGraph, Vec<String>) {
        let mut g = PhyloSplitsGraph::new();

        let internal = g.base.new_node();
        let leaf1 = g.base.new_node_with_label("leaf1");
        let leaf2 = g.base.new_node_with_label("leaf2");
        let leaf3 = g.base.new_node_with_label("leaf3");

        // Map taxa (1-based) to leaf nodes
        g.base.add_taxon(leaf1, 1);
        g.base.add_taxon(leaf2, 2);
        g.base.add_taxon(leaf3, 3);

        let e1 = g.base.new_edge(internal, leaf1).unwrap();
        let e2 = g.base.new_edge(internal, leaf2).unwrap();
        let e3 = g.base.new_edge(internal, leaf3).unwrap();

        g.base.set_weight(e1, 1.0);
        g.base.set_weight(e2, 2.0);
        g.base.set_weight(e3, 0.5);

        g.set_split(e1, 1);
        g.set_split(e2, 2);
        g.set_split(e3, 3);

        let labels = vec!["Alpha".to_string(), "Beta".to_string(), "Gamma".to_string()];
        (g, labels)
    }

    // ---------- node_degree ----------

    #[test]
    fn node_degree_leaf_is_one() {
        let (g, _labels) = make_star_graph();
        // Leaf nodes should have degree 1
        for v in g.base.graph.node_indices() {
            if g.base.node_label(v).is_some() {
                // this is a leaf node (has a label)
                assert_eq!(node_degree(&g.base, v), 1, "Leaf node should have degree 1");
            }
        }
    }

    #[test]
    fn node_degree_internal_is_three() {
        let (g, _labels) = make_star_graph();
        // The internal node (no label) should have degree 3
        for v in g.base.graph.node_indices() {
            if g.base.node_label(v).is_none() {
                assert_eq!(
                    node_degree(&g.base, v),
                    3,
                    "Internal node in 3-leaf star should have degree 3"
                );
            }
        }
    }

    #[test]
    fn node_degree_isolated_node_is_zero() {
        let mut pg = PhyloGraph::new();
        let v = pg.new_node();
        assert_eq!(node_degree(&pg, v), 0, "Isolated node has degree 0");
    }

    // ---------- leaf_label ----------

    #[test]
    fn leaf_label_from_taxon_mapping() {
        let (g, labels) = make_star_graph();
        // Find a leaf with taxon mapping
        for v in g.base.graph.node_indices() {
            if let Some(n2t) = g.base.node2taxa() {
                if let Some(list) = n2t.get(&v) {
                    if list.len() == 1 {
                        let taxon = list[0];
                        let lbl = leaf_label(&g.base, v, &labels);
                        assert!(lbl.is_some(), "Leaf with taxon mapping should have a label");
                        assert_eq!(
                            lbl.unwrap(),
                            labels[taxon - 1],
                            "Label should come from taxa_labels_1based"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn leaf_label_falls_back_to_node_label() {
        let mut pg = PhyloGraph::new();
        let v = pg.new_node_with_label("fallback_name");
        let labels = vec!["A".to_string(), "B".to_string()];

        // No taxon mapping, so should fall back to node label
        let lbl = leaf_label(&pg, v, &labels);
        assert_eq!(lbl, Some("fallback_name".to_string()));
    }

    #[test]
    fn leaf_label_no_label_no_taxon_returns_none() {
        let mut pg = PhyloGraph::new();
        let v = pg.new_node(); // no label, no taxon mapping
        let labels = vec!["A".to_string()];

        let lbl = leaf_label(&pg, v, &labels);
        assert!(
            lbl.is_none(),
            "Node with no label and no taxon mapping returns None"
        );
    }

    #[test]
    fn leaf_label_taxon_out_of_range() {
        let mut pg = PhyloGraph::new();
        let v = pg.new_node_with_label("backup");
        pg.add_taxon(v, 99); // taxon 99 is out of range for a 2-element label list

        let labels = vec!["A".to_string(), "B".to_string()];
        let lbl = leaf_label(&pg, v, &labels);
        // Taxon 99 is out of range, should fall back to node label
        assert_eq!(lbl, Some("backup".to_string()));
    }

    // ---------- write_network_block_splits ----------

    #[test]
    fn network_block_has_expected_structure() {
        let (g, labels) = make_star_graph();

        // Build simple coordinates for each node
        let mut coords = HashMap::new();
        for (i, v) in g.base.graph.node_indices().enumerate() {
            let angle = (i as f64) * std::f64::consts::TAU / 4.0;
            coords.insert(v, Pt(angle.cos(), angle.sin()));
        }

        let mut buf = String::new();
        write_network_block_splits(&mut buf, &g, &coords, &labels).unwrap();

        assert!(buf.contains("BEGIN Network;"), "Must open Network block");
        assert!(
            buf.contains("DIMENSIONS ntax=3 nvertices=4 nedges=3;"),
            "Must have correct dimensions: got\n{}",
            buf
        );
        assert!(buf.contains("DRAW to_scale;"), "Must have DRAW line");
        assert!(buf.contains("TRANSLATE"), "Must have TRANSLATE section");
        assert!(buf.contains("VERTICES"), "Must have VERTICES section");
        assert!(buf.contains("EDGES"), "Must have EDGES section");
        assert!(buf.contains("END; [Network]"), "Must close Network block");
    }

    #[test]
    fn network_block_translate_lists_leaves() {
        let (g, labels) = make_star_graph();

        let mut coords = HashMap::new();
        for v in g.base.graph.node_indices() {
            coords.insert(v, Pt(0.0, 0.0));
        }

        let mut buf = String::new();
        write_network_block_splits(&mut buf, &g, &coords, &labels).unwrap();

        // All three leaf labels should appear in the TRANSLATE section
        assert!(buf.contains("'Alpha'"), "Must translate Alpha");
        assert!(buf.contains("'Beta'"), "Must translate Beta");
        assert!(buf.contains("'Gamma'"), "Must translate Gamma");
    }

    #[test]
    fn network_block_edges_have_split_and_weight() {
        let (g, labels) = make_star_graph();

        let mut coords = HashMap::new();
        for v in g.base.graph.node_indices() {
            coords.insert(v, Pt(0.0, 0.0));
        }

        let mut buf = String::new();
        write_network_block_splits(&mut buf, &g, &coords, &labels).unwrap();

        // Edges should contain s= and w= markers
        assert!(buf.contains("s=1"), "Must reference split id 1");
        assert!(buf.contains("s=2"), "Must reference split id 2");
        assert!(buf.contains("s=3"), "Must reference split id 3");
        assert!(buf.contains("w=1"), "Must have weight 1");
        assert!(buf.contains("w=2"), "Must have weight 2");
        assert!(buf.contains("w=0.5"), "Must have weight 0.5");
    }

    #[test]
    fn network_block_vertices_have_coordinates() {
        let (g, labels) = make_star_graph();

        let mut coords = HashMap::new();
        let nodes: Vec<_> = g.base.graph.node_indices().collect();
        coords.insert(nodes[0], Pt(1.5, 2.5));
        coords.insert(nodes[1], Pt(3.0, 4.0));
        coords.insert(nodes[2], Pt(-1.0, 0.0));
        coords.insert(nodes[3], Pt(0.0, -1.0));

        let mut buf = String::new();
        write_network_block_splits(&mut buf, &g, &coords, &labels).unwrap();

        // Check that specific coordinate values appear
        assert!(buf.contains("1.5"), "Must contain x-coord 1.5");
        assert!(buf.contains("2.5"), "Must contain y-coord 2.5");
        assert!(buf.contains("-1"), "Must contain negative coordinate");
    }

    #[test]
    fn network_block_missing_coords_default_to_zero() {
        let (g, labels) = make_star_graph();

        // Provide coords for only some nodes
        let coords = HashMap::new(); // empty - all nodes will use default Pt(0,0)

        let mut buf = String::new();
        write_network_block_splits(&mut buf, &g, &coords, &labels).unwrap();

        // Should still write valid output (with 0 coordinates)
        assert!(buf.contains("BEGIN Network;"));
        assert!(buf.contains("END; [Network]"));
        // All vertex lines should have "0 0 s=n,"
        let vertex_lines: Vec<&str> = buf.lines().filter(|l| l.contains("s=n,")).collect();
        assert_eq!(vertex_lines.len(), 4, "Should have 4 vertex lines");
        for line in &vertex_lines {
            assert!(
                line.contains("0 0 s=n,"),
                "Missing coords should default to 0 0, got: {}",
                line
            );
        }
    }
}
