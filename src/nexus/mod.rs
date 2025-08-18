use std::fmt::Write as FmtWrite;

use anyhow::Result;
use fixedbitset::FixedBitSet;

pub mod network_writer;
pub mod nexus_writer;

/* ---------- helpers ---------- */

/// Helper: format float with up to 7 significant digits, trimming trailing zeros.
fn fmt_f(x: f64) -> String {
    // 7 decimal digits is usually enough for NEXUS dumps
    let mut s = format!("{:.7}", x);
    // trim trailing zeros and possible trailing dot
    while s.contains('.') && s.ends_with('0') {
        s.pop();
    }
    if s.ends_with('.') {
        s.pop();
    }
    if s.is_empty() {
        s.push('0');
    }
    s
}

fn trim_float(x: f64, digits: usize) -> String {
    // format with fixed precision, then trim trailing zeros
    let mut s = format!("{:.1$}", x, digits);
    while s.ends_with('0') && s.contains('.') {
        s.pop();
    }
    if s.ends_with('.') {
        s.pop();
    }
    if s.is_empty() {
        s.push('0');
    }
    s
}

/// Convert FixedBitSet (1-based) into "i j k" string; skips bit 0 if present.
fn bitset_to_string(bs: &FixedBitSet) -> String {
    let mut out = String::new();
    let mut first = true;
    for t in bs.ones() {
        if t == 0 {
            continue;
        }
        if !first {
            out.push(' ');
        }
        first = false;
        out.push_str(&t.to_string());
    }
    out
}

fn escape_label(s: &str) -> String {
    s.replace('\'', "''")
}

fn write_title_and_link<W: FmtWrite>(
    mut w: W,
    title: Option<&str>,
    link: Option<&str>,
) -> Result<()> {
    if let Some(t) = title {
        writeln!(w, "TITLE '{}';", escape_label(t))?;
    }
    if let Some(l) = link {
        writeln!(w, "LINK '{}';", escape_label(l))?;
    }
    Ok(())
}

// fn fold_256(s: &str) -> String {
//     const MAX: usize = 256;
//     if s.len() <= MAX { s.to_string() } else { s[..MAX].to_string() }
// }
