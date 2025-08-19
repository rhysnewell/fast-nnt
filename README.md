# fast-nnt
fast-nnt (read Fast Ent) is a simple Rust implementation of the Neighbor Net algorithm


### Installation
Install Rust via [rustup](https://rustup.rs/).

Clone and install this repo via:
```
git clone https://github.com/yourusername/fast-nnt.git
cd fast-nnt
cargo install --path .
```

### Usage

To generate a split nexus file (mostly) identical to SplitsTree4 and SplitsTree6:
```
fast_nnt neighbour_net -t 4 -i test/data/large_dist_matrix.csv -d output_dir -o prefix -O splits-tree4
```

Use the new Huson 2023 ordering algorithm (default):
```
fast_nnt neighbour_net -t 4 -i test/data/large_dist_matrix.csv -d output_dir -o prefix -O huson2023
```


### Known issues
- Floating point drift in the CGNR function, as observed in the smoke_30 test.
- Not sure where it is happening, but the final results on real data end up looking pretty much the same so not sure if it is an issue. Will need to be fixed at some point.


### TODO
- Work on parallelism. Not a priority as the program is fast enough.
- Test on giant datasets.