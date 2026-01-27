use anyhow::{Context, anyhow};
use clap::{Parser, crate_name, crate_version};

use anon_nnt::{
    cli::{ProgramArgs, ProgramSubcommand},
    neighbour_net::neighbour_net::NeighbourNet,
    set_log_level,
};
use log::{error, info};

fn main() {
    let app = ProgramArgs::parse();

    if let Err(err) = rayon::ThreadPoolBuilder::new()
        .num_threads(app.threads)
        .build_global()
    {
        error!("Failed to initialize Rayon thread pool: {:#}", err);
        std::process::exit(1);
    }

    set_log_level(&app, true, crate_name!(), crate_version!());
    info!("Rayon threads: {}", rayon::current_num_threads());

    // Dispatch subcommands
    let result = match app.subcommand {
        ProgramSubcommand::NeighborNet(args) => {
            let runner = NeighbourNet::new(app.output_directory, args)
                .context("creating NeighbourNet")
                .expect("Failed to create NeighbourNet");
            runner.run()
        }

        #[allow(unreachable_patterns)]
        other => {
            error!("Subcommand not implemented: {:?}", other);
            Err(anyhow!("unsupported subcommand"))
        }
    };

    if let Err(err) = result {
        error!("{:#}", err);
        std::process::exit(1);
    }
}
