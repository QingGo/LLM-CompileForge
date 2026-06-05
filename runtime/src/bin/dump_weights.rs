//! Thin wrapper — delegates to `dump_weights_runner::run()` in the lib crate.
//!
//! Usage:
//!   cargo run --bin dump_weights -- --compiled-dir outputs/compiled/tiny_llama_fresh

extern crate llm_serveforge_runtime;

use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut compiled_dir = PathBuf::from("outputs/compiled/tiny_llama_fresh");
    let mut output_dir: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--compiled-dir" => {
                i += 1;
                compiled_dir = PathBuf::from(&args[i]);
            }
            "--output-dir" => {
                i += 1;
                output_dir = Some(PathBuf::from(&args[i]));
            }
            _ => {
                eprintln!("Usage: dump_weights --compiled-dir <path> [--output-dir <path>]");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    llm_serveforge_runtime::check::dump_weights_runner::run(compiled_dir, output_dir)
}
