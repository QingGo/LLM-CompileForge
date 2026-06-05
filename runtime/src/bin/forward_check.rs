//! Thin wrapper — delegates to `forward_check_runner::run()` in the lib crate.
//!
//! Usage: cargo run --bin forward_check

extern crate llm_serveforge_runtime;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    llm_serveforge_runtime::check::forward_check_runner::run()
}
