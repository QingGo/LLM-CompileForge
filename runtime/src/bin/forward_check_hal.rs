//! Thin wrapper — delegates to `forward_check_hal_runner::run()` in the lib crate.
//!
//! Usage:
//!   cargo run --bin forward_check_hal --features hal-rust

extern crate llm_serveforge_runtime;

#[cfg(not(feature = "hal-rust"))]
fn main() {
    eprintln!(
        "forward_check_hal requires --features hal-rust.  \
         Build with: cargo build --bin forward_check_hal --features hal-rust"
    );
    std::process::exit(1);
}

#[cfg(feature = "hal-rust")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    llm_serveforge_runtime::forward_check_hal_runner::run()
}
