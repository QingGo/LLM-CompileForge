//! Phase S2 Day 1 bounded spike: real 12-layer F16 weight streaming GEMV.
//!
//! S0 measured one layer in L3.  This binary preloads all 12 OPT decoder
//! layers (q/k/v/out/fc1/fc2) plus lm_head into owned F16 buffers first,
//! then times full passes that touch ~170 MB of weights — an order of
//! magnitude beyond this machine's L3, so every measured pass is a DRAM
//! stream, not an L3-resident repeat.
//!
//! This is still a spike: it does not touch the ABI or serveforge path.

use std::path::PathBuf;
use std::time::Instant;

use llm_serveforge_runtime::engine::spikes::{
    f16_weight_to_f32, f32_gemv_blas_threaded, fp16_gemv_f16c, fp16_gemv_threaded,
    load_f16_tensor, min_elapsed_ms, F16TensorView,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Qkv,
    Out,
    Fc1,
    Fc2,
}

struct LayerWeight {
    kind: Kind,
    tensor: F16TensorView,
}

#[derive(Clone, Copy, Debug, Default)]
struct PassTiming {
    total_ms: f64,
    qkv_ms: f64,
    out_ms: f64,
    fc1_ms: f64,
    fc2_ms: f64,
    bytes: usize,
}

struct Args {
    safetensors: PathBuf,
    iters: usize,
}

fn default_safetensors() -> Option<PathBuf> {
    let hub = std::path::PathBuf::from(
        std::env::var("HOME").unwrap_or_else(|_| ".".to_string()),
    )
    .join(".cache/huggingface/hub/models--facebook--opt-125m/snapshots");
    let mut found = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&hub) {
        for entry in entries.flatten() {
            let cand = entry.path().join("model.safetensors");
            if cand.exists() {
                found.push(cand);
            }
        }
    }
    found.sort();
    found.pop()
}

fn parse_args() -> Result<Args, anyhow::Error> {
    let mut args = std::env::args().skip(1).peekable();
    let mut safetensors = None;
    let mut iters = 3usize;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--safetensors" => {
                let path = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("missing value for --safetensors"))?;
                safetensors = Some(PathBuf::from(path));
            }
            "--iters" => {
                let value = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("missing value for --iters"))?;
                iters = value.parse()?;
            }
            "--help" | "-h" => {
                eprintln!(
                    "usage: fp16_layer_stream_spike [--safetensors PATH] [--iters N]\n\
                     Measures 12-layer F16 weight streaming GEMV; JSON report on stdout."
                );
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown argument: {}", other),
        }
    }
    let safetensors = safetensors
        .or_else(default_safetensors)
        .ok_or_else(|| anyhow::anyhow!("no --safetensors and no HF cache model.safetensors found"))?;
    anyhow::ensure!(safetensors.exists(), "safetensors not found: {}", safetensors.display());
    anyhow::ensure!(iters > 0, "--iters must be >= 1");
    Ok(Args { safetensors, iters })
}

fn load_layer_weights(st: &std::path::Path) -> Result<Vec<LayerWeight>, anyhow::Error> {
    let mut weights = Vec::with_capacity(12 * 6);
    for layer in 0..12usize {
        let prefix = format!("model.decoder.layers.{layer}.self_attn.");
        for (kind, suffix) in [
            (Kind::Qkv, "q_proj.weight"),
            (Kind::Qkv, "k_proj.weight"),
            (Kind::Qkv, "v_proj.weight"),
            (Kind::Out, "out_proj.weight"),
        ] {
            let key = format!("{prefix}{suffix}");
            let tensor = load_f16_tensor(st, &key)
                .map_err(|e| anyhow::anyhow!("load {key}: {e}"))?;
            weights.push(LayerWeight { kind, tensor });
        }
        for (kind, suffix) in [
            (Kind::Fc1, "fc1.weight"),
            (Kind::Fc2, "fc2.weight"),
        ] {
            let key = format!("model.decoder.layers.{layer}.{suffix}");
            let tensor = load_f16_tensor(st, &key)
                .map_err(|e| anyhow::anyhow!("load {key}: {e}"))?;
            weights.push(LayerWeight { kind, tensor });
        }
    }
    Ok(weights)
}

fn x_for_len(k: usize) -> Vec<f32> {
    (0..k)
        .map(|i| ((i % 64) as f32 - 31.5) * 0.02)
        .collect()
}

fn run_pass(weights: &[LayerWeight], fc_threads: usize) -> PassTiming {
    let mut timing = PassTiming::default();
    let x768 = x_for_len(768);
    let x3072 = x_for_len(3072);

    let mut kind_t0 = Instant::now();
    let mut current = Kind::Qkv;
    for weight in weights {
        if weight.kind != current {
            let elapsed = kind_t0.elapsed().as_secs_f64() * 1e3;
            match current {
                Kind::Qkv => timing.qkv_ms += elapsed,
                Kind::Out => timing.out_ms += elapsed,
                Kind::Fc1 => timing.fc1_ms += elapsed,
                Kind::Fc2 => timing.fc2_ms += elapsed,
            }
            current = weight.kind;
            kind_t0 = Instant::now();
        }
        let x: &[f32] = if weight.tensor.k == 768 { &x768 } else { &x3072 };
        let out = if fc_threads > 1 && matches!(weight.kind, Kind::Fc1 | Kind::Fc2) {
            fp16_gemv_threaded(x, weight.tensor.n, weight.tensor.k, &weight.tensor.data, fc_threads)
        } else {
            fp16_gemv_f16c(x, weight.tensor.n, weight.tensor.k, &weight.tensor.data)
        };
        debug_assert_eq!(out.len(), weight.tensor.n);
        timing.bytes += weight.tensor.n.saturating_mul(weight.tensor.k) * 2;
    }
    let elapsed = kind_t0.elapsed().as_secs_f64() * 1e3;
    match current {
        Kind::Qkv => timing.qkv_ms += elapsed,
        Kind::Out => timing.out_ms += elapsed,
        Kind::Fc1 => timing.fc1_ms += elapsed,
        Kind::Fc2 => timing.fc2_ms += elapsed,
    }
    timing.total_ms = timing.qkv_ms + timing.out_ms + timing.fc1_ms + timing.fc2_ms;
    timing
}

fn bench_pass(weights: &[LayerWeight], fc_threads: usize, iters: usize) -> PassTiming {
    for _ in 0..2 {
        let _ = run_pass(weights, fc_threads);
    }
    let mut best = PassTiming {
        total_ms: f64::INFINITY,
        ..PassTiming::default()
    };
    for _ in 0..iters {
        let pass = run_pass(weights, fc_threads);
        if pass.total_ms < best.total_ms {
            best = pass;
        }
    }
    best
}

fn main() -> Result<(), anyhow::Error> {
    let args = parse_args()?;
    let weights = load_layer_weights(&args.safetensors)?;
    let layer_bytes: usize = weights.iter().map(|w| w.tensor.n * w.tensor.k * 2).sum();
    anyhow::ensure!(
        weights.len() == 72,
        "expected 72 layer weights, loaded {}",
        weights.len()
    );

    let mut report = serde_json::Map::new();
    report.insert(
        "safetensors".to_string(),
        serde_json::Value::String(args.safetensors.display().to_string()),
    );
    report.insert(
        "layer_weight_bytes".to_string(),
        serde_json::Value::from(layer_bytes),
    );

    // Full 12-layer streaming pass variants.  fc1/fc2 are large enough to
    // benefit from row blocking; qkv/out stay single-thread (small N).
    let mut variants = serde_json::Map::new();
    let mut best_variant: Option<(usize, PassTiming)> = None;
    for threads in [1usize, 2, 4] {
        let pass = bench_pass(&weights, threads, args.iters);
        let entry = serde_json::json!({
            "total_ms": pass.total_ms,
            "qkv_ms": pass.qkv_ms,
            "out_ms": pass.out_ms,
            "fc1_ms": pass.fc1_ms,
            "fc2_ms": pass.fc2_ms,
            "bytes": pass.bytes,
            "effective_gb_s": pass.bytes as f64 / pass.total_ms / 1e6,
        });
        if best_variant
            .as_ref()
            .map(|(_, best)| pass.total_ms < best.total_ms)
            .unwrap_or(true)
        {
            best_variant = Some((threads, pass));
        }
        variants.insert(threads.to_string(), entry);
        eprintln!(
            "[stream] fc_threads={}: total={:.3}ms qkv={:.3} out={:.3} fc1={:.3} fc2={:.3} bandwidth={:.1}GB/s",
            threads,
            pass.total_ms,
            pass.qkv_ms,
            pass.out_ms,
            pass.fc1_ms,
            pass.fc2_ms,
            pass.bytes as f64 / pass.total_ms / 1e6,
        );
    }
    report.insert("layer_variants".to_string(), serde_json::Value::Object(variants));

    // Vocabulary projection (77 MB; measured separately, as in S0).
    let vocab = load_f16_tensor(&args.safetensors, "lm_head.weight")
        .map_err(|e| anyhow::anyhow!("load lm_head.weight: {e}"))?;
    let x_vocab = x_for_len(vocab.k);
    let mut vocab_scaling = serde_json::Map::new();
    let mut best_vocab_ms = f64::INFINITY;
    for threads in [1usize, 2, 4, 6] {
        let ms = min_elapsed_ms(
            || {
                let _ = fp16_gemv_threaded(&x_vocab, vocab.n, vocab.k, &vocab.data, threads);
            },
            2,
            args.iters,
        )?;
        let entry = serde_json::json!({
            "ms": ms,
            "bytes": vocab.n * vocab.k * 2,
            "effective_gb_s": (vocab.n * vocab.k * 2) as f64 / ms / 1e6,
        });
        if ms < best_vocab_ms {
            best_vocab_ms = ms;
        }
        vocab_scaling.insert(threads.to_string(), entry);
        eprintln!(
            "[stream] vocab threads={}: {:.3}ms ({:.1}GB/s)",
            threads,
            ms,
            (vocab.n * vocab.k * 2) as f64 / ms / 1e6,
        );
    }
    report.insert("vocab_thread_scaling".to_string(), serde_json::Value::Object(vocab_scaling));

    // f32 BLAS control for vocab only (same input/output contract).
    let vocab_f32 = f16_weight_to_f32(&vocab.data);
    let vocab_f32_ms = min_elapsed_ms(
        || {
            let _ = f32_gemv_blas_threaded(&x_vocab, vocab.n, vocab.k, &vocab_f32, 6);
        },
        1,
        args.iters,
    )?;
    report.insert(
        "vocab_f32_control_ms_6threads".to_string(),
        serde_json::Value::from(vocab_f32_ms),
    );

    let (best_threads, best_pass) =
        best_variant.expect("at least one layer variant was measured");
    let projection_ms = best_pass.total_ms + best_vocab_ms;
    let zero_overhead_tok_s = 1000.0 / projection_ms;
    let mut overhead_grid = Vec::new();
    for overhead_ms in [0.0f64, 2.0, 4.0, 5.0, 6.0, 6.5, 7.0, 8.0] {
        let step_ms = projection_ms + overhead_ms;
        let tok_s = 1000.0 / step_ms;
        overhead_grid.push(serde_json::json!({
            "overhead_ms": overhead_ms,
            "step_ms": step_ms,
            "tok_s": tok_s,
        }));
    }
    let projection = serde_json::json!({
        "best_layer_fc_threads": best_threads,
        "layer_pass_ms": best_pass.total_ms,
        "layer_qkv_ms": best_pass.qkv_ms,
        "layer_out_ms": best_pass.out_ms,
        "layer_fc1_ms": best_pass.fc1_ms,
        "layer_fc2_ms": best_pass.fc2_ms,
        "layer_effective_gb_s": best_pass.bytes as f64 / best_pass.total_ms / 1e6,
        "vocab_best_ms": best_vocab_ms,
        "projection_ms": projection_ms,
        "projection_tok_s_zero_overhead": zero_overhead_tok_s,
        "overhead_grid": overhead_grid,
        "pass_40_at_6_5ms_overhead": 1000.0 / (projection_ms + 6.5) >= 40.0,
    });
    report.insert("projection".to_string(), projection);

    println!("{}", serde_json::to_string_pretty(&serde_json::Value::Object(report))?);
    Ok(())
}
