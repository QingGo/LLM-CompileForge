//! Phase S0 fp16 GEMV spike CLI.
//!
//! Measures real F16 safetensors weights with the AVX2+F16C kernel in
//! `engine::spikes` against the f32 `cblas_sgemm` control, then projects
//! the decode-only throughput with the hard formula from
//! `.omo/plans/p1-post-d2-development-plan.md`.
//!
//! This binary is intentionally not part of the serveforge inference path.

use std::path::PathBuf;

use llm_serveforge_runtime::engine::spikes::{
    f16_weight_to_f32, f32_gemv_blas, f32_gemv_blas_threaded, fp16_gemv_f16c,
    fp16_gemv_threaded, load_f16_tensor, min_elapsed_ms,
};

const DEFAULT_KEYS: [(&str, &str, &str); 4] = [
    (
        "qkv",
        "model.decoder.layers.0.self_attn.q_proj.weight",
        "N=768 K=768 (q/k/v/out proxy)",
    ),
    (
        "fc1",
        "model.decoder.layers.0.fc1.weight",
        "N=3072 K=768",
    ),
    (
        "fc2",
        "model.decoder.layers.0.fc2.weight",
        "N=768 K=3072",
    ),
    (
        "vocab",
        "lm_head.weight",
        "N=50272 K=768",
    ),
];

struct Args {
    safetensors: PathBuf,
    key_qkv: String,
    key_fc1: String,
    key_fc2: String,
    key_vocab: String,
    overhead_ms: f64,
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
    let mut key_qkv = DEFAULT_KEYS[0].1.to_string();
    let mut key_fc1 = DEFAULT_KEYS[1].1.to_string();
    let mut key_fc2 = DEFAULT_KEYS[2].1.to_string();
    let mut key_vocab = DEFAULT_KEYS[3].1.to_string();
    let mut overhead_ms = 0.0f64;

    while let Some(arg) = args.next() {
        let mut value = |name: &str| -> Result<String, anyhow::Error> {
            args.next()
                .ok_or_else(|| anyhow::anyhow!("missing value for {}", name))
        };
        match arg.as_str() {
            "--safetensors" => safetensors = Some(PathBuf::from(value("--safetensors")?)),
            "--key-qkv" => key_qkv = value("--key-qkv")?,
            "--key-fc1" => key_fc1 = value("--key-fc1")?,
            "--key-fc2" => key_fc2 = value("--key-fc2")?,
            "--key-vocab" => key_vocab = value("--key-vocab")?,
            "--overhead-ms" => overhead_ms = value("--overhead-ms")?.parse()?,
            "--help" | "-h" => {
                eprintln!(
                    "usage: fp16_gemv_spike [--safetensors PATH] [--overhead-ms MS]\n\
                     keys: --key-qkv/--key-fc1/--key-fc2/--key-vocab"
                );
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown argument: {}", other),
        }
    }

    let safetensors = match safetensors {
        Some(p) => p,
        None => default_safetensors()
            .ok_or_else(|| anyhow::anyhow!("no --safetensors and no HF cache model.safetensors found"))?,
    };
    Ok(Args {
        safetensors,
        key_qkv,
        key_fc1,
        key_fc2,
        key_vocab,
        overhead_ms,
    })
}

fn main() -> Result<(), anyhow::Error> {
    let args = parse_args()?;
    let st = &args.safetensors;
    anyhow::ensure!(st.exists(), "safetensors not found: {}", st.display());

    let mut report = serde_json::Map::new();
    report.insert(
        "safetensors".to_string(),
        serde_json::Value::String(st.display().to_string()),
    );
    report.insert(
        "cpu".to_string(),
        serde_json::Value::String(
            std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".to_string()),
        ),
    );

    // Kernel feature report.
    #[cfg(target_arch = "x86_64")]
    {
        report.insert(
            "avx2".to_string(),
            serde_json::Value::Bool(std::arch::is_x86_feature_detected!("avx2")),
        );
        report.insert(
            "f16c".to_string(),
            serde_json::Value::Bool(std::arch::is_x86_feature_detected!("f16c")),
        );
    }

    // Load the four real weight matrices from the f16 safetensors.
    let specs = [
        ("qkv", &args.key_qkv),
        ("fc1", &args.key_fc1),
        ("fc2", &args.key_fc2),
        ("vocab", &args.key_vocab),
    ];
    let mut weights = serde_json::Map::new();
    let mut f32_weights: Vec<(&str, Vec<f32>, usize, usize)> = Vec::new();
    let mut vocab_u16: Option<(Vec<u16>, usize, usize)> = None;
    for (name, key) in specs {
        let t = load_f16_tensor(st, key)
            .map_err(|e| anyhow::anyhow!("load {} from {}: {}", key, st.display(), e))?;
        let mut entry = serde_json::Map::new();
        entry.insert("key".to_string(), serde_json::Value::String(key.clone()));
        entry.insert("n".to_string(), serde_json::Value::from(t.n));
        entry.insert("k".to_string(), serde_json::Value::from(t.k));
        weights.insert(name.to_string(), serde_json::Value::Object(entry));
        let f32_control = f16_weight_to_f32(&t.data);
        f32_weights.push((name, f32_control, t.n, t.k));
        if name == "vocab" {
            vocab_u16 = Some((t.data.clone(), t.n, t.k));
        }

        // Deterministic "hidden state" input: non-trivial, in the weight
        // dynamic range, and identical for every measurement group.
        let x: Vec<f32> = (0..t.k)
            .map(|i| ((i % 64) as f32 - 31.5) * 0.02)
            .collect();

        let iters = match name {
            "qkv" => 60,
            "fc1" => 30,
            "fc2" => 30,
            "vocab" => 20,
            _ => 10,
        };
        let f32_control = f32_weights.last().unwrap().1.as_slice();
        let f16_ms = min_elapsed_ms(|| {
            let _ = fp16_gemv_f16c(&x, t.n, t.k, &t.data);
        }, 4, iters)?;
        let f32_ms = min_elapsed_ms(|| {
            let _ = f32_gemv_blas(&x, t.n, t.k, f32_control);
        }, 2, (iters / 2).max(2))?;

        let mut entry = weights.get_mut(name).unwrap().as_object_mut().unwrap();
        entry.insert(
            "f16_us".to_string(),
            serde_json::Value::from((f16_ms * 1000.0 * 1000.0).round()),
        );
        entry.insert(
            "f32_control_us".to_string(),
            serde_json::Value::from((f32_ms * 1000.0 * 1000.0).round()),
        );
        eprintln!(
            "[spike] {name}: f16 {:.1}us f32-control {:.1}us",
            f16_ms * 1e3,
            f32_ms * 1e3,
        );
    }
    report.insert("weights".to_string(), serde_json::Value::Object(weights));

    // Vocabulary thread scaling (the only N large enough to matter).
    let (vocab_w, vocab_n, vocab_k) = match f32_weights.iter().find(|(n, _, _, _)| *n == "vocab") {
        Some((_, w, n, k)) => (w, *n, *k),
        None => anyhow::bail!("vocab weight not measured"),
    };
    let (vocab_data, vn, vk) = vocab_u16.expect("vocab weight must be loaded");
    debug_assert_eq!((vocab_n, vocab_k), (vn, vk));
    let x_vocab: Vec<f32> = (0..vocab_k)
        .map(|i| ((i % 64) as f32 - 31.5) * 0.02)
        .collect();
    let mut thread_scaling = serde_json::Map::new();
    for threads in [1usize, 2, 4, 6] {
        let f16_ms = min_elapsed_ms(|| {
            let _ = fp16_gemv_threaded(&x_vocab, vocab_n, vocab_k, &vocab_data, threads);
        }, 2, 12)?;
        let f32_ms = min_elapsed_ms(|| {
            let _ = f32_gemv_blas_threaded(&x_vocab, vocab_n, vocab_k, vocab_w, threads);
        }, 2, 4)?;
        let mut entry = serde_json::Map::new();
        entry.insert("f16_us".to_string(), serde_json::Value::from((f16_ms * 1e6).round()));
        entry.insert(
            "f32_control_us".to_string(),
            serde_json::Value::from((f32_ms * 1e6).round()),
        );
        thread_scaling.insert(threads.to_string(), serde_json::Value::Object(entry));
        eprintln!(
            "[spike] vocab threads={}: f16 {:.1}us f32-control {:.1}us",
            threads,
            f16_ms * 1e3,
            f32_ms * 1e3,
        );
    }
    report.insert("vocab_thread_scaling".to_string(), serde_json::Value::Object(thread_scaling));

    // Hard projection formula (us values, min across measurements):
    //   per layer = 3*qkv + out(qkv proxy) + fc1 + fc2
    //   step = 12*layer + vocab + calibrated non-projection overhead
    let get_us = |name: &str| -> f64 {
        report["weights"][name]["f16_us"]
            .as_f64()
            .expect("f16_us must be a number")
    };
    let qkv_us = get_us("qkv");
    let fc1_us = get_us("fc1");
    let fc2_us = get_us("fc2");
    let vocab_us = get_us("vocab");
    let per_layer_us = 3.0 * qkv_us + qkv_us + fc1_us + fc2_us;
    let projection_total_us = 12.0 * per_layer_us + vocab_us;

    let mut overhead_grid = Vec::new();
    for overhead_ms in [0.0f64, 1.0, 2.0, 2.5, 3.0, 4.0] {
        let step_ms = projection_total_us / 1e3 + overhead_ms;
        let tok_s = 1000.0 / step_ms;
        let mut entry = serde_json::Map::new();
        entry.insert("overhead_ms".to_string(), serde_json::Value::from(overhead_ms));
        entry.insert("step_ms".to_string(), serde_json::Value::from(step_ms));
        entry.insert("tok_s".to_string(), serde_json::Value::from(tok_s));
        overhead_grid.push(serde_json::Value::Object(entry));
    }

    let chosen_overhead = args.overhead_ms;
    let chosen_step_ms = projection_total_us / 1e3 + chosen_overhead;
    let chosen_tok_s = 1000.0 / chosen_step_ms;
    let mut projection = serde_json::Map::new();
    projection.insert("per_layer_us".to_string(), serde_json::Value::from(per_layer_us));
    projection.insert("vocab_us".to_string(), serde_json::Value::from(vocab_us));
    projection.insert(
        "projection_total_ms".to_string(),
        serde_json::Value::from(projection_total_us / 1e3),
    );
    projection.insert(
        "projection_tok_s_at_zero_overhead".to_string(),
        serde_json::Value::from(1000.0 / (projection_total_us / 1e3)),
    );
    projection.insert(
        "chosen_overhead_ms".to_string(),
        serde_json::Value::from(chosen_overhead),
    );
    projection.insert(
        "projected_step_ms".to_string(),
        serde_json::Value::from(chosen_step_ms),
    );
    projection.insert(
        "projected_tok_s".to_string(),
        serde_json::Value::from(chosen_tok_s),
    );
    projection.insert(
        "pass_125".to_string(),
        serde_json::Value::Bool(chosen_tok_s >= 125.0),
    );
    projection.insert("overhead_grid".to_string(), serde_json::Value::Array(overhead_grid));
    report.insert("projection".to_string(), serde_json::Value::Object(projection));

    println!("{}", serde_json::to_string_pretty(&serde_json::Value::Object(report))?);
    Ok(())
}
