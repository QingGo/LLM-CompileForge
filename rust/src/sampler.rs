//! Token sampling strategies for autoregressive generation.
//!
//! Supports greedy (argmax), temperature scaling, top-k filtering,
//! and top-p (nucleus) filtering.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub struct Sampler {
    rng: StdRng,
}

#[derive(Debug, Clone)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
        }
    }
}

impl SamplerConfig {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
        }
    }
}

impl Sampler {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn sample(&mut self, logits: &[f32], config: &SamplerConfig) -> u32 {
        if logits.is_empty() {
            return 0;
        }

        // Guard: non-finite logits (NaN, ±inf) → fall back to greedy
        // on the finite subset, or return 0 if none are finite.
        if logits.iter().any(|v| !v.is_finite()) {
            let max_idx = logits
                .iter()
                .enumerate()
                .filter(|(_, &v)| v.is_finite())
                .fold((0u32, f32::NEG_INFINITY), |(mi, mv), (i, &v)| {
                    if v > mv { (i as u32, v) } else { (mi, mv) }
                });
            return max_idx.0;
        }

        if config.temperature <= 0.0 {
            return Self::greedy(logits);
        }

        // Guard: extremely small temperature can overflow to infinity.
        let t = if config.temperature < 1e-7 {
            eprintln!(
                "warning: temperature {} is very small, clamping to 1e-7",
                config.temperature
            );
            1e-7f32
        } else {
            config.temperature
        };

        let mut probs: Vec<f32> = logits.to_vec();

        if (t - 1.0).abs() > f32::EPSILON {
            apply_temperature(&mut probs, t);
        }

        if config.top_k > 0 && config.top_k < probs.len() {
            apply_top_k(&mut probs, config.top_k);
        }

        if config.top_p > 0.0 && config.top_p < 1.0 {
            apply_top_p(&mut probs, config.top_p);
        }

        // Guard: if all logits were filtered to -inf, softmax produces
        // uniform probabilities. Detect this and fall back to greedy.
        let all_neg_inf = probs.iter().all(|&v| v == f32::NEG_INFINITY);
        if all_neg_inf {
            return Self::greedy(logits);
        }

        softmax_inplace(&mut probs);
        multinomial(&mut self.rng, &probs)
    }

    pub fn greedy(logits: &[f32]) -> u32 {
        let mut max_val = f32::NEG_INFINITY;
        let mut max_idx = 0u32;
        for (i, &v) in logits.iter().enumerate() {
            if v > max_val {
                max_val = v;
                max_idx = i as u32;
            }
        }
        max_idx
    }
}

fn apply_temperature(logits: &mut [f32], t: f32) {
    let inv_t = 1.0 / t;
    for v in logits.iter_mut() {
        *v *= inv_t;
    }
}

fn apply_top_k(logits: &mut [f32], k: usize) {
    let n = logits.len();
    if k >= n {
        return;
    }
    let mut indexed: Vec<(f32, usize)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i))
        .collect();
    let split_idx = n - k;
    indexed.select_nth_unstable_by(split_idx, |a, b| {
        a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
    });
    let threshold = indexed[split_idx].0;
    for v in logits.iter_mut() {
        if *v < threshold {
            *v = f32::NEG_INFINITY;
        }
    }
}

fn apply_top_p(logits: &mut [f32], p: f32) {
    let n = logits.len();
    let mut indexed: Vec<(f32, usize)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i))
        .collect();
    indexed.sort_unstable_by(|a, b| {
        b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut exp_sum = 0.0f32;
    let mut exps: Vec<f32> = Vec::with_capacity(n);
    for &(val, _) in &indexed {
        let e = val.exp();
        exps.push(e);
        exp_sum += e;
    }

    let mut cumsum = 0.0f32;
    let mut cutoff = 0usize;
    for (i, (&(val, _), &e)) in indexed.iter().zip(exps.iter()).enumerate() {
        cumsum += e / exp_sum;
        cutoff = i;
        if cumsum >= p {
            break;
        }
    }

    let threshold = if cutoff < n - 1 {
        indexed[cutoff + 1].0
    } else {
        f32::NEG_INFINITY
    };

    for v in logits.iter_mut() {
        if *v < threshold {
            *v = f32::NEG_INFINITY;
        }
    }
}

fn softmax_inplace(probs: &mut [f32]) {
    let max_val = probs
        .iter()
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum = 0.0f32;
    for v in probs.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in probs.iter_mut() {
            *v /= sum;
        }
    }
}

fn multinomial(rng: &mut StdRng, probs: &[f32]) -> u32 {
    let r: f32 = rng.gen();
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_argmax() {
        let logits = [0.1, 0.5, 0.3, 0.9, 0.2];
        assert_eq!(Sampler::greedy(&logits), 3);
    }

    #[test]
    fn test_greedy_all_equal() {
        let logits = [1.0; 10];
        assert_eq!(Sampler::greedy(&logits), 0);
    }

    #[test]
    fn test_greedy_empty_returns_zero() {
        let logits: [f32; 0] = [];
        assert_eq!(Sampler::greedy(&logits), 0);
    }

    #[test]
    fn test_sample_with_temperature_zero_is_greedy() {
        let mut s = Sampler::new(42);
        let logits = [1.0, 2.0, 3.0, 4.0, 5.0];
        let token = s.sample(&logits, &SamplerConfig::greedy());
        assert_eq!(token, 4);
    }

    #[test]
    fn test_sample_with_temperature_produces_valid_token() {
        let mut s = Sampler::new(42);
        let logits = [0.1, 0.2, 0.3, 0.4, 5.0];
        let cfg = SamplerConfig {
            temperature: 0.8,
            top_p: 1.0,
            top_k: 0,
        };
        let token = s.sample(&logits, &cfg);
        assert!(token < 5);
    }

    #[test]
    fn test_top_k_excludes_low_values() {
        let logits = [0.1, 0.2, 0.3, 0.4, 0.5];
        let cfg = SamplerConfig {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 2,
        };
        let mut s = Sampler::new(42);
        for _ in 0..20 {
            let token = s.sample(&logits, &cfg);
            assert!(token >= 3); // only top 2 (indices 3,4) should be chosen
        }
    }

    #[test]
    fn test_top_p_excludes_low_probability() {
        let mut logits = [0.0f32; 100];
        logits[50] = 10.0;
        logits[51] = 8.0;
        let cfg = SamplerConfig {
            temperature: 1.0,
            top_p: 0.9,
            top_k: 0,
        };
        let mut s = Sampler::new(42);
        for _ in 0..20 {
            let token = s.sample(&logits, &cfg);
            assert!(token == 50 || token == 51);
        }
    }

    #[test]
    fn test_non_finite_logits_fallback() {
        let logits = [1.0, f32::NAN, 2.0, f32::INFINITY, 3.0];
        let cfg = SamplerConfig::default();
        let mut s = Sampler::new(42);
        let token = s.sample(&logits, &cfg);
        // Should pick the max among finite values: 3.0 at index 4
        assert_eq!(token, 4);
    }

    #[test]
    fn test_all_neg_inf_logits_fallback() {
        let logits = [f32::NEG_INFINITY; 5];
        let cfg = SamplerConfig::default();
        let mut s = Sampler::new(42);
        let token = s.sample(&logits, &cfg);
        // All -inf → falls back to greedy, returns index 0
        assert_eq!(token, 0);
    }

    #[test]
    fn test_tiny_temperature_clamped() {
        let logits = [0.0, 1e6_f32];
        let cfg = SamplerConfig {
            temperature: 1e-15,
            top_p: 1.0,
            top_k: 0,
        };
        let mut s = Sampler::new(42);
        let token = s.sample(&logits, &cfg);
        // With tiny temperature, index 1 should dominate
        assert_eq!(token, 1);
    }

    #[test]
    fn test_top_k_with_ties() {
        let logits = [0.1, 0.5, 0.5, 0.5, 0.9];
        let cfg = SamplerConfig {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 3,
        };
        let mut s = Sampler::new(42);
        for _ in 0..20 {
            let token = s.sample(&logits, &cfg);
            // Top 3 should be indices 1,2,3,4 (4 values tie at top-3 boundary)
            assert!(token >= 1);
        }
    }

    #[test]
    fn test_top_k_greater_than_vocab() {
        let logits = [0.1, 0.2, 0.3];
        let cfg = SamplerConfig {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 10,
        };
        let mut s = Sampler::new(42);
        let token = s.sample(&logits, &cfg);
        assert!(token < 3);
    }
}
