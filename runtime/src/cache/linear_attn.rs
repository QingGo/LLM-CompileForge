//! Fixed-size linear-attention state cache (E11).
//!
//! GatedDeltaNet layers keep two per-layer states that are not paged:
//! a recurrent delta state `[heads, key_dim, value_dim]` and a short-conv
//! state `[conv_channels, kernel]`.  This module owns the flat storage and
//! gives the future op-plan/linear-attn executor a place to read/write them
//! without conflating with paged K/V.

pub struct LinearAttnCache {
    pub num_layers: usize,
    pub heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub conv_channels: usize,
    pub conv_kernel: usize,
    recurrent: Vec<f32>,
    conv: Vec<f32>,
}

impl LinearAttnCache {
    pub fn new(
        num_layers: usize,
        heads: usize,
        key_dim: usize,
        value_dim: usize,
        conv_channels: usize,
        conv_kernel: usize,
    ) -> Self {
        let recurrent_len = num_layers * heads * key_dim * value_dim;
        let conv_len = num_layers * conv_channels * conv_kernel;
        Self {
            num_layers,
            heads,
            key_dim,
            value_dim,
            conv_channels,
            conv_kernel,
            recurrent: vec![0.0; recurrent_len],
            conv: vec![0.0; conv_len],
        }
    }

    pub fn recurrent_state(&self, layer: usize) -> &[f32] {
        let start = layer * self.heads * self.key_dim * self.value_dim;
        &self.recurrent[start..start + self.heads * self.key_dim * self.value_dim]
    }

    pub fn recurrent_state_mut(&mut self, layer: usize) -> &mut [f32] {
        let start = layer * self.heads * self.key_dim * self.value_dim;
        &mut self.recurrent[start..start + self.heads * self.key_dim * self.value_dim]
    }

    pub fn conv_state(&self, layer: usize) -> &[f32] {
        let start = layer * self.conv_channels * self.conv_kernel;
        &self.conv[start..start + self.conv_channels * self.conv_kernel]
    }

    pub fn conv_state_mut(&mut self, layer: usize) -> &mut [f32] {
        let start = layer * self.conv_channels * self.conv_kernel;
        &mut self.conv[start..start + self.conv_channels * self.conv_kernel]
    }

    pub fn reset(&mut self) {
        self.recurrent.fill(0.0);
        self.conv.fill(0.0);
    }

    pub fn total_bytes(&self) -> usize {
        (self.recurrent.len() + self.conv.len()) * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_slices_match_contract() {
        let mut cache = LinearAttnCache::new(2, 4, 8, 16, 32, 3);
        assert_eq!(cache.recurrent.len(), 2 * 4 * 8 * 16);
        assert_eq!(cache.conv.len(), 2 * 32 * 3);

        cache.recurrent_state_mut(1).copy_from_slice(&vec![7.0; 4 * 8 * 16]);
        cache.conv_state_mut(0).copy_from_slice(&vec![3.0; 32 * 3]);
        assert!(cache.recurrent_state(0).iter().all(|&x| x == 0.0));
        assert!(cache.recurrent_state(1).iter().all(|&x| x == 7.0));
        assert!(cache.conv_state(1).iter().all(|&x| x == 0.0));
        assert!(cache.conv_state(0).iter().all(|&x| x == 3.0));

        cache.reset();
        assert!(cache.recurrent.iter().all(|&x| x == 0.0));
        assert!(cache.conv.iter().all(|&x| x == 0.0));
    }
}
