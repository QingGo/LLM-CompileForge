//! KV-cache block with tensor storage.
//!
//! Provides [`KVCacheBlock`] which owns the actual key/value tensor data
//! for a physical cache block.  The [`BlockManager`](crate::cache::block::BlockManager)
//! uses this instead of bare [`Block`](crate::cache::block::Block) when the user
//! creates it via [`new_with_cache`](crate::cache::block::BlockManager::new_with_cache).
//!
//! For cache policy types, see [`crate::cache::policy`].

/// A physical KV cache block that owns its tensor data.
#[derive(Debug, Clone)]
pub struct KVCacheBlock {
    pub block_id: usize,
    pub key_cache: Vec<f32>,
    pub value_cache: Vec<f32>,
    pub ref_count: usize,
}

impl KVCacheBlock {
    pub fn new(
        block_id: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        let cache_size = block_size * num_kv_heads * head_dim;
        Self {
            block_id,
            key_cache: vec![0.0f32; cache_size],
            value_cache: vec![0.0f32; cache_size],
            ref_count: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_block_create() {
        let block = KVCacheBlock::new(7, 16, 8, 64);
        let expected = 16 * 8 * 64;
        assert_eq!(block.block_id, 7);
        assert_eq!(block.key_cache.len(), expected);
        assert_eq!(block.value_cache.len(), expected);
        assert!(block.key_cache.iter().all(|&x| x == 0.0));
        assert!(block.value_cache.iter().all(|&x| x == 0.0));
        assert_eq!(block.ref_count, 0);
    }

    #[test]
    fn test_kv_cache_block_sizes() {
        let block = KVCacheBlock::new(0, 16, 12, 64);
        let expected = 16 * 12 * 64;
        assert_eq!(block.key_cache.len(), expected);
        assert_eq!(block.value_cache.len(), expected);
    }

    #[test]
    fn test_kv_cache_block_different_sizes() {
        let block = KVCacheBlock::new(42, 32, 4, 128);
        let expected = 32 * 4 * 128;
        assert_eq!(block.key_cache.len(), expected);
        assert_eq!(block.value_cache.len(), expected);
    }
}
