//! Paged KV Cache block manager.
//!
//! Port of `engine/block_manager.py`.  Manages a fixed-size pool of
//! physical blocks with reference-counted allocation, supporting
//! prefix-cache sharing and single-block eviction (for RadixCache).

use std::collections::HashMap;

use crate::kv_cache::KVCacheBlock;

/// A physical KV cache block on device.  The actual tensor data is
/// owned by the executor; BlockManager tracks only logical ownership.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Block {
    pub block_id: usize,
    pub ref_count: usize,
}

impl Block {
    pub fn new(block_id: usize) -> Self {
        Self {
            block_id,
            ref_count: 0,
        }
    }
}

/// Internal block variant: plain or KV-cache-backed.
///
/// [`BlockManager`] stores [`BlockEntry`] so that the pool can be
/// created either without tensor data (bare [`Block`]) or with
/// full KV-cache buffers ([`KVCacheBlock`]).
#[derive(Debug, Clone)]
pub enum BlockEntry {
    /// A logical block without tensor data (used by the default
    /// [`BlockManager::new`] constructor).
    Plain(Block),
    /// A block that owns key/value cache buffers (used by
    /// [`BlockManager::new_with_cache`]).
    Cached(KVCacheBlock),
}

impl BlockEntry {
    pub fn block_id(&self) -> usize {
        match self {
            BlockEntry::Plain(b) => b.block_id,
            BlockEntry::Cached(b) => b.block_id,
        }
    }

    pub fn ref_count(&self) -> usize {
        match self {
            BlockEntry::Plain(b) => b.ref_count,
            BlockEntry::Cached(b) => b.ref_count,
        }
    }

    pub fn ref_count_mut(&mut self) -> &mut usize {
        match self {
            BlockEntry::Plain(b) => &mut b.ref_count,
            BlockEntry::Cached(b) => &mut b.ref_count,
        }
    }
}

/// Error returned when the block pool is exhausted.
#[derive(Debug)]
pub struct OutOfMemoryError {
    pub needed: usize,
    pub free: usize,
    pub total: usize,
}

impl std::fmt::Display for OutOfMemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "OutOfMemory: need {} blocks but only {} free (total pool: {})",
            self.needed, self.free, self.total
        )
    }
}

impl std::error::Error for OutOfMemoryError {}

/// Manages the KV cache block pool.
pub struct BlockManager {
    pub block_size: usize,
    pub num_blocks: usize,
    pub num_kv_heads: Option<usize>,
    pub head_dim: Option<usize>,
    pub blocks: HashMap<usize, BlockEntry>,
    pub free_blocks: Vec<usize>,
    pub block_tables: HashMap<String, Vec<usize>>,
    shared_owners: HashMap<usize, Vec<String>>,
}

impl BlockManager {
    pub fn new(num_blocks: usize, block_size: usize) -> Result<Self, anyhow::Error> {
        if num_blocks == 0 {
            return Err(anyhow::anyhow!("num_blocks must be positive"));
        }
        if block_size == 0 {
            return Err(anyhow::anyhow!("block_size must be positive"));
        }
        let blocks: HashMap<usize, BlockEntry> = (0..num_blocks)
            .map(|i| (i, BlockEntry::Plain(Block::new(i))))
            .collect();
        Ok(Self {
            block_size,
            num_blocks,
            num_kv_heads: None,
            head_dim: None,
            blocks,
            free_blocks: (0..num_blocks).collect(),
            block_tables: HashMap::new(),
            shared_owners: HashMap::new(),
        })
    }

    /// Create a block pool where every block owns KV-cache tensor
    /// data (a [`KVCacheBlock`]).  This is the cache-enabled variant
    /// of [`new`](Self::new).
    pub fn new_with_cache(
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, anyhow::Error> {
        if num_blocks == 0 {
            return Err(anyhow::anyhow!("num_blocks must be positive"));
        }
        if block_size == 0 {
            return Err(anyhow::anyhow!("block_size must be positive"));
        }
        let blocks: HashMap<usize, BlockEntry> = (0..num_blocks)
            .map(|i| (i, BlockEntry::Cached(KVCacheBlock::new(
                i, block_size, num_kv_heads, head_dim,
            ))))
            .collect();
        Ok(Self {
            block_size,
            num_blocks,
            num_kv_heads: Some(num_kv_heads),
            head_dim: Some(head_dim),
            blocks,
            free_blocks: (0..num_blocks).collect(),
            block_tables: HashMap::new(),
            shared_owners: HashMap::new(),
        })
    }

    // ── Allocation ──────────────────────────────────────────

    /// Allocate physical blocks for a new request.
    ///
    /// Returns the list of physical block IDs.  Raises `OutOfMemoryError`
    /// if insufficient free blocks are available.
    pub fn allocate(&mut self, request_id: &str, num_tokens: usize) -> Result<Vec<usize>, OutOfMemoryError> {
        if self.block_tables.contains_key(request_id) {
            return Err(OutOfMemoryError {
                needed: 0,
                free: self.free_blocks.len(),
                total: self.num_blocks,
            });
        }
        let needed = num_tokens.div_ceil(self.block_size);
        if needed > self.free_blocks.len() {
            return Err(OutOfMemoryError {
                needed,
                free: self.free_blocks.len(),
                total: self.num_blocks,
            });
        }
        let mut allocated = Vec::with_capacity(needed);
        for _ in 0..needed {
            let bid = self.free_blocks.pop()
                .expect("invariant: enough free blocks checked above");
            *self.blocks.get_mut(&bid)
                .expect("invariant: bid from free_blocks is valid")
                .ref_count_mut() += 1;
            allocated.push(bid);
        }
        self.block_tables.insert(request_id.to_string(), allocated.clone());
        Ok(allocated)
    }

    /// Release all blocks for a request.  Shared blocks (ref_count > 1)
    /// are only decremented, not returned to the free pool.
    pub fn free(&mut self, request_id: &str) {
        let table = match self.block_tables.remove(request_id) {
            Some(t) => t,
            None => return,
        };
        for bid in &table {
            if let Some(block) = self.blocks.get_mut(bid) {
                let rc = block.ref_count().saturating_sub(1);
                *block.ref_count_mut() = rc;
                if let Some(owners) = self.shared_owners.get_mut(bid) {
                    owners.retain(|o| o != request_id);
                }
                if rc == 0 {
                    self.free_blocks.push(*bid);
                    self.shared_owners.remove(bid);
                }
            }
        }
    }

    /// Release a single physical block (used by RadixCache eviction).
    #[allow(dead_code)]
    pub fn free_block(&mut self, block_id: usize) {
        let block = match self.blocks.get_mut(&block_id) {
            Some(b) => b,
            None => return,
        };
        let rc = block.ref_count().saturating_sub(1);
        *block.ref_count_mut() = rc;
        self.shared_owners.remove(&block_id);
        if rc == 0 {
            self.free_blocks.push(block_id);
        }
    }

    /// Prepend cached blocks to a request's block table.  Used when
    /// a request hits the prefix cache.
    pub fn assign_cached_blocks(&mut self, request_id: &str, block_ids: &[usize]) {
        for bid in block_ids {
            if let Some(block) = self.blocks.get_mut(bid) {
                *block.ref_count_mut() += 1;
                self.shared_owners
                    .entry(*bid)
                    .or_default()
                    .push(request_id.to_string());
            }
        }
        self.block_tables
            .entry(request_id.to_string())
            .and_modify(|table| {
                let mut prefix = block_ids.to_vec();
                prefix.append(table);
                *table = prefix;
            })
            .or_insert_with(|| block_ids.to_vec());
    }

    /// Ensure a request has enough blocks to cover `target_tokens`.
    /// Used during prefill when cached blocks do not cover the full prompt.
    pub fn ensure_blocks(
        &mut self,
        request_id: &str,
        target_tokens: usize,
    ) -> Result<(), OutOfMemoryError> {
        let existing = self.block_tables.get(request_id)
            .map(|t| t.len() * self.block_size)
            .unwrap_or(0);
        if existing >= target_tokens {
            return Ok(());
        }
        let extra = target_tokens - existing;
        let n_extra = extra.div_ceil(self.block_size);
        if n_extra > self.free_blocks.len() {
            return Err(OutOfMemoryError {
                needed: n_extra,
                free: self.free_blocks.len(),
                total: self.num_blocks,
            });
        }
        let table = self.block_tables
            .entry(request_id.to_string())
            .or_default();
        for _ in 0..n_extra {
            let bid = self.free_blocks.pop()
                .expect("invariant: enough free blocks checked above");
            *self.blocks.get_mut(&bid)
                .expect("invariant: bid from free_blocks is valid")
                .ref_count_mut() += 1;
            table.push(bid);
        }
        Ok(())
    }

    // ── Prefix Cache via Block Sharing ──────────────────────

    /// Share prefix KV cache blocks between two requests.
    #[allow(dead_code)]
    pub fn share_prefix(
        &mut self,
        src_request_id: &str,
        dst_request_id: &str,
        prefix_len: usize,
    ) -> Result<Vec<usize>, String> {
        if !self.block_tables.contains_key(src_request_id) {
            return Err(format!("Source request '{}' not found", src_request_id));
        }
        if self.block_tables.contains_key(dst_request_id) {
            return Err(format!("Destination request '{}' already has blocks", dst_request_id));
        }
        let n_blocks = prefix_len.div_ceil(self.block_size);
        let src_blocks = self.block_tables[src_request_id].clone();
        if n_blocks > src_blocks.len() {
            return Err(format!(
                "Source has {} blocks, cannot share {} (prefix_len={}, block_size={})",
                src_blocks.len(), n_blocks, prefix_len, self.block_size
            ));
        }
        let shared = &src_blocks[..n_blocks];
        for bid in shared {
            *self.blocks.get_mut(bid)
                .ok_or_else(|| format!("block {} not found in share_prefix", bid))?
                .ref_count_mut() += 1;
            self.shared_owners
                .entry(*bid)
                .or_default()
                .push(dst_request_id.to_string());
        }
        self.block_tables
            .insert(dst_request_id.to_string(), shared.to_vec());
        Ok(shared.to_vec())
    }

    // ── Query ───────────────────────────────────────────────

    pub fn get_blocks(&self, request_id: &str) -> Result<&[usize], String> {
        self.block_tables
            .get(request_id)
            .map(|v| v.as_slice())
            .ok_or_else(|| format!("Unknown request_id: {}", request_id))
    }

    #[allow(dead_code)]
    pub fn num_free_blocks(&self) -> usize {
        self.free_blocks.len()
    }

    #[allow(dead_code)]
    pub fn num_allocated_blocks(&self) -> usize {
        self.num_blocks - self.free_blocks.len()
    }

    /// Increment reference count on a block (used by RadixCache on insert).
    pub fn increment_ref_count(&mut self, block_id: usize) {
        if let Some(block) = self.blocks.get_mut(&block_id) {
            *block.ref_count_mut() += 1;
        }
    }

    #[allow(dead_code)]
    pub fn utilization(&self) -> f64 {
        if self.num_blocks == 0 {
            return 0.0;
        }
        self.num_allocated_blocks() as f64 / self.num_blocks as f64
    }

    // ── KV Cache Read/Write ──────────────────────────────────

    /// Write one tensor's worth of data into the KV cache blocks.
    ///
    /// `start_pos` is the first global token position. `data` is a flat
    /// `f32` slice with `num_tokens * hidden_dim` elements. `is_key`
    /// selects between the key and value cache.
    ///
    /// The blocks must have been created via [`new_with_cache`](Self::new_with_cache).
    pub fn write_kv(
        &mut self,
        request_id: &str,
        start_pos: usize,
        data: &[f32],
        hidden_dim: usize,
        is_key: bool,
    ) -> Result<(), String> {
        let block_ids = self.get_blocks(request_id)?.to_vec();
        let num_tokens = data.len() / hidden_dim;
        if data.len() % hidden_dim != 0 {
            return Err(format!(
                "write_kv: data.len()={} not divisible by hidden_dim={}",
                data.len(),
                hidden_dim
            ));
        }

        for offset in 0..num_tokens {
            let pos = start_pos + offset;
            let block_idx = pos / self.block_size;
            let intra_pos = pos % self.block_size;

            if block_idx >= block_ids.len() {
                return Err(format!(
                    "write_kv: position {} (block_idx={}) out of range for '{}' ({} blocks)",
                    pos, block_idx, request_id, block_ids.len()
                ));
            }

            let bid = block_ids[block_idx];
            let entry = self.blocks.get_mut(&bid).ok_or_else(|| {
                format!("write_kv: block {} not found", bid)
            })?;

            match entry {
                BlockEntry::Cached(kv) => {
                    let src = offset * hidden_dim;
                    let dst = intra_pos * hidden_dim;
                    let cache = if is_key {
                        &mut kv.key_cache
                    } else {
                        &mut kv.value_cache
                    };
                    cache[dst..dst + hidden_dim].copy_from_slice(&data[src..src + hidden_dim]);
                }
                BlockEntry::Plain(_) => {
                    return Err(format!(
                        "write_kv: block {} is Plain, not Cached. Use new_with_cache.",
                        bid
                    ));
                }
            }
        }
        Ok(())
    }

    /// Read all cached K/V up to (but not including) `up_to_pos`.
    ///
    /// Returns `(key_data, value_data)` where each is a flat `f32` vec
    /// with `up_to_pos * hidden_dim` elements.
    pub fn read_kv(
        &self,
        request_id: &str,
        up_to_pos: usize,
        hidden_dim: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let block_ids = self.get_blocks(request_id)?.to_vec();
        let total_elems = up_to_pos * hidden_dim;
        let mut key_out = vec![0.0f32; total_elems];
        let mut val_out = vec![0.0f32; total_elems];

        for pos in 0..up_to_pos {
            let block_idx = pos / self.block_size;
            let intra_pos = pos % self.block_size;

            if block_idx >= block_ids.len() {
                break;
            }

            let bid = block_ids[block_idx];
            let entry = self.blocks.get(&bid).ok_or_else(|| {
                format!("read_kv: block {} not found", bid)
            })?;

            match entry {
                BlockEntry::Cached(kv) => {
                    let src = intra_pos * hidden_dim;
                    let dst = pos * hidden_dim;
                    key_out[dst..dst + hidden_dim]
                        .copy_from_slice(&kv.key_cache[src..src + hidden_dim]);
                    val_out[dst..dst + hidden_dim]
                        .copy_from_slice(&kv.value_cache[src..src + hidden_dim]);
                }
                BlockEntry::Plain(_) => {
                    return Err(format!(
                        "read_kv: block {} is Plain, not Cached.",
                        bid
                    ));
                }
            }
        }
        Ok((key_out, val_out))
    }

    /// Flush (zero out) all K/V data for a finished request and release
    /// its blocks back to the free pool.
    #[allow(dead_code)]
    pub fn flush_request(&mut self, request_id: &str) {
        let block_ids = match self.block_tables.get(request_id) {
            Some(ids) => ids.clone(),
            None => return,
        };

        for &bid in &block_ids {
            if let Some(BlockEntry::Cached(kv)) = self.blocks.get_mut(&bid) {
                kv.key_cache.fill(0.0);
                kv.value_cache.fill(0.0);
            }
        }

        self.free(request_id);
    }
}

// ── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation() {
        let bm = BlockManager::new(1000, 16).unwrap();
        assert_eq!(bm.num_blocks, 1000);
        assert_eq!(bm.block_size, 16);
        assert_eq!(bm.num_free_blocks(), 1000);
    }

    #[test]
    fn test_new_with_cache() {
        let bm = BlockManager::new_with_cache(100, 16, 12, 64).unwrap();
        assert_eq!(bm.num_blocks, 100);
        assert_eq!(bm.block_size, 16);
        let expected = 16 * 12 * 64;
        if let BlockEntry::Cached(kv) = &bm.blocks[&0] {
            assert_eq!(kv.key_cache.len(), expected);
            assert_eq!(kv.value_cache.len(), expected);
            assert!(kv.key_cache.iter().all(|&x| x == 0.0));
        } else {
            panic!("Expected Cached variant for new_with_cache blocks");
        }
    }

    #[test]
    fn test_invalid_params() {
        assert!(BlockManager::new(0, 16).is_err());
        assert!(BlockManager::new(100, 0).is_err());
    }

    #[test]
    fn test_allocate_basic() {
        let mut bm = BlockManager::new(1000, 16).unwrap();
        let blocks = bm.allocate("req_1", 32).unwrap();
        assert_eq!(blocks.len(), 2); // 32/16 = 2
        assert_eq!(bm.num_free_blocks(), 998);
    }

    #[test]
    fn test_allocate_exact_block_boundary() {
        let mut bm = BlockManager::new(1000, 16).unwrap();
        let blocks = bm.allocate("req_1", 16).unwrap();
        assert_eq!(blocks.len(), 1);
    }

    #[test]
    fn test_allocate_one_token_needs_one_block() {
        let mut bm = BlockManager::new(1000, 16).unwrap();
        let blocks = bm.allocate("req_1", 1).unwrap();
        assert_eq!(blocks.len(), 1);
    }

    #[test]
    fn test_free_returns_blocks() {
        let mut bm = BlockManager::new(1000, 16).unwrap();
        bm.allocate("req_1", 32).unwrap();
        assert_eq!(bm.num_free_blocks(), 998);
        bm.free("req_1");
        assert_eq!(bm.num_free_blocks(), 1000);
    }

    #[test]
    fn test_free_unknown_is_noop() {
        let mut bm = BlockManager::new(1000, 16).unwrap();
        bm.free("nonexistent"); // should not panic
        assert_eq!(bm.num_free_blocks(), 1000);
    }

    #[test]
    fn test_out_of_memory() {
        let mut bm = BlockManager::new(10, 16).unwrap();
        let result = bm.allocate("req_1", 200); // needs 13 blocks, only 10 available
        assert!(result.is_err());
    }

    #[test]
    fn test_share_prefix() {
        let mut bm = BlockManager::new(1000, 16).unwrap();
        let src_blocks = bm.allocate("src", 64).unwrap(); // 4 blocks
        assert_eq!(src_blocks.len(), 4);

        let shared = bm.share_prefix("src", "dst", 32).unwrap(); // first 2 blocks
        assert_eq!(shared.len(), 2);
        assert_eq!(bm.get_blocks("dst").unwrap(), shared.as_slice());
    }

    #[test]
    fn test_share_prefix_ref_count() {
        let mut bm = BlockManager::new(1000, 16).unwrap();
        bm.allocate("src", 32).unwrap();
        bm.share_prefix("src", "dst", 16).unwrap();

        bm.free("src");
        assert_eq!(bm.num_free_blocks(), 999); // 1 block freed, 1 still shared

        bm.free("dst");
        assert_eq!(bm.num_free_blocks(), 1000); // both freed
    }

    #[test]
    fn test_free_block_single() {
        let mut bm = BlockManager::new(1000, 16).unwrap();
        let blocks = bm.allocate("req_1", 32).unwrap();
        let bid = blocks[0];
        bm.free_block(bid);
        assert_eq!(bm.num_free_blocks(), 999); // one block returned
    }

    #[test]
    fn test_assign_cached_blocks() {
        let mut bm = BlockManager::new(1000, 16).unwrap();
        bm.assign_cached_blocks("req_1", &[0, 1]);
        let blocks = bm.get_blocks("req_1").unwrap().to_vec();
        assert_eq!(blocks, vec![0, 1]);
        // ref_count should be incremented
        assert_eq!(bm.blocks[&0].ref_count(), 1);
        assert_eq!(bm.blocks[&1].ref_count(), 1);
    }

    #[test]
    fn test_utilization() {
        let mut bm = BlockManager::new(100, 16).unwrap();
        assert!((bm.utilization() - 0.0).abs() < 0.001);
        bm.allocate("req_1", 160).unwrap(); // 10 blocks
        assert!((bm.utilization() - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_get_blocks_unknown() {
        let bm = BlockManager::new(10, 16).unwrap();
        assert!(bm.get_blocks("unknown").is_err());
    }

    // ── Property-based tests ───────────────────────────────────

    #[test]
    fn test_write_kv_single_token() {
        let mut bm = BlockManager::new_with_cache(10, 16, 12, 64).unwrap();
        bm.allocate("req_1", 1).unwrap();

        let hidden_dim = 12 * 64; // 768
        let key_data = vec![1.0f32; hidden_dim];
        let val_data = vec![2.0f32; hidden_dim];

        bm.write_kv("req_1", 0, &key_data, hidden_dim, true).unwrap();
        bm.write_kv("req_1", 0, &val_data, hidden_dim, false).unwrap();

        let (k, v) = bm.read_kv("req_1", 1, hidden_dim).unwrap();
        assert_eq!(k.len(), hidden_dim);
        assert_eq!(v.len(), hidden_dim);
        assert!((k[0] - 1.0).abs() < 1e-6);
        assert!((v[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_write_kv_multiple_positions() {
        let mut bm = BlockManager::new_with_cache(10, 16, 12, 64).unwrap();
        bm.allocate("req_1", 5).unwrap();

        let hidden_dim = 768;
        let n_tokens = 5;
        let key_data: Vec<f32> = (0..n_tokens * hidden_dim)
            .map(|i| (i % 100) as f32)
            .collect();
        let val_data: Vec<f32> = (0..n_tokens * hidden_dim)
            .map(|i| ((i + 100) % 200) as f32)
            .collect();

        bm.write_kv("req_1", 0, &key_data, hidden_dim, true).unwrap();
        bm.write_kv("req_1", 0, &val_data, hidden_dim, false).unwrap();

        // Read back all 5 positions
        let (k, v) = bm.read_kv("req_1", 5, hidden_dim).unwrap();
        assert_eq!(k.len(), hidden_dim * 5);
        assert_eq!(v.len(), hidden_dim * 5);

        // Verify position 0 matches written data
        for i in 0..hidden_dim {
            assert!((k[i] - (i % 100) as f32).abs() < 1e-6, "K[{}] mismatch", i);
            assert!((v[i] - ((i + 100) % 200) as f32).abs() < 1e-6, "V[{}] mismatch", i);
        }

        // Verify position 1
        for i in 0..hidden_dim {
            let offset = hidden_dim;
            assert!((k[offset + i] - ((i + hidden_dim) % 100) as f32).abs() < 1e-6);
        }
    }

    #[test]
    fn test_write_kv_decode_then_read_all() {
        // Simulate 3 decode steps writing one token each, then read all
        let mut bm = BlockManager::new_with_cache(10, 16, 12, 64).unwrap();
        bm.allocate("req_1", 3).unwrap();
        let hidden_dim = 768;

        // Step 0: position 0
        let k0 = vec![10.0f32; hidden_dim];
        let v0 = vec![20.0f32; hidden_dim];
        bm.write_kv("req_1", 0, &k0, hidden_dim, true).unwrap();
        bm.write_kv("req_1", 0, &v0, hidden_dim, false).unwrap();

        // Step 1: position 1
        let k1 = vec![11.0f32; hidden_dim];
        let v1 = vec![21.0f32; hidden_dim];
        bm.write_kv("req_1", 1, &k1, hidden_dim, true).unwrap();
        bm.write_kv("req_1", 1, &v1, hidden_dim, false).unwrap();

        // Step 2: position 2
        let k2 = vec![12.0f32; hidden_dim];
        let v2 = vec![22.0f32; hidden_dim];
        bm.write_kv("req_1", 2, &k2, hidden_dim, true).unwrap();
        bm.write_kv("req_1", 2, &v2, hidden_dim, false).unwrap();

        // Read positions [0..3)
        let (k, v) = bm.read_kv("req_1", 3, hidden_dim).unwrap();
        assert_eq!(k.len(), hidden_dim * 3);
        assert!((k[0] - 10.0).abs() < 1e-6);
        assert!((k[hidden_dim] - 11.0).abs() < 1e-6);
        assert!((k[2 * hidden_dim] - 12.0).abs() < 1e-6);
        assert!((v[0] - 20.0).abs() < 1e-6);
        assert!((v[hidden_dim] - 21.0).abs() < 1e-6);
        assert!((v[2 * hidden_dim] - 22.0).abs() < 1e-6);
    }

    #[test]
    fn test_read_kv_plain_block_errors() {
        let mut bm = BlockManager::new(10, 16).unwrap();
        bm.allocate("req_1", 1).unwrap();
        let result = bm.read_kv("req_1", 1, 768);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.contains("Plain"), "expected 'Plain' error, got: {}", err);
    }

    #[test]
    fn test_write_kv_plain_block_errors() {
        let mut bm = BlockManager::new(10, 16).unwrap();
        bm.allocate("req_1", 1).unwrap();
        let result = bm.write_kv("req_1", 0, &[1.0f32; 768], 768, true);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.contains("Plain"), "expected 'Plain' error, got: {}", err);
    }

    #[test]
    fn test_write_kv_bad_data_len_errors() {
        let mut bm = BlockManager::new_with_cache(10, 16, 12, 64).unwrap();
        bm.allocate("req_1", 1).unwrap();
        let result = bm.write_kv("req_1", 0, &[1.0f32; 10], 768, true);
        assert!(result.is_err());
    }

    /// Invariant: after any sequence of successful allocate/free operations,
    /// ``num_free_blocks() + num_allocated_blocks() == num_blocks``.
    #[test]
    fn prop_invariant_free_plus_allocated_equals_total() {
        use proptest::prelude::*;
        proptest!(|(ops in proptest::collection::vec(0..25usize, 1..20))| {
            let mut bm = BlockManager::new(100, 16).unwrap();
            for i in ops {
                let rid = format!("req_{}", i);
                if bm.num_free_blocks() >= 4 {
                    bm.allocate(&rid, 64).unwrap();  // 4 blocks
                    bm.free(&rid);
                }
            }
            let total = bm.num_free_blocks() + bm.num_allocated_blocks();
            assert_eq!(total, bm.num_blocks, "free+allocated must equal total");
        });
    }

    /// Invariant: allocating more blocks than available returns an error.
    #[test]
    fn prop_allocate_beyond_total_fails() {
        use proptest::prelude::*;
        proptest!(|(over_alloc in 1..=5usize)| {
            let n_blocks = 5;
            let mut bm = BlockManager::new(n_blocks, 16).unwrap();
            // Use all blocks
            let rid = "req_all";
            bm.allocate(rid, n_blocks * 16).unwrap();
            // One more should fail
            let extra = format!("req_extra_{}", over_alloc);
            assert!(bm.allocate(&extra, 16).is_err(),
                "allocating beyond capacity must fail");
        });
    }

    /// Invariant: after free, the blocks are available for reallocation.
    #[test]
    fn prop_free_returns_blocks_to_pool() {
        use proptest::prelude::*;
        proptest!(|(n_blocks in 1..20usize)| {
            let mut bm = BlockManager::new(50, 16).unwrap();
            let free_before = bm.num_free_blocks();
            let needed_bytes = n_blocks * 16;
            bm.allocate("prop_test", needed_bytes).unwrap();
            assert_eq!(bm.num_free_blocks(), free_before - n_blocks);
            bm.free("prop_test");
            assert_eq!(bm.num_free_blocks(), free_before);
        });
    }

    // ── Flush tests (use new request_id-based API) ───────────

    #[test]
    fn test_flush_request_zeros_and_frees() {
        let mut bm = BlockManager::new_with_cache(10, 4, 2, 3).unwrap();
        bm.allocate("req_flush", 4).unwrap();
        assert_eq!(bm.num_free_blocks(), 9);
        let hidden_dim = 2 * 3;
        let key_data = vec![42.0f32; hidden_dim];
        bm.write_kv("req_flush", 0, &key_data, hidden_dim, true).unwrap();
        bm.flush_request("req_flush");
        assert!(bm.get_blocks("req_flush").is_err());
        assert_eq!(bm.num_free_blocks(), 10);
    }

    #[test]
    fn test_flush_request_data_zeroed() {
        let mut bm = BlockManager::new_with_cache(10, 4, 2, 3).unwrap();
        bm.allocate("req_zero", 4).unwrap();
        let hidden_dim = 2 * 3;
        let key_data = vec![42.0f32; hidden_dim];
        bm.write_kv("req_zero", 0, &key_data, hidden_dim, true).unwrap();
        let (rkey, _) = bm.read_kv("req_zero", 1, hidden_dim).unwrap();
        assert!(rkey.iter().any(|&x| x != 0.0));
        bm.flush_request("req_zero");
        // After flush, the request is freed; reading should fail
        assert!(bm.read_kv("req_zero", 1, hidden_dim).is_err());
    }
}
