//! Paged KV Cache block manager.
//!
//! Port of `engine/block_manager.py`.  Manages a fixed-size pool of
//! physical blocks with reference-counted allocation, supporting
//! prefix-cache sharing and single-block eviction (for RadixCache).

use std::collections::HashMap;

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
    pub blocks: HashMap<usize, Block>,
    pub free_blocks: Vec<usize>,
    pub block_tables: HashMap<String, Vec<usize>>,
    shared_owners: HashMap<usize, Vec<String>>,
}

impl BlockManager {
    pub fn new(num_blocks: usize, block_size: usize) -> Result<Self, String> {
        if num_blocks == 0 {
            return Err("num_blocks must be positive".into());
        }
        if block_size == 0 {
            return Err("block_size must be positive".into());
        }
        let blocks: HashMap<usize, Block> = (0..num_blocks)
            .map(|i| (i, Block::new(i)))
            .collect();
        Ok(Self {
            block_size,
            num_blocks,
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
            let bid = self.free_blocks.pop().unwrap();
            self.blocks.get_mut(&bid).unwrap().ref_count += 1;
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
            let block = self.blocks.get_mut(bid).unwrap();
            block.ref_count = block.ref_count.saturating_sub(1);
            if let Some(owners) = self.shared_owners.get_mut(bid) {
                owners.retain(|o| o != request_id);
            }
            if block.ref_count == 0 {
                self.free_blocks.push(*bid);
                self.shared_owners.remove(bid);
            }
        }
    }

    /// Release a single physical block (used by RadixCache eviction).
    pub fn free_block(&mut self, block_id: usize) {
        let block = match self.blocks.get_mut(&block_id) {
            Some(b) => b,
            None => return,
        };
        block.ref_count = block.ref_count.saturating_sub(1);
        self.shared_owners.remove(&block_id);
        if block.ref_count == 0 {
            self.free_blocks.push(block_id);
        }
    }

    /// Prepend cached blocks to a request's block table.  Used when
    /// a request hits the prefix cache.
    pub fn assign_cached_blocks(&mut self, request_id: &str, block_ids: &[usize]) {
        for bid in block_ids {
            if let Some(block) = self.blocks.get_mut(bid) {
                block.ref_count += 1;
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
            let bid = self.free_blocks.pop().unwrap();
            self.blocks.get_mut(&bid).unwrap().ref_count += 1;
            table.push(bid);
        }
        Ok(())
    }

    // ── Prefix Cache via Block Sharing ──────────────────────

    /// Share prefix KV cache blocks between two requests.
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
            self.blocks.get_mut(bid).unwrap().ref_count += 1;
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

    pub fn num_free_blocks(&self) -> usize {
        self.free_blocks.len()
    }

    pub fn num_allocated_blocks(&self) -> usize {
        self.num_blocks - self.free_blocks.len()
    }

    /// Increment reference count on a block (used by RadixCache on insert).
    pub fn increment_ref_count(&mut self, block_id: usize) {
        if let Some(block) = self.blocks.get_mut(&block_id) {
            block.ref_count += 1;
        }
    }

    pub fn utilization(&self) -> f64 {
        if self.num_blocks == 0 {
            return 0.0;
        }
        self.num_allocated_blocks() as f64 / self.num_blocks as f64
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
        assert_eq!(bm.blocks[&0].ref_count, 1);
        assert_eq!(bm.blocks[&1].ref_count, 1);
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
}
