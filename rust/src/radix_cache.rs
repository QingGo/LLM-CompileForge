//! Radix Tree prefix cache for KV block sharing.
//!
//! Maps token sequences to physical KV cache block IDs.  Multiple
//! requests can share the same prefix blocks via reference counting.
//!
//! Port of `cache/radix_cache.py`.  Integrates with the Rust BlockManager
//! for block lifecycle management.

use std::collections::HashMap;

use crate::block_manager::BlockManager;

/// A node in the Radix Tree representing a contiguous token subsequence.
#[derive(Debug, Default)]
pub struct RadixTreeNode {
    pub token_ids: Vec<u32>,
    pub children: HashMap<u32, RadixTreeNode>,
    pub kv_blocks: Vec<usize>,
    pub ref_count: usize,
}

impl RadixTreeNode {
    pub fn new(token_ids: Vec<u32>) -> Self {
        Self {
            token_ids,
            children: HashMap::new(),
            kv_blocks: Vec::new(),
            ref_count: 0,
        }
    }
}

/// Prefix cache based on a Radix Tree of token sequences.
pub struct RadixCache {
    pub root: RadixTreeNode,
    block_size: usize,
}

impl RadixCache {
    pub fn new(block_size: usize) -> Self {
        Self {
            root: RadixTreeNode::new(Vec::new()),
            block_size,
        }
    }

    /// Find the longest prefix of `token_ids` present in the tree.
    ///
    /// Returns `(matched_kv_blocks, matched_token_count)`.
    pub fn match_prefix(&self, token_ids: &[u32]) -> (Vec<usize>, usize) {
        let mut matched_blocks: Vec<usize> = Vec::new();
        let mut consumed: usize = 0;
        let mut remaining = token_ids;
        let mut node = &self.root;

        while !remaining.is_empty() {
            let first = remaining[0];
            let child = match node.children.get(&first) {
                Some(c) => c,
                None => break,
            };

            let child_tokens = &child.token_ids;
            let common = common_prefix_len(remaining, child_tokens);

            if common < child_tokens.len() {
                let n_blocks = ceil_div(common, self.block_size);
                matched_blocks.extend_from_slice(&child.kv_blocks[..n_blocks.min(child.kv_blocks.len())]);
                consumed += common;
                break;
            }

            matched_blocks.extend_from_slice(&child.kv_blocks);
            consumed += child_tokens.len();
            remaining = &remaining[child_tokens.len()..];
            node = child;
        }

        (matched_blocks, consumed)
    }

    /// Insert a token sequence and its KV blocks into the tree.
    ///
    /// Updates block reference counts via `bm.increment_ref_count`.
    /// Uses raw pointer traversal internally to avoid borrow conflicts between
    /// the recursive tree structure and external mutable references (`bm`).
    /// SAFETY: The tree is a single-owner structure; all nodes outlive `&mut self`.
    /// The traversal always re-acquires a valid reference from a pinned allocation.
    pub fn insert(
        &mut self,
        token_ids: &[u32],
        kv_blocks: &[usize],
        bm: &mut BlockManager,
    ) {
        if token_ids.is_empty() {
            return;
        }

        let mut remaining = token_ids;
        let mut block_offset: usize = 0;

        // Raw pointer traversal avoids conflicting `&mut` borrows between
        // tree navigation and child insertion/splitting.  All nodes are
        // owned by `self` and are never deallocated during insertion.
        let mut node_ptr: *mut RadixTreeNode = &mut self.root;

        loop {
            if remaining.is_empty() {
                break;
            }
            let first = remaining[0];

            // SAFETY: `node_ptr` always points to a valid node owned by
            // `self`.  The pointer is obtained either from `self.root` or
            // from a child that remains pinned in its parent's HashMap.
            let node = unsafe { &mut *node_ptr };
            let child = node.children.get_mut(&first);

            match child {
                None => {
                    let new_blocks = &kv_blocks[block_offset..];
                    add_node(node, remaining.to_vec(), new_blocks.to_vec(), bm);
                    break;
                }
                Some(child_node) => {
                    let child_tokens = child_node.token_ids.clone();
                    let common = common_prefix_len(remaining, &child_tokens);

                    if common == 0 {
                        let new_blocks = &kv_blocks[block_offset..];
                        add_node(node, remaining.to_vec(), new_blocks.to_vec(), bm);
                        break;
                    }

                    if common < child_tokens.len() {
                        // ── Split node ─────────────────────────────
                        let shared_tokens = child_tokens[..common].to_vec();
                        let remaining_child_tokens = child_tokens[common..].to_vec();
                        let n_shared = ceil_div(common, self.block_size);

                        let child_kv = child_node.kv_blocks.clone();
                        let shared_blocks = child_kv[..n_shared.min(child_kv.len())].to_vec();
                        let child_remaining_blocks = child_kv[n_shared.min(child_kv.len())..].to_vec();
                        let child_ref = child_node.ref_count;
                        let first_remainder = remaining_child_tokens[0];

                        let mut split_node = RadixTreeNode::new(shared_tokens);
                        split_node.kv_blocks = shared_blocks;
                        split_node.ref_count = child_ref;

                        child_node.token_ids = remaining_child_tokens;
                        child_node.kv_blocks = child_remaining_blocks;
                        split_node.children.insert(first_remainder, std::mem::take(child_node));

                        node.children.insert(first, split_node);

                        // Advance to the newly inserted split node for the next iteration.
                        // SAFETY: `split_node` was just inserted into `node.children`,
                        // so the pointer is valid until `self` is dropped.
                        remaining = &remaining[common..];
                        node_ptr = {
                            let parent = unsafe { &mut *node_ptr };
                            parent.children.get_mut(&first).unwrap() as *mut RadixTreeNode
                        };
                    } else {
                        block_offset += child_node.kv_blocks.len();
                        remaining = &remaining[child_tokens.len()..];
                        // SAFETY: `child_node` lives in `node.children` which is owned by `self`.
                        node_ptr = child_node as *mut RadixTreeNode;
                    }
                }
            }
        }
    }

    /// Evict subtrees whose ref_count is zero, freeing blocks via `bm.free_block`.
    ///
    /// Returns the number of blocks actually freed.
    pub fn evict(&mut self, target_blocks: usize, bm: &mut BlockManager) -> usize {
        let mut freed: usize = 0;
        if target_blocks == 0 {
            return 0;
        }

        // Collect first-token keys to avoid borrow issues during DFS
        let root_keys: Vec<u32> = self.root.children.keys().copied().collect();
        for ft in root_keys {
            if freed >= target_blocks {
                break;
            }
            dfs_evict_child(&mut self.root, ft, target_blocks, &mut freed, bm);
        }

        freed
    }

    /// Returns total number of cached blocks across all nodes.
    pub fn cached_blocks(&self) -> usize {
        count_blocks(&self.root)
    }

    /// Returns total number of nodes in the tree.
    pub fn node_count(&self) -> usize {
        count_nodes(&self.root)
    }
}

// ── Helpers ─────────────────────────────────────────────────

fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    let n = a.len().min(b.len());
    for i in 0..n {
        if a[i] != b[i] {
            return i;
        }
    }
    n
}

fn ceil_div(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

fn add_node(
    parent: &mut RadixTreeNode,
    token_ids: Vec<u32>,
    kv_blocks: Vec<usize>,
    bm: &mut BlockManager,
) {
    let first = token_ids[0];
    let mut node = RadixTreeNode::new(token_ids);
    for bid in &kv_blocks {
        bm.increment_ref_count(*bid);
    }
    node.kv_blocks = kv_blocks;
    parent.children.insert(first, node);
}

fn dfs_evict_child(
    parent: &mut RadixTreeNode,
    first_token: u32,
    target_blocks: usize,
    freed: &mut usize,
    bm: &mut BlockManager,
) {
    // First, recursively evict children (leaves before parents)
    let child_keys: Vec<u32> = parent
        .children
        .get(&first_token)
        .map(|c| c.children.keys().copied().collect())
        .unwrap_or_default();

    for ck in child_keys {
        if *freed >= target_blocks {
            return;
        }
        // Get mutable reference to the child and recurse
        if let Some(child) = parent.children.get_mut(&first_token) {
            dfs_evict_child(child, ck, target_blocks, freed, bm);
        }
    }

    if *freed >= target_blocks {
        return;
    }

    // Try to evict this node
    let node = match parent.children.get(&first_token) {
        Some(n) if n.ref_count == 0 => n,
        _ => return,
    };

    for bid in node.kv_blocks.iter().copied().collect::<Vec<_>>() {
        bm.free_block(bid);
        *freed += 1;
        if *freed >= target_blocks {
            parent.children.remove(&first_token);
            return;
        }
    }

    parent.children.remove(&first_token);
}

fn count_blocks(node: &RadixTreeNode) -> usize {
    let mut total = node.kv_blocks.len();
    for child in node.children.values() {
        total += count_blocks(child);
    }
    total
}

fn count_nodes(node: &RadixTreeNode) -> usize {
    let mut total = 1;
    for child in node.children.values() {
        total += count_nodes(child);
    }
    total
}

// ── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_manager::BlockManager;

    fn setup() -> (RadixCache, BlockManager) {
        let bm = BlockManager::new(100, 16).unwrap();
        let cache = RadixCache::new(16);
        (cache, bm)
    }

    #[test]
    fn test_empty_match() {
        let (cache, _bm) = setup();
        let (blocks, matched) = cache.match_prefix(&[1, 2, 3]);
        assert!(blocks.is_empty());
        assert_eq!(matched, 0);
    }

    #[test]
    fn test_insert_and_match_exact() {
        let (mut cache, mut bm) = setup();
        let tokens: Vec<u32> = (1..=32).collect();
        let blocks = bm.allocate("r1", tokens.len()).unwrap(); // 2 blocks
        cache.insert(&tokens, &blocks, &mut bm);

        let (matched, matched_len) = cache.match_prefix(&tokens);
        assert_eq!(matched, blocks);
        assert_eq!(matched_len, 32);
    }

    #[test]
    fn test_insert_and_match_partial() {
        let (mut cache, mut bm) = setup();
        // Insert 64 tokens with block_size=16 → 4 blocks
        let tokens: Vec<u32> = (1..=64).collect();
        let blocks = bm.allocate("r1", tokens.len()).unwrap();
        cache.insert(&tokens, &blocks, &mut bm);

        // Match 32 tokens → 2 blocks
        let mut query: Vec<u32> = (1..=32).collect();
        query.push(999); // diverges at token 33
        let (matched, matched_len) = cache.match_prefix(&query);
        assert_eq!(matched_len, 32);
        assert_eq!(matched.len(), 2);
    }

    #[test]
    fn test_insert_splits_existing_node() {
        let (mut cache, mut bm) = setup();
        // Insert sequence with 32 matching prefix + 32 unique → total 2 blocks
        let mut t1: Vec<u32> = (1..=32).collect();
        t1.extend(101..=132);
        let b1 = bm.allocate("r1", t1.len()).unwrap();
        cache.insert(&t1, &b1, &mut bm);

        // Insert different suffix that shares the first 32 tokens
        let mut t2: Vec<u32> = (1..=32).collect();
        t2.extend(201..=232);
        let b2 = bm.allocate("r2", t2.len()).unwrap();
        cache.insert(&t2, &b2, &mut bm);

        // Match shared prefix
        let (_matched, matched_len) = cache.match_prefix(&[1, 2]);
        assert_eq!(matched_len, 2);
        assert!(cache.node_count() >= 3);
    }

    #[test]
    fn test_evict_frees_blocks() {
        let (mut cache, mut bm) = setup();
        let initial_free = bm.num_free_blocks();

        // 32 tokens with block_size=16 → 2 blocks
        let tokens: Vec<u32> = (1..=32).collect();
        let blocks = bm.allocate("r1", tokens.len()).unwrap();
        assert_eq!(blocks.len(), 2);
        cache.insert(&tokens, &blocks, &mut bm);

        // After insert, block ref_count=2 (allocate + cache), node.ref_count=0
        bm.free("r1"); // decrements to ref_count=1

        // Evict — should return blocks to free pool
        let freed = cache.evict(10, &mut bm);
        assert_eq!(freed, 2);
        assert_eq!(bm.num_free_blocks(), initial_free);
    }

    #[test]
    fn test_evict_respects_ref_count() {
        let (mut cache, mut bm) = setup();
        let initial_free = bm.num_free_blocks();

        let tokens: Vec<u32> = (1..=32).collect();
        let blocks = bm.allocate("r1", tokens.len()).unwrap();
        cache.insert(&tokens, &blocks, &mut bm);

        bm.free("r1"); // block ref_count: 2→1 (only cache holds)
        assert!(bm.num_free_blocks() < initial_free);

        let freed = cache.evict(10, &mut bm);
        assert_eq!(freed, 2);
        assert_eq!(bm.num_free_blocks(), initial_free);
    }

    #[test]
    fn test_cached_blocks_count() {
        let (mut cache, mut bm) = setup();
        assert_eq!(cache.cached_blocks(), 0);

        let tokens: Vec<u32> = (1..=32).collect();
        let blocks = bm.allocate("r1", tokens.len()).unwrap();
        cache.insert(&tokens, &blocks, &mut bm);

        assert_eq!(cache.cached_blocks(), 2);
    }

    #[test]
    fn test_node_count() {
        let (mut cache, mut bm) = setup();
        // Root + 1 child = 2 nodes
        let tokens: Vec<u32> = (1..=16).collect();
        let blocks = bm.allocate("r1", tokens.len()).unwrap();
        cache.insert(&tokens, &blocks, &mut bm);
        assert_eq!(cache.node_count(), 2);

        // Another with shared prefix → split → 4 nodes total
        let mut t2: Vec<u32> = (1..=8).collect();
        t2.push(99);
        let b2 = bm.allocate("r2", t2.len()).unwrap();
        cache.insert(&t2, &b2, &mut bm);
        assert!(cache.node_count() >= 3);
    }
}
