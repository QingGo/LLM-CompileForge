//! Python bindings for the LLM-ServeForge Rust runtime.
//!
//! Enabled with ``--features python-bindings``.

pub mod model;
pub mod engine;
pub mod check;
pub mod hal;
pub mod debug;
pub mod cache;
pub mod kv_cache;

#[cfg(test)]
#[path = "tests/integration_tests.rs"]
mod integration_tests;

#[cfg(test)]
#[path = "tests/kv_cache_tests.rs"]
mod kv_cache_tests;

#[cfg(test)]
#[path = "tests/contract_determinism_tests.rs"]
mod contract_determinism_tests;

#[cfg(test)]
#[path = "tests/contract_precision_tests.rs"]
mod contract_precision_tests;

#[cfg(test)]
#[path = "tests/golden_tests.rs"]
mod golden_tests;

#[cfg(test)]
#[path = "tests/golden_reader.rs"]
mod golden_reader;

#[cfg(test)]
#[path = "tests/internal_e2e_tests.rs"]
mod internal_e2e_tests;

#[cfg(test)]
#[path = "tests/precision_contract_tests.rs"]
mod precision_contract_tests;

#[cfg(test)]
#[path = "tests/contract_weight_tests.rs"]
mod contract_weight_tests;

#[cfg(test)]
#[path = "tests/e2e_tests.rs"]
mod e2e_tests;

#[cfg(test)]
#[path = "tests/function_output_tests.rs"]
mod function_output_tests;

#[cfg(test)]
#[path = "tests/diagnostic_tests.rs"]
mod diagnostic_tests;

#[cfg(test)]
#[path = "tests/runner_consistency_tests.rs"]
mod runner_consistency_tests;

#[cfg(test)]
#[path = "tests/sdpa_decode_shape_tests.rs"]
mod sdpa_decode_shape_tests;

#[cfg(test)]
#[path = "tests/dylib_lock.rs"]
pub mod dylib_lock;

#[cfg(feature = "python-bindings")]
mod py_bindings {
    use crate::cache::block::BlockManager;
    use crate::cache::radix::RadixCache;
    use crate::engine::scheduler::Scheduler;
    use crate::engine::types::{Batch, PrefixCacheHit};
    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PyList};

    #[pyclass]
    pub struct PyBlockManager {
        inner: BlockManager,
    }

    #[pymethods]
    impl PyBlockManager {
        #[new]
        fn new(num_blocks: usize, block_size: usize) -> PyResult<Self> {
            let inner = BlockManager::new(num_blocks, block_size)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            Ok(Self { inner })
        }

        fn allocate(
            &mut self,
            request_id: &str,
            num_tokens: usize,
        ) -> PyResult<Vec<usize>> {
            self.inner
                .allocate(request_id, num_tokens)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        }

        fn free(&mut self, request_id: &str) {
            self.inner.free(request_id);
        }

        fn free_block(&mut self, block_id: usize) {
            self.inner.free_block(block_id);
        }

        fn get_blocks(&self, request_id: &str) -> PyResult<Vec<usize>> {
            self.inner
                .get_blocks(request_id)
                .map(|v| v.to_vec())
                .map_err(|e| pyo3::exceptions::PyKeyError::new_err(e.to_string()))
        }

        fn assign_cached_blocks(
            &mut self,
            request_id: &str,
            block_ids: Vec<usize>,
        ) {
            self.inner.assign_cached_blocks(request_id, &block_ids);
        }

        fn ensure_blocks(
            &mut self,
            request_id: &str,
            target_tokens: usize,
        ) -> PyResult<()> {
            self.inner
                .ensure_blocks(request_id, target_tokens)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        }

        fn share_prefix(
            &mut self,
            src_request_id: &str,
            dst_request_id: &str,
            prefix_len: usize,
        ) -> PyResult<Vec<usize>> {
            self.inner
                .share_prefix(src_request_id, dst_request_id, prefix_len)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
        }

        #[getter]
        fn block_size(&self) -> usize {
            self.inner.block_size
        }

        #[getter]
        fn num_blocks(&self) -> usize {
            self.inner.num_blocks
        }

        fn num_free_blocks(&self) -> usize {
            self.inner.num_free_blocks()
        }

        fn utilization(&self) -> f64 {
            self.inner.utilization()
        }

        fn increment_ref_count(&mut self, block_id: usize) {
            self.inner.increment_ref_count(block_id);
        }
    }

    #[pyclass]
    pub struct PyScheduler {
        inner: Scheduler,
    }

    #[pymethods]
    impl PyScheduler {
        #[new]
        #[pyo3(signature = (max_batch_size, max_tokens_per_step, chunk_size, use_kv_cache=false))]
        fn new(
            max_batch_size: usize,
            max_tokens_per_step: usize,
            chunk_size: usize,
            use_kv_cache: bool,
        ) -> PyResult<Self> {
            let inner = Scheduler::new(max_batch_size, max_tokens_per_step, chunk_size, use_kv_cache)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            Ok(Self { inner })
        }

        fn add_request(
            &mut self,
            prompt_tokens: Vec<u32>,
            priority: i32,
            max_tokens: usize,
            stop_token_ids: Vec<u32>,
            request_id: Option<String>,
        ) -> String {
            self.inner.add_request(
                prompt_tokens,
                priority,
                max_tokens,
                stop_token_ids,
                request_id,
            )
        }

        fn schedule(
            &mut self,
            py: Python<'_>,
            block_manager: &mut PyBlockManager,
            cache_hits: Vec<(String, Vec<usize>, usize)>,
        ) -> PyResult<PyObject> {
            let hits: Vec<PrefixCacheHit> = cache_hits
                .into_iter()
                .map(|(rid, blocks, tokens)| PrefixCacheHit {
                    request_id: rid,
                    matched_blocks: blocks,
                    matched_tokens: tokens,
                })
                .collect();

            let batch: Batch = self.inner.schedule(&mut block_manager.inner, &hits);

            let result = PyDict::new(py);
            let py_requests = PyList::empty(py);

            for req in &batch.requests {
                let d = PyDict::new(py);
                d.set_item("request_id", &req.request_id)?;
                d.set_item("input_ids", req.input_ids.clone())?;
                d.set_item("positions", req.positions.clone())?;
                d.set_item("state", req.state.to_string())?;
                d.set_item("block_table", req.block_table.clone())?;
                d.set_item("n_tokens", req.n_tokens)?;
                py_requests.append(d)?;
            }

            result.set_item("requests", py_requests)?;
            result.set_item("total_tokens", batch.total_tokens)?;

            Ok(result.into())
        }

        fn record_output(&mut self, request_id: &str, token_id: u32) -> bool {
            self.inner.record_output(request_id, token_id)
        }

        fn waiting_count(&self) -> usize {
            self.inner.waiting.len()
        }

        fn running_count(&self) -> usize {
            self.inner.running.len()
        }

        fn has_work(&self) -> bool {
            self.inner.has_work()
        }
    }

    #[pyclass]
    pub struct PyRadixCache {
        inner: RadixCache,
    }

    unsafe impl Sync for PyRadixCache {}

    #[pymethods]
    impl PyRadixCache {
        #[new]
        fn new(block_size: usize) -> PyResult<Self> {
            Ok(Self {
                inner: RadixCache::new(block_size),
            })
        }

        fn match_prefix(&self, token_ids: Vec<u32>) -> (Vec<usize>, usize) {
            self.inner.match_prefix(&token_ids)
        }

        fn insert(
            &mut self,
            token_ids: Vec<u32>,
            kv_blocks: Vec<usize>,
            block_manager: &mut PyBlockManager,
        ) {
            self.inner.insert(&token_ids, &kv_blocks, &mut block_manager.inner);
        }

        fn evict(
            &mut self,
            target_blocks: usize,
            block_manager: &mut PyBlockManager,
        ) -> usize {
            self.inner.evict(target_blocks, &mut block_manager.inner)
        }

        fn cached_blocks(&self) -> usize {
            self.inner.cached_blocks()
        }

        fn node_count(&self) -> usize {
            self.inner.node_count()
        }
    }

    #[pymodule]
    fn llm_serveforge_runtime(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyScheduler>()?;
        m.add_class::<PyBlockManager>()?;
        m.add_class::<PyRadixCache>()?;
        Ok(())
    }
}
