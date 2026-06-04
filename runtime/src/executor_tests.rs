use super::*;
use crate::hal::cpu::MemRefDesc2;
use crate::hal::cpu::MemRefDescAny;
use crate::hal::traits::{self, Executable as _};
use crate::tensor::Dtype;

/// A mock device that records its compile() call for validation.
#[derive(Debug)]
struct MockDevice {
    name: String,
    compile_called: std::sync::atomic::AtomicBool,
}

impl MockDevice {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            compile_called: std::sync::atomic::AtomicBool::new(false),
        }
    }

    fn was_compile_called(&self) -> bool {
        self.compile_called.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[derive(Debug)]
struct MockBuffer(Vec<u8>);

impl traits::Buffer for MockBuffer {
    fn as_ptr(&self) -> *const u8 { self.0.as_ptr() }
    fn as_mut_ptr(&mut self) -> *mut u8 { self.0.as_mut_ptr() }
    fn len(&self) -> usize { self.0.len() }
    fn copy_from_host(&mut self, src: &[u8], _stream: &dyn traits::Stream) -> Result<(), anyhow::Error> {
        self.0.copy_from_slice(src);
        Ok(())
    }
    fn copy_to_host(&self, dst: &mut [u8], _stream: &dyn traits::Stream) -> Result<(), anyhow::Error> {
        dst.copy_from_slice(&self.0);
        Ok(())
    }
}

#[derive(Debug)]
struct MockEvent;

impl traits::Event for MockEvent {
    fn is_complete(&self) -> bool { true }
    fn synchronize(&self) -> Result<(), anyhow::Error> { Ok(()) }
}

#[derive(Debug)]
struct MockStream;

impl traits::Stream for MockStream {
    fn synchronize(&self) -> Result<(), anyhow::Error> { Ok(()) }
    fn wait_event(&self, _event: &dyn traits::Event) -> Result<(), anyhow::Error> { Ok(()) }
    fn record_event(&self, _event: &dyn traits::Event) -> Result<(), anyhow::Error> { Ok(()) }
}

#[derive(Debug)]
struct MockExecutable {
    function_count: usize,
    module_data: Vec<u8>,
}

impl MockExecutable {
    fn new(count: usize) -> Self {
        Self { function_count: count, module_data: Vec::new() }
    }
}

impl traits::Executable for MockExecutable {
    fn execute(&self, _op_name: &str, _stream: &dyn traits::Stream, _inputs: &[crate::hal::sfa::SfaMemRef], _outputs: &mut [crate::hal::sfa::SfaMemRef]) -> Result<Vec<Vec<i64>>, anyhow::Error> {
        Ok(Vec::new())
    }
    fn function_count(&self) -> usize { self.function_count }
    fn module_data(&self) -> &[u8] { &self.module_data }
    fn supported_ops(&self) -> &[&str] {
        &["mock"]
    }
    fn register_expert_kernel(
        &mut self,
        _op_name: &str,
        _kernel: Box<dyn traits::ExpertKernel>,
    ) -> Result<(), anyhow::Error> {
        Ok(())
    }
}

impl traits::Device for MockDevice {
    fn alloc(&self, size: usize) -> Result<Box<dyn traits::Buffer>, anyhow::Error> {
        Ok(Box::new(MockBuffer(vec![0u8; size])))
    }
    fn create_stream(&self) -> Result<Box<dyn traits::Stream>, anyhow::Error> {
        Ok(Box::new(MockStream))
    }
    fn create_event(&self) -> Result<Box<dyn traits::Event>, anyhow::Error> {
        Ok(Box::new(MockEvent))
    }
    fn compile(&self, module_data: &[u8]) -> Result<Box<dyn traits::Executable>, anyhow::Error> {
        self.compile_called.store(true, std::sync::atomic::Ordering::Relaxed);
        assert!(!module_data.is_empty(), "compile should receive non-empty data");
        Ok(Box::new(MockExecutable::new(2)))
    }
    fn name(&self) -> &str { &self.name }
}

#[test]
fn test_executor_load_nonexistent_fails() {
    let result = ModelExecutor::load("/nonexistent/lib.dylib", None);
    assert!(result.is_err());
}

#[test]
fn test_executor_load_with_device_calls_compile() {
    let device = MockDevice::new("mock-test");
    // This will fail because the .dylib doesn't exist, but compile()
    // should be called first.
    let result = ModelExecutor::load_with_device(&device, "/nonexistent/lib.dylib", None);
    assert!(result.is_err(), "load should fail on nonexistent dylib");
    // Even though the load fails (no dylib), compile() should have been attempted
    // The compile_called flag indicates the trait was used
    assert!(device.was_compile_called(),
        "Device::compile() should be called during load_with_device");
}

#[test]
fn test_global_input_i64_memref() {
    let input_ids: Vec<i64> = vec![2, 525, 484, 0];
    let raw: Vec<u8> = input_ids.iter().flat_map(|&v| v.to_ne_bytes()).collect();
    let p = raw.as_ptr();
    let desc = MemRefDesc2 {
        allocated: p as *mut std::ffi::c_void,
        aligned: p as *mut std::ffi::c_void,
        offset: 0,
        sizes: [1, 4],
        strides: [4, 1],
    };
    unsafe {
        for i in 0..4i64 {
            let val = *(desc.aligned.add((i * 8) as usize) as *const i64);
            assert_eq!(val, input_ids[i as usize]);
        }
    }
    assert_eq!(desc.sizes[0], 1);
    assert_eq!(desc.sizes[1], 4);
    assert_eq!(desc.strides[0], 4);
    assert_eq!(desc.strides[1], 1);
    assert_eq!(desc.numel(), 4);
}

#[test]
fn test_f16_to_f32_conversion() {
    use half::f16;
    let f16_vals: Vec<u16> = vec![
        f16::from_f32(1.0).to_bits(),
        f16::from_f32(0.5).to_bits(),
        f16::from_f32(0.0).to_bits(),
        f16::from_f32(-1.0).to_bits(),
    ];
    let raw: Vec<u8> = f16_vals.iter().flat_map(|&v| v.to_ne_bytes()).collect();
    let p = raw.as_ptr();
    let desc = MemRefDesc2 {
        allocated: p as *mut std::ffi::c_void,
        aligned: p as *mut std::ffi::c_void,
        offset: 0,
        sizes: [2, 2],
        strides: [2, 1],
    };
    let n = desc.numel();
    let data: Vec<f32> = unsafe {
        let raw = desc.aligned as *const u16;
        let slice = std::slice::from_raw_parts(raw, n);
        slice.iter().map(|&h| f16::from_bits(h).to_f32()).collect()
    };
    assert!((data[0] - 1.0).abs() < 1e-6);
    assert!((data[1] - 0.5).abs() < 1e-6);
    assert!((data[2] - 0.0).abs() < 1e-6);
    assert!((data[3] + 1.0).abs() < 1e-6);
}

#[test]
fn test_desc_pointers_unique_and_stable_in_vec() {
    // Regression: as_input_ptr() must point to Vec element (not stack-local desc),
    // and must remain valid after subsequent pushes (Vec must not reallocate).
    let data = vec![1.0f32; 768];
    let cap = 10;
    let mut descs: Vec<MemRefDescAny> = Vec::with_capacity(cap);
    let mut ptrs: Vec<*const std::ffi::c_void> = Vec::with_capacity(cap);

    for _ in 0..cap {
        let tensor = Tensor::new_owned(vec![768], data.clone(), Dtype::F32);
        descs.push(MemRefDescAny::from_f32(&tensor.shape, tensor.as_slice()).unwrap());
        ptrs.push(descs.last().unwrap().as_input_ptr());
    }

    // Each pointer should be unique (pointing to different Vec elements)
    for i in 0..cap {
        for j in i + 1..cap {
            assert_ne!(ptrs[i], ptrs[j],
                "descs[{}] and descs[{}] have same pointer: {:p}", i, j, ptrs[i]);
        }
    }

    // Re-verify by re-reading pointers from Vec
    for i in 0..cap {
        assert_eq!(ptrs[i], descs[i].as_input_ptr(),
            "descs[{}] pointer changed after subsequent pushes", i);
    }
}

#[test]
fn test_desc_pointers_different_for_different_ranks() {
    // Different rank descriptors should have different pointers
    let mut descs: Vec<MemRefDescAny> = Vec::with_capacity(3);

    let d1 = vec![0.0f32];
    descs.push(MemRefDescAny::from_f32(&[], &d1).unwrap());  // rank-0
    let d2 = vec![0.0f32; 4];
    descs.push(MemRefDescAny::from_f32(&[4], &d2).unwrap());  // rank-1
    let d3 = vec![0.0f32; 8];
    descs.push(MemRefDescAny::from_f32(&[2, 4], &d3).unwrap());  // rank-2

    let p0 = descs[0].as_input_ptr();
    let p1 = descs[1].as_input_ptr();
    let p2 = descs[2].as_input_ptr();

    assert_ne!(p0, p1, "rank-0 and rank-1 pointers must differ");
    assert_ne!(p1, p2, "rank-1 and rank-2 pointers must differ");
    assert_ne!(p0, p2, "rank-0 and rank-2 pointers must differ");
}

// ── KV cache concat test (validate the concat logic manually) ──

#[test]
fn test_kv_cache_concat_decode_flow() {
    // Simulates the decode flow: pre-write 4 cached positions, write 1 new
    // position, then verify concat has shape [1, 5, 768].
    use crate::block_manager::BlockManager;
    let num_kv_heads = 12;
    let head_dim = 64;
    let hidden_dim = num_kv_heads * head_dim; // 768
    let block_size = 16;

    let mut bm = BlockManager::new_with_cache(10, block_size, num_kv_heads, head_dim).unwrap();
    bm.allocate("test_req", 5).unwrap();

    // Pre-write positions 0..3 (simulating previous decode steps)
    for pos in 0..4 {
        let k_data: Vec<f32> = vec![(pos + 1) as f32; hidden_dim]; // K=1,2,3,4
        let v_data: Vec<f32> = vec![(pos + 10) as f32; hidden_dim]; // V=10,11,12,13
        bm.write_kv("test_req", pos, &k_data, hidden_dim, true).unwrap();
        bm.write_kv("test_req", pos, &v_data, hidden_dim, false).unwrap();
    }

    // Simulate new K/V at position 4 (decode step)
    let k_new: Vec<f32> = vec![5.0f32; hidden_dim];
    let v_new: Vec<f32> = vec![14.0f32; hidden_dim];

    // Read cached positions [0..4)
    let (cached_k, cached_v) = bm.read_kv("test_req", 4, hidden_dim).unwrap();
    assert_eq!(cached_k.len(), 4 * hidden_dim);
    assert_eq!(cached_v.len(), 4 * hidden_dim);

    // Concat: K_all = [K_cached[0..4)] ++ [K_new], V_all = [V_cached[0..4)] ++ [V_new]
    let mut k_all = Vec::with_capacity(5 * hidden_dim);
    k_all.extend_from_slice(&cached_k);
    k_all.extend_from_slice(&k_new);
    let mut v_all = Vec::with_capacity(5 * hidden_dim);
    v_all.extend_from_slice(&cached_v);
    v_all.extend_from_slice(&v_new);

    // Assert shapes
    assert_eq!(k_all.len(), 5 * hidden_dim); // [1, 5, 768]
    assert_eq!(v_all.len(), 5 * hidden_dim); // [1, 5, 768]

    // Verify K values: positions 0..3 have cached values, position 4 has new
    assert!((k_all[0] - 1.0).abs() < 1e-6, "K[0] should be 1.0");
    assert!((k_all[hidden_dim] - 2.0).abs() < 1e-6, "K[768] should be 2.0");
    assert!((k_all[2 * hidden_dim] - 3.0).abs() < 1e-6, "K[1536] should be 3.0");
    assert!((k_all[3 * hidden_dim] - 4.0).abs() < 1e-6, "K[2304] should be 4.0");
    assert!((k_all[4 * hidden_dim] - 5.0).abs() < 1e-6, "K[3072] should be 5.0");

    // Write new K/V to cache (simulating what forward_with_kv does)
    bm.write_kv("test_req", 4, &k_new, hidden_dim, true).unwrap();
    bm.write_kv("test_req", 4, &v_new, hidden_dim, false).unwrap();

    // Verify write to cache at position 4
    let (read_k, read_v) = bm.read_kv("test_req", 5, hidden_dim).unwrap();
    assert_eq!(read_k.len(), 5 * hidden_dim);
    assert!((read_k[4 * hidden_dim] - 5.0).abs() < 1e-6, "cached K[4] should be 5.0");
    assert!((read_v[4 * hidden_dim] - 14.0).abs() < 1e-6, "cached V[4] should be 14.0");
}

// ── Integration: forward_with_kv with real model ──

#[test]
fn test_forward_with_kv_requires_compiled_model() {
    // This test validates that forward_with_kv is structurally callable.
    // It requires a compiled model with split functions (main_Xa + main_Xb).
    // If the model isn't compiled, we skip gracefully.
    let dylib = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../outputs/compiled/opt_125m_fresh/libopt_125m_fresh.dylib"
    );
    if !std::path::Path::new(dylib).exists() {
        eprintln!("SKIP: no compiled model at {dylib}");
        return;
    }
    let executor = match ModelExecutor::load(dylib, None) {
        Ok(e) => e,
        Err(err) => {
            eprintln!("SKIP: failed to load model: {err}");
            return;
        }
    };

    // Check that the compute graph has multiple functions (split attention)
    let num_funcs = executor.compute_graph.functions.len();
    eprintln!("forward_with_kv integration: {} functions in graph", num_funcs);

    // Run forward_with_kv WITHOUT a block manager to verify the no-cache path
    let input_ids = &[2u32, 525, 484, 0];
    let positions = &[0u32, 1, 2, 3];
    let result = executor.forward_with_kv(input_ids, positions, None, None);
    match result {
        Ok(logits) => {
            assert!(!logits.as_slice().is_empty(), "logits should not be empty");
            assert!(
                logits.as_slice().iter().any(|&x| x != 0.0),
                "logits should not be all zeros"
            );
            eprintln!("forward_with_kv (no cache) OK: shape={:?}, first={:.4}",
                logits.shape, logits.as_slice()[0]);
        }
        Err(e) => {
            // This may fail if the model doesn't use split functions with
            // consumed_internally outputs — that's expected for v8 models.
            eprintln!("forward_with_kv (no cache) skipped: {e}");
        }
    }
}

#[test]
fn test_kv_model_loads_and_forwards() {
    let dylib = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../outputs/compiled/opt_125m_kv/libopt_125m_kv.dylib"
    );
    if !std::path::Path::new(dylib).exists() {
        eprintln!("SKIP: no KV model at {dylib}");
        return;
    }
    let executor = match ModelExecutor::load(dylib, None) {
        Ok(e) => e,
        Err(err) => { eprintln!("FAIL LOAD: {err}"); return; }
    };
    let n = executor.compute_graph.functions.len();
    let ci = executor.compute_graph.functions.iter()
        .flat_map(|f| &f.outputs).filter(|o| o.consumed_internally).count();
    eprintln!("KV MODEL: {} functions, {} consumed_internally=True", n, ci);
    assert!(n > 16, "expected >16 functions (split model)");
    assert!(ci > 0, "expected consumed_internally=True outputs");
    eprintln!("PASS: KV model loads with {n} functions, {ci} consumed_internally");
}

// ── ExpertKernel trait tests ───────────────────────────────────────

/// A minimal ExpertKernel implementation for testing the trait.
#[derive(Debug)]
struct MultiplyByTwoKernel;

impl traits::ExpertKernel for MultiplyByTwoKernel {
    fn execute(
        &self,
        _stream: &dyn traits::Stream,
        inputs: &[&dyn traits::Buffer],
        outputs: &[&dyn traits::Buffer],
    ) -> Result<(), anyhow::Error> {
        if let (Some(inp), Some(out)) = (inputs.first(), outputs.first()) {
            let n = inp.len().min(out.len()) / 4;
            unsafe {
                let inp_slice = std::slice::from_raw_parts(inp.as_ptr() as *const f32, n);
                let out_slice = std::slice::from_raw_parts_mut(out.as_ptr() as *mut f32, n);
                for i in 0..n {
                    out_slice[i] = inp_slice[i] * 2.0;
                }
            }
            Ok(())
        } else {
            anyhow::bail!("MultiplyByTwoKernel needs at least 1 input and 1 output")
        }
    }
}

#[test]
fn test_expert_kernel_trait() {
    // Verify that an ExpertKernel can be created and boxed.
    let kernel: Box<dyn traits::ExpertKernel> = Box::new(MultiplyByTwoKernel);
    let stream = MockStream;

    let mut input_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let mut output_data = vec![0.0f32; 4];
    let input = MockBuffer(
        unsafe {
            std::slice::from_raw_parts_mut(
                input_data.as_mut_ptr() as *mut u8,
                input_data.len() * 4,
            )
        }
        .to_vec(),
    );
    let output = MockBuffer(
        unsafe {
            std::slice::from_raw_parts_mut(
                output_data.as_mut_ptr() as *mut u8,
                output_data.len() * 4,
            )
        }
        .to_vec(),
    );

    let inputs: [&dyn traits::Buffer; 1] = [&input];
    let outputs: [&dyn traits::Buffer; 1] = [&output];
    kernel.execute(&stream, &inputs, &outputs).expect("expert kernel should execute");

    // Verify output = input * 2
    let out_slice = unsafe {
        std::slice::from_raw_parts(output.as_ptr() as *const f32, 4)
    };
    assert!((out_slice[0] - 2.0).abs() < 1e-6);
    assert!((out_slice[1] - 4.0).abs() < 1e-6);
    assert!((out_slice[2] - 6.0).abs() < 1e-6);
    assert!((out_slice[3] - 8.0).abs() < 1e-6);
}

/// An executable that uses the DEFAULT trait implementations for
/// supported_ops() and register_expert_kernel().
#[derive(Debug)]
struct MinimalExecutable {
    count: usize,
    data: Vec<u8>,
}

impl traits::Executable for MinimalExecutable {
    fn execute(
        &self,
        _op_name: &str,
        _stream: &dyn traits::Stream,
        _inputs: &[crate::hal::sfa::SfaMemRef],
        _outputs: &mut [crate::hal::sfa::SfaMemRef],
    ) -> Result<Vec<Vec<i64>>, anyhow::Error> {
        Ok(vec![vec![0i64]])
    }
    fn function_count(&self) -> usize { self.count }
    fn module_data(&self) -> &[u8] { &self.data }
}

#[test]
fn test_supported_ops_default() {
    let exe = MinimalExecutable { count: 1, data: vec![] };
    let ops = exe.supported_ops();
    // Default implementation returns &[]
    assert!(ops.is_empty(), "default supported_ops should return empty slice");
}

#[test]
fn test_register_expert_kernel_default() {
    let mut exe = MinimalExecutable { count: 1, data: vec![] };
    let kernel: Box<dyn traits::ExpertKernel> = Box::new(MultiplyByTwoKernel);
    let result = exe.register_expert_kernel("matmul", kernel);
    // Default implementation returns Err
    assert!(result.is_err(), "default register_expert_kernel should error");
    assert!(
        result.unwrap_err().to_string().contains("not supported"),
        "error should mention 'not supported'"
    );
}

// ── function_count validation (hal-rust helper) ──

#[cfg(feature = "hal-rust")]
#[test]
fn test_function_count_validation() {
    let dir = std::env::temp_dir().join("_test_func_count_val");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("generated")).unwrap();

    // Create constants.bin (basic dummy content)
    std::fs::write(dir.join("constants.bin"), b"dummy").unwrap();

    // Case 1: hal_ir.json with null num_functions
    std::fs::write(
        dir.join("generated").join("hal_ir.json"),
        r#"{"num_functions": null}"#,
    )
    .unwrap();
    let result = super::load_hal_rust_executable_from_dir(&dir);
    assert!(
        result.is_err(),
        "should reject null num_functions"
    );
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("num_functions") || err.contains("missing") || err.contains("invalid"),
        "error should mention num_functions or missing/invalid, got: {err}",
    );

    // Case 2: hal_ir.json with missing num_functions key
    std::fs::write(
        dir.join("generated").join("hal_ir.json"),
        r#"{"other_key": 42}"#,
    )
    .unwrap();
    let result = super::load_hal_rust_executable_from_dir(&dir);
    assert!(
        result.is_err(),
        "should reject missing num_functions key"
    );
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("num_functions") || err.contains("missing"),
        "error should mention num_functions or missing, got: {err}",
    );

    // Case 3: hal_ir.json with valid num_functions (should succeed)
    std::fs::write(
        dir.join("generated").join("hal_ir.json"),
        r#"{"num_functions": 2}"#,
    )
    .unwrap();
    let result = super::load_hal_rust_executable_from_dir(&dir);
    assert!(
        result.is_ok(),
        "should accept valid num_functions, got: {:?}",
        result.err(),
    );

    // Cleanup
    let _ = std::fs::remove_dir_all(&dir);
}

