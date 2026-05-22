use super::*;
use crate::hal::traits::{self, Device as _, Executable as _};

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
struct MockStream;

impl traits::Stream for MockStream {
    fn synchronize(&self) -> Result<(), anyhow::Error> { Ok(()) }
}

#[derive(Debug)]
struct MockExecutable {
    entry_count: usize,
}

impl traits::Executable for MockExecutable {
    fn execute(&self, _stream: &dyn traits::Stream, _inputs: &[&dyn traits::Buffer], _outputs: &[&dyn traits::Buffer]) -> Result<(), anyhow::Error> {
        Ok(())
    }
    fn entry_count(&self) -> usize { self.entry_count }
}

impl traits::Device for MockDevice {
    fn alloc(&self, size: usize) -> Result<Box<dyn traits::Buffer>, anyhow::Error> {
        Ok(Box::new(MockBuffer(vec![0u8; size])))
    }
    fn create_stream(&self) -> Result<Box<dyn traits::Stream>, anyhow::Error> {
        Ok(Box::new(MockStream))
    }
    fn compile(&self, module_data: &[u8]) -> Result<Box<dyn traits::Executable>, anyhow::Error> {
        self.compile_called.store(true, std::sync::atomic::Ordering::Relaxed);
        assert!(!module_data.is_empty(), "compile should receive non-empty data");
        Ok(Box::new(MockExecutable { entry_count: 2 }))
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
fn test_parse_sret_descriptor_rank1() {
    let data: Vec<f32> = vec![1.0, 2.0, 3.0];
    let dummy_allocated = data.as_ptr() as u64;
    let mut buf = Vec::<u8>::new();
    buf.extend_from_slice(&dummy_allocated.to_ne_bytes());
    buf.extend_from_slice(&dummy_allocated.to_ne_bytes());
    buf.extend_from_slice(&0u64.to_ne_bytes());
    buf.extend_from_slice(&3u64.to_ne_bytes());
    buf.extend_from_slice(&1u64.to_ne_bytes());
    assert_eq!(buf.len(), 40);
    let (aligned, sizes) = unsafe { parse_sret_descriptor(&buf, 1).unwrap() };
    assert_eq!(aligned as u64, dummy_allocated);
    assert_eq!(sizes, vec![3]);
}

#[test]
fn test_parse_sret_descriptor_rank4() {
    let data: Vec<f32> = vec![0.0; 10];
    let p = data.as_ptr() as u64;
    let mut buf = Vec::<u8>::new();
    buf.extend_from_slice(&p.to_ne_bytes());
    buf.extend_from_slice(&p.to_ne_bytes());
    buf.extend_from_slice(&0u64.to_ne_bytes());
    buf.extend_from_slice(&2u64.to_ne_bytes());
    buf.extend_from_slice(&1u64.to_ne_bytes());
    buf.extend_from_slice(&4u64.to_ne_bytes());
    buf.extend_from_slice(&4u64.to_ne_bytes());
    buf.extend_from_slice(&4u64.to_ne_bytes());
    buf.extend_from_slice(&4u64.to_ne_bytes());
    buf.extend_from_slice(&1u64.to_ne_bytes());
    buf.extend_from_slice(&1u64.to_ne_bytes());
    assert_eq!(buf.len(), 88);
    let (aligned, sizes) = unsafe { parse_sret_descriptor(&buf, 4).unwrap() };
    assert_eq!(aligned as u64, p);
    assert_eq!(sizes, vec![2, 1, 4, 4]);
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
