//! Tensor data type — the runtime representation of model inputs, weights,
//! intermediate activations, and outputs.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Dtype {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    I64 = 3,
    I32 = 4,
    I8 = 5,
    U8 = 6,
}

impl Dtype {
    #[allow(dead_code)]
    pub fn element_size(&self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::F16 => 2,
            Dtype::BF16 => 2,
            Dtype::I64 => 8,
            Dtype::I32 => 4,
            Dtype::I8 => 1,
            Dtype::U8 => 1,
        }
    }

    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Dtype::F32),
            1 => Some(Dtype::F16),
            2 => Some(Dtype::BF16),
            3 => Some(Dtype::I64),
            4 => Some(Dtype::I32),
            5 => Some(Dtype::I8),
            6 => Some(Dtype::U8),
            _ => None,
        }
    }
}

impl fmt::Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dtype::F32 => write!(f, "f32"),
            Dtype::F16 => write!(f, "f16"),
            Dtype::BF16 => write!(f, "bf16"),
            Dtype::I64 => write!(f, "i64"),
            Dtype::I32 => write!(f, "i32"),
            Dtype::I8 => write!(f, "i8"),
            Dtype::U8 => write!(f, "u8"),
        }
    }
}

pub enum TensorData<'a> {
    Owned(Vec<f32>),
    #[allow(dead_code)]
    Borrowed(&'a [f32]),
}

pub struct Tensor<'a> {
    pub data: TensorData<'a>,
    pub shape: Vec<usize>,
    pub dtype: Dtype,
}

impl<'a> Tensor<'a> {
    pub fn new_owned(shape: Vec<usize>, data: Vec<f32>, dtype: Dtype) -> Self {
        debug_assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "Tensor::new_owned: data len {} != shape product {:?}",
            data.len(),
            shape.iter().product::<usize>()
        );
        Self {
            data: TensorData::Owned(data),
            shape,
            dtype,
        }
    }

    #[allow(dead_code)]
    pub fn from_borrowed(shape: Vec<usize>, data: &'a [f32], dtype: Dtype) -> Self {
        debug_assert_eq!(data.len(), shape.iter().product::<usize>());
        Self {
            data: TensorData::Borrowed(data),
            shape,
            dtype,
        }
    }

    #[allow(dead_code)]
    pub fn scalar(value: f32) -> Self {
        Self {
            data: TensorData::Owned(vec![value]),
            shape: vec![],
            dtype: Dtype::F32,
        }
    }

    pub fn as_slice(&self) -> &[f32] {
        match &self.data {
            TensorData::Owned(v) => v.as_slice(),
            TensorData::Borrowed(s) => s,
        }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    #[allow(dead_code)]
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn to_owned(&self) -> Tensor<'static> {
        Tensor {
            data: TensorData::Owned(self.as_slice().to_vec()),
            shape: self.shape.clone(),
            dtype: self.dtype,
        }
    }
}

impl Clone for Tensor<'_> {
    fn clone(&self) -> Self {
        match &self.data {
            TensorData::Owned(v) => Self {
                data: TensorData::Owned(v.clone()),
                shape: self.shape.clone(),
                dtype: self.dtype,
            },
            TensorData::Borrowed(s) => Self {
                data: TensorData::Borrowed(s),
                shape: self.shape.clone(),
                dtype: self.dtype,
            },
        }
    }
}

impl fmt::Debug for Tensor<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let data_preview = if self.numel() <= 6 {
            format!("{:?}", self.as_slice())
        } else {
            format!(
                "[{}, {}, {}, ..., {}, {}, {}]",
                self.as_slice()[0],
                self.as_slice()[1],
                self.as_slice()[2],
                self.as_slice()[self.numel() - 3],
                self.as_slice()[self.numel() - 2],
                self.as_slice()[self.numel() - 1],
            )
        };
        write!(
            f,
            "Tensor(dtype={}, shape={:?}, data={})",
            self.dtype, self.shape, data_preview
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_codes_roundtrip() {
        for code in 0u8..=6 {
            let dt = Dtype::from_code(code).expect("valid code");
            assert_eq!(dt as u8, code);
        }
    }

    #[test]
    fn test_new_owned() {
        let t = Tensor::new_owned(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Dtype::F32);
        assert_eq!(t.rank(), 2);
        assert_eq!(t.numel(), 6);
    }

    #[test]
    fn test_scalar() {
        let t = Tensor::scalar(3.14);
        assert_eq!(t.rank(), 0);
        assert_eq!(t.numel(), 1);
    }

    #[test]
    fn test_to_owned() {
        let data = [1.0, 2.0, 3.0];
        let borrowed = Tensor::from_borrowed(vec![3], &data, Dtype::F32);
        let owned = borrowed.to_owned();
        assert_eq!(owned.as_slice(), &[1.0, 2.0, 3.0]);
    }
}
