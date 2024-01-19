use core::panic;

use ndarray::{Array, IxDyn};
use onnx::onnx::TensorProto_DataType;
use ndarray_rand::RandomExt;
use rand::distributions::{Uniform, uniform::SampleUniform};

use crate::onnx::{TensorShapeProto, tensor_shape_proto::dimension::Value};
use crate::operations::*;

#[derive(Debug, Clone)]
pub enum ArrayMultiType {
    FLOAT(Array<f32, IxDyn>),
    UINT8(Array<u8, IxDyn>),
    INT8(Array<i8, IxDyn>),
    UINT16(Array<u16, IxDyn>),
    INT16(Array<i16, IxDyn>),
    INT32(Array<i32, IxDyn>),
    INT64(Array<i64, IxDyn>),
    BOOL(Array<bool, IxDyn>)
}
pub trait FromBytes {
    fn from_le_bytes(bytes: &[u8]) -> Self;
}

impl FromBytes for f32 {
    fn from_le_bytes(bytes: &[u8]) -> Self {
        f32::from_le_bytes(bytes.try_into().unwrap())
    }
}

impl FromBytes for u8 {
    fn from_le_bytes(bytes: &[u8]) -> Self {
        u8::from_le_bytes(bytes.try_into().unwrap())
    }
}

impl FromBytes for i8 {
    fn from_le_bytes(bytes: &[u8]) -> Self {
        i8::from_le_bytes(bytes.try_into().unwrap())
    }
}

impl FromBytes for u16 {
    fn from_le_bytes(bytes: &[u8]) -> Self {
        u16::from_le_bytes(bytes.try_into().unwrap())
    }
}

impl FromBytes for i16 {
    fn from_le_bytes(bytes: &[u8]) -> Self {
        i16::from_le_bytes(bytes.try_into().unwrap())
    }
}

impl FromBytes for i32 {
    fn from_le_bytes(bytes: &[u8]) -> Self {
        i32::from_le_bytes(bytes.try_into().unwrap())
    }
}

impl FromBytes for i64 {
    fn from_le_bytes(bytes: &[u8]) -> Self {
        i64::from_le_bytes(bytes.try_into().unwrap())
    }
}

impl FromBytes for bool {
    fn from_le_bytes(bytes: &[u8]) -> Self {
        bool::from_le_bytes(bytes.try_into().unwrap())
    }
}

fn from_bytes<A: FromBytes>(bytes: &Vec<u8>, dims: &Vec<i64>) -> Result<Array<A, IxDyn>, &'static str>  {
    let size_of_t = std::mem::size_of::<A>();
    let shape = dims.iter().map(|&d| d as usize).collect::<Vec<usize>>();

    if bytes.len() % size_of_t != 0 {
        return Err("Invalid number of bytes");
    }

    let num_elements = bytes.len() / size_of_t;
    let mut result = Vec::with_capacity(num_elements);

    for i in 0..num_elements {
        let binary  = &bytes[i*size_of_t..(i+1)*size_of_t];
        let value = A::from_le_bytes(binary);
        result.push(value);
    }
    Array::from_shape_vec(shape, result).map_err(|_| "Invalid initializer shape")
}

fn from_vec<A: Clone>(vec: &Vec<A>, dims: &Vec<i64>) -> Result<Array<A, IxDyn>, &'static str> {
    let shape = dims.iter().map(|&d| d as usize).collect::<Vec<usize>>();
    Array::from_shape_vec(shape, vec.clone()).map_err(|_| "Invalid initializer shape")
}

fn random<A: SampleUniform>(tensor_shape: &TensorShapeProto, range_low: A, range_high: A) -> Array<A, IxDyn> {
    let shape = tensor_shape.dim.iter().map(|d| {
        match &d.value {
            Some(Value::DimValue(v)) => *v as usize,
            Some(Value::DimParam(_)) => 1 as usize,
            None => 0,
        }
    }).collect::<Vec<usize>>();
    
    Array::<A, IxDyn>::random(shape, Uniform::new(range_low, range_high))
}

impl ArrayMultiType {
    pub fn from_bytes(bytes: &Vec<u8>, dims: &Vec<i64>, data_type: i32) -> Result<ArrayMultiType, &'static str> {
        match data_type {
            i if i == TensorProto_DataType::FLOAT as i32 => Ok(ArrayMultiType::FLOAT(from_bytes::<f32>(bytes, dims)?)),
            i if i == TensorProto_DataType::UINT8 as i32 => Ok(ArrayMultiType::UINT8(from_bytes::<u8>(bytes, dims)?)),
            i if i == TensorProto_DataType::INT8 as i32 => Ok(ArrayMultiType::INT8(from_bytes::<i8>(bytes, dims)?)),
            i if i == TensorProto_DataType::UINT16 as i32 => Ok(ArrayMultiType::UINT16(from_bytes::<u16>(bytes, dims)?)),
            i if i == TensorProto_DataType::INT16 as i32 => Ok(ArrayMultiType::INT16(from_bytes::<i16>(bytes, dims)?)),
            i if i == TensorProto_DataType::INT32 as i32 => Ok(ArrayMultiType::INT32(from_bytes::<i32>(bytes, dims)?)),
            i if i == TensorProto_DataType::INT64 as i32 => Ok(ArrayMultiType::INT64(from_bytes::<i64>(bytes, dims)?)),
            i if i == TensorProto_DataType::BOOL as i32 => Ok(ArrayMultiType::BOOL(from_bytes::<bool>(bytes, dims)?)),
            _ => Err("Data type not supported")
        }
    }

    pub fn from_vec_f32(vec: &Vec<f32>, dims: &Vec<i64>) -> Result<ArrayMultiType, &'static str> {
        Ok(ArrayMultiType::FLOAT(from_vec::<f32>(vec, dims)?))
    }

    pub fn from_vec_i32(vec: &Vec<i32>, dims: &Vec<i64>) -> Result<ArrayMultiType, &'static str> {
        Ok(ArrayMultiType::INT32(from_vec::<i32>(vec, dims)?))
    }

    pub fn from_vec_i64(vec: &Vec<i64>, dims: &Vec<i64>) -> Result<ArrayMultiType, &'static str> {
        Ok(ArrayMultiType::INT64(from_vec::<i64>(vec, dims)?))
    }

    pub fn random(shape: &TensorShapeProto, data_type: i32) -> ArrayMultiType {
        match data_type {
            i if i == TensorProto_DataType::FLOAT as i32 => ArrayMultiType::FLOAT(random::<f32>(shape, -1., 1.)),
            i if i == TensorProto_DataType::UINT8 as i32 => ArrayMultiType::UINT8(random::<u8>(shape, 0, 255)),
            i if i == TensorProto_DataType::INT8 as i32 => ArrayMultiType::INT8(random::<i8>(shape, -128, 127)),
            i if i == TensorProto_DataType::UINT16 as i32 => ArrayMultiType::UINT16(random::<u16>(shape, 0, 512)),
            i if i == TensorProto_DataType::INT16 as i32 => ArrayMultiType::INT16(random::<i16>(shape, -256, 255)),
            i if i == TensorProto_DataType::INT32 as i32 => ArrayMultiType::INT32(random::<i32>(shape, -256, 255)),
            i if i == TensorProto_DataType::INT64 as i32 => ArrayMultiType::INT64(random::<i64>(shape, -256, 255)),
            _ => panic!("Random op does not support this data type")
        }
    }

    pub fn to_vec_usize(&self) -> Vec<usize> {
        match self {
            ArrayMultiType::FLOAT(a) => a.shape().to_vec().iter().map(|&x| x as usize).collect::<Vec<usize>>(),
            ArrayMultiType::UINT8(a) => a.shape().to_vec().iter().map(|&x| x as usize).collect::<Vec<usize>>(),
            ArrayMultiType::INT8(a) => a.shape().to_vec().iter().map(|&x| x as usize).collect::<Vec<usize>>(),
            ArrayMultiType::UINT16(a) => a.shape().to_vec().iter().map(|&x| x as usize).collect::<Vec<usize>>(),
            ArrayMultiType::INT16(a) => a.shape().to_vec().iter().map(|&x| x as usize).collect::<Vec<usize>>(),
            ArrayMultiType::INT32(a) => a.shape().to_vec().iter().map(|&x| x as usize).collect::<Vec<usize>>(),
            ArrayMultiType::INT64(a) => a.shape().to_vec().iter().map(|&x| x as usize).collect::<Vec<usize>>(),
            ArrayMultiType::BOOL(a) => a.shape().to_vec().iter().map(|&x| x as usize).collect::<Vec<usize>>()
        }
    }

    pub fn multiply(array_a: &ArrayMultiType, array_b: &ArrayMultiType) -> ArrayMultiType {
        match (array_a, array_b) {
            (ArrayMultiType::FLOAT(a), ArrayMultiType::FLOAT(b)) => ArrayMultiType::FLOAT(multiply(a, b)),
            (ArrayMultiType::UINT8(a), ArrayMultiType::UINT8(b)) => ArrayMultiType::UINT8(multiply(a, b)),
            (ArrayMultiType::INT8(a), ArrayMultiType::INT8(b)) => ArrayMultiType::INT8(multiply(a, b)),
            (ArrayMultiType::UINT16(a), ArrayMultiType::UINT16(b)) => ArrayMultiType::UINT16(multiply(a, b)),
            (ArrayMultiType::INT16(a), ArrayMultiType::INT16(b)) => ArrayMultiType::INT16(multiply(a, b)),
            (ArrayMultiType::INT32(a), ArrayMultiType::INT32(b)) => ArrayMultiType::INT32(multiply(a, b)),
            (ArrayMultiType::INT64(a), ArrayMultiType::INT64(b)) => ArrayMultiType::INT64(multiply(a, b)),
            _ => panic!("Multiply op does not support this data type")
        }
    }

    pub fn add(array_a: &ArrayMultiType, array_b: &ArrayMultiType) -> ArrayMultiType {
        match (array_a, array_b) {
            (ArrayMultiType::FLOAT(a), ArrayMultiType::FLOAT(b)) => ArrayMultiType::FLOAT(add(a, b)),
            (ArrayMultiType::UINT8(a), ArrayMultiType::UINT8(b)) => ArrayMultiType::UINT8(add(a, b)),
            (ArrayMultiType::INT8(a), ArrayMultiType::INT8(b)) => ArrayMultiType::INT8(add(a, b)),
            (ArrayMultiType::UINT16(a), ArrayMultiType::UINT16(b)) => ArrayMultiType::UINT16(add(a, b)),
            (ArrayMultiType::INT16(a), ArrayMultiType::INT16(b)) => ArrayMultiType::INT16(add(a, b)),
            (ArrayMultiType::INT32(a), ArrayMultiType::INT32(b)) => ArrayMultiType::INT32(add(a, b)),
            (ArrayMultiType::INT64(a), ArrayMultiType::INT64(b)) => ArrayMultiType::INT64(add(a, b)),
            _ => panic!("Add op does not support this data type")
        }
    }

    pub fn exp(array: &ArrayMultiType) -> ArrayMultiType {
        match array {
            ArrayMultiType::FLOAT(a) => ArrayMultiType::FLOAT(exp(a)),
            _ => panic!("Exp op does not support this data type")
        }
    }

    pub fn floor(array: &ArrayMultiType) -> ArrayMultiType {
        match array {
            ArrayMultiType::FLOAT(a) => ArrayMultiType::FLOAT(floor(a)),
            _ => panic!("Floor op does not support this data type")
        }
    }

    pub fn log(array: &ArrayMultiType) -> ArrayMultiType {
        match array {
            ArrayMultiType::FLOAT(a) => ArrayMultiType::FLOAT(log(a)),
            _ => panic!("Log op does not support this data type")
        }
    }

    pub fn greater(array_a: &ArrayMultiType, array_b: &ArrayMultiType) -> ArrayMultiType {
        match (array_a, array_b) {
            (ArrayMultiType::FLOAT(a), ArrayMultiType::FLOAT(b)) => ArrayMultiType::BOOL(greater(a, b)),
            (ArrayMultiType::UINT8(a), ArrayMultiType::UINT8(b)) => ArrayMultiType::BOOL(greater(a, b)),
            (ArrayMultiType::INT8(a), ArrayMultiType::INT8(b)) => ArrayMultiType::BOOL(greater(a, b)),
            (ArrayMultiType::UINT16(a), ArrayMultiType::UINT16(b)) => ArrayMultiType::BOOL(greater(a, b)),
            (ArrayMultiType::INT16(a), ArrayMultiType::INT16(b)) => ArrayMultiType::BOOL(greater(a, b)),
            (ArrayMultiType::INT32(a), ArrayMultiType::INT32(b)) => ArrayMultiType::BOOL(greater(a, b)),
            (ArrayMultiType::INT64(a), ArrayMultiType::INT64(b)) => ArrayMultiType::BOOL(greater(a, b)),
            _ => panic!("Greater op does not support this data type")
        }
    }

   pub fn greater_or_equal(array_a: &ArrayMultiType, array_b: &ArrayMultiType) -> ArrayMultiType {
        match (array_a, array_b) {
            (ArrayMultiType::FLOAT(a), ArrayMultiType::FLOAT(b)) => ArrayMultiType::BOOL(greater_or_equal(a, b)),
            (ArrayMultiType::UINT8(a), ArrayMultiType::UINT8(b)) => ArrayMultiType::BOOL(greater_or_equal(a, b)),
            (ArrayMultiType::INT8(a), ArrayMultiType::INT8(b)) => ArrayMultiType::BOOL(greater_or_equal(a, b)),
            (ArrayMultiType::UINT16(a), ArrayMultiType::UINT16(b)) => ArrayMultiType::BOOL(greater_or_equal(a, b)),
            (ArrayMultiType::INT16(a), ArrayMultiType::INT16(b)) => ArrayMultiType::BOOL(greater_or_equal(a, b)),
            (ArrayMultiType::INT32(a), ArrayMultiType::INT32(b)) => ArrayMultiType::BOOL(greater_or_equal(a, b)),
            (ArrayMultiType::INT64(a), ArrayMultiType::INT64(b)) => ArrayMultiType::BOOL(greater_or_equal(a, b)),
            _ => panic!("Greater or equal op does not support this data type")
        }
    }

    pub fn less(array_a: &ArrayMultiType, array_b: &ArrayMultiType) -> ArrayMultiType {
        match (array_a, array_b) {
            (ArrayMultiType::FLOAT(a), ArrayMultiType::FLOAT(b)) => ArrayMultiType::BOOL(less(a, b)),
            (ArrayMultiType::UINT8(a), ArrayMultiType::UINT8(b)) => ArrayMultiType::BOOL(less(a, b)),
            (ArrayMultiType::INT8(a), ArrayMultiType::INT8(b)) => ArrayMultiType::BOOL(less(a, b)),
            (ArrayMultiType::UINT16(a), ArrayMultiType::UINT16(b)) => ArrayMultiType::BOOL(less(a, b)),
            (ArrayMultiType::INT16(a), ArrayMultiType::INT16(b)) => ArrayMultiType::BOOL(less(a, b)),
            (ArrayMultiType::INT32(a), ArrayMultiType::INT32(b)) => ArrayMultiType::BOOL(less(a, b)),
            (ArrayMultiType::INT64(a), ArrayMultiType::INT64(b)) => ArrayMultiType::BOOL(less(a, b)),
            _ => panic!("Less op does not support this data type")
        }
    }

    pub fn less_or_equal(array_a: &ArrayMultiType, array_b: &ArrayMultiType) -> ArrayMultiType {
        match (array_a, array_b) {
            (ArrayMultiType::FLOAT(a), ArrayMultiType::FLOAT(b)) => ArrayMultiType::BOOL(less_or_equal(a, b)),
            (ArrayMultiType::UINT8(a), ArrayMultiType::UINT8(b)) => ArrayMultiType::BOOL(less_or_equal(a, b)),
            (ArrayMultiType::INT8(a), ArrayMultiType::INT8(b)) => ArrayMultiType::BOOL(less_or_equal(a, b)),
            (ArrayMultiType::UINT16(a), ArrayMultiType::UINT16(b)) => ArrayMultiType::BOOL(less_or_equal(a, b)),
            (ArrayMultiType::INT16(a), ArrayMultiType::INT16(b)) => ArrayMultiType::BOOL(less_or_equal(a, b)),
            (ArrayMultiType::INT32(a), ArrayMultiType::INT32(b)) => ArrayMultiType::BOOL(less_or_equal(a, b)),
            (ArrayMultiType::INT64(a), ArrayMultiType::INT64(b)) => ArrayMultiType::BOOL(less_or_equal(a, b)),
            _ => panic!("Less or equal op does not support this data type")
        }
    }

    pub fn equal(array_a: &ArrayMultiType, array_b: &ArrayMultiType) -> ArrayMultiType {
        match (array_a, array_b) {
            (ArrayMultiType::FLOAT(a), ArrayMultiType::FLOAT(b)) => ArrayMultiType::BOOL(equal(a, b)),
            (ArrayMultiType::UINT8(a), ArrayMultiType::UINT8(b)) => ArrayMultiType::BOOL(equal(a, b)),
            (ArrayMultiType::INT8(a), ArrayMultiType::INT8(b)) => ArrayMultiType::BOOL(equal(a, b)),
            (ArrayMultiType::UINT16(a), ArrayMultiType::UINT16(b)) => ArrayMultiType::BOOL(equal(a, b)),
            (ArrayMultiType::INT16(a), ArrayMultiType::INT16(b)) => ArrayMultiType::BOOL(equal(a, b)),
            (ArrayMultiType::INT32(a), ArrayMultiType::INT32(b)) => ArrayMultiType::BOOL(equal(a, b)),
            (ArrayMultiType::INT64(a), ArrayMultiType::INT64(b)) => ArrayMultiType::BOOL(equal(a, b)),
            _ => panic!("Equal op does not support this data type")
        }
    }

    pub fn not(array: &ArrayMultiType) -> ArrayMultiType {
        match array {
            ArrayMultiType::BOOL(a) => ArrayMultiType::BOOL(not(a)),
            _ => panic!("Not op does not support this data type")
        }
    }

    pub fn matmul(array_a: &ArrayMultiType, array_b: &ArrayMultiType) -> ArrayMultiType {
        match (array_a, array_b) {
            (ArrayMultiType::FLOAT(a), ArrayMultiType::FLOAT(b)) => ArrayMultiType::FLOAT(matmul(a, b).unwrap()),
            _ => panic!("Matmul op does not support this data type")
        }
    }

    pub fn global_average_pool(array: &ArrayMultiType) -> ArrayMultiType {
        match array {
            ArrayMultiType::FLOAT(a) => ArrayMultiType::FLOAT(global_average_pool(a)),
            _ => panic!("Global average pool op does not support this data type")
        }
    }

    pub fn concat(arrays: Vec<&ArrayMultiType>, axis: isize) -> ArrayMultiType {
        match arrays {
            arrays if arrays.iter().all(|array| matches!(array, ArrayMultiType::FLOAT(_))) => {
                let arrays = arrays.iter().map(|array| {
                    match array {
                        ArrayMultiType::FLOAT(a) => a,
                        _ => panic!("Concat op does not support this data type")
                    }
                }).collect::<Vec<&Array<f32, IxDyn>>>();
                ArrayMultiType::FLOAT(concat(arrays, axis))
            },
            arrays if arrays.iter().all(|array| matches!(array, ArrayMultiType::UINT8(_))) => {
                let arrays = arrays.iter().map(|array| {
                    match array {
                        ArrayMultiType::UINT8(a) => a,
                        _ => panic!("Concat op does not support this data type")
                    }
                }).collect::<Vec<&Array<u8, IxDyn>>>();
                ArrayMultiType::UINT8(concat(arrays, axis))
            },
            arrays if arrays.iter().all(|array| matches!(array, ArrayMultiType::INT8(_))) => {
                let arrays = arrays.iter().map(|array| {
                    match array {
                        ArrayMultiType::INT8(a) => a,
                        _ => panic!("Concat op does not support this data type")
                    }
                }).collect::<Vec<&Array<i8, IxDyn>>>();
                ArrayMultiType::INT8(concat(arrays, axis))
            },
            arrays if arrays.iter().all(|array| matches!(array, ArrayMultiType::UINT16(_))) => {
                let arrays = arrays.iter().map(|array| {
                    match array {
                        ArrayMultiType::UINT16(a) => a,
                        _ => panic!("Concat op does not support this data type")
                    }
                }).collect::<Vec<&Array<u16, IxDyn>>>();
                ArrayMultiType::UINT16(concat(arrays, axis))
            },
            arrays if arrays.iter().all(|array| matches!(array, ArrayMultiType::INT16(_))) => {
                let arrays = arrays.iter().map(|array| {
                    match array {
                        ArrayMultiType::INT16(a) => a,
                        _ => panic!("Concat op does not support this data type")
                    }
                }).collect::<Vec<&Array<i16, IxDyn>>>();
                ArrayMultiType::INT16(concat(arrays, axis))
            },
            arrays if arrays.iter().all(|array| matches!(array, ArrayMultiType::INT32(_))) => {
                let arrays = arrays.iter().map(|array| {
                    match array {
                        ArrayMultiType::INT32(a) => a,
                        _ => panic!("Concat op does not support this data type")
                    }
                }).collect::<Vec<&Array<i32, IxDyn>>>();
                ArrayMultiType::INT32(concat(arrays, axis))
            },
            arrays if arrays.iter().all(|array| matches!(array, ArrayMultiType::INT64(_))) => {
                let arrays = arrays.iter().map(|array| {
                    match array {
                        ArrayMultiType::INT64(a) => a,
                        _ => panic!("Concat op does not support this data type")
                    }
                }).collect::<Vec<&Array<i64, IxDyn>>>();
                ArrayMultiType::INT64(concat(arrays, axis))
            },
            _ => panic!("Concat op does not support this data type")
        }
    }

    pub fn relu(array: &ArrayMultiType) -> ArrayMultiType {
        match array {
            ArrayMultiType::FLOAT(a) => ArrayMultiType::FLOAT(relu(a)),
            ArrayMultiType::UINT8(a) => ArrayMultiType::UINT8(relu(a)),
            ArrayMultiType::INT8(a) => ArrayMultiType::INT8(relu(a)),
            ArrayMultiType::UINT16(a) => ArrayMultiType::UINT16(relu(a)),
            ArrayMultiType::INT16(a) => ArrayMultiType::INT16(relu(a)),
            ArrayMultiType::INT32(a) => ArrayMultiType::INT32(relu(a)),
            ArrayMultiType::INT64(a) => ArrayMultiType::INT64(relu(a)),
            _ => panic!("Relu op does not support this data type")
        }
    }

    pub fn convolution(
        input: &ArrayMultiType, // Input tensor X
        weights: &ArrayMultiType, // Weight tensor W
        bias: Option<&ArrayMultiType>, // Optional Bias tensor B
        auto_pad: &str, // auto_pad attribute
        dilations: &Vec<i64>, // dilations attribute
        group: i64, // group attribute
        kernel_shape: &Vec<i64>, // kernel_shape attribute
        pads: &Vec<i64>, // pads attribute
        strides: &Vec<i64>, // strides attribute
    ) -> ArrayMultiType {
        match (input, weights, bias) {
            (ArrayMultiType::FLOAT(input), ArrayMultiType::FLOAT(weights), Some(ArrayMultiType::FLOAT(bias))) => {
                ArrayMultiType::FLOAT(convolution(input, weights, Some(bias), auto_pad, dilations, group, kernel_shape, pads, strides))
            },
            (ArrayMultiType::FLOAT(input), ArrayMultiType::FLOAT(weights), None) => {
                ArrayMultiType::FLOAT(convolution(input, weights, None, auto_pad, dilations, group, kernel_shape, pads, strides))
            },
            (ArrayMultiType::UINT8(input), ArrayMultiType::UINT8(weights), Some(ArrayMultiType::UINT8(bias))) => {
                ArrayMultiType::UINT8(convolution(input, weights, Some(bias), auto_pad, dilations, group, kernel_shape, pads, strides))
            },
            (ArrayMultiType::UINT8(input), ArrayMultiType::UINT8(weights), None) => {
                ArrayMultiType::UINT8(convolution(input, weights, None, auto_pad, dilations, group, kernel_shape, pads, strides))
            },
            (ArrayMultiType::INT8(input), ArrayMultiType::INT8(weights), Some(ArrayMultiType::INT8(bias))) => {
                ArrayMultiType::INT8(convolution(input, weights, Some(bias), auto_pad, dilations, group, kernel_shape, pads, strides))
            },
            (ArrayMultiType::INT8(input), ArrayMultiType::INT8(weights), None) => {
                ArrayMultiType::INT8(convolution(input, weights, None, auto_pad, dilations, group, kernel_shape, pads, strides))
            },
            (ArrayMultiType::UINT16(input), ArrayMultiType::UINT16(weights), Some(ArrayMultiType::UINT16(bias))) => {
                ArrayMultiType::UINT16(convolution(input, weights, Some(bias), auto_pad, dilations, group, kernel_shape, pads, strides))
            },
            (ArrayMultiType::UINT16(input), ArrayMultiType::UINT16(weights), None) => {
                ArrayMultiType::UINT16(convolution(input, weights, None, auto_pad, dilations, group, kernel_shape, pads, strides))
            },
            (ArrayMultiType::INT16(input), ArrayMultiType::INT16(weights), Some(ArrayMultiType::INT16(bias))) => {
                ArrayMultiType::INT16(convolution(input, weights, Some(bias), auto_pad, dilations, group, kernel_shape, pads, strides))
            },
            (ArrayMultiType::INT16(input), ArrayMultiType::INT16(weights), None) => {
                ArrayMultiType::INT16(convolution(input, weights, None, auto_pad, dilations, group, kernel_shape, pads, strides))
            },
            (ArrayMultiType::INT32(input), ArrayMultiType::INT32(weights), Some(ArrayMultiType::INT32(bias))) => {
                ArrayMultiType::INT32(convolution(input, weights, Some(bias), auto_pad, dilations, group, kernel_shape, pads, strides))
            },
            (ArrayMultiType::INT32(input), ArrayMultiType::INT32(weights), None) => {
                ArrayMultiType::INT32(convolution(input, weights, None, auto_pad, dilations, group, kernel_shape, pads, strides))
            },
            (ArrayMultiType::INT64(input), ArrayMultiType::INT64(weights), Some(ArrayMultiType::INT64(bias))) => {
                ArrayMultiType::INT64(convolution(input, weights, Some(bias), auto_pad, dilations, group, kernel_shape, pads, strides))
            },
            (ArrayMultiType::INT64(input), ArrayMultiType::INT64(weights), None) => {
                ArrayMultiType::INT64(convolution(input, weights, None, auto_pad, dilations, group, kernel_shape, pads, strides))
            },
            _ => panic!("Convolution op does not support this data type")
        }
    }

    pub fn transpose(array: &ArrayMultiType, axes: Option<Vec<usize>>) -> ArrayMultiType {
        match array {
            ArrayMultiType::FLOAT(a) => ArrayMultiType::FLOAT(transpose(a, axes)),
            ArrayMultiType::UINT8(a) => ArrayMultiType::UINT8(transpose(a, axes)),
            ArrayMultiType::INT8(a) => ArrayMultiType::INT8(transpose(a, axes)),
            ArrayMultiType::UINT16(a) => ArrayMultiType::UINT16(transpose(a, axes)),
            ArrayMultiType::INT16(a) => ArrayMultiType::INT16(transpose(a, axes)),
            ArrayMultiType::INT32(a) => ArrayMultiType::INT32(transpose(a, axes)),
            ArrayMultiType::INT64(a) => ArrayMultiType::INT64(transpose(a, axes)),
            _ => panic!("Transpose op does not support this data type")
        }
    }

    pub fn tile(array: &ArrayMultiType, repeats: &Vec<usize>) -> ArrayMultiType {
        match array {
            ArrayMultiType::FLOAT(a) => ArrayMultiType::FLOAT(tile(a, repeats)),
            ArrayMultiType::UINT8(a) => ArrayMultiType::UINT8(tile(a, repeats)),
            ArrayMultiType::INT8(a) => ArrayMultiType::INT8(tile(a, repeats)),
            ArrayMultiType::UINT16(a) => ArrayMultiType::UINT16(tile(a, repeats)),
            ArrayMultiType::INT16(a) => ArrayMultiType::INT16(tile(a, repeats)),
            ArrayMultiType::INT32(a) => ArrayMultiType::INT32(tile(a, repeats)),
            ArrayMultiType::INT64(a) => ArrayMultiType::INT64(tile(a, repeats)),
            _ => panic!("Tile op does not support this data type")
        }
    }

    pub fn gather(array: &ArrayMultiType, indices: &Vec<usize>, axis: usize) -> ArrayMultiType {
        match array {
            ArrayMultiType::FLOAT(a) => ArrayMultiType::FLOAT(gather(a, indices, axis)),
            ArrayMultiType::UINT8(a) => ArrayMultiType::UINT8(gather(a, indices, axis)),
            ArrayMultiType::INT8(a) => ArrayMultiType::INT8(gather(a, indices, axis)),
            ArrayMultiType::UINT16(a) => ArrayMultiType::UINT16(gather(a, indices, axis)),
            ArrayMultiType::INT16(a) => ArrayMultiType::INT16(gather(a, indices, axis)),
            ArrayMultiType::INT32(a) => ArrayMultiType::INT32(gather(a, indices, axis)),
            ArrayMultiType::INT64(a) => ArrayMultiType::INT64(gather(a, indices, axis)),
            _ => panic!("Gather op does not support this data type")
        }
            
    }
}
