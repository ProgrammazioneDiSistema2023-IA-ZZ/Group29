use core::panic;

use ndarray::{Array, IxDyn};
use onnx::onnx::TensorProto_DataType;
use ndarray_rand::RandomExt;
use rand::distributions::{Uniform, uniform::SampleUniform};

use crate::onnx::{TensorShapeProto, tensor_shape_proto::dimension::Value};
use crate::operations::*;

#[derive(Debug)]
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
        match d.value.clone().unwrap() {
            Value::DimValue(v) => v as usize,
            Value::DimParam(_) => 1 as usize
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

    pub fn matmul(array_a: &ArrayMultiType, array_b: &ArrayMultiType) -> ArrayMultiType {
        match (array_a, array_b) {
            (ArrayMultiType::FLOAT(a), ArrayMultiType::FLOAT(b)) => ArrayMultiType::FLOAT(matmul(a, b).unwrap()),
            _ => panic!("Matmul op does not support this data type")
        }
    }


}


            
