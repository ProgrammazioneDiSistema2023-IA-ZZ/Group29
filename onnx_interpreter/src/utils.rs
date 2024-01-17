use std::collections::HashMap;

use crate::onnx::type_proto::Value as ProtoValue;
use crate::onnx::{ValueInfoProto, GraphProto, TensorProto};
use crate::array::ArrayMultiType;

fn init_array(initializer: &TensorProto) -> Result<(String, ArrayMultiType), &'static str> {  
    let array = if initializer.float_data.len() > 0 {
        ArrayMultiType::from_vec_f32(&initializer.float_data, &initializer.dims)?
    } else if initializer.int32_data.len() > 0 {
        ArrayMultiType::from_vec_i32(&initializer.int32_data, &initializer.dims)?
    } else if initializer.int64_data.len() > 0 {
        ArrayMultiType::from_vec_i64(&initializer.int64_data, &initializer.dims)?
    } else {
        ArrayMultiType::from_bytes(&initializer.raw_data, &initializer.dims, initializer.data_type)?
    };
    
    let name = initializer.name.clone();
    Ok((name, array))
}

fn random_array(info: &ValueInfoProto) -> Result<(String, ArrayMultiType), &'static str> {
    let tensor_info = match &info.r#type {
        Some(proto) => match &proto.value {
            Some(ProtoValue::TensorType(tensor)) => tensor,
            _ => return Err("TypeProto is not a TensorType")
        },
        _ => return Err("Invalid tensor info")
    };

    let shape = match &tensor_info.shape {
        Some(shape) => shape,
        _ => return Err("Invalid tensor shape")
    };

    let array = ArrayMultiType::random(shape, tensor_info.elem_type);
    let name = info.name.clone();
    Ok((name, array))
}
pub fn get_inputs(graph: &GraphProto) -> Result<HashMap<String, ArrayMultiType>, &'static str> {
    let mut inputs = HashMap::new();
    
    for initializer in graph.initializer.iter() {
        let (name, array) = init_array(&initializer)?;
        inputs.insert(name, array);
    }

    for input in graph.input.iter() {
        if !inputs.contains_key(&input.name) {
            let (name, array) = random_array(&input)?;
            inputs.insert(name, array);
        }
    }

    Ok(inputs)
}



