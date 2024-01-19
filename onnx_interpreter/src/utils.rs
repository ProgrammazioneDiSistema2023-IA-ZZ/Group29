use std::collections::HashMap;
use std::thread;
use std::sync::{Arc, Mutex};

use crate::onnx::attribute_proto::AttributeType;
use crate::onnx::type_proto::Value as ProtoValue;
use crate::onnx::{ValueInfoProto, GraphProto, TensorProto, AttributeProto, NodeProto};
use crate::array::ArrayMultiType;
use crate::attribute::Attribute;

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
    let shared_inputs = Arc::new(Mutex::new(HashMap::new()));
    let mut threads = Vec::new();

    for input in graph.input.iter() {
        let input_clone = input.clone(); // Clone the input variable
        let shared_inputs_clone = Arc::clone(&shared_inputs); // Clone the shared_inputs variable
        threads.push(thread::spawn(move || {
            let (name, array) = random_array(&input_clone).unwrap();
            let mut inputs = shared_inputs_clone.lock().unwrap();
            inputs.insert(name, array);
        }));
    }

    for initializer in graph.initializer.iter() {
        let initializer_clone = initializer.clone(); // Clone the initializer variable
        let shared_inputs_clone = Arc::clone(&shared_inputs); // Clone the shared_inputs variable
        threads.push(thread::spawn(move || {
            let (name, array) = init_array(&initializer_clone).unwrap();
            let mut inputs = shared_inputs_clone.lock().unwrap();
            inputs.insert(name, array);
        }));
    }

    for thread in threads {
        thread.join().unwrap();
    }

    let inputs = shared_inputs.lock().unwrap();
    Ok(inputs.clone())
}

pub fn get_attributes(node: &NodeProto) -> Result<HashMap<String, Attribute>, &'static str> {
    let shared_attributes = Arc::new(Mutex::new(HashMap::new()));
    let mut threads = Vec::new();

    for attribute in node.attribute.iter() {
        let attribute_clone = attribute.clone(); // Clone the attribute variable
        let shared_attributes_clone = Arc::clone(&shared_attributes); // Clone the shared_attributes variable
        threads.push(thread::spawn(move || {
            let attribute = Attribute::from_proto(&attribute_clone).unwrap();
            let mut attributes = shared_attributes_clone.lock().unwrap();
            attributes.insert(attribute_clone.name.clone(), attribute);
        }));
    }

    for thread in threads {
        thread.join().unwrap();
    }

    let attributes = shared_attributes.lock().unwrap();
    Ok(attributes.clone())
}


