use crate::onnx::{AttributeProto, attribute_proto::AttributeType};
use crate::array::ArrayMultiType;
use crate::utils::init_array;

#[derive(Debug, Clone)]
pub enum Attribute {
    Float(f32),
    Int(i64),
    String(String),
    Tensor(ArrayMultiType),
    Floats(Vec<f32>),
    Ints(Vec<i64>),
    Strings(Vec<String>),
    Tensors(Vec<ArrayMultiType>)
}

impl Attribute {
    pub fn from_proto(attribute: &AttributeProto) -> Result<Attribute, &'static str> {
        match attribute {
            AttributeProto { ref r#type, ref f, ref i, ref s, ref t, ref floats, ref ints, ref strings, ref tensors, .. } => {
                match *r#type {
                    x if x == AttributeType::Float as i32 => Ok(Attribute::Float(*f)),
                    x if x == AttributeType::Int as i32 => Ok(Attribute::Int(*i)),
                    x if x == AttributeType::String as i32 => Ok(Attribute::String(String::from_utf8(s.to_vec()).unwrap())),
                    x if x == AttributeType::Tensor as i32 => {
                        let tensor = match t {
                            Some(tensor) => init_array(tensor)?,
                            _ => return Err("Invalid tensor")
                        };

                        Ok(Attribute::Tensor(tensor.1))
                    },
                    x if x == AttributeType::Floats as i32 => Ok(Attribute::Floats(floats.clone())),
                    x if x == AttributeType::Ints as i32 => Ok(Attribute::Ints(ints.clone())),
                    x if x == AttributeType::Strings as i32 => Ok(Attribute::Strings(strings.into_iter().map(|s| String::from_utf8(s.to_vec()).unwrap()).collect())),
                    x if x == AttributeType::Tensors as i32 => {
                        let mut tensors_vec= tensors.iter().map(|tensor| init_array(tensor).unwrap().1).collect::<Vec<ArrayMultiType>>();

                        Ok(Attribute::Tensors(tensors_vec))
                    },
                    _ => Err("Attribute type does not implemented")
                }
            }
        }
    }
}

