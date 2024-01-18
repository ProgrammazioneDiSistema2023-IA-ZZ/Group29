use crate::onnx::{AttributeProto, attribute_proto::AttributeType};

pub enum Attribute {
    Float(f32),
    Int(i64),
    String(String),
    Floats(Vec<f32>),
    Ints(Vec<i64>),
    Strings(Vec<String>)
}

impl Attribute {
    pub fn from_proto(attribute: &AttributeProto) -> Result<Attribute, &'static str> {
        match attribute {
            AttributeProto { ref r#type, ref f, ref i, ref s, ref floats, ref ints, ref strings, .. } => {
                match *r#type {
                    x if x == AttributeType::Float as i32 => Ok(Attribute::Float(*f)),
                    x if x == AttributeType::Int as i32 => Ok(Attribute::Int(*i)),
                    x if x == AttributeType::String as i32 => Ok(Attribute::String(String::from_utf8(s.to_vec()).unwrap())),
                    x if x == AttributeType::Floats as i32 => Ok(Attribute::Floats(floats.clone())),
                    x if x == AttributeType::Ints as i32 => Ok(Attribute::Ints(ints.clone())),
                    x if x == AttributeType::Strings as i32 => Ok(Attribute::Strings(strings.into_iter().map(|s| String::from_utf8(s.to_vec()).unwrap()).collect())),
                    _ => Err("Attribute type does not implemented")
                }
            }
        }
    }
}

