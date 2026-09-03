use std::borrow::Cow;
use std::collections::HashMap;

use super::{FunctionRepr, Object};

#[derive(Clone)]
pub struct Env {
    pub constants: HashMap<String, Object>,
    pub functions: HashMap<String, FunctionRepr>
}

impl Env {
    /// For every non-`DirectFunction`-entry in `other`, updates `self` to correspond to that entry.
    pub fn update(&mut self, other: Env) {
        for (s, c) in other.constants {
            self.constants.insert(s, c);
        }
        for (s, f) in other.functions {
            if let FunctionRepr::ByExpression(..) = &f {
                self.functions.insert(s, f);
            }
        }
    }
}


pub trait VarStackLookup {
    fn lookup<'a>(&'a self, key: &String) -> Option<&'a Object>;
}

// #[derive(Debug)]
// pub enum VarStack<'a> {
//     Empty,
//     Frame {
//         vars: Cow<'a, HashMap<&'a String, Cow<'a, Object>>>,
//         parent: &'a VarStack<'a>
//     }
// }
pub enum VarStack<'a, 'p> {
    Empty,
    Frame {
        vars: Cow<'a, HashMap<&'a String, Cow<'a, Object>>>,
        parent: &'p dyn VarStackLookup
    }
}

// impl<'a> VarStack<'a> {
//     pub fn lookup(&self, key: &String) -> Option<&Object> {
//         match self {
//             VarStack::Empty => None,
//             VarStack::Frame { vars, parent } => {
//                 vars.get(key)
//                 .map(|cow| cow.as_ref())
//                 .or_else(|| parent.lookup(key))
//             }
//         }
//     }

//     /// Adds a frame `(ident, value)` on top of `self` and returns the result.
//     // pub fn with<'b>(&'a self, ident: &'b String, value: &'b Object) -> VarStack<'b> where 'a: 'b {
//     //     VarStack::Frame {
//     //         vars: Cow::Owned(HashMap::from([(ident, Cow::Borrowed(value))])),
//     //         parent: self
//     //     }
//     // }
//     pub fn with(&'a self, ident: &'a String, value: &'a Object) -> VarStack<'a> {
//         VarStack::Frame {
//             vars: Cow::Owned(HashMap::from([(ident, Cow::Borrowed(value))])),
//             parent: self
//         }
//     }

// }

impl<'a, 'p> VarStackLookup for VarStack<'a, 'p> {
    fn lookup(&self, key: &String) -> Option<&Object> {
        match self {
            VarStack::Empty => None,
            VarStack::Frame { vars, parent } => {
                vars.get(key)
                .map(|cow| cow.as_ref())
                .or_else(|| parent.lookup(key))
            }
        }
    }
}

impl<'a, 'p> VarStack<'a, 'p> {
    pub fn get_top_level(&self) -> Option<&Cow<'a, HashMap<&'a String, Cow<'a, Object>>>> {
        match self {
            VarStack::Empty => None,
            VarStack::Frame { vars, parent: _ } => Some(&vars)
        }
    }

    /// Adds a frame `(ident, value)` on top of `self` and returns the result.
    pub fn with<'v, 'q>(&'q self, ident: &'v String, value: Cow<'v, Object>) -> VarStack<'v, 'q> {
        VarStack::Frame {
            vars: Cow::Owned(HashMap::from([(ident, value)])),
            parent: self
        }
    }

    /// Adds a frame with the given information on top of `self` and returns the result.
    pub fn with_multiple<'v, 'q>(&'q self, identifiers: impl Iterator<Item=&'v String>, values: impl Iterator<Item=&'v Object>) -> VarStack<'v, 'q>
    {
        VarStack::Frame {
            vars: Cow::Owned(identifiers.zip(values.map(Cow::Borrowed)).collect()),
            parent: self
        }
    }
}