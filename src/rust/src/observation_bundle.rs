use ordered_float::OrderedFloat;
use std::collections::HashSet;

// ObservationBundle Struct Associates Several observations with the same predictor value into a
// bundle, so they can all be removed / added to a leaf at once
#[derive(Debug, Clone)]
pub struct ObservationBundle{
    pub cut_point : OrderedFloat<f64>,
    pub indexes : HashSet<usize>
}

impl ObservationBundle{
    pub fn new(cut_point: OrderedFloat<f64>, index : usize) -> Self{
        ObservationBundle{
            cut_point,
            indexes : HashSet::from([index])
        }
    }

    pub fn add(&mut self, index : usize) {
        self.indexes.insert(index);
    }

    pub fn remove(&mut self, index : usize) {
        self.indexes.remove(&index);
    }
}
