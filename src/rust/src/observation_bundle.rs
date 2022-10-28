use ordered_float::OrderedFloat;

// ObservationBundle Struct Associates Several observations with the same predictor value into a
// bundle, so they can all be removed / added to a leaf at once
pub struct ObservationBundle{
    pub cut_point : OrderedFloat<f64>,
    pub indexes : Vec<usize>
}

impl ObservationBundle{
    pub fn new(cut_point: OrderedFloat<f64>, index : usize) -> Self{
        ObservationBundle{
            cut_point : cut_point,
            indexes : vec![index]
        }
    }

    pub fn add(&mut self, index : usize) {
        self.indexes.push(index)
    }
}
