use ordered_float::OrderedFloat;

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
