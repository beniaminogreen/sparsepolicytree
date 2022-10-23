use extendr_api::prelude::*;
use iter_utils::argmax;
use ordered_float::OrderedFloat;

use rayon::prelude::*;

use std::collections::BTreeMap;

pub mod node;
use crate::node::Node;

// struct ObservationBundle{
//     cut_point : OrderedFloat<f64>,
//     indexes : Vec<usize>
// }

#[derive(Debug, Clone)]
struct SortedSets<'a> {
    sets: Vec<BTreeMap<OrderedFloat<f64>, Vec<usize>>>,
    dataset: ArrayView2<'a, OrderedFloat<f64>>,
    scores: ArrayView2<'a, OrderedFloat<f64>>,
    max_treatment_utils: Array1<OrderedFloat<f64>>,
}

impl<'a> SortedSets<'a> {
    fn new_empty(
        dataset: ArrayView2<'a, OrderedFloat<f64>>,
        scores: ArrayView2<'a, OrderedFloat<f64>>,
    ) -> Self {
        let mut output = SortedSets {
            sets: Vec::new(),
            dataset: dataset,
            scores: scores,
            max_treatment_utils: Array1::from_elem(scores.dim().1, OrderedFloat(0.0)),
        };

        for _ in dataset.axis_iter(Axis(1)) {
            let tree: BTreeMap<OrderedFloat<f64>, Vec<usize>> = BTreeMap::new();
            output.sets.push(tree);
        }
        output
    }

    fn new_populated(
        dataset: ArrayView2<'a, OrderedFloat<f64>>,
        scores: ArrayView2<'a, OrderedFloat<f64>>,
    ) -> Self {
        let mut tree = Self::new_empty(dataset, scores);

        for i in 0..dataset.dim().0 {
            tree.add_by_index(i);
        }

        tree
    }

    fn remove_by_key(&mut self, starting_index: usize, key: &OrderedFloat<f64>) {
        let (_, row_idxes) = self.sets[starting_index]
            .remove_entry(key)
            .expect("Key Not Found While Removing");

        for i in row_idxes {
            for j in 0..self.dataset.dim().1 {
                if j != starting_index {
                    let key = OrderedFloat(self.dataset[[i, j]]);
                    let arr: &mut Vec<usize> = self.sets[j].get_mut(&key).expect("Unable to find");
                    if arr.len() == 1 {
                        self.sets[j].remove(&key).expect("failed to remove!");
                    } else {
                        arr.retain(|x| x != &i);
                    }
                }
            }
            self.max_treatment_utils -= &self.scores.slice(s![i, ..]);
        }
    }

    fn add_by_index(&mut self, index: usize) {
        let row = self.dataset.slice(s![index, ..]);
        for (i, entry) in row.iter().enumerate() {
            if !self.sets[i].contains_key(entry) {
                self.sets[i].insert(*entry, vec![index]);
            } else {
                let arr = self.sets[i].get_mut(entry).unwrap();
                arr.push(index);
            }
        }
        self.max_treatment_utils += &self.scores.slice(s![index, ..]);
    }

    fn search_single_split(&self) -> Node {
        let nd: usize = self.scores.dim().1;
        let np: usize = self.sets.len();

        let mut best_r_leaf = Node::new_leaf(OrderedFloat(-999999999999.9), 0);
        let mut best_l_leaf = Node::new_leaf(OrderedFloat(-999999999999.9), 0);

        let mut best_axis: usize = 0;
        let mut best_cut_point: OrderedFloat<f64> = OrderedFloat(-999999999999.9);

        for p in 0..np {
            let mut current_l_rewards = Array1::from_elem(nd, OrderedFloat(0.0));
            let mut current_r_rewards = self.max_treatment_utils.clone();
            for (cut_point, row_indexes) in self.sets[p].iter() {
                for row_idx in row_indexes {
                    current_l_rewards += &self.scores.slice(s![*row_idx, ..]);
                    current_r_rewards -= &self.scores.slice(s![*row_idx, ..]);
                }

                let current_l_idx = argmax(current_l_rewards.iter()).unwrap();
                let current_r_idx = argmax(current_r_rewards.iter()).unwrap();

                let current_l_reward = current_l_rewards[current_l_idx];
                let current_r_reward = current_r_rewards[current_r_idx];

                if (current_l_reward + current_r_reward) > (best_l_leaf.reward + best_r_leaf.reward)
                {
                    best_axis = p;
                    best_cut_point = *cut_point;

                    best_l_leaf.reward = current_l_reward;
                    best_r_leaf.reward = current_r_reward;

                    best_l_leaf.action = Some(current_l_idx);
                    best_r_leaf.action = Some(current_r_idx);
                }
            }
        }
        Node::new_branch(best_l_leaf, best_r_leaf, best_axis, best_cut_point)
    }

    fn single_dimension_recursive_search(&self, dim: usize, depth: usize) -> Node {
        let mut best_r_tree = Node::new_leaf(OrderedFloat(-999999999999.9), 0);
        let mut best_l_tree = Node::new_leaf(OrderedFloat(-999999999999.9), 0);

        let mut best_split_point: OrderedFloat<f64> = OrderedFloat(0.0);
        let mut best_reward: OrderedFloat<f64> = OrderedFloat(-99999999999.99);

        let mut sets_r = self.clone();
        let mut sets_l = Self::new_empty(self.dataset, self.scores);

        let n_cuts = sets_r.sets[dim].len();
        for _ in 0..(n_cuts) {
            let (cut_point, ids) = sets_r.sets[dim].first_key_value().unwrap();
            let cut_point = cut_point.clone();

            // TODO: This is extremely expensive. Try and cut it out

            for index in ids {
                sets_l.add_by_index(*index);
            }

            sets_r.remove_by_key(dim, &cut_point);

            let tree_l = sets_l.recursive_tree_search(depth - 1, false);
            let tree_r = sets_r.recursive_tree_search(depth - 1, false);

            let current_reward = tree_l.reward + tree_r.reward;

            if current_reward > best_reward {
                best_l_tree = tree_l;
                best_r_tree = tree_r;
                best_reward = current_reward;
                best_split_point = cut_point;
            }
        }

        return Node::new_branch(best_l_tree, best_r_tree, dim, best_split_point);
    }

    fn recursive_tree_search(&self, depth: usize, top: bool) -> Node {
        if depth == 1 {
            return self.search_single_split();
        } else if top {
            let np: usize = self.sets.len();

            (0..np)
                .into_par_iter()
                .map(|dim| self.single_dimension_recursive_search(dim, depth))
                .max()
                .unwrap()
        } else {
            let np: usize = self.sets.len();

            let mut best_r_tree = Node::new_leaf(OrderedFloat(-999999999999.9), 0);
            let mut best_l_tree = Node::new_leaf(OrderedFloat(-999999999999.9), 0);
            let mut best_split_axis: usize = 0;
            let mut best_split_point: OrderedFloat<f64> = OrderedFloat(0.0);
            let mut best_reward: OrderedFloat<f64> = OrderedFloat(-99999999999.99);

            for p in 0..np {
                let mut sets_r = self.clone();
                let mut sets_l = Self::new_empty(self.dataset, self.scores);

                let n_cuts = sets_r.sets[p].len();
                for _ in 0..(n_cuts) {
                    let (cut_point, ids) = sets_r.sets[p].first_key_value().unwrap();
                    let cut_point = cut_point.clone();

                    // TODO: very expensive. try and remove
                    for index in ids {
                        sets_l.add_by_index(*index);
                    }

                    sets_r.remove_by_key(p, &cut_point);

                    let tree_l = sets_l.recursive_tree_search(depth - 1, false);
                    let tree_r = sets_r.recursive_tree_search(depth - 1, false);

                    let current_reward = tree_l.reward + tree_r.reward;

                    if current_reward > best_reward {
                        best_l_tree = tree_l;
                        best_r_tree = tree_r;
                        best_reward = current_reward;
                        best_split_axis = p;
                        best_split_point = cut_point;
                    }
                }
            }

            return Node::new_branch(best_l_tree, best_r_tree, best_split_axis, best_split_point)
        }
    }
}


#[extendr]
fn rust_exhaustive_tree(x_robj: Robj, gamma_robj: Robj, depth: i64) -> List {
    let x_mat = <ArrayView2<f64>>::from_robj(&x_robj)
        .unwrap()
        .to_owned()
        .map(|x| OrderedFloat(*x));
    let scores_mat = <ArrayView2<f64>>::from_robj(&gamma_robj)
        .unwrap()
        .to_owned()
        .map(|x| OrderedFloat(*x));

    let test = SortedSets::new_populated(x_mat.view(), scores_mat.view());

    let search_results = test.recursive_tree_search(depth as usize, true);

    search_results.r_representation()
}

// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod exhaustivetree;
    fn rust_exhaustive_tree;
}
