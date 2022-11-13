use extendr_api::prelude::*;
use iter_utils::argmax;
use ordered_float::OrderedFloat;
use rayon::prelude::*;
use std::collections::BTreeMap;

use std::collections::HashSet;

pub mod node;
use crate::node::Node;

pub mod observation_bundle;
use crate::observation_bundle::ObservationBundle;

// Creates Sorted Sets from a view of the original datasets. Sorts sets using binary trees, then
// turns them into a vector of observation bundles to save time
fn new_sorted_sets(dataset: ArrayView2<OrderedFloat<f64>>) -> (Vec<Vec<ObservationBundle>>, Array2<usize>)
                                                               {
    // Create new vetor to store binary tree maps
    let mut btree_vec = Vec::new();
    for _ in dataset.axis_iter(Axis(1)) {
        let tree: BTreeMap<OrderedFloat<f64>, ObservationBundle> = BTreeMap::new();
        btree_vec.push(tree);
    }

    // enter every row of the dataset into the sorted sets
    for (x, row) in dataset.axis_iter(Axis(0)).enumerate() {
        for (y, entry) in row.iter().enumerate() {
            if !btree_vec[y].contains_key(entry) {
                btree_vec[y].insert(*entry, ObservationBundle::new(*entry, x));
            } else {
                let arr = btree_vec[y].get_mut(entry).unwrap();
                arr.add(x);
            }
        }
    }

    // move from binary trees to vectors of ObservationBundles
    let mut sorted_sets = Vec::new();
    for btree in btree_vec {
        let mut sorted_set = Vec::new();
        for obs_bundle in btree.into_values() {
            sorted_set.push(obs_bundle);
        }
        sorted_sets.push(sorted_set);
    }

    let mut obs_to_bundle : Array2<usize> = Array2::from_elem(dataset.dim(), 0);
    for (col, set) in sorted_sets.iter().enumerate() {
        for (set_index, bundle) in set.iter().enumerate() {
            for obs_index in &bundle.indexes {
                obs_to_bundle[[*obs_index,col]] = set_index;
            };
        };
    };


    (sorted_sets, obs_to_bundle)

}

// Tree Search Struct. Keeps a reference to the sorted sets, but does not change them so they don't
// have to be copied / modified. The observations that are in consideration are stored in the
// `active` field, which is a boolean vector Also keeps track of the utility from giving every unit
// each of the possible treatments, which cuts out the use of an array in the `search single
// dimension` part of the algotithm
#[derive(Clone)]
struct TreeSearcher<'a> {
    sets: Vec<Vec<ObservationBundle>>,
    scores: ArrayView2<'a, OrderedFloat<f64>>,
    sample_to_bundle: ArrayView2<'a, usize>,
    max_treatment_utils: Array1<OrderedFloat<f64>>,
}

impl<'a> TreeSearcher<'a> {
    fn new_empty(
        sets: & Vec<Vec<ObservationBundle>>,
        scores: ArrayView2<'a, OrderedFloat<f64>>,
        sample_to_bundle: ArrayView2<'a, usize>
    ) -> Self {
        let mut empty_sets_vec : Vec<Vec<ObservationBundle>> = Vec::new();

        for set in sets {
            let mut temp_vec : Vec<ObservationBundle> = Vec::new();
            for bundle in set {
                temp_vec.push(ObservationBundle { cut_point: bundle.cut_point, indexes: HashSet::new() })
            }
            empty_sets_vec.push(temp_vec);
        }


        TreeSearcher {
            sets: empty_sets_vec,
            scores,
            sample_to_bundle,
            max_treatment_utils: Array1::from_elem(scores.dim().1, OrderedFloat(0.0)),
        }
    }

    fn new_full(
        sets: Vec<Vec<ObservationBundle>>,
        scores: ArrayView2<'a, OrderedFloat<f64>>,
        sample_to_bundle: ArrayView2<'a, usize>
    ) -> Self {
        let out = TreeSearcher {
            sets,
            scores,
            sample_to_bundle,
            max_treatment_utils: scores.sum_axis(Axis(0)),
        };

        out
    }

    fn add(&mut self, index: usize) {
        self.max_treatment_utils += &self.scores.index_axis(Axis(0), index);

        for (col, bundle_index) in self.sample_to_bundle.index_axis(Axis(0), index).iter().enumerate() {
            self.sets[col][*bundle_index].add(index)
        };

    }

    fn remove(&mut self, index: usize) {
        self.max_treatment_utils -= &self.scores.index_axis(Axis(0), index);

        for (col, bundle_index) in self.sample_to_bundle.index_axis(Axis(0), index).iter().enumerate() {
            self.sets[col][*bundle_index].remove(index)
        };
    }

    // Search Single Split. Direct Analogue of the algorithm from the paper, but the rewards from
    // assigning every unit a treatment are already calculated, so no arrays are needed.
    fn search_single_split(&self) -> Node {
        let nd: usize = self.scores.dim().1;
        let np: usize = self.sets.len();

        let mut best_r_leaf = Node::new_leaf(OrderedFloat(-f64::INFINITY), 0);
        let mut best_l_leaf = Node::new_leaf(OrderedFloat(-f64::INFINITY), 0);

        let mut best_axis: usize = 0;
        let mut best_cut_point: OrderedFloat<f64> = OrderedFloat(0.0);

        for p in 0..np {
            let mut current_l_rewards = Array1::from_elem(nd, OrderedFloat(0.0));
            let mut current_r_rewards = self.max_treatment_utils.clone();

            for bundle in self.sets[p].iter() {
                for row_idx in bundle.indexes.iter() {
                    current_l_rewards += &self.scores.index_axis(Axis(0), *row_idx);
                    current_r_rewards -= &self.scores.index_axis(Axis(0), *row_idx);
                }

                let current_l_idx = argmax(current_l_rewards.iter()).unwrap();
                let current_r_idx = argmax(current_r_rewards.iter()).unwrap();

                let current_l_reward = current_l_rewards[current_l_idx];
                let current_r_reward = current_r_rewards[current_r_idx];

                if (current_l_reward + current_r_reward) > (best_l_leaf.reward + best_r_leaf.reward)
                {
                    best_axis = p;
                    best_cut_point = bundle.cut_point;

                    best_l_leaf.reward = current_l_reward;
                    best_r_leaf.reward = current_r_reward;

                    best_l_leaf.action = Some(current_l_idx);
                    best_r_leaf.action = Some(current_r_idx);
                }
            }
        }

        Node::new_branch(best_l_leaf, best_r_leaf, best_axis, best_cut_point)
    }

    // Single dimension recursive search. Runs an exhaustive search, but is only able to consider splits along one axis in the top node.
    // Used for paralleization with Rayon.
    fn single_dimension_recursive_search(&self, dim: usize, depth: usize) -> Node {
        let mut best_r_tree = Node::new_leaf(OrderedFloat(-f64::INFINITY), 0);
        let mut best_l_tree = Node::new_leaf(OrderedFloat(-f64::INFINITY), 0);

        let mut best_split_point: OrderedFloat<f64> = OrderedFloat(0.0);
        let mut best_reward: OrderedFloat<f64> = OrderedFloat(-f64::INFINITY);

        let mut sets_r = self.clone();
        let mut sets_l = Self::new_empty(&self.sets, self.scores, self.sample_to_bundle);

        for bundle_index in 0..sets_r.sets[dim].len() {
            let cut_point = sets_r.sets[dim][bundle_index].cut_point;

            for index in sets_r.sets[dim][bundle_index].indexes.clone() {
                sets_l.add(index);
                sets_r.remove(index);
            }


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

    // Proper recursive tree search. Taken almost directly from policytree package.
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

            let mut best_r_tree = Node::new_leaf(OrderedFloat(-f64::INFINITY), 0);
            let mut best_l_tree = Node::new_leaf(OrderedFloat(-f64::INFINITY), 0);
            let mut best_split_axis: usize = 0;
            let mut best_split_point: OrderedFloat<f64> = OrderedFloat(0.0);
            let mut best_reward: OrderedFloat<f64> = OrderedFloat(-f64::INFINITY);

            for p in 0..np {
                let mut sets_l = Self::new_empty(&self.sets, self.scores, self.sample_to_bundle);
                let mut sets_r = self.clone();

                for bundle_index in 0..sets_r.sets[p].len() {
                    let cut_point = sets_r.sets[p][bundle_index].cut_point;

                    for index in sets_r.sets[p][bundle_index].indexes.clone() {
                        sets_l.add(index);
                        sets_r.remove(index);
                    }

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

            return Node::new_branch(best_l_tree, best_r_tree, best_split_axis, best_split_point);
        }
    }
}

// function called from R. Process data into matrix of OrderedFloats, then run search.
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

    let (sorted_sets, obs_to_bundle) = new_sorted_sets(x_mat.view());

    let searcher = TreeSearcher::new_full(sorted_sets, scores_mat.view(), obs_to_bundle.view());

    let search_results = searcher.recursive_tree_search(depth as usize, true);

    search_results.r_representation()
}

// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod parallelpolicytree;
    fn rust_exhaustive_tree;
}
