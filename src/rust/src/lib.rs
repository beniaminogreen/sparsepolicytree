// #![feature(map_first_last)]
use extendr_api::prelude::*;
use iter_utils::argmax;
use ordered_float::OrderedFloat;

use rayon::prelude::*;

// use std::collections::HashSet;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::collections::VecDeque;

fn variant_eq<T>(a: &T, b: &T) -> bool {
    std::mem::discriminant(a) == std::mem::discriminant(b)
}

// struct ObservationBundle<'a>{
//     scores: ArrayView2<'a, OrderedFloat<f64>>,
//     indexes : HashSet<usize>,
//     max_treatment_utils: Array1<OrderedFloat<f64>>,
// }

// impl <'a> ObservationBundle<'a> {
//     fn new(index : usize, scores : ArrayView2<'a, OrderedFloat<f64>>) -> Self {
//         let mut indexes : HashSet<usize> = HashSet::new();
//         indexes.insert(index);

//         Self{
//             scores : scores,
//             indexes : indexes,
//             max_treatment_utils : scores.slice(s![index, ..]).to_owned()
//         }
//     }
//     fn add(&mut self, index : usize){
//         self.indexes.insert(index);
//         self.max_treatment_utils += &self.scores.slice(s![index, ..]);
//     }
//     fn remove(&mut self, index : usize){
//         self.indexes.remove(&index);
//         self.max_treatment_utils -= &self.scores.slice(s![index, ..]);
//     }
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

        return Node {
            node_type: NodeType::Branch,
            reward: best_reward,
            action: None,
            left_child: Some(Box::new(best_l_tree)),
            right_child: Some(Box::new(best_r_tree)),
            cut_axis: Some(dim),
            cut_point: Some(best_split_point),
        };
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
            if best_l_tree.action.unwrap() != best_r_tree.action.unwrap() {
                return Node {
                    node_type: NodeType::Branch,
                    reward: best_reward,
                    action: None,
                    left_child: Some(Box::new(best_l_tree)),
                    right_child: Some(Box::new(best_r_tree)),
                    cut_axis: Some(best_split_axis),
                    cut_point: Some(best_split_point),
                }
            } else{
                return Node {
                    node_type: NodeType::Leaf,
                    reward: best_reward,
                    action: best_l_tree.action,
                    left_child: None,
                    right_child: None,
                    cut_axis: None,
                    cut_point: None,
                }
            }
        }
    }
}

#[derive(Debug, Eq, PartialEq, Clone)]
enum NodeType {
    Leaf,
    Branch,
}

#[derive(Debug, Clone)]
struct Node {
    node_type: NodeType,
    reward: OrderedFloat<f64>,
    action: Option<usize>,
    left_child: Option<Box<Node>>,
    right_child: Option<Box<Node>>,
    cut_axis: Option<usize>,
    cut_point: Option<OrderedFloat<f64>>,
}

impl Node {
    fn new_leaf(reward: OrderedFloat<f64>, action: usize) -> Self {
        Self {
            node_type: NodeType::Leaf,
            action: Some(action),
            reward: reward,
            left_child: None,
            right_child: None,
            cut_axis: None,
            cut_point: None,
        }
    }
    fn new_branch(
        left_child: Node,
        right_child: Node,
        axis: usize,
        cut_point: OrderedFloat<f64>,
    ) -> Self {
        Self {
            node_type: NodeType::Branch,
            action: None,
            reward: left_child.reward + right_child.reward,
            left_child: Some(Box::new(left_child)),
            right_child: Some(Box::new(right_child)),
            cut_axis: Some(axis),
            cut_point: Some(cut_point),
        }
    }
    fn r_representation(&self) -> List {
        let mut queue: VecDeque<Node> = VecDeque::new();
        let mut output: Vec<List> = Vec::new();

        let mut next_avaliable = 2;

        queue.push_back(self.clone());
        while queue.len() != 0 {
            let current = queue.pop_front().unwrap();

            if variant_eq(&current.node_type, &NodeType::Leaf) {
                output.push(list!(is_leaf = true, action = current.action.unwrap()+1));
            } else {
                let left_child = *current.left_child.unwrap();
                let right_child = *current.right_child.unwrap();

                queue.push_back(left_child);
                queue.push_back(right_child);

                output.push(list!(
                    is_leaf = false,
                    split_variable = current.cut_axis.unwrap() + 1,
                    split_value = f64::from(current.cut_point.unwrap()),
                    left_child = next_avaliable,
                    right_child = next_avaliable + 1,
                ));
                next_avaliable += 2;
            }
        }

        List::from_values(output)
    }
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        self.reward.cmp(&other.reward)
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.reward == other.reward
    }
}

impl Eq for Node {}

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
