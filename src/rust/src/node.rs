use extendr_api::prelude::*;

use ordered_float::OrderedFloat;
use std::cmp::Ordering;
use std::collections::VecDeque;

#[derive(Debug, Eq, PartialEq, Clone)]
pub enum NodeType {
    Leaf,
    Branch,
}

#[derive(Debug, Clone)]
pub struct Node {
    pub node_type: NodeType,
    pub reward: OrderedFloat<f64>,
    pub action: Option<usize>,
    pub left_child: Option<Box<Node>>,
    pub right_child: Option<Box<Node>>,
    pub cut_axis: Option<usize>,
    pub cut_point: Option<OrderedFloat<f64>>,
}

impl Node {
    pub fn new_leaf(reward: OrderedFloat<f64>, action: usize) -> Self {
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
    pub fn new_branch(
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
    pub fn r_representation(&self) -> List {
        let mut queue: VecDeque<Node> = VecDeque::new();
        let mut output: Vec<List> = Vec::new();

        let mut next_avaliable = 2;

        queue.push_back(self.clone());
        while queue.len() != 0 {
            let current = queue.pop_front().unwrap();

            match current.node_type {
                NodeType::Leaf => {
                    output.push(list!(is_leaf = true, action = current.action.unwrap()+1));
                }
                NodeType::Branch => {
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
