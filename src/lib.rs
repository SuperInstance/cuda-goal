/*!
# cuda-goal

Hierarchical goal management for agents.

An agent without goals is noise. An agent with one flat goal is a tool.
An agent with a stack of decomposed goals is intelligent.

This crate provides:
- Goals with priority, deadline, and confidence
- Hierarchical decomposition (parent → children)
- Goal stack (LIFO) for active work
- Completion tracking with partial credit
- Goal conflict detection
- Motivation scoring (want-to vs need-to vs have-to)
*/

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Goal motivation level
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Motivation {
    WantTo,    // intrinsically motivated (play, explore)
    NeedTo,    // externally required (survival, compliance)
    HaveTo,    // obligation (assigned task)
}

/// A single goal
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Goal {
    pub id: u64,
    pub description: String,
    pub motivation: Motivation,
    pub priority: f64,          // 0-1
    pub confidence: f64,        // belief in achievability
    pub parent: Option<u64>,    // parent goal
    pub children: Vec<u64>,     // subgoals
    pub deadline: Option<u64>,
    pub created: u64,
    pub completed: bool,
    pub progress: f64,          // 0-1
    pub blocked_by: Vec<u64>,   // prerequisite goal ids
    pub tags: Vec<String>,
}

impl Goal {
    pub fn new(id: u64, desc: &str, motivation: Motivation) -> Self {
        Goal { id, description: desc.to_string(), motivation, priority: 0.5, confidence: 0.5, parent: None, children: vec![], deadline: None, created: now(), completed: false, progress: 0.0, blocked_by: vec![], tags: vec![] }
    }

    pub fn with_priority(mut self, p: f64) -> Self { self.priority = p.clamp(0.0, 1.0); self }
    pub fn with_confidence(mut self, c: f64) -> Self { self.confidence = c.clamp(0.0, 1.0); self }
    pub fn with_deadline(mut self, d: u64) -> Self { self.deadline = Some(d); self }

    /// Composite priority = motivation weight * priority * urgency * confidence
    pub fn composite_score(&self, now: u64) -> f64 {
        let motivation_weight = match self.motivation {
            Motivation::HaveTo => 1.0,
            Motivation::NeedTo => 0.8,
            Motivation::WantTo => 0.5,
        };
        let urgency = match self.deadline {
            Some(dl) => {
                let remaining = dl.saturating_sub(now) as f64;
                let elapsed = now.saturating_sub(self.created) as f64;
                if elapsed < 1.0 { 0.5 } else { 1.0 - (remaining / elapsed).min(2.0) / 2.0 }
            }
            None => 0.5,
        };
        motivation_weight * self.priority * urgency * self.confidence
    }

    /// Complete this goal with given progress
    pub fn complete(&mut self, progress: f64) {
        self.completed = true;
        self.progress = progress.clamp(0.0, 1.0);
    }
}

/// Goal tree — hierarchical goal structure
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GoalTree {
    pub goals: HashMap<u64, Goal>,
    pub roots: Vec<u64>,
    pub next_id: u64,
}

impl GoalTree {
    pub fn new() -> Self { GoalTree { goals: HashMap::new(), roots: vec![], next_id: 1 } }

    pub fn add_goal(&mut self, mut goal: Goal) -> u64 {
        if goal.id == 0 { goal.id = self.next_id; self.next_id += 1; }
        let id = goal.id;
        if let Some(parent_id) = goal.parent {
            if let Some(parent) = self.goals.get_mut(&parent_id) {
                parent.children.push(id);
            }
        } else {
            self.roots.push(id);
        }
        self.goals.insert(id, goal);
        self.next_id = self.next_id.max(id + 1);
        id
    }

    /// Decompose a goal into subgoals
    pub fn decompose(&mut self, parent_id: u64, subgoals: Vec<Goal>) -> Vec<u64> {
        let mut ids = vec![];
        for mut sg in subgoals {
            sg.id = self.next_id;
            sg.parent = Some(parent_id);
            self.next_id += 1;
            let id = sg.id;
            ids.push(id);
        }
        // Set children on parent
        if let Some(parent) = self.goals.get_mut(&parent_id) {
            parent.children = ids.clone();
        }
        for sg in subgoals.into_iter().enumerate() {
            let mut goal = sg.1;
            goal.id = ids[sg.0];
            self.goals.insert(goal.id, goal);
        }
        ids
    }

    /// Get leaf goals (no children, actionable)
    pub fn leaf_goals(&self) -> Vec<&Goal> {
        self.goals.values().filter(|g| g.children.is_empty() && !g.completed).collect()
    }

    /// Ancestors of a goal
    pub fn ancestors(&self, goal_id: u64) -> Vec<&Goal> {
        let mut chain = vec![];
        let mut current = goal_id;
        while let Some(goal) = self.goals.get(&current) {
            chain.push(goal);
            match goal.parent { Some(p) => current = p, None => break }
        }
        chain
    }

    /// Propagate progress up from children
    pub fn propagate_progress(&mut self) {
        // Post-order traversal
        fn propagate(goals: &mut HashMap<u64, Goal>, goal_id: u64) -> f64 {
            let goal = match goals.get(&goal_id) { Some(g) => g.clone(), None => return 0.0 };
            if goal.children.is_empty() { return goal.progress; }
            let child_progress: f64 = goal.children.iter()
                .map(|&cid| propagate(goals, cid))
                .sum::<f64>() / goal.children.len() as f64;
            if let Some(g) = goals.get_mut(&goal_id) {
                g.progress = child_progress;
                g.completed = g.children.iter().all(|&cid| goals.get(&cid).map_or(true, |c| c.completed));
            }
            child_progress
        }
        for &root in &self.roots.clone() {
            propagate(&mut self.goals, root);
        }
    }

    /// Detect conflicting goals
    pub fn conflicts(&self) -> Vec<(u64, u64)> {
        let mut conflicts = vec![];
        let goals: Vec<_> = self.goals.values().filter(|g| !g.completed).collect();
        for i in 0..goals.len() {
            for j in (i+1)..goals.len() {
                // Goals conflict if they share tags but have opposing priorities
                let shared: Vec<_> = goals[i].tags.iter().filter(|t| goals[j].tags.contains(t)).collect();
                if !shared.is_empty() && goals[i].priority > 0.7 && goals[j].priority > 0.7 {
                    conflicts.push((goals[i].id, goals[j].id));
                }
            }
        }
        conflicts
    }

    /// Check if blocked_by dependencies are met
    pub fn is_unblocked(&self, goal_id: u64) -> bool {
        let goal = match self.goals.get(&goal_id) { Some(g) => g, None => return false };
        goal.blocked_by.iter().all(|&dep| self.goals.get(&dep).map_or(false, |d| d.completed))
    }

    /// Best next goal to work on (highest composite score, unblocked, uncompleted)
    pub fn best_next(&self, now: u64) -> Option<u64> {
        self.goals.values()
            .filter(|g| !g.completed)
            .filter(|g| g.children.is_empty()) // leaf only
            .filter(|g| self.is_unblocked(g.id))
            .max_by(|a, b| a.composite_score(now).partial_cmp(&b.composite_score(now)).unwrap())
            .map(|g| g.id)
    }

    /// Summary stats
    pub fn stats(&self) -> GoalStats {
        let total = self.goals.len();
        let completed = self.goals.values().filter(|g| g.completed).count();
        let leaves = self.leaf_goals().len();
        let avg_progress = if total > 0 {
            self.goals.values().map(|g| g.progress).sum::<f64>() / total as f64
        } else { 0.0 };
        GoalStats { total, completed, leaves, avg_progress, roots: self.roots.len() }
    }
}

#[derive(Clone, Debug)]
pub struct GoalStats {
    pub total: usize,
    pub completed: usize,
    pub leaves: usize,
    pub avg_progress: f64,
    pub roots: usize,
}

/// Goal stack — LIFO active goals
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GoalStack {
    pub stack: Vec<u64>,
    pub capacity: usize,
}

impl GoalStack {
    pub fn new(capacity: usize) -> Self { GoalStack { stack: vec![], capacity } }

    pub fn push(&mut self, goal_id: u64) -> bool {
        if self.stack.len() >= self.capacity { return false; }
        self.stack.push(goal_id);
        true
    }

    pub fn pop(&mut self) -> Option<u64> { self.stack.pop() }
    pub fn peek(&self) -> Option<u64> { self.stack.last().copied() }
    pub fn depth(&self) -> usize { self.stack.len() }

    /// Suspend current goal, push new goal
    pub fn suspend_and_push(&mut self, new_goal: u64) -> bool { self.push(new_goal) }

    /// Drop all goals below a depth
    pub fn truncate(&mut self, depth: usize) {
        self.stack.truncate(depth);
    }
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_goal_creation() {
        let g = Goal::new(1, "survive", Motivation::NeedTo);
        assert_eq!(g.motivation, Motivation::NeedTo);
        assert!(!g.completed);
    }

    #[test]
    fn test_goal_composite_score() {
        let mut g = Goal::new(1, "urgent", Motivation::HaveTo).with_priority(0.9).with_confidence(0.8);
        g.deadline = Some(now() + 100);
        let score = g.composite_score(now());
        assert!(score > 0.3);
    }

    #[test]
    fn test_motivation_weights() {
        let have = Goal::new(1, "a", Motivation::HaveTo).with_priority(0.5).with_confidence(0.5);
        let need = Goal::new(2, "b", Motivation::NeedTo).with_priority(0.5).with_confidence(0.5);
        let want = Goal::new(3, "c", Motivation::WantTo).with_priority(0.5).with_confidence(0.5);
        assert!(have.composite_score(now()) > need.composite_score(now()));
        assert!(need.composite_score(now()) > want.composite_score(now()));
    }

    #[test]
    fn test_tree_add_decompose() {
        let mut tree = GoalTree::new();
        let parent = tree.add_goal(Goal::new(0, "build shelter", Motivation::NeedTo));
        let children = tree.decompose(parent, vec![
            Goal::new(0, "gather wood", Motivation::HaveTo),
            Goal::new(0, "find nails", Motivation::HaveTo),
        ]);
        assert_eq!(children.len(), 2);
        assert_eq!(tree.leaf_goals().len(), 2);
    }

    #[test]
    fn test_tree_propagate() {
        let mut tree = GoalTree::new();
        let parent = tree.add_goal(Goal::new(0, "parent", Motivation::NeedTo));
        let children = tree.decompose(parent, vec![
            Goal::new(0, "child1", Motivation::HaveTo),
            Goal::new(0, "child2", Motivation::HaveTo),
        ]);
        // Complete child1 fully
        tree.goals.get_mut(&children[0]).unwrap().complete(1.0);
        tree.propagate_progress();
        let parent = tree.goals.get(&parent).unwrap();
        assert!(!parent.completed); // child2 not done
        assert!((parent.progress - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_tree_ancestors() {
        let mut tree = GoalTree::new();
        let root = tree.add_goal(Goal::new(0, "root", Motivation::NeedTo));
        let child = tree.decompose(root, vec![Goal::new(0, "child", Motivation::HaveTo)])[0];
        let grand = tree.decompose(child, vec![Goal::new(0, "grand", Motivation::WantTo)])[0];
        let ancestors = tree.ancestors(grand);
        assert_eq!(ancestors.len(), 3);
    }

    #[test]
    fn test_blocked_goals() {
        let mut tree = GoalTree::new();
        let g1 = tree.add_goal(Goal::new(0, "prereq", Motivation::NeedTo));
        let mut g2 = Goal::new(0, "dependent", Motivation::HaveTo);
        g2.blocked_by = vec![g1];
        let g2_id = tree.add_goal(g2);
        assert!(!tree.is_unblocked(g2_id));
        tree.goals.get_mut(&g1).unwrap().complete(1.0);
        assert!(tree.is_unblocked(g2_id));
    }

    #[test]
    fn test_best_next() {
        let mut tree = GoalTree::new();
        tree.add_goal(Goal::new(0, "low", Motivation::WantTo).with_priority(0.1));
        tree.add_goal(Goal::new(0, "high", Motivation::HaveTo).with_priority(0.9));
        let best = tree.best_next(now());
        assert!(best.is_some());
    }

    #[test]
    fn test_goal_stack() {
        let mut stack = GoalStack::new(5);
        assert!(stack.push(1));
        assert!(stack.push(2));
        assert_eq!(stack.pop(), Some(2));
        assert_eq!(stack.peek(), Some(1));
        assert_eq!(stack.depth(), 1);
    }

    #[test]
    fn test_stack_capacity() {
        let mut stack = GoalStack::new(2);
        assert!(stack.push(1));
        assert!(stack.push(2));
        assert!(!stack.push(3)); // over capacity
    }

    #[test]
    fn test_tree_stats() {
        let mut tree = GoalTree::new();
        tree.add_goal(Goal::new(0, "a", Motivation::NeedTo));
        let s = tree.stats();
        assert_eq!(s.total, 1);
        assert_eq!(s.completed, 0);
    }

    #[test]
    fn test_conflicts() {
        let mut tree = GoalTree::new();
        let mut g1 = Goal::new(0, "a", Motivation::HaveTo).with_priority(0.9);
        g1.tags = vec!["combat".into()];
        let mut g2 = Goal::new(0, "b", Motivation::HaveTo).with_priority(0.8);
        g2.tags = vec!["combat".into()];
        tree.add_goal(g1);
        tree.add_goal(g2);
        let conflicts = tree.conflicts();
        assert_eq!(conflicts.len(), 1);
    }
}
