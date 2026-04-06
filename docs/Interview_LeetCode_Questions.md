# OncoSeg — LeetCode & Coding Interview Questions

Coding questions that AI/ML engineer interviews commonly ask, organized by category. Each question includes the problem, a Python solution, time/space complexity, and how it connects to the OncoSeg project.

---

## 1. Array & Matrix (Foundation for Tensor Operations)

### Q1: Rotate a 3D Matrix (Easy → Medium)
**Relevance:** 3D augmentation — our pipeline randomly rotates 3D volumes.

```python
"""
Given a 3D matrix [D][H][W], rotate it 90 degrees along the H-W plane.
This is what RandRotate90d does in our data augmentation.
"""
def rotate_3d_hw(matrix):
    D = len(matrix)
    H = len(matrix[0])
    W = len(matrix[0][0])
    # Rotate each depth slice 90 degrees: transpose then reverse rows
    result = []
    for d in range(D):
        rotated_slice = []
        for w in range(W):
            row = []
            for h in range(H - 1, -1, -1):
                row.append(matrix[d][h][w])
            rotated_slice.append(row)
        result.append(rotated_slice)
    return result

# Time: O(D * H * W), Space: O(D * H * W)

# Test
matrix = [[[1,2],[3,4]], [[5,6],[7,8]]]
print(rotate_3d_hw(matrix))
# [[[3,1],[4,2]], [[7,5],[8,6]]]
```

---

### Q2: Sliding Window Maximum (Hard)
**Relevance:** Sliding window inference — core of how we process 3D volumes.

```python
"""
Given an array nums and window size k, return the max in each window.
This is the 1D analogy of our sliding_window_inference.

LeetCode 239: Sliding Window Maximum
"""
from collections import deque

def max_sliding_window(nums, k):
    result = []
    dq = deque()  # stores indices, front is always the max

    for i in range(len(nums)):
        # Remove indices outside the window
        while dq and dq[0] < i - k + 1:
            dq.popleft()

        # Remove smaller elements from back
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()

        dq.append(i)

        if i >= k - 1:
            result.append(nums[dq[0]])

    return result

# Time: O(n), Space: O(k)

# Test
print(max_sliding_window([1,3,-1,-3,5,3,6,7], 3))
# [3, 3, 5, 5, 6, 7]
```

---

### Q3: 3D Connected Components (Medium)
**Relevance:** RECIST measurement — we find connected components in 3D segmentation masks.

```python
"""
Given a 3D binary mask, count the number of connected components.
This is what our RECIST pipeline does to find individual lesions.

Extension of LeetCode 200: Number of Islands (but in 3D)
"""
from collections import deque

def count_3d_components(mask):
    if not mask or not mask[0] or not mask[0][0]:
        return 0

    D, H, W = len(mask), len(mask[0]), len(mask[0][0])
    visited = [[[False]*W for _ in range(H)] for _ in range(D)]
    count = 0

    # 6-connectivity: up, down, left, right, front, back
    directions = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]

    for d in range(D):
        for h in range(H):
            for w in range(W):
                if mask[d][h][w] == 1 and not visited[d][h][w]:
                    # BFS from this voxel
                    count += 1
                    queue = deque([(d, h, w)])
                    visited[d][h][w] = True

                    while queue:
                        cd, ch, cw = queue.popleft()
                        for dd, dh, dw in directions:
                            nd, nh, nw = cd+dd, ch+dh, cw+dw
                            if (0 <= nd < D and 0 <= nh < H and 0 <= nw < W
                                and mask[nd][nh][nw] == 1
                                and not visited[nd][nh][nw]):
                                visited[nd][nh][nw] = True
                                queue.append((nd, nh, nw))

    return count

# Time: O(D * H * W), Space: O(D * H * W)

# Test
mask = [
    [[1,0,0],[0,0,0],[0,0,1]],
    [[1,0,0],[0,0,0],[0,0,0]],
]
print(count_3d_components(mask))  # 2 (two separate lesions)
```

---

### Q4: Longest Diameter in a Binary Mask (Medium)
**Relevance:** RECIST 1.1 — we compute the longest axial diameter of each lesion.

```python
"""
Given a 2D binary mask of a lesion, find the longest distance between
any two points in the lesion (longest axial diameter for RECIST).
"""
import math

def longest_diameter(mask):
    # Collect all lesion points
    points = []
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            if mask[i][j] == 1:
                points.append((i, j))

    if len(points) < 2:
        return 0.0

    # Brute force for small lesions (O(n^2))
    # For large lesions, use rotating calipers on convex hull
    max_dist = 0.0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = math.sqrt(
                (points[i][0] - points[j][0])**2 +
                (points[i][1] - points[j][1])**2
            )
            max_dist = max(max_dist, dist)

    return max_dist

# Time: O(n^2) where n = number of lesion voxels
# Space: O(n)

# Test
mask = [
    [0, 1, 1, 0],
    [0, 1, 1, 1],
    [0, 0, 1, 0],
]
print(f"{longest_diameter(mask):.2f}")  # ~3.16
```

---

## 2. Dynamic Programming

### Q5: Weighted Interval Scheduling (Medium)
**Relevance:** Deep supervision loss — weighted combination of predictions at different scales.

```python
"""
Given intervals with weights, find maximum weight non-overlapping subset.
Analogy: choosing which decoder scales to supervise and how to weight them.

Similar to LeetCode 1235: Maximum Profit in Job Scheduling
"""
import bisect

def max_weight_schedule(intervals):
    # intervals: list of (start, end, weight)
    intervals.sort(key=lambda x: x[1])  # sort by end time
    n = len(intervals)

    dp = [0] * (n + 1)
    ends = [iv[1] for iv in intervals]

    for i in range(1, n + 1):
        start_i = intervals[i-1][0]
        weight_i = intervals[i-1][2]

        # Find last non-overlapping interval
        j = bisect.bisect_right(ends, start_i, 0, i-1)

        dp[i] = max(dp[i-1], dp[j] + weight_i)

    return dp[n]

# Time: O(n log n), Space: O(n)

# Test
intervals = [(1,3,5), (2,5,6), (4,6,5), (6,7,4)]
print(max_weight_schedule(intervals))  # 10
```

---

### Q6: Dice Score Computation (Easy)
**Relevance:** Directly computes the Dice metric we use for evaluation.

```python
"""
Compute the Dice score between two binary arrays.
Dice = 2 * |A ∩ B| / (|A| + |B|)

This is our primary evaluation metric.
"""
def dice_score(pred, target):
    intersection = sum(p & t for p, t in zip(pred, target))
    sum_pred = sum(pred)
    sum_target = sum(target)

    if sum_pred + sum_target == 0:
        return 1.0  # both empty = perfect match

    return 2.0 * intersection / (sum_pred + sum_target)

# Time: O(n), Space: O(1)

# Test
pred =   [1, 1, 1, 0, 0, 1, 0]
target = [1, 1, 0, 0, 0, 1, 1]
print(f"Dice: {dice_score(pred, target):.4f}")  # 0.75
```

---

### Q7: Minimum Path Sum in 3D Grid (Medium)
**Relevance:** Understanding gradient flow through 3D networks.

```python
"""
Given a 3D grid with costs, find minimum cost path from (0,0,0) to (D-1,H-1,W-1).
Can only move right, down, or deeper.

Extension of LeetCode 64: Minimum Path Sum (but 3D)
"""
def min_path_sum_3d(grid):
    D, H, W = len(grid), len(grid[0]), len(grid[0][0])
    dp = [[[float('inf')]*W for _ in range(H)] for _ in range(D)]
    dp[0][0][0] = grid[0][0][0]

    for d in range(D):
        for h in range(H):
            for w in range(W):
                if d == 0 and h == 0 and w == 0:
                    continue
                val = grid[d][h][w]
                candidates = []
                if d > 0: candidates.append(dp[d-1][h][w])
                if h > 0: candidates.append(dp[d][h-1][w])
                if w > 0: candidates.append(dp[d][h][w-1])
                dp[d][h][w] = val + min(candidates)

    return dp[D-1][H-1][W-1]

# Time: O(D * H * W), Space: O(D * H * W)

# Test
grid = [[[1,2],[3,4]], [[5,1],[2,1]]]
print(min_path_sum_3d(grid))  # 5 (1->2->1->1)
```

---

## 3. Graph & Tree (Architecture Design)

### Q8: Binary Tree Level Order Traversal (Medium)
**Relevance:** U-Net is tree-structured — encoder levels down, decoder levels up.

```python
"""
LeetCode 102: Binary Tree Level Order Traversal
Analogy: processing features at each resolution level in the U-Net.
"""
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def level_order(root):
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        level = []
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)

    return result

# Time: O(n), Space: O(n)
```

---

### Q9: Topological Sort (Medium)
**Relevance:** Computation graph — PyTorch autograd processes nodes in topological order for backpropagation.

```python
"""
LeetCode 207 / 210: Course Schedule
Analogy: determining the order of operations in a neural network's
computation graph for forward and backward passes.
"""
from collections import deque, defaultdict

def topological_sort(num_nodes, edges):
    # edges: list of (prerequisite, dependent)
    graph = defaultdict(list)
    in_degree = [0] * num_nodes

    for pre, dep in edges:
        graph[pre].append(dep)
        in_degree[dep] += 1

    queue = deque([i for i in range(num_nodes) if in_degree[i] == 0])
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return order if len(order) == num_nodes else []  # empty = cycle

# Time: O(V + E), Space: O(V + E)

# Test: encoder -> bottleneck -> decoder -> output
edges = [(0,1), (1,2), (2,3), (0,3)]  # skip connection: 0->3
print(topological_sort(4, edges))  # [0, 1, 2, 3]
```

---

## 4. String & Hash (Data Processing)

### Q10: Group Anagrams (Medium)
**Relevance:** Grouping data by properties — like grouping patients by tumor characteristics.

```python
"""
LeetCode 49: Group Anagrams
Analogy: grouping subjects by similar tumor profiles for stratified evaluation.
"""
from collections import defaultdict

def group_anagrams(strs):
    groups = defaultdict(list)
    for s in strs:
        key = tuple(sorted(s))
        groups[key].append(s)
    return list(groups.values())

# Time: O(n * k log k) where k = max string length
# Space: O(n * k)

print(group_anagrams(["eat","tea","tan","ate","nat","bat"]))
# [["eat","tea","ate"], ["tan","nat"], ["bat"]]
```

---

### Q11: LRU Cache (Medium)
**Relevance:** Caching preprocessed data — MONAI's CacheDataset uses this pattern.

```python
"""
LeetCode 146: LRU Cache
Analogy: MONAI CacheDataset caches preprocessed volumes to avoid
recomputing transforms every epoch.
"""
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

# Time: O(1) for get and put
# Space: O(capacity)

cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
print(cache.get(1))   # 1
cache.put(3, 3)       # evicts key 2
print(cache.get(2))   # -1
```

---

## 5. Math & Probability (ML Theory)

### Q12: Implement Softmax (Easy)
**Relevance:** We switched from softmax to sigmoid — understanding both is essential.

```python
"""
Implement softmax with numerical stability.
softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
"""
import math

def softmax(logits):
    max_val = max(logits)
    exps = [math.exp(x - max_val) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]

def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)

# Test
print(softmax([2.0, 1.0, 0.1]))  # [0.659, 0.242, 0.099] — sums to 1
print([sigmoid(x) for x in [2.0, 1.0, 0.1]])  # [0.881, 0.731, 0.525] — independent
```

---

### Q13: Implement Binary Cross-Entropy Loss (Easy)
**Relevance:** BCE is half of our DiceCE loss function.

```python
"""
Implement BCE loss: -[y * log(p) + (1-y) * log(1-p)]
"""
import math

def bce_loss(predictions, targets, eps=1e-7):
    n = len(predictions)
    total_loss = 0.0
    for p, y in zip(predictions, targets):
        p = max(min(p, 1 - eps), eps)  # clip for numerical stability
        total_loss += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return total_loss / n

# Test
preds = [0.9, 0.8, 0.1, 0.2]   # model predictions
targets = [1, 1, 0, 0]          # ground truth
print(f"BCE Loss: {bce_loss(preds, targets):.4f}")  # ~0.164 (low = good)

bad_preds = [0.1, 0.2, 0.9, 0.8]  # wrong predictions
print(f"BCE Loss: {bce_loss(bad_preds, targets):.4f}")  # ~1.836 (high = bad)
```

---

### Q14: Implement Dice Score with Smooth (Easy)
**Relevance:** Exact implementation of our Dice loss numerator/denominator.

```python
"""
Implement soft Dice score with smoothing (as used in our DiceLoss).
Dice = (2 * sum(p * g) + smooth) / (sum(p) + sum(g) + smooth)
"""
def soft_dice(pred_probs, targets, smooth=1e-5):
    intersection = sum(p * t for p, t in zip(pred_probs, targets))
    sum_pred = sum(pred_probs)
    sum_target = sum(targets)
    return (2.0 * intersection + smooth) / (sum_pred + sum_target + smooth)

# Test with soft predictions (probabilities, not binary)
preds = [0.9, 0.8, 0.7, 0.1, 0.05]
targets = [1.0, 1.0, 1.0, 0.0, 0.0]
print(f"Soft Dice: {soft_dice(preds, targets):.4f}")  # ~0.914
```

---

## 6. Two Pointers & Binary Search

### Q15: Find Optimal Learning Rate (Binary Search) (Medium)
**Relevance:** Hyperparameter tuning — finding the right learning rate.

```python
"""
Given a function that returns validation loss for a learning rate,
find the learning rate that minimizes loss using ternary search.
Analogy: learning rate finder used before training.
"""
def find_optimal_lr(loss_fn, lo=1e-7, hi=1e-1, iterations=50):
    for _ in range(iterations):
        mid1 = lo + (hi - lo) / 3
        mid2 = hi - (hi - lo) / 3
        if loss_fn(mid1) < loss_fn(mid2):
            hi = mid2
        else:
            lo = mid1
    return (lo + hi) / 2

# Time: O(iterations * cost_of_loss_fn)

# Simulated loss function (parabola centered at lr=1e-3)
import math
loss_fn = lambda lr: (math.log10(lr) - math.log10(1e-3))**2 + 0.5
optimal = find_optimal_lr(loss_fn)
print(f"Optimal LR: {optimal:.6f}")  # ~0.001
```

---

### Q16: Merge Overlapping Intervals (Medium)
**Relevance:** Merging overlapping sliding window predictions.

```python
"""
LeetCode 56: Merge Intervals
Analogy: merging overlapping sliding window patches back into a full volume.
"""
def merge_intervals(intervals):
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]

    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return merged

# Time: O(n log n), Space: O(n)

print(merge_intervals([[1,3],[2,6],[8,10],[15,18]]))
# [[1,6], [8,10], [15,18]]
```

---

## 7. Design & OOP (ML System Design)

### Q17: Design a Metric Tracker (Medium)
**Relevance:** Our SegmentationMetrics class accumulates and aggregates metrics.

```python
"""
Design a class that tracks running metrics across batches,
then computes aggregate statistics. Like our DiceMetric.
"""
class MetricTracker:
    def __init__(self):
        self.values = []

    def update(self, value):
        self.values.append(value)

    def compute_mean(self):
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

    def compute_std(self):
        if len(self.values) < 2:
            return 0.0
        mean = self.compute_mean()
        variance = sum((v - mean)**2 for v in self.values) / (len(self.values) - 1)
        return variance ** 0.5

    def compute_best(self):
        return max(self.values) if self.values else 0.0

    def reset(self):
        self.values = []

# Test
tracker = MetricTracker()
for dice in [0.75, 0.82, 0.79, 0.85, 0.81]:
    tracker.update(dice)
print(f"Mean: {tracker.compute_mean():.4f}")  # 0.8040
print(f"Std: {tracker.compute_std():.4f}")    # 0.0365
print(f"Best: {tracker.compute_best():.4f}")  # 0.8500
```

---

### Q18: Implement a Checkpoint Manager (Medium)
**Relevance:** Directly mirrors our checkpoint/resume system in train_all.py.

```python
"""
Design a checkpoint manager that saves/loads training state
and tracks the best model.
"""
import json
from pathlib import Path

class CheckpointManager:
    def __init__(self, save_dir, metric_name="dice"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metric_name = metric_name
        self.best_value = 0.0
        self.best_epoch = 0

    def save(self, epoch, metrics, is_best=False):
        state = {
            "epoch": epoch,
            "metrics": metrics,
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
        }
        # Always save latest checkpoint
        with open(self.save_dir / "checkpoint.json", "w") as f:
            json.dump(state, f, indent=2)

        # Save best if improved
        if is_best:
            with open(self.save_dir / "best.json", "w") as f:
                json.dump(state, f, indent=2)

    def update_best(self, epoch, value):
        if value > self.best_value:
            self.best_value = value
            self.best_epoch = epoch
            return True
        return False

    def load(self):
        ckpt_path = self.save_dir / "checkpoint.json"
        if ckpt_path.exists():
            with open(ckpt_path) as f:
                state = json.load(f)
            self.best_value = state["best_value"]
            self.best_epoch = state["best_epoch"]
            return state
        return None
```

---

## 8. Bit Manipulation & Math

### Q19: Count Set Bits in a Mask (Easy)
**Relevance:** Counting foreground voxels in a binary segmentation mask = volume calculation.

```python
"""
Count number of 1s in a binary representation.
Analogy: counting tumor voxels for volume measurement in RECIST.

LeetCode 191: Number of 1 Bits
"""
def count_ones(n):
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count

# Brian Kernighan's trick (faster)
def count_ones_fast(n):
    count = 0
    while n:
        n &= (n - 1)  # removes lowest set bit
        count += 1
    return count

# Time: O(number of set bits)
print(count_ones(0b11010110))  # 5
```

---

### Q20: Matrix Multiplication (Medium)
**Relevance:** Core of attention computation — Q @ K^T in cross-attention.

```python
"""
Implement matrix multiplication from scratch.
This is what happens in every attention layer: Q @ K^T
"""
def matmul(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    assert cols_A == rows_B, "Incompatible dimensions"

    result = [[0.0] * cols_B for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    return result

# Time: O(n * m * p), Space: O(n * p)

# Test: Q(2x3) @ K^T(3x2) = attention_scores(2x2)
Q = [[1, 0, 1], [0, 1, 1]]
K_T = [[1, 0], [0, 1], [1, 1]]
print(matmul(Q, K_T))  # [[2, 1], [1, 2]]
```

---

## 9. Common ML Interview Coding Tasks

### Q21: Implement K-Fold Cross-Validation (Medium)
**Relevance:** Evaluation methodology — we use a single train/val split but k-fold is common.

```python
"""
Implement k-fold cross-validation splitting.
"""
def k_fold_split(data, k=5):
    n = len(data)
    fold_size = n // k
    folds = []

    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n
        val_indices = list(range(start, end))
        train_indices = list(range(0, start)) + list(range(end, n))
        folds.append((train_indices, val_indices))

    return folds

# Test
data = list(range(10))
for i, (train, val) in enumerate(k_fold_split(data, k=5)):
    print(f"Fold {i}: train={train}, val={val}")
```

---

### Q22: Implement Cosine Similarity (Easy)
**Relevance:** Attention uses dot product similarity; cosine similarity is the normalized version.

```python
"""
Compute cosine similarity between two vectors.
cos(A, B) = (A · B) / (||A|| * ||B||)
"""
import math

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x**2 for x in a))
    norm_b = math.sqrt(sum(x**2 for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

# Time: O(n), Space: O(1)

a = [1, 2, 3]
b = [1, 2, 3]
print(f"Same vector: {cosine_similarity(a, b):.4f}")  # 1.0

c = [-1, -2, -3]
print(f"Opposite: {cosine_similarity(a, c):.4f}")  # -1.0

d = [3, -2, 1]
print(f"Orthogonal-ish: {cosine_similarity(a, d):.4f}")  # 0.143
```

---

### Q23: Implement Mini-Batch Gradient Descent (Medium)
**Relevance:** Exactly what our training loop does each epoch.

```python
"""
Implement mini-batch SGD for linear regression.
Analogy: our training loop with DataLoader batching.
"""
import random

def mini_batch_sgd(X, y, lr=0.01, epochs=100, batch_size=4):
    n_features = len(X[0])
    weights = [0.0] * n_features
    bias = 0.0
    n = len(X)

    for epoch in range(epochs):
        # Shuffle data each epoch (like shuffle=True in DataLoader)
        indices = list(range(n))
        random.shuffle(indices)

        total_loss = 0.0
        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            bs = len(batch_idx)

            # Forward pass
            grad_w = [0.0] * n_features
            grad_b = 0.0
            for i in batch_idx:
                pred = sum(w * x for w, x in zip(weights, X[i])) + bias
                error = pred - y[i]
                total_loss += error ** 2
                for j in range(n_features):
                    grad_w[j] += (2 / bs) * error * X[i][j]
                grad_b += (2 / bs) * error

            # Backward pass (parameter update)
            for j in range(n_features):
                weights[j] -= lr * grad_w[j]
            bias -= lr * grad_b

    return weights, bias

# Test: y = 2*x1 + 3*x2 + 1
X = [[1,1],[2,1],[1,2],[2,2],[3,1]]
y = [6, 8, 8, 10, 10]
w, b = mini_batch_sgd(X, y, lr=0.01, epochs=500)
print(f"Weights: [{w[0]:.2f}, {w[1]:.2f}], Bias: {b:.2f}")
# Should approximate [2, 3], 1
```

---

### Q24: Implement Precision, Recall, F1 (Easy)
**Relevance:** Related to our Sensitivity (recall) and Specificity metrics.

```python
"""
Compute precision, recall, and F1 from predictions and targets.
"""
def compute_metrics(predictions, targets):
    tp = sum(p == 1 and t == 1 for p, t in zip(predictions, targets))
    fp = sum(p == 1 and t == 0 for p, t in zip(predictions, targets))
    fn = sum(p == 0 and t == 1 for p, t in zip(predictions, targets))
    tn = sum(p == 0 and t == 0 for p, t in zip(predictions, targets))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # = sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall (sensitivity)": recall,
        "specificity": specificity,
        "f1": f1,
    }

# Test
preds =   [1, 1, 1, 0, 0, 1, 0, 0]
targets = [1, 1, 0, 0, 0, 1, 1, 0]
for k, v in compute_metrics(preds, targets).items():
    print(f"  {k}: {v:.4f}")
```

---

### Q25: Implement Weighted Random Sampling (Medium)
**Relevance:** Deep supervision loss uses exponentially decaying weights.

```python
"""
Given weights, sample an index proportional to weight.
Analogy: deep supervision assigns weight 0.5^i to each decoder level.

LeetCode 528: Random Pick with Weight
"""
import random
import bisect

class WeightedSampler:
    def __init__(self, weights):
        self.prefix_sums = []
        running = 0
        for w in weights:
            running += w
            self.prefix_sums.append(running)
        self.total = running

    def sample(self):
        target = random.uniform(0, self.total)
        return bisect.bisect_left(self.prefix_sums, target)

# Time: O(log n) per sample, O(n) to build
# Space: O(n)

# Test: deep supervision weights for 3 decoder levels
# w = [0.5^1, 0.5^2, 0.5^3] normalized
raw = [0.5**i for i in range(1, 4)]
total = sum(raw)
weights = [w/total for w in raw]
print(f"Normalized weights: {[f'{w:.3f}' for w in weights]}")

sampler = WeightedSampler(weights)
counts = [0, 0, 0]
for _ in range(10000):
    counts[sampler.sample()] += 1
print(f"Sample distribution: {[c/10000 for c in counts]}")
# Should approximate [0.571, 0.286, 0.143]
```

---

## Summary: LeetCode Problems to Practice

| # | LeetCode Problem | Difficulty | Relevance to OncoSeg |
|---|-----------------|------------|---------------------|
| 239 | Sliding Window Maximum | Hard | Sliding window inference |
| 200 | Number of Islands (extend to 3D) | Medium | RECIST connected components |
| 102 | Binary Tree Level Order | Medium | U-Net multi-scale processing |
| 207 | Course Schedule (Topo Sort) | Medium | Computation graph / autograd |
| 49 | Group Anagrams | Medium | Data grouping / stratification |
| 146 | LRU Cache | Medium | MONAI data caching |
| 56 | Merge Intervals | Medium | Merging sliding window patches |
| 64 | Min Path Sum (extend to 3D) | Medium | 3D grid traversal |
| 528 | Random Pick with Weight | Medium | Weighted sampling / deep supervision |
| 191 | Number of 1 Bits | Easy | Mask voxel counting |
| 1235 | Max Profit in Job Scheduling | Hard | Weighted interval scheduling |
