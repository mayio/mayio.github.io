---
layout: post
title: 'Faster Data Association with Max-Sum Belief Propagation (MASDA)'
subtitle: 'A max-sum message passing algorithm for associating measurements with objects under clutter and misdetection, derived from the binary variable model.'
thumbnail-img: /assets/img/2025-11-26-Faster-Data-Association-with-Max-Sum-Loopy-Belief-Propagation-MASDA_files/2025-11-26-Faster-Data-Association-with-Max-Sum-Loopy-Belief-Propagation-MASDA_3_0.png
date: '2025-11-26 18:50:00 +0200'
categories: association
comments: false
mathjax: true
author: Mario Lüder
tags: [belief-propagation, data-association, autonomous-driving, algorithms]
---

This presents a data association algorithm I call MASDA (Max-Sum Algorithm Data Association) which relies on message passing in a factor graph. 

I derived the formula presented here using the very same approach as in the paper:

> Givoni, Inmar & Frey, Brendan. (2009). [A Binary Variable Model for Affinity Propagation](https://www.researchgate.net/publication/23975825_A_Binary_Variable_Model_for_Affinity_Propagation). Neural Computation. 21. 1589-1600. 10.1162/neco.2009.05-08-785.

Further, I describe the relation to SPADA (Sum-Product Algorithm Data Association). The Paper *Message Passing Algorithms for Scalable Multitarget Tracking* by Florian Meyer discusses SPADA for object tracking purposes in detail.

> Meyer, Florian & Kropfreiter, Thomas & Williams, Jason & Lau, Roslyn & Hlawatsch, Franz & Braca, Paolo & Win, Moe. (2018). Message Passing Algorithms for Scalable Multitarget Tracking. Proceedings of the IEEE. 106. 221-259. 10.1109/JPROC.2018.2789427.

At the end there is a naive Python implementation, so the mathematics above has a working example beside it.

## Introduction

Data association is a fundamental problem in robotics, computer vision and autonomous systems. The task is to link incoming data — sensor measurements, say — to the entities or tracks that already exist.

**Example 1: Measurement-to-Object Association**

Consider a scenario in autonomous driving or surveillance where a vehicle is equipped with sensors — radar, lidar, a camera — that detect objects in its environment. At each time step, these sensors generate a set of 'measurements' – detections of potential objects. Simultaneously, the system maintains a set of 'objects' (or tracks) that it believes exist and is currently monitoring. The core challenge of measurement-to-object association is to decide, for each new measurement, which of four cases applies:

- it corresponds to an existing object;
- it is a new, previously unobserved object (a *birth*);
- an existing object was not detected in this scan (a *missed detection*, or *misdetection*);
- or the measurement is spurious (*clutter*).

For example, suppose a radar detects a blob at a certain position. In that case, the data association algorithm must decide if this blob is the same car it measured in a previous scan, a newly appeared pedestrian, or just sensor noise. Incorrectly associating a new measurement with an old track can lead to track swapping, where the system believes it's still tracking the original object but is actually tracking a different one. This can have severe consequences, such as an autonomous vehicle mispredicting the behavior of surrounding traffic.

**Example 2: Data Association in Simultaneous Localization and Mapping (SLAM)**

In SLAM, a mobile robot or agent builds a map of an unknown environment while simultaneously estimating its own position within that map. Get the association wrong and both the map and the robot's own position are wrong with it. A robot observes a feature with its sensors: a corner, a door, a landmark. It then has to decide whether that feature is already present in its growing map, or is a new one to be added. This is known as feature association or loop closure detection (when the robot revisits a previously mapped area).

Suppose the robot associates a newly observed feature with an existing map feature that is actually a different one. That introduces errors into both the map and its own estimated pose: duplicate features, and skewed map geometry. The opposite failure costs as much. If the robot does not recognise a feature it has seen before, it concludes it is somewhere new, and the map grows a second copy of a place it had already mapped.

Data association algorithms are also needed in other fields, such as medicine, statistics, biology, and weather forecasting, to name a few. The terminology differs in those fields. This article uses *measurements* and *detections* throughout, for the entities describing a state of the system perceived without prior knowledge from a previous scan. These might be raw measurements, such as positions, or already processed data, such as clusters and features. Objects describe an accumulated, maybe "learned" state or just a previous state of the system. The two words can also separate two sensors rather than two times. In a multi-sensor setup with overlapping fields of view, "measurements" may be the data from one sensor and "objects" the data from another.

Objects and measurements are considered throughout this article as independent and identically distributed (i.i.d.).

**Hungarian-Derived Algorithms**

Hungarian-derived algorithms are the classic methods for solving optimal assignment problems. The Munkres algorithm for the Assignment Problem is the usual example.

Data association can be framed as exactly that task: find a one-to-one mapping between two sets of items, minimising total cost or maximising total similarity. What rules these algorithms out is not accuracy but cost.

The standard Hungarian algorithm runs in $$O(N^3)$$ for $$N$$ items. A tracking or mapping problem routinely carries hundreds or thousands of potential associations. Cubic growth over that many, inside a frame budget and on an embedded processor, is what makes it unusable rather than merely slow.

**The Jonker-Volgenant Algorithm**

The Jonker-Volgenant Algorithm was developed to improve upon the Hungarian Algorithm and is now the preferred method for solving the Linear Assignment Problem (LAP) for large, square cost matrices. Both algorithms reach a theoretical worst-case time complexity of $$O(n^3)$$ in their improved versions. Here $$n$$ is the number of rows or columns of the cost matrix. The Jonker-Volgenant method was nevertheless designed to be practically faster.

## Factor Graphs for Data Association

Four properties of the factor graph matter for data association. Each one is about what the representation makes cheap, not about how it looks.

**The decisions and the rules are separate nodes.** Variable nodes carry the decisions: whether measurement $$i$$ is associated with object $$j$$, and whether it is clutter instead. Factor nodes carry everything that scores or restricts those decisions: the similarity of a pair, and the rule that a measurement may be used once. Keeping them apart is what makes the other three properties possible.

**A new rule is a new factor, not a new algorithm.** Each factor is a local function over the variables it touches. To add a rule — that two objects cannot be arbitrarily close, say — you add a factor and derive the message it sends. The solver itself does not change. [Part 1 of the stereo series](https://www.mariolueder.com/2026-08-07-MASDA-for-Sparse-Stereo-Matching/#6-can-masda-express-the-ordering-constraint) does exactly this for the ordering constraint, and measures what it costs.

The factor graph below and the notation is used throughout this article.
    
![png](/assets/img/2025-11-26-Faster-Data-Association-with-Max-Sum-Loopy-Belief-Propagation-MASDA_files/2025-11-26-Faster-Data-Association-with-Max-Sum-Loopy-Belief-Propagation-MASDA_3_0.png)
    


**Every dependency is drawn.** Every decision connects to what scores it and to what restricts it. Each association decision $$c_{ij}$$, each clutter decision $$e_i$$ and each misdetection decision $$d_j$$ reaches its own similarity score, and the two exclusivity constraints $$I_i$$ and $$E_j$$. A rule the graph does not contain is a rule the solver cannot enforce, and the drawing is where you check which is which.

**Message passing runs directly on the structure.** Max-Sum, used in MASDA, and Sum-Product, used in SPADA, both work by sending local messages along the edges until the assignment stops changing. Each pixel's or measurement's preference reaches the rest of the graph through its neighbours, so a global assignment comes out of local arithmetic. The cost per iteration follows the number of *edges*, not the number of possible assignments.

Data association is inherently a combinatorial problem. Factor graphs and message-passing algorithms search that space without enumerating every possible assignment. What comes back is optimal on a tree, and on a loopy graph usually close to it.

The graph carries two kinds of term:

  * Unary potentials: Costs/similarities for individual assignments ($$S_{ij}$$), clutter ($$\Lambda_i$$), or misdetections ($$\Gamma_j$$).
  * High-order potentials and constraints: relationships involving several variables at once. The mutual exclusivity constraints $$I_i$$ and $$E_j$$ are the ones used here. $$I_i$$ ensures each measurement associates with at most one object or with clutter; $$E_j$$ ensures each object associates with at most one measurement or with a misdetection.

Depending on the message-passing algorithm used:

  * Max-Sum (MASDA) finds the maximum a posteriori assignment: the single most likely set of associations.
  * Sum-Product (SPADA) computes a marginal probability for each possible association. A consumer that needs to weigh an uncertain association rather than accept or reject it wants those numbers.

### SPADA (Sum Product Algorithm Data Association) and MASDA (Max-Sum Algorithm Data Association)

SPADA is a method for solving data association problems using the Sum-Product Algorithm (SPA), also known as Belief Propagation. The Max-Sum algorithm, demonstrated below, finds the single best assignment: the maximum a posteriori, or MAP, estimate. The Sum-Product Algorithm instead computes a marginal probability for every possible assignment. SPADA therefore gives a more complete probabilistic account of the associations, at the cost of more computation.

*Here's a breakdown:*

Instead of just finding the most likely configuration, SPADA computes the marginal probability for each potential association variable: the probability that measurement i is associated with object j, for instance.
It operates on a factor graph, similar to Max-Sum, but the messages passed between nodes represent probability distributions (or likelihoods in the log-domain), rather than just 'scores' or 'costs'.

*Factor Graph Representation*

Variable nodes represent potential associations: $$c_{ij}$$ for associating measurement $$i$$ with object $$j$$, $$e_i$$ for clutter, $$d_j$$ for misdetection. Factor nodes represent the relationships and constraints between those variables — the similarity costs, and the mutual exclusivity constraints.

*Messages in SPADA*

In the Sum-Product Algorithm, messages are typically represented as functions (or vectors) over the possible states of the variable they are being sent to.

**Variable-to-Factor Messages:** A message from a variable node to a factor node is the product of all incoming messages from other factor nodes connected to that variable.

**Factor-to-Variable Messages:** a message from a factor node to a variable node is computed in two steps.

1. Multiply the factor's function by every incoming message from the other variable nodes connected to that factor.
2. Sum, or integrate, over all variables except the recipient.

The 'sum' in Sum-Product is that second step. Max-Sum maximises there instead, which is its 'max'.

Messages are iteratively passed until they converge (or for a fixed number of iterations). Once messages converge, the belief (or marginal probability distribution) for each variable node can be computed by multiplying all messages received by that variable node.
These beliefs are the marginal probabilities of each possible association: $$P(c_{ij}=1)$$, $$P(e_i=1)$$ and $$P(d_j=1)$$.

*Key Differences from Max-Sum (MASDA)*

Max-Sum aims for the MAP assignment, SPADA for marginal probabilities. Therefore Max-Sum uses max and sum operations, while SPADA uses sum and product operations.

In the log-domain SPADA uses log-sum-exp and sum instead. That prevents underflow, and it replaces multiplication with addition.
Max-Sum typically gives a hard assignment (0 or 1 for each $$c_{ij}$$). SPADA provides probabilities (values between 0 and 1) for each association.
Where the right answer is ambiguous, SPADA says so with a number instead of picking one.

*Advantages of SPADA*

SPADA returns the uncertainty rather than hiding it. Where several assignments are plausible it gives each a non-zero probability instead of choosing between them. Those marginals are numbers a consumer can carry: an expected value, a variance, or a weight on a track that does not have to be accepted or discarded outright.

*Applications*

SPADA is used where the probability itself is wanted, not just the winner: multi-target tracking, sensor fusion, and the part of SLAM that has to weigh one candidate map update against another.
In short: Max-Sum finds the best way to associate measurements with objects. SPADA says how likely each possible association is, which is a more complete picture of the uncertainty in the system.


## MASDA

MASDA, or Max-Sum Algorithm Data Association, is an iterative message-passing algorithm used to find the most likely (Maximum A Posteriori or MAP) assignment in data association problems. It operates on a factor graph, which visually represents the variables involved and the relationships (factors) between them.

### Factor Graph Structure

The factor graph structure for MASDA remains the same as before. The architecture and the derivation below follows closely the ideas in [A Binary Variable Model for Affinity Propagation](https://www.researchgate.net/publication/23975825_A_Binary_Variable_Model_for_Affinity_Propagation)

**Variable Nodes** (circles — $$c_{ij}$$, $$e_i$$, $$d_j$$): These represent the decisions or unknowns we want to infer:

  * $$c_{ij}$$: Binary variables indicating whether measurement $$i$$ is associated with object $$j$$ ($$c_{ij}=1$$) or not ($$c_{ij}=0$$).
  * $$e_i$$: Binary variables indicating whether measurement $$i$$ is 'clutter' (not associated with any existing object).
  * $$d_j$$: Binary variables indicating whether object $$j$$ is 'misdetected' (not associated with any measurement).

**Factor Nodes** (squares — $$s_{ij}$$, $$\Lambda_i$$, $$\Gamma_j$$, $$E_j$$, $$I_i$$): These represent functions that define relationships, costs, or constraints between the connected variable nodes:

*Similarity Factors* ($$s_{ij}$$): a factor $$S_{ij}(c_{ij})$$ encodes the cost or similarity of associating measurement $$i$$ with object $$j$$. When $$c_{ij}=1$$, its value is $$s(i,j)$$ (the similarity score); otherwise, it's 0. These are essentially priors for specific associations.

$$
S_{ij}(c_{ij})  =
\begin{cases}
    s(i,j),& \text{if } c_{ij} = 1\\
    0,              & \text{otherwise}
\end{cases}
$$

*Clutter Factors* ($$\Lambda_i$$): a factor $$\Lambda_i(e_i)$$ represents the cost or prior probability that measurement $$i$$ is clutter. If $$e_i=1$$, its value is $$\lambda(i)$$; otherwise, it's 0.

$$
\Lambda_{i}(e_{i})  =
\begin{cases}
    \lambda(i),& \text{if } e_{i} = 1\\
    0,              & \text{otherwise}
\end{cases}
$$

*Misdetection Factors* ($$\Gamma_j$$): a factor $$\Gamma_j(d_j)$$ represents the cost or prior probability that object $$j$$ is misdetected. If $$d_j=1$$, its value is $$\gamma(j)$$; otherwise, it's 0.

$$
\Gamma_{j}(d_{j})  =
\begin{cases}
    \gamma(j),& \text{if } d_{j} = 1\\
    0,              & \text{otherwise}
\end{cases}
$$

Constraint Factors ($$I_i$$ and $$E_j$$): these enforce the rules of data association.

*Measurement Constraint Factors* ($$I_i$$): (For each measurement $$i$$) This factor ensures that each measurement $$i$$ is either associated with exactly one object, or it is declared as clutter ($$e_i=1$$). The function $$I_i(c_{i1}, \dots, c_{iN}, e_i)$$ evaluates to $$-\infty$$ if $$e_i + \sum_j c_{ij} \neq 1$$, and to $$0$$ otherwise. In the log-domain a score of $$-\infty$$ is not an expensive option, it is an unavailable one: nothing finite can be added to it to bring it back.

$$
I_{i}(c_{i1},\dots,c_{iN}, e_i)  =
\begin{cases}
    -∞,& \text{if } e_i + \sum_i c_{ij} \neq 1\\
    0,              & \text{otherwise}
\end{cases}
$$

*Object Constraint Factors* ($$E_j$$): (For each object $$j$$) This factor ensures that each object $$j$$ is either associated with exactly one measurement, or it is declared as misdetected ($$d_j=1$$). The function $$E_j(c_{1j}, \dots, c_{Nj}, d_j)$$ evaluates to $$-\infty$$ if $$d_j + \sum_i c_{ij} \neq 1$$. Otherwise, it's 0.


$$
E_{j}(c_{1j},\dots,c_{Nj},d_j)  =
\begin{cases}
    -∞,& \text{if } d_j + \sum_j c_{ij} \neq 1\\
    0,              & \text{otherwise}
\end{cases}
$$

### Objective Function:
The goal of MASDA is to find the assignment of all $$c_{ij}$$, $$e_i$$, and $$d_j$$ variables that maximizes the total score (or minimizes total cost) as defined by the objective function $$\mathcal{S}$$:

$$
\mathcal{S}(\dots) = \sum_{i,j}S_{ij}(c_{ij}) + \sum_{j}\Gamma_{j}(d_{j}) + \sum_{i}\Lambda_{i}(e_{i}) + \sum_{i}I_{i}(c_{i1},\dots,c_{iN}, e_i) + \sum_{j}E_{j}(c_{1j},\dots,c_{Nj}, d_j)
$$

This function sums up all the similarity/cost factors for associations, clutter, misdetections, and critically, applies the penalties for violating the mutual exclusivity constraints.



### Message Passing (The "Max-Sum" part):
MASDA works by iteratively passing "messages" between the variable and factor nodes. These messages represent beliefs or preferences about the state of the variables. The algorithm is typically run in the log-domain, so "sum" replaces "product" and "max" replaces "sum" from the original Sum-Product algorithm. The messages passed are essentially scores.

**Variable-to-Factor Messages:** A variable node $$x$$ sends a message to a factor node $$f$$ that is the sum of all incoming messages to $$x$$ from other factor nodes (excluding $$f$$).

$$
\mu_{x\rightarrow f}(x) = \sum_{\{l|f_l \in \text{ne}(x) \setminus f\}}\mu_{f_l\rightarrow x}(x)
$$

**Factor-to-Variable Messages:** a factor node $$f$$ sends a message to a variable node $$x$$ in three steps.

1. Sum the messages from the other variables connected to $$f$$, excluding $$x$$.
2. Add $$f$$'s own function value.
3. Take the maximum over all possible assignments of those other variables.

$$
\mu_{f_l\rightarrow x}(x) = \max_{x_1,\dots,x_M}\left[f(x, x_1, \dots, x_m) + \sum_{\{m|x_m \in \text{ne}(f) \setminus x\}}\right]
$$

    
![png](/assets/img/2025-11-26-Faster-Data-Association-with-Max-Sum-Loopy-Belief-Propagation-MASDA_files/2025-11-26-Faster-Data-Association-with-Max-Sum-Loopy-Belief-Propagation-MASDA_10_0.png)
    
    
![png](/assets/img/2025-11-26-Faster-Data-Association-with-Max-Sum-Loopy-Belief-Propagation-MASDA_files/2025-11-26-Faster-Data-Association-with-Max-Sum-Loopy-Belief-Propagation-MASDA_11_0.png)
    
    
![png](/assets/img/2025-11-26-Faster-Data-Association-with-Max-Sum-Loopy-Belief-Propagation-MASDA_files/2025-11-26-Faster-Data-Association-with-Max-Sum-Loopy-Belief-Propagation-MASDA_12_0.png)
    


The messages for the binary nodes $$c_{ij}$$, $$e_i$$, $$d_j$$ can be theoretically expressed with with a two-valued vector. But this is for our purpose of finding the best association, a MAP estimate, not necessary. Propagating its differences, a scalar, is enough as the two values can be recovered from this difference up to an additive constant factor.

For example: The message from Factor $$I_j$$ to the variable $$c_{ij}$$ is denoted with $$\beta_{ij}(m)$$
$$
\beta_{ij}(m) = \mu_{c_{ij} \rightarrow I_i}(m)
$$
where
$$
m \in \{0,1\}
$$
The scalar message difference is denoted
$$
\beta_{ij} = \beta_{ij}(1) - \beta_{ij}(0)
$$
This notation is also used for the messages $$\alpha_{ij}$$, $$\rho_{ij}$$ and $$\eta_{ij}$$


### Variable to Factor Messages

These messages are just computed by the sum of the incoming messages, as stated above.

In order to simplify the notation we substitude the variables of the nodes and factors as follows

$$
c_{i,n+1} = e_i
$$

$$
c_{m+1,j} = d_j
$$

$$
S_{i,n+1} = \Lambda_i
$$

$$
S_{m+1,j} = \Gamma_j
$$

$$
s_{i,n+1} = \lambda_i
$$

$$
s_{m+1,j} = \gamma_j
$$

#### Message To $$I$$

for $$c_{ij} = 1$$:

$$
\begin{align}
\beta_{ij}(1) & = \mu_{c_{ij} \rightarrow I_i}(1) \\
& = \sum_{\{l|f_l \in \text{ne}(c_{ij}) \setminus I_i\}}\mu_{f_l\rightarrow c_{ij}}(1) \\
& = S_{ij}(1) + \alpha_{ij}(1)
\end{align}
$$

for $$c_{ij} = 0$$:

$$
\beta_{ij}(0) = S_{ij}(0) + \alpha_{ij}(0)
$$

Scalar message

$$
\begin{align}
\beta_{ij} & = \beta_{ij}(1) - \beta_{ij}(0) \\
& = (S_{ij}(1) - S_{ij}(0)) + (\alpha_{ij}(1) - \alpha_{ij}(0)) \\
& = s(i,j) + \alpha_{ij} \\
\beta_{i,n+1} & = \lambda(i)
\end{align}
$$

#### Message To $$E$$

The derivation is similar to $$I$$

$$
\begin{align}
\rho_{ij}(1) & = S_{ij}(1) + \eta_{ij}(1) \\
\rho_{ij}(0) & = S_{ij}(0) + \eta_{ij}(0) \\
\rho_{ij} & = \rho_{ij}(1) - \rho_{ij}(0) \\
& = (S_{ij}(1) - S_{ij}(0)) + (\eta_{ij}(1) - \eta_{ij}(0)) \\
& = s(i,j) + \eta_{ij}\\
\rho_{m+1,j}(1) & = \gamma(j)
\end{align}
$$

There is no message to the $$S$$ factor at it is used as a prior.

### Factor to Variable Messages

We make use of the previously given factor to variable message update equation. Again, we will fix $$c_{ij}$$ to $$0$$ and $$1$$ respectively. Any combination that violates the constraints evaluates to $$\infty$$ and is non optimal.


#### From $$I$$

This constrains only rows, ensuring that one measurement is associated to one object or is clutter/birth.

for $$c_{ij} = 1$$, the association case

$$
\begin{align}
\eta_{ij}(1) & = \mu_{I_i\rightarrow c_{ij}}(1) \\
& = \max_{c_{ik}, k \ne j}\left[I_i(c_{i1},\dots, c_{ik} = 1, \dots, c_{iN}) + \sum_{\{t|c_{it} \in \text{ne}(I_i) \setminus c_{ij}\}} \mu_{c_{it} \rightarrow I_i}(c_{it}) \right] \\
& = \max_{c_{ik}, k \ne j}\left[I_i(c_{i1},\dots, c_{ik} = 1, \dots, c_{iN}) + \sum_{\{t|c_{it} \in \text{ne}(I_i) \setminus c_{ij}\}} \beta_{it}(c_{it}) \right] \\
& = \sum_{t \ne j} \beta_{it}(0)
\end{align}
$$

This is as only one node in the neighborhood of $$I$$ can be set to $$1$$

for $$c_{ij} = 0$$, the non association case

$$
\begin{align}
\eta_{ij}(0) & = \mu_{I_i\rightarrow c_{ij}}(0) \\
& = \max_{c_{ik}, k \ne j}\left[I_i(c_{i1},\dots, c_{ik} = 0, \dots, c_{iN}) + \sum_{\{t|c_{it} \in \text{ne}(I_i) \setminus c_{ij}\}} \mu_{c_{it} \rightarrow I_i}(c_{it}) \right] \\
& = \max_{k \ne j}\left[\beta_{ik}(1) + \sum_{t \notin \{k,j\} } \beta_{it}(0) \right]
\end{align}
$$

Exactly one of the variables in a row must be set to $$1$$, all the others must be set to $$0$$

The scalar message is

$$
\begin{align}
\eta_{ij} & = \eta_{ij}(1) - \eta_{ij}(0) \\
& = - (\eta_{ij}(0) - \eta_{ij}(1)) \\
& = - \max_{k \ne j}\left[\beta_{ik}(1) + \sum_{t \notin \{k,j\} } \beta_{it}(0) - \sum_{t \ne j} \beta_{it}(0) \right] \\
& = - \max_{k \ne j}\left[\beta_{ik}(1) - \beta_{ik}(0) \right] \\
& = - \max_{k \ne j}\beta_{ik}
\end{align}
$$

#### From $$E$$

This can be derived similarly as the constraint is the same but ensures that only one vertical node is active (ensuring that one object is associated to one measurement)

for $$c_{ij} = 1$$ the association case

$$
\begin{align}
\alpha_{ij}(1) & = \mu_{E_j\rightarrow c_{ij}}(1) \\
& = \max_{c_{kj}, k \ne i}\left[E_j(c_{1j},\dots, c_{kj} = 1, \dots, c_{Nj}) + \sum_{\{t|c_{tj} \in \text{ne}(E_j) \setminus c_{ij}\}} \mu_{c_{tj} \rightarrow E_j}(c_{tj}) \right] \\
& = \max_{c_{kj}, k \ne i}\left[E_j(c_{1j},\dots, c_{kj} = 1, \dots, c_{Nj}) + \sum_{\{t|c_{tj} \in \text{ne}(E_j) \setminus c_{ij}\}} \rho_{tj}(c_{tj}) \right] \\
& = \sum_{t \ne i} \rho_{tj}(0)
\end{align}
$$

for $$c_{ij} = 0$$, the non association case

$$
\begin{align}
\alpha_{ij}(0) & = \mu_{E_j\rightarrow c_{ij}}(0) \\
& = \max_{c_{kj}, k \ne i}\left[E_j(c_{1j},\dots, c_{kj} = 0, \dots, c_{Nj}) + \sum_{\{t|c_{tj} \in \text{ne}(E_j) \setminus c_{ij}\}} \mu_{c_{tj} \rightarrow E_j}(c_{tj}) \right] \\
& = \max_{k \ne i}\left[\rho_{kj}(1) + \sum_{t \notin \{k,i\} } \rho_{tj}(0) \right]
\end{align}
$$

Exactly one of the variables in a column must be set to $$1$$, all the others must be set to $$0$$

The scalar message is

$$
\begin{align}
\alpha_{ij} & = \alpha_{ij}(1) - \alpha_{ij}(0) \\
& = - (\alpha_{ij}(0) - \alpha_{ij}(1)) \\
& = - \max_{k \ne i}\left[\rho_{kj}(1) + \sum_{t \notin \{k,i\} } \rho_{tj}(0) - \sum_{t \ne i} \rho_{tj}(0) \right] \\
& = - \max_{k \ne i}\left[\rho_{kj}(1) - \rho_{kj}(0) \right] \\
& = - \max_{k \ne i}\rho_{kj}
\end{align}
$$

#### From $$S$$

The scalar message send from the $$S_{ij}$$ function is always the similarity between $$i$$ and $$j$$, since $$s(i,j) = S_{ij}(1) - S_{ij}(0)$$


## Summary

As

$$
\begin{align}
\beta_{ij} & = s(i,j) + \alpha_{ij} \\
\beta_{i,n+1} & = s(i,n+1)  = \lambda_i \\
\rho_{ij} & = s(i,j) + \eta_{ij} \\
\rho_{m+1,j} & = s(m+1,j) = \gamma_j \\
\eta_{ij} & = - \max_{k \ne j}\beta_{ik} \\
\alpha_{ij} & = - \max_{k \ne i}\rho_{kj}
\end{align}
$$

### Combined

$$
\begin{align}
\beta_{ij} & = s(i,j) - \max_{k \ne i}\rho_{kj} \\
\rho_{ij} & = s(i,j) - \max_{k \ne j}\beta_{ik} \\
\end{align}
$$

### The belief

Is just the sum of the messages towards a variable node.

$$
\mu_{x}(x) = \sum_{\{l|f_l \in \text{ne}(x) \}}\mu_{f_l\rightarrow x}(x)
$$

Belief association between measurement and object

$$
b_{ij} = \alpha_{ij} + \eta_{ij} + s_{ij}
$$

Belief measurement is clutter or a new born object

$$
b_{i,n+1} = \eta_{i,n+1} + \lambda_{i}
$$

Belief that an object was misdetected

$$
b_{m + 1, j} = \alpha_{m+1, j} + \gamma_{j}
$$


### Iterative Process

These messages $$\beta$$ and $$\rho$$ are iteratively updated until convergence or for a fixed number of iterations. As messages propagate, information about similarities, clutter/misdetection likelihoods, and constraints are shared across the graph. This allows the algorithm to find a globally consistent assignment that satisfies as many preferences and constraints as possible.


### Final Beliefs and Assignment

After message passing, the final "belief" for each variable, $$c_{ij}$$ among them, is computed by summing all incoming messages to that variable from its connected factors. The assignment that maximizes this belief for each variable is chosen, resulting in a set of binary associations, cluttered measurements, and misdetected objects.

That decode is the whole output. It is MASDA's answer to the data association problem, reached by passing local preferences and global constraints along the edges until they settle, and approximate because the graph is loopy.

### Computational Complexity: MASDA vs. Jonker-Volgenant

The Jonker-Volgenant algorithm solves the Linear Assignment Problem exactly, in $$O(N^3)$$. Here $$N$$ is the number of items to be assigned — measurements or objects — after the problem has been augmented with the extra rows and columns that clutter and misdetection need.

This complexity holds when you have a well-defined cost matrix and are seeking a strict one-to-one assignment that minimizes total cost. JV is known for its speed and guaranteed optimality for LAP.

MASDA is an iterative inference algorithm, a form of Belief Propagation. Its complexity, similar to SPADA, depends on the graph structure and the number of iterations.

In a data association factor graph with $$M$$ measurements and $$N$$ objects, computing messages for MASDA involves performing 'max' operations over the neighbors of each node. The complexity per iteration is roughly proportional to two things: the number of edges, and the number of states a variable can take within a factor. That usually scales as O(M * N * K). Here K is related to the maximum degree of the nodes, or to the size of the variable domains within factors.

Loopy factor graphs are the normal case in data association, and on them MASDA is an approximate inference algorithm. It is not guaranteed to reach the globally optimal MAP assignment, and it is not guaranteed to converge at all. In practice it runs for a fixed number of iterations $$T$$, which makes the total complexity $$O(T \cdot M \cdot N \cdot K)$$.

Note that for our purpose of data association, K is set 1.

*Key Differences:*

JV solves the Linear Assignment Problem provably optimally. MASDA solves the MAP problem on loopy graphs approximately. Its answers are usually close, and nothing guarantees it.

Because MASDA passes messages, complex constraints go directly into the factor graph structure. One measurement per object, one object per measurement, clutter and misdetection all enter that way, with none of the matrix augmentation JV usually needs.

For a pure LAP at moderate $$N$$, JV's $$O(N^3)$$ is the one to use. Here $$N$$ is the number of potential associations.

At very large N, MASDA can scale better in practice, provided its factor graph is sparse and few iterations T are needed. That applies to problems JV would struggle to model directly without heavy pre-processing or approximation.

In short: if the problem can be framed accurately as a pure Linear Assignment Problem, Jonker-Volgenant gives a guaranteed optimal solution.

MASDA is the alternative where that framing does not hold. Three cases put it there:

- probabilistic dependencies JV cannot express;
- assignments that are not strictly one-to-one;
- clutter and misdetection carried in the model rather than added by augmentation.

Its solutions are approximate, and at a fixed number of iterations it is often faster as well.

### Computational Complexity: MASDA vs. SPADA

Both SPADA (Sum-Product Algorithm Data Association) and MASDA (Max-Sum Algorithm Data Association) are iterative message-passing algorithms operating on factor graphs.

SPADA generally involves higher computational efforts compared to MASDA because it aims to compute marginal probabilities for each possible assignment, rather than just the single most likely assignment.
No formula for it is derived here. Being iterative on a factor graph, SPADA has the same structure of complexity per iteration as MASDA. What differs is the operation: marginalisation needs 'sum', or log-sum-exp in the log-domain, and both are more demanding than MASDA's 'max'.
The message computations in SPADA represent probability distributions, adding to the complexity compared to MASDA's scalar scores.

## Implementation

The implementation below is written to be read rather than to be fast. Much of it can be optimised, and the largest gain is in the sparse-matrix form.

The problem shown here is an artificial one. Five measurements are given as two-dimensional positions. We got as well four object positions in 2D. The one-to-one association of measurements with objects might appear ambiguous for the human eye. One measurement can be associated to exacly one object or is clutter/new born. One object can be associated to exactly one measurement or is misdetected. Using just the euclidean distance might lead to a violation of these constraints.

### Includes
```python
import numpy as np
import torch
import random
import os
import matplotlib.pyplot as plt
import sys
```

### Measurements and Objects

```python
# Measurements
points = torch.tensor([
    [40.1732, 3.66123],
    [40.2158, 3.43958],
    [40.2211, 3.21301],
    [40.3561, 2.99893],
    [42.4198, 2.91224],
])

points = torch.t(points)
meas_x = points[0,:]
meas_y = points[1,:]

# Objects
objs = torch.tensor([
    [40.7211, 3.41301],
    [40.7158, 3.63958],
    [40.6732, 3.96123],
    [43.7526, 2.19003]
])

objs = torch.t(objs)
objs_x = objs[0,:]
objs_y = objs[1,:]

# plot the measurements meas and the objects

plt.figure(figsize=(8, 6)) # Create a new figure for clarity
plt.scatter(meas_x.numpy(), meas_y.numpy(), label='Measurements (meas)', color='blue')
plt.scatter(objs_x.numpy(), objs_y.numpy(), label='Objects (objs)', color='red')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.title('Plot of Measurements and Objects')
plt.legend()
plt.grid(True)
plt.show()
```
    
![png](/assets/img/2025-11-26-Faster-Data-Association-with-Max-Sum-Loopy-Belief-Propagation-MASDA_files/2025-11-26-Faster-Data-Association-with-Max-Sum-Loopy-Belief-Propagation-MASDA_25_0.png)
    

### Shuffle the measurements and object

```python
# Get the number of measurements and objects
num_measurements = meas_x.shape[0]
num_objects = objs_x.shape[0]

# Create shuffled indices for measurements and objects
shuffled_meas_indices = torch.randperm(num_measurements)
shuffled_objs_indices = torch.randperm(num_objects)

# Shuffle the measurements and objects using the shuffled indices
shuffled_meas_x = meas_x[shuffled_meas_indices]
shuffled_meas_y = meas_y[shuffled_meas_indices]

shuffled_objs_x = objs_x[shuffled_objs_indices]
shuffled_objs_y = objs_y[shuffled_objs_indices]

```

### Compute a distance matrix using the shuffeled measurements

```python
# Reshape shuffled measurements and objects for broadcasting
shuffled_meas = torch.stack((shuffled_meas_x, shuffled_meas_y), dim=1)
shuffled_objs = torch.stack((shuffled_objs_x, shuffled_objs_y), dim=1)

shuffled_meas = torch.stack((meas_x, meas_y), dim=1)
shuffled_objs = torch.stack((objs_x, objs_y), dim=1)

# Compute the squared difference in x and y coordinates
diff_x = shuffled_meas[:, None, 0] - shuffled_objs[None, :, 0]
diff_y = shuffled_meas[:, None, 1] - shuffled_objs[None, :, 1]


# Compute the squared Euclidean distance
dist_sq = diff_x**2 + diff_y**2

# Take the square root to get the Euclidean distance matrix
#distance_matrix = -torch.sqrt(dist_sq)
distance_matrix = -dist_sq

# distance_matrix = torch.cat((distance_matrix, torch.ones(num_measurements, 1) * -2.), dim=1) # column for clutter
# distance_matrix = torch.cat((distance_matrix, torch.ones(1, num_objects + 1) * -2.), dim=0) # row for misdetections

print("Distance Matrix:")
distance_matrix
```

    Distance Matrix:





    tensor([[ -0.3618,  -0.2949,  -0.3400, -14.9766],
            [ -0.2560,  -0.2900,  -0.4813, -14.0703],
            [ -0.2900,  -0.4267,  -0.7642, -13.5180],
            [ -0.3047,  -0.5398,  -1.0266, -12.1906],
            [ -3.1364,  -3.4326,  -4.1510,  -2.2979]])



### MASDA - iteration and compute the belief

```python
# both beta and rho matrices are initialized with a large positive value (1e+3)
# for all their elements. In the context of the Max-Sum algorithm, where
# messages represent scores or preferences, initializing with a large positive
# value like this often serves as a 'neutral' or 'uninformative' starting point.
# It essentially means that, initially, there is no strong preference for any
# particular association (or non-association) based on prior messages.
beta = torch.ones(distance_matrix.shape) * 1e+3
rho = torch.ones(distance_matrix.shape) * 1e+3
n_meas_assoc = distance_matrix.shape[0]
n_obj_assoc = distance_matrix.shape[1]

# Define the constant costs/scores for misdetection and clutter.
# These represent the prior belief or preference for an object to be misdetected
# or a measurement to be clutter/newly born.
misdetection = -1.0
clutter = misdetection


# print("n_objs:", n_obj_assoc)
# print("n_points:", n_meas_assoc)
# print("n_objs:", num_objects)
# print("n_points:", num_measurements)

# Damping factor to stabilize message updates, especially in loopy graphs.
# A value of 0.0 means no damping (messages are fully replaced each iteration).
# A value closer to 1.0 would mean messages change slowly (weighted average of old and new).
damping = 0.0

# Initialize belief matrix. It will store the final scores for each possible assignment,
# including clutter and misdetection cases.
belief = torch.ones((n_meas_assoc + 1, n_obj_assoc + 1)) * misdetection

# Function to compute the final beliefs after message passing.
# The belief for each assignment (measurement i to object j, clutter, or misdetection)
# is the sum of relevant incoming messages and the similarity score.
def compute_belief():
    # Iterate through each measurement to calculate its association beliefs
    for i in range(num_measurements):
        # Mask for object-specific calculations: ensures we don't consider
        # the current object's own message when calculating max for alpha.
        # This large value effectively 'removes' the current element from max operations
        # by making it so large that it will never be selected.
        mask_meas = torch.zeros(n_meas_assoc)
        mask_meas[i] = 1.e+20

        # Calculate beliefs for measurement-to-object associations
        for j in range(num_objects):
            # Mask for measurement-specific calculations: similar to mask_meas but for beta.
            mask_obj = torch.zeros(n_obj_assoc)
            mask_obj[j] = 1.e+20
            # eta_ij = -max_{k!=j} beta_ik (message from I_i to c_ij)
            # It represents the penalty if measurement 'i' were associated with 'j',
            # considering alternatives to 'j'.
            eta = - torch.max(torch.max(beta[i,:] - mask_obj), torch.tensor(clutter))
            # alpha_ij = -max_{k!=i} rho_kj (message from E_j to c_ij)
            # It represents the penalty if object 'j' were associated with 'i',
            # considering alternatives to 'i'.
            alpha = - torch.max(torch.max(rho[:,j] - mask_meas), torch.tensor(clutter))
            # The belief for associating measurement 'i' with object 'j' is the sum of
            # these two messages and the direct similarity score.
            belief[i,j] = eta + alpha + distance_matrix[i,j]

    # Calculate beliefs for measurements being clutter (new born object)
    for i in range(num_measurements):
        j = num_objects # This represents the 'clutter' column
        # eta_i,n+1 = -max_{k} beta_ik (message from I_i to e_i)
        # It represents the penalty if measurement 'i' is clutter, considering all object alternatives.
        eta = - torch.max(beta[i,:])
        # Belief for measurement 'i' being clutter is this message plus the clutter cost.
        belief[i,j] = eta + clutter

    # Calculate beliefs for objects being misdetected
    for j in range(num_objects):
        i = num_measurements # This represents the 'misdetection' row
        # alpha_m+1,j = -max_{k} rho_kj (message from E_j to d_j)
        # It represents the penalty if object 'j' is misdetected, considering all measurement alternatives.
        alpha = - torch.max(rho[:,j])
        # Belief for object 'j' being misdetected is this message plus the misdetection cost.
        belief[i,j] = alpha + misdetection

    # print("Belief:\n", belief)
    # print(torch.sum(belief))

# Main message passing loop
for iter in range(6):
   # Update beta messages (variable to factor I_i)
   for i in range(num_measurements):
        # Create a mask to exclude the current measurement's own message when calculating
        # the maximum over rho_kj in the beta update.
        mask = torch.zeros(n_meas_assoc)
        mask[i] = 1.e+20 # Large value to ensure it's not chosen by torch.max

        for j in range(num_objects):
            # Beta update: beta_ij = s(i,j) - max_{k!=i} rho_kj
            # max_val here computes max_{k!=i} rho_kj or clutter
            max_val = torch.max(torch.max(rho[:,j] - mask), torch.tensor(clutter))
            # Apply damping to the message update
            beta[i,j] = beta[i,j] * damping + (distance_matrix[i,j] - max_val) * (1. - damping)

   # Update rho messages (variable to factor E_j)
   for j in range(num_objects):
        # Create a mask to exclude the current object's own message when calculating
        # the maximum over beta_ik in the rho update.
        mask = torch.zeros(n_obj_assoc)
        mask[j] = 1.e+20 # Large value to ensure it's not chosen by torch.max

        for i in range(num_measurements):
            # Rho update: rho_ij = s(i,j) - max_{k!=j} beta_ik
            # max_val here computes max_{k!=j} beta_ik or misdetection
            max_val = torch.max(torch.max(beta[i,:] - mask), torch.tensor(misdetection))
            # Apply damping to the message update
            rho[i,j] = rho[i,j] * damping + (distance_matrix[i,j] - max_val) * (1. - damping)

   # print rho and beta
   # print("beta: ", beta)
   # print("rho: ", rho)

# Recompute beliefs after message passing
compute_belief()
# print(belief)
```

### Plot the association

```python
# get the maximum index from beta and plot the point association using this information

beta_misdetection = torch.cat((beta, torch.ones(num_measurements, 1) * misdetection), dim=1) # column for clutter
# distance_matrix = torch.cat((distance_matrix, torch.ones(1, num_objects + 1) * -2.), dim=0) # row for misdetections

beta_misdetection = belief
# print(beta_misdetection)

# Get the index of the maximum value in beta
max_beta_index = torch.argmax(beta_misdetection,1)
# print(max_beta_index)

# Convert the linear index to row and column indices
row_index = range(num_measurements)
col_index = max_beta_index

# print(f"Maximum value of beta is at index: ({row_index}, {col_index})")

# Extract the associated points from the shuffled measurements and objects
associated_measurement = shuffled_meas[row_index]

# print(f"Associated Measurement Point:\n{associated_measurement.numpy()}")
# print(f"Associated Object Point:\n{associated_object.numpy()}")

# Plot the associated points and a line connecting them
plt.figure(figsize=(8, 6))
plt.scatter(shuffled_meas_x.numpy(), shuffled_meas_y.numpy(), label='Shuffled Measurements', color='blue', alpha=0.5)
plt.scatter(shuffled_objs_x.numpy(), shuffled_objs_y.numpy(), label='Shuffled Objects', color='red', alpha=0.5)
for i in range(num_measurements):
    if max_beta_index[i] < num_objects:
        plt.plot([shuffled_meas[i][0].numpy(), shuffled_objs[max_beta_index[i]][0].numpy()],
            [shuffled_meas[i][1].numpy(), shuffled_objs[max_beta_index[i]][1].numpy()],
                color='green', linestyle='--', label='Association')

plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.title('Point Association based on Maximum Beta')
plt.legend()
plt.grid(True)
plt.show()
```

![png](/assets/img/2025-11-26-Faster-Data-Association-with-Max-Sum-Loopy-Belief-Propagation-MASDA_files/2025-11-26-Faster-Data-Association-with-Max-Sum-Loopy-Belief-Propagation-MASDA_29_0.png)
    

### Discussion

The hypothesis is a sensible one. The top-left measurements go to the top-left objects. The other two measurements have no object anywhere near them and stay unassociated, which leaves the bottom-right object undetected.

The computation of the similarities is too simplistic here to be considered a real-world example. One would incorporate uncertainties. Further, with objects and measurements streaming from a radar or a lidar, detection probabilities and clutter probabilities belong in the model too. Both come from functions describing what the sensor can do and what the environment constrains.
