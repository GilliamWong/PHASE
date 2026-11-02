# PHASE: Algorithmic Contribution

November 1, 2026 · William Gong

PHASE splits each human intervention in two: imitate the human's actions while they pull the robot out of a bad state, then stop copying actions and instead reward the robot for reaching the states the human took it to.

## Core idea

Humans tend to intervene when the robot's progress stalls or declines. What they do after taking over carries two different kinds of information:

- **Recovery (early in the intervention).** The robot is in a failure state its own policy reached. Its own actions there are known to be bad, and recovery is hard to discover by random exploration, so the human's *actions* are worth imitating. This is DAgger's core argument.
- **Cruising (later).** The robot is back on track and the human is just driving. Their actions are now mostly style: axis-aligned, saturated teleoperation inputs, and one of several valid solution modes. What is worth keeping is *where* they took the robot.

Cloning every intervention action copies suboptimal and stylistic behavior, averages across correction modes (a Gaussian policy fit to left-side and right-side grasps lands in between), and biases the final policy if its weight never decays. Q-OIL's decoupled bonus does not fix this: because its optimistic critic bootstraps from the task critic, the bonus only favors human actions once the robot is already in an intervention state. Nothing attracts the robot toward those states.

PHASE uses each signal where it is trustworthy: behavior cloning on recovery steps, and potential-based reward shaping toward post-recovery states. The shaping propagates through the critic, so it delivers the attraction toward human-guided regions that Q-OIL claimed but does not have. Because potential-based shaping leaves the optimal policy unchanged, a poor human route can slow learning but cannot make the final policy worse.

**Novelty, scoped.** Shaping from human data has prior work, notably Brys et al. (IJCAI 2015), which builds a potential from demonstrations, as well as state-only imitation (learning from observation). The likely-new part is splitting individual interventions into an action phase and a state phase. Confirm with a literature search before claiming it.

## The algorithm

PHASE adds two terms to a standard off-policy actor-critic (RLPD): a behavior-cloning term on recovery steps and a shaping reward toward post-recovery states. It needs no accurate value function, keeps every intervention, and runs on a real robot.

### Notation

An intervention starts when the human takes over and ends when they hand back, $K$ steps later. $\tau = 0, 1, \dots, K$ counts steps since takeover; $s_\tau$ is the state after $\tau$ human steps; $s_0$ is the takeover state and $s_K$ the handback state.

### 1. Switch point

Recovery ends once the human has brought the robot somewhere that looks like successful behavior:

$$
\hat\tau^* = \min\{\tau : d(s_\tau, \mathcal{D}_{\text{good}}) < \epsilon\}, \qquad \mathcal{D}_{\text{good}} = \{\text{earlier handback states}\} \cup \{\text{states from successful episodes}\}
$$

$d$ is a nearest-neighbor distance over a few normalized features (gripper and object positions). Handback states are a free label: a human hands back when they judge the robot is on track. No demonstrations are needed. Until the reference set has entries, treat the whole segment as recovery, which is plain DAgger-style imitation and right while the policy is poor.

| Variant | Switch point | Role |
| --- | --- | --- |
| Proximity | first step within $\epsilon$ of the reference set | main method |
| Fixed schedule | weight $e^{-\tau/\tau_0}$ on actions, the rest on states | simplest baseline |
| Oracle | handback curve from counterfactual rollouts (next section) | simulation-only upper bound |

### 2. Recovery behavior cloning

Imitate the human's actions on steps before the switch point. Two implementations differ in an important way.

- **Averaged over the intervention buffer (Q-OIL's form).** Its weight in the loss stays at $\lambda$ forever, no matter how much other data accumulates.
- **Sampled from the main replay buffer, applied only to recovery steps.** Its effective weight is $\lambda$ times the fraction of recovery steps in the buffer, so it dilutes automatically as successful data accumulates. The fraction shrinks slowly, roughly like $1/t$.

$$
\mathcal{L}_{\text{BC}} = -\lambda\, \mathbb{E}_{(s,a)\sim\mathcal{D}_\pi}\big[\mathbb{1}[\text{recovery step}]\cdot \log \pi_\theta(a \mid s)\big]
$$

**Open question: should recovery BC decay at all?**

- *Keep it.* Recovery states are failure states. As the policy improves it rarely visits them, so BC there stops affecting normal behavior, and keeping it preserves the recovery skill for rare failures.
- *Decay it.* Human recoveries are still suboptimal, so once RL can recover better on its own, BC holds it back. Intervention states may also sit close to normal successful states, so BC on them can leak into ordinary behavior.

Resolve it with a three-way ablation, using recovery under perturbation as the metric that would expose any cost: fixed $\lambda$; diluted (main-buffer sampling, above); and annealed by the intervention rate, $\lambda_t = \lambda_0 \cdot \bar\rho_t / \bar\rho_0$, where $\bar\rho_t$ is a moving average of the intervention rate. If recovery BC never decays, claim "no asymptotic bias" only for the shaping half.

### 3. State shaping

The potential $\Phi(s)$ scores how much a state resembles where humans took the robot after recovering. Build it as a kernel density estimate: a small Gaussian bump on every stored post-switch state, summed.

$$
\Phi(s) = c \cdot \frac{1}{Z} \sum_i w_i \exp\!\Big(-\frac{\lVert f(s) - f(s_i)\rVert^2}{2\sigma^2}\Big), \qquad F(s, s') = \gamma\,\Phi(s') - \Phi(s)
$$

- $s_i$ are stored states with $\tau \ge \hat\tau^*$. Takeover states are excluded, so the robot is never drawn toward where it was failing.
- $f(s)$ is a few normalized features. For Lift: gripper position, cube position, gripper-to-cube offset, gripper open or closed, cube height.
- $\sigma$ is the bump width, $Z$ normalizes the peak to 1, and $c$ sets the scale relative to the task reward of 10 (try 1 to 2). $\Phi = 0$ at terminal states.

Why this form:

- **Can't be farmed.** Summed along any trajectory, the shaping rewards telescope to $\gamma^T \Phi(s_T) - \Phi(s_0)$, so hovering or looping near human states earns nothing. Ng, Harada and Russell (1999) proved this leaves the optimal policy unchanged.
- **Propagates.** The shaping reward enters the critic's targets, so value flows backward and the robot is drawn toward human-guided regions from anywhere.
- **Multimodal.** A sum of bumps has a peak at every mode, for example left-side and right-side approaches, instead of averaging them the way Gaussian BC does.

Implementation notes: keep a capped reservoir of about 5,000 states and sum over the $k$ nearest only. Compute $F$ from the current $\Phi$ each time a batch is sampled; never store shaped rewards, since $\Phi$ changes as interventions arrive. A small classifier separating post-switch states from ordinary replay states is a scalable alternative to the density.

### 4. Full objective

$$
\text{critic target: } r^{\text{task}} + F(s,s') + \gamma\,\mathbb{E}_{a'\sim\pi_\theta}\,Q(s',a'), \qquad \text{actor: } \max_\theta\; \mathbb{E}_{s}\big[Q(s,\pi_\theta(s))\big] - \mathcal{L}_{\text{BC}}
$$

One critic, no bonus terms, no decoupling. Every intervention is kept; none are discarded or down-weighted by an estimated score.

## Counterfactual oracle

Counterfactual rollouts measure what each intervention was worth and where its recovery phase really ended. They are used for analysis and as an upper bound, not to train the main method.

### Procedure

When the human takes over, save the simulator state. Also save it at a few points along the human's segment. Later, restore each saved state $s_\tau$ in a separate simulator copy and roll out the current policy with no human. $G_\pi(s)$ is that rollout's return: $10\gamma^T$ if it succeeds $T$ steps later, otherwise 0.

$$
J(\tau) = \gamma^{\tau}\, G_\pi(s_\tau), \qquad \Delta = J(K) - J(0) = \gamma^{K} G_\pi(s_K) - G_\pi(s_0), \qquad \tau^* = \min\{\tau : J(\tau) \ge J(K)\}
$$

- $J(\tau)$ answers: if the human had handed back after $\tau$ steps, how well would the robot have done?
- $\Delta$ is how much the intervention improved the robot's prospects, charged for the $K$ steps the human spent. It credits interventions that only correct the robot back to a good state, even when the human never finishes the task.
- $\tau^*$ is the earliest point where handing back would have been as good as letting the human finish: the true end of recovery.

**Worked example (Lift, $\gamma = 0.99$).** The human realigns over the cube in 6 steps, then grasps and lifts, handing back at $K = 15$. From $s_0$ the policy fails, so $J(0) = 0$. From $s_6$ it succeeds in 25 steps: $J(6) = 0.99^6 \cdot 10 \cdot 0.99^{25} \approx 7.32$. From $s_{15}$ it succeeds in 16: $J(15) \approx 7.32$. So $\tau^* = 6$ and $\Delta \approx 7.32$: imitate the first 6 actions, shape toward the rest.

### What it does and does not estimate

- It estimates $V^\pi$, the **current learner's** value, not $V^*$. "The counterfactual beat the human" means the human did worse than the learner at that state, not that the human is suboptimal in an absolute sense.
- With a deterministic policy (mean action), MuJoCo makes each rollout exact, so one rollout suffices. With the stochastic training policy, use $N$ rollouts; at a 50% success rate, $N = 5$ gives a standard error of about 1.6 on a 0 to 10 scale.
- There are three outcomes, not two. If the policy fails from both $s_0$ and $s_K$, the comparison is **uninformative**, not evidence the human didn't help. This is the normal case early in training, so report it separately.

### Independent evaluators

Rolling out the learner is uninformative early in training, because it fails from almost every state. A second measure fixes this: after training, roll out an ensemble of strong policies trained without the intervention data (pure RLPD with task reward, several seeds) from every recorded intervention state, and average their returns.

$$
\Delta_{\text{eval}} = \gamma^{K} G_{\text{eval}}(s_K) - G_{\text{eval}}(s_0), \qquad \tau^*_{\text{eval}} = \min\{\tau : J_{\text{eval}}(\tau) \ge J_{\text{eval}}(K)\}
$$

This asks whether the human left the robot in a state from which a competent policy does better: a judgment about the state, close to measuring against $V^*$, rather than about what the learner could do at the time.

- **Why independent.** A final policy trained on these interventions is best from states it trained on, which can inflate $G(s_K)$. A single evaluator may also prefer a different solution mode than the human set up. Several seeds trained without intervention data avoid both.
- **Why rollouts.** Only states are queried, and actual outcomes are observed. This avoids scoring a critic on human actions it has never seen, the problem with Q-OIL's Section 4. It is the Monte Carlo fix for Q-OIL's Finding 2.
- **Limitation.** Evaluators may fail from unusual but fine human-reached states, which understates the human's contribution.
- **Not training usefulness.** This does not say how much an intervention helped train the final policy. That needs data ablations, such as dropping a random 20% of interventions and retraining. Optional.

| | Learner-relative | Evaluator-relative |
| --- | --- | --- |
| Evaluator | the learner at the moment of intervention | independent strong policies, run after training |
| Question | did the intervention help this learner, then? | did the human objectively improve the robot's situation? |
| Early in training | mostly uninformative | informative |
| Use | training dynamics; grading proxies | human-quality analysis; the premise check |

$\tau^*_{\text{eval}}$ depends on the state, not on the learner's competence at the time. With the learner-relative version, early interventions all show $\tau^* = K$ just because the learner cannot finish from anywhere, so $\tau^*_{\text{eval}}$ is the cleaner test of whether interventions have a recovery phase. It is cheap: post hoc, batched, parallel across CPU cores, deterministic, one rollout per evaluator per state.

### Why it is not part of the main method

Weighting or discarding interventions by $\Delta$ risks throwing away good data whenever the value estimate is wrong. That is most likely early in training, exactly when interventions matter most, and it can slow the policy, which keeps the estimate wrong. Its off-simulator proxy, the learned critic, is equally unreliable early.

The oracle instead has three jobs:

1. **Test the premise.** Do interventions actually have a recovery phase and a cruising phase? Use the evaluator-relative $\tau^*_{\text{eval}}$ and look at its distribution relative to $K$. If $\tau^*$ is almost always near $K$ (all recovery) or near 0 (all cruising), the split adds little.
2. **Set the ceiling.** Run PHASE with the oracle $\tau^*$ and compare it against proximity-based PHASE. The gap is the cost of not knowing the true switch point.
3. **Grade the proxies.** Measure how often the proximity $\hat\tau^*$ and a critic-based $\hat\Delta = \gamma^K \hat V(s_K) - \hat V(s_0)$ agree with the exact values.

An optional ablation adds $\Delta$-based weights to PHASE, testing the prediction that discarding interventions hurts early in training.

Cost control: branch at only 3 to 5 points per intervention and on a random subset of interventions, only in measurement runs. Count every branch rollout as environment steps, and report oracle variants as upper bounds, not competitors.

## Evaluation

Synthetic humans carry the method comparison; real humans check the premise. Results are reported on Robomimic Lift, then Can, using RLPD as the base learner.

### Human models

**Synthetic humans (main sweep).** Each has a gating rule (when to intervene), a corrector (how to act) and a handback rule.

- **Gating:** progress-based, with the intervention probability set from the model's precision, not its recall. Run both a stagnation-triggered and a decline-triggered version, since Q-OIL's fitted threshold suggests humans react to declines. PHASE itself only uses what humans do after taking over, so it does not depend on which is right.
- **Correctors:** RL checkpoints at about 70% and under 50% success, plus a scripted waypoint controller. Add multiple solution modes and SpaceMouse-style noise: axis-aligned, saturated, repeated actions and reaction lag. Do not script recovery behavior in, or the premise is true by construction.
- **Handback:** when progress resumes, with some hysteresis.
- **Validation:** match real data on intervention latency, segment lengths, action style and the within-segment progress profile.
- **Caveat:** with an RL checkpoint as the corrector, $\Delta$ partly measures the checkpoint's quality relative to the learner, and phase structure partly reflects the handback rule. Synthetic humans support comparisons between methods, not claims about human behavior. Report results across all variants as a robustness sweep.

**Real humans (premise check and a small head-to-head).** The author plus 2 to 3 others, using keyboard teleoperation in simulation. Background workers run counterfactual rollouts from saved states, so participants do nothing extra. Blind participants to the method and randomize order. Add Q-OIL's logs if they include simulator states. Check with the instructor about human-subjects approval.

### Baselines and ablations

| Group | Variants |
| --- | --- |
| Baselines | HIL; BC+RL with fixed $\lambda$; BC+RL with decaying $\lambda$; Q-OIL with and without BC |
| Mechanism ablations | action-only; state-only; reversed (states early, actions late) |
| Switch point | proximity (main); fixed schedule; oracle $\tau^*$ (upper bound) |
| Recovery BC weight | fixed; diluted; annealed |
| Verification | PHASE plus $\Delta$-based weights |

The reversed variant is the key falsification check: if it matches PHASE, the phase structure isn't what's helping.

### Metrics

1. Success rate, with 5 or more seeds and per-seed curves alongside the mean.
2. Intervention count.
3. Attraction vs. chaining: how often the policy enters human-guided regions from outside, vs. how far it follows a human route once on it.
4. Copying: similarity to human actions at held-out intervention states, plus the fraction of axis-aligned or saturated actions.
5. Recovery under perturbation: shove the object or open the gripper mid-episode at evaluation time.

Train past Q-OIL's budgets so asymptotic differences can appear.

### Predictions

Written down before running anything:

- PHASE copies less and attracts more than action-only methods, including Q-OIL.
- PHASE recovers from perturbations as well as action-only methods.
- The reversed variant does worse than PHASE.
- PHASE's advantage grows as corrector quality drops; with a near-perfect corrector, methods converge.
- Verification weights hurt early in training.

## Open questions and risks

| Item | Why it matters | How to handle it |
| --- | --- | --- |
| Interventions may not have two phases | The split adds little if $\tau^*$ is almost always near 0 or near $K$ | Measure $\tau^*$ on real interventions first; fall back to state-only shaping |
| Whether recovery BC should decay | Fixed BC may hold back recovery once RL surpasses the human | Three-way ablation: fixed, diluted, annealed |
| Proximity threshold $\epsilon$ and features $f(s)$ | The switch point may be sensitive to both | Sweep $\epsilon$; compare against the oracle $\tau^*$ |
| Quality of $\Phi$ | A poor potential slows learning, though it cannot change the optimum | Tune $\sigma$ and $c$; try the classifier alternative |
| Style leaking through recovery BC | Recovery imitation still copies some teleoperation style | Track the copying metric on recovery steps separately |
| Synthetic humans may not resemble real ones | Method rankings could depend on the human model | Robustness sweep across human variants; validate on real statistics |
| Novelty | Shaping from human data has prior work | Literature search before claiming novelty; cite Brys et al. (2015) |

To-dos:

- [ ] Literature search on shaping from demonstrations and interventions.
- [ ] Ask whether Q-OIL's released logs include full simulator states.
- [ ] Ask the instructor about human-subjects approval for the keyboard sessions.
- [ ] Pilot on Lift with 3 seeds to measure throughput before the full sweep.
