# PHASE: Algorithmic Contribution

November 5, 2026 · William Gong

PHASE splits each human intervention in two: imitate the human's actions until the robot is back somewhere that has led to success before, then stop copying actions and instead shape the robot toward the states the human took it to.

## Core idea

Humans tend to intervene when the robot's progress stalls or declines. What they do after taking over carries two kinds of information:

- **Recovery (early).** The robot is in a failure state its own policy reached. Its actions there are known to be bad and recovery is hard to find by exploration, so the human's *actions* are worth imitating (DAgger's argument).
- **Cruising (later).** The robot is back on a route that has led to success before. The human's actions now add mostly style (axis-aligned, saturated teleop inputs, one of several valid modes). What is worth keeping is *where* they took the robot.

Cloning every intervention action copies style and suboptimality, averages across modes, and keeps pulling the policy toward the human if its weight never decays. Decaying the BC weight with training time fades human actions but never separates recovery from cruising and never uses the states.

PHASE uses each signal where it is expected to be informative: BC on recovery steps, and potential-based shaping toward states visited after recovery. Shaping enters the critic's targets and steers exploration toward human-guided regions while the critic is still learning, and for a fixed potential it preserves the optimal-policy set of the objective.

**Intended lifecycle.** Early in training, many intervention steps should count as recovery, so PHASE behaves much like DAgger. As the robot succeeds near human routes, interventions there shift from BC to shaping. Recovery BC expires with age, so where humans stop needing to intervene, BC stops; where they keep intervening in the same place, PHASE keeps cloning. This lifecycle is a hypothesis, not a consequence of the rule: demo states are in the reference set from the start, so early interventions near the demonstrated route can be labeled cruise immediately. The pilot measures it.

**The claim being tested.** Allocating BC and state shaping to different portions of the same intervention improves learning beyond a well-tuned combination of both signals, with continuation tests explaining why.

**Related work and novelty.**

- Shaping from demonstrations: Brys et al. (2015) use a max of Gaussian similarities to demonstrated state-action pairs as a potential. PHASE's max-kernel over states is an established building block, not a contribution.
- Rewards or values from takeovers: RLIF penalizes takeover states, the mirror image of PHASE's attraction toward post-recovery states; also Li et al. (2022) and PVP.
- Weighted or filtered BC on interventions and demos: IWR, Sirius, and the Q-filter (Nair et al., 2018), which applies BC only where the critic prefers the demonstrated action.
- Adaptive imitation strength: SiLRI (Zhao et al., 2026) constrains the policy to stay near a learned human policy with state-wise Lagrange multipliers, so it imitates where human actions are consistent and relies on RL elsewhere. It is the closest prior work on *where* to imitate; PHASE decides by intervention phase and adds state shaping.
- Stages within interventions: ROVE (Xiao et al., 2026) splits humanoid interventions into rollout, adaptation and recovery, and treats the first moments after takeover as hesitant rather than useful recovery. That cuts against cloning the earliest human actions; with keyboard teleop in simulation the onset lag is likely smaller, but the pilot checks for it.
- Guidance retired as the learner becomes competent: Jump-Start RL (Uchendu et al., 2023); reverse curricula from demonstrations (Salimans & Chen, 2018).
- Q-OIL: its optimistic value bonus favors human actions at intervention states, and its authors describe this as encouraging revisitation of intervention states. PHASE differs in using an explicit state-only potential over a *selected portion* of each intervention, and in assigning actions and states to different portions. (Confirm the exact Q-OIL formulation with its authors.)

The likely-new part is splitting each intervention into an action portion and a state portion. Without the shaping half, PHASE becomes filtered intervention BC and sits close to IWR, Sirius and the Q-filter.

## Algorithm

PHASE adds two terms to an off-policy actor-critic (RLPD). All intervention transitions go in the main replay buffer immediately, so the critic sees their rewards and shaping terms. BC is selected by a mask over the main buffer, not by a separate buffer.

### Conventions

**Environment.** A wrapper gives sparse reward 10 on success and sets $\text{done} = 1$ on success (robosuite environments do not terminate on success by default). The horizon is treated as an artificial cutoff of a continuing task, so timeouts bootstrap ($\text{done} = 0$). If finishing within the horizon becomes part of the objective, remaining time must go into the state instead.

**Offline data.** RLPD's offline half (50% of each batch) is a small set of human demonstrations, identical for every method, baselines included. Demo transitions are relabeled with the same reward and termination conventions as online data; the dataset's own `done` flags are not trusted. Demos are used only as RLPD uses them: they are not behavior-cloned and are not shaping centers. Their states do join the reference set $\mathcal{D}_{\text{good}}$.

**RLPD configuration.** One configuration, recorded in a table and shared by every method: critic ensemble of 10 with LayerNorm, targets from the min over a random 2-member subset, entropy term in the backup, 50/50 symmetric sampling, and fixed update-to-data ratio, batch size, learning rates, $\gamma$ and target-update rate (set in the pilot).

**Storage and indices.** The main buffer is FIFO, so an evicted intervention loses its BC, one more form of expiry. The reference set $\mathcal{D}_{\text{good}}$, the shaping support, and the takeover history are stored separately and never evicted. $\mathcal{D}_{\text{good}}$ and the shaping support contain different points, so each gets its own nearest-neighbor index.

**Feature metric.** $f$ is a few task features. For Lift: end-effector-to-cube vector (not absolute positions, since cube placement is randomized), gripper width, and cube height. Each feature is normalized by its standard deviation over demo states, computed **once before training** and frozen, with a floor for near-constant features. $\sigma$ and $c$ are also fixed. A metric that changed during training would move old shaping centers and break the properties in Section 3.

**Notation.** An intervention has states $s_0, \dots, s_K$ (takeover $s_0$, handback $s_K$) and human actions $a^h_0, \dots, a^h_{K-1}$, with $a^h_\tau$ taken at $s_\tau$. $m_\tau \in \{0,1\}$ marks actions that receive BC. $\alpha$ is SAC's entropy temperature.

### 1. Labeling an intervention

Labels are computed once, when the intervention arrives.

**Reference set.**

$$
\mathcal{D}_{\text{good}} = \{\text{confirmed handback states}\} \cup \{\text{robot success states}\} \cup \{\text{demo states}\}
$$

A handback is a candidate until the episode finishes; it is *confirmed*, and joins $\mathcal{D}_{\text{good}}$, only if the robot then reached success with no further intervention. Takeover and recovery states are never in $\mathcal{D}_{\text{good}}$.

**Repeat check.** Keep a recent-takeover history $\mathcal{H}$: the takeover states of interventions that arrived within the last $N$ interventions and the last $E$ environment steps. Membership depends only on age, not on whether an intervention received BC; a first-time stall labeled entirely cruise must still enter $\mathcal{H}$, or a repeated stall in the same place would never be caught. If

$$
\min_{h \in \mathcal{H}} \big\lVert f(s_0) - f(h) \big\rVert < \epsilon_t,
$$

the robot is failing where it recently failed, so set $\hat\tau^* = K+1$ (below).

**Otherwise, switch by proximity.**

$$
\hat\tau^* = \min\{\tau \in \{0, \dots, K\} : d_f(s_\tau, \mathcal{D}_{\text{good}}) < \epsilon\}, \qquad \hat\tau^* = K+1 \text{ if no state qualifies}
$$

where $d_f$ is nearest-neighbor distance under the frozen metric, and $\epsilon$ is set relative to the typical nearest-neighbor spacing within the demos.

**Masks.** BC and shaping support are defined separately:

- BC on actions $a^h_\tau$ with $\tau < \min(\hat\tau^*, K)$.
- Shaping support: states $s_\tau$ with $\max(\hat\tau^*, 1) \le \tau \le K$. The takeover state is never a center.

| Case | BC | Shaping support |
| --- | --- | --- |
| $\hat\tau^* = 0$: takeover already near $\mathcal{D}_{\text{good}}$ | none | $s_1, \dots, s_K$ |
| $1 \le \hat\tau^* \le K$: split | $a^h_0, \dots, a^h_{\hat\tau^*-1}$ | $s_{\hat\tau^*}, \dots, s_K$ |
| $\hat\tau^* = K+1$: repeat, or no state qualified | all actions | none |

$\hat\tau^* = K$ (only the handback qualifies) still shapes toward $s_K$, so it is reported separately from $\hat\tau^* = K+1$, which is genuinely BC-only.

**Expiry.** A BC-labeled action leaves the active set after $N$ newer interventions or $E$ environment steps, whichever comes first. Expiry by count alone would freeze the last few interventions in BC forever once interventions stop.

**Known limitations of the proximity rule.**

- *Proximity is not competence.* The intended boundary is "the current learner can finish from here." The rule measures "this state is near one that appeared on a successful trajectory." Demos pass through states the learner cannot handle, and successful episodes can contain stalls. The rule is a heuristic, validated against continuation by the frozen learner in pilot P2 before any large sweep.
- *First-time stalls.* A stall at a position successful trajectories pass through is labeled cruise at takeover; the repeat check catches the second occurrence.
- *Feature aliasing.* States that agree on $f$ can differ in orientation, velocity or contact. If success and failure continuations overlap in $f$, no threshold fixes it; P2 checks separability.
- *Non-monotonic recovery.* A weak corrector may recover, err, and recover again. The first-hit rule labels everything after the first match as cruise. P2 inspects this.

**Switch variants** (ablations; the main method is proximity with the repeat check):

- *Proximity only:* no repeat check.
- *Progress-gated proximity:* label each state of a successful episode with steps-to-go $h^e_t = T_e - t$, and let $\hat h(s)$ be the median over its $k$ nearest successful states, using demos and robot successes (demos alone until the robot has succeeded). Recovery ends at the first state near $\mathcal{D}_{\text{good}}$ that, for an on-route takeover ($d_f(s_0, \mathcal{D}_{\text{good}}) < \epsilon$), also satisfies $\hat h(s_\tau) \le \hat h(s_0) - \Delta$. This catches first-time stalls at the cost of $\hat h$, $\Delta$ and $k$.
- *Agreement:* end recovery after $k$ consecutive non-idle steps on which the frozen pre-intervention learner matches every key the human pressed ($A_\tau = 1$; see (b)). It handles stalls but labels a competent robot that prefers a different route as recovery.
- *Fixed fraction:* $\hat\tau^* = \lfloor \beta K \rfloor$, for the sweep in (a).

### 2. Recovery BC

$$
\mathcal{L}_{\text{BC}} = -\lambda\, \mathbb{E}_{(s,a^h)\sim\mathcal{D}_\pi}\Big[m(s, a^h) \cdot \log \pi_\theta\big(\operatorname{clip}(a^h, -1 + \epsilon_a, 1 - \epsilon_a) \mid s\big)\Big]
$$

$m(s, a^h) = 1$ only for active, unexpired BC-labeled actions.

**Numerics.** Teleop actions are often saturated at $\pm 1$, where the inverse tanh is undefined, so the log-likelihood of a saturated action diverges to $-\infty$ and its gradient can dominate. Clip with a small $\epsilon_a$ (e.g. $10^{-3}$), evaluate the log-probability in stable form (pre-squash Gaussian plus the tanh correction), and bound the policy's log-std.

**Dilution.** Sampled from the main buffer, the effective BC weight is $\lambda$ times the fraction of active BC-labeled samples, so it dilutes as data accumulates and shrinks further as labels expire. (Averaging over a separate intervention buffer keeps it at $\lambda$ forever and resamples the same few segments.)

**BC decay.** Expiry is the default. It removes the explicit BC term in finite time, and once interventions stop, all BC ends within $E$ environment steps. It does not undo BC's effect on the data collected or the networks trained along the way, so final performance remains an empirical question, not a guarantee. *Keep more:* recovery states are failure states the improving policy rarely visits, and BC preserves recovery skill. *Decay more:* human recoveries are suboptimal and may leak into nearby normal states. Ablation, scored on recovery under perturbation:

- expiry window (default), sweeping $N$ and $E$;
- never expire, fixed $\lambda$;
- never expire, diluted;
- annealed by intervention rate, $\lambda_t = \lambda_0\, \bar\rho_t / \bar\rho_0$;
- age-decayed per sample, $\lambda(i+1) = (1-\epsilon_\lambda)\,\lambda(i)$ per actor update.

### 3. State shaping

$\Phi$ is a max-kernel over the shaping support, a smooth function of the distance to the nearest support state:

$$
\Phi(s) = c \cdot \max_i \exp\!\Big(-\frac{\lVert f(s) - f(s_i)\rVert^2}{2\sigma^2}\Big) = c \cdot \exp\!\Big(-\frac{d_{\text{NN}}(s)^2}{2\sigma^2}\Big), \qquad F(s, s') = \gamma\,(1 - \text{done})\,\Phi(s') - \Phi(s)
$$

$c$ sets scale against the task reward of 10. $\Phi(s')$ is zeroed only at true terminals; zeroing it at timeouts would give every timed-out episode near a human region a spurious penalty of about $-c$ on its last step. $F$ is computed at sampling time from the current $\Phi$, never stored.

**Why a max, not a sum.** A sum of kernels measures sample density: it peaks where humans lingered, moved slowly, or used one mode more often, and normalizing its peak to 1 divides every region by the densest one, so a new intervention in a new region gets the weakest pull. The max gives every covered region the same height $c$ and needs no normalizer.

**Properties, stated precisely.**

- **Policy invariance, for a fixed $\Phi$.** For any fixed potential, discounted shaping telescopes to $-\Phi(s_0)$ along a trajectory ending at a true terminal, and the shaped Q-function of every policy is $Q^\pi(s,a) - \Phi(s)$, an action-independent shift; this holds for the entropy-regularized objective too (Ng et al., 1999). So each frozen version of $\Phi$ preserves the optimal-policy set. When $\Phi$ changes between versions, extra terms remain; recomputing $F$ at sampling time means each update uses the current version's invariant objective, but the sequence of objectives is not a single shaped MDP (dynamic shaping needs a separate construction; Devlin & Kudenko, 2012). Preserving the optimal-policy set also does not guarantee that a neural actor-critic learns that policy.
- **Scaffolding.** Before the critic is accurate, the per-step bonus $\gamma\Phi(s')$ differs by action while the critic has not yet learned the matching payback, so actions toward higher $\Phi$ look better. Wiewiora (2003) proves shaping equivalent to optimistic Q-initialization for matched tabular updates; for SAC with ensembles and target networks this is motivation, not an identity. The pull fades as backups grounded in real outcomes (true terminals, or many discounted hops) overwrite it. Under a finite budget, it can still decide whether the robot finds the reward at all.
- **$\Phi$ only grows**, given the frozen metric, $\sigma$ and $c$. New support states raise $\Phi$ locally and never lower it. Old centers are kept: lowering $\Phi$ at a state the critic has already learned makes it look worse than it is until the critic relearns, a temporary repulsion. Keeping old centers is a design choice with costs (computation, and an exploration bias that persists wherever values are still inaccurate). Recency is handled on the BC side, by expiry.
- **Takeover neighborhoods can still have high $\Phi$.** Excluding takeover and recovery states as centers does not exclude their neighborhoods: a support state within $\sigma$ of the takeover gives $\Phi(s_0) \ge c\,e^{-1/2} \approx 0.61c$, and within $0.1\sigma$, about $0.995c$. $\Phi$ at takeover states is measured.
- **Dips between centers.** Between two centers at distance $d$, the midpoint potential is $c\exp(-d^2/8\sigma^2)$, about $0.88c$ at $\sigma = d$. Choosing $\sigma$ at least the typical spacing between consecutive human states reduces, but does not remove, the resulting oscillation in $F$.
- **Attraction, not direction.** $\Phi \approx c$ along a covered route, so moving along it in either direction costs a small $(1-\gamma)c$ per step. Shaping pulls the robot onto human-guided regions without favoring a direction of travel; forward progress comes from the task reward. This is a design choice: a progress-weighted potential is still a state potential, not BC, but it would bias exploration toward the human's direction of travel, which PHASE deliberately leaves to RL.
- **Multimodal.** Every covered mode gets the same peak height, instead of being averaged or weighted by how often it was used.

### 4. Full objective

Critic target, for each ensemble member:

$$
y = r^{\text{task}} + F(s,s') + \gamma\,(1 - \text{done})\Big(\min_{j \in \mathcal{M}} Q_{\bar\phi_j}(s', \tilde a') - \alpha \log \pi_\theta(\tilde a' \mid s')\Big), \qquad \tilde a' \sim \pi_\theta(\cdot \mid s')
$$

$\mathcal{M}$ is a random two-member subset of the critic ensemble.

Actor:

$$
\min_\theta\; \mathbb{E}_{s \sim \mathcal{D},\, \xi \sim \mathcal{N}(0, I)}\Big[\alpha \log \pi_\theta(\tilde a \mid s) - \bar Q(s, \tilde a)\Big] + \mathcal{L}_{\text{BC}}, \qquad \tilde a = \tanh\big(\mu_\theta(s) + \sigma_\theta(s) \odot \xi\big)
$$

$\bar Q$ is the ensemble mean, and $\alpha$ is auto-tuned as in SAC. No extra critics or bonus terms. Every intervention is kept.

**Temperature.** BC changes the policy's entropy at labeled states (usually lowering it, though fitting inconsistent human actions can raise it), and auto-tuning compensates elsewhere, so variants with different amounts of BC can reach different temperatures. Log $\alpha$ and realized policy entropy for every run. This reveals temperature differences; it does not rule them out as an explanation. If variants differ noticeably, run a small fixed-$\alpha$ comparison.

## Evaluation

Robomimic Lift, then Can; RLPD as the base learner.

**Protocol.**

- Success curves come from autonomous evaluation episodes with intervention disabled.
- Report human-controlled steps alongside intervention count; one long takeover can be more assistance than several short ones.
- Count training interaction, diagnostic rollouts, and evaluation separately.
- Tune on pilot seeds; report on separate evaluation seeds (5+). Training runs are the independent units. Report interquartile means with bootstrap confidence intervals (Agarwal et al., 2021).
- Pre-specify a small number of primary comparisons: PHASE vs. BC-all + shape-all, and PHASE vs. recovery-BC only.
- "Variant ≈ PHASE" is claimed only with a stated practical-equivalence margin and a confidence interval inside it; an inconclusive difference is not equivalence.
- Training past Q-OIL's budgets measures later finite-budget behavior, not asymptotic convergence.

### Pilot (required, before the learning grid)

**P1: Does Lift separate PHASE from BC?** If failure, takeover and success states cluster together, the repeat check fires on most interventions and PHASE reduces to intervention BC. Measure the fraction of interventions with $\hat\tau^* = K+1$, split into repeat-check and no-match cases. If the repeat check dominates, adopt progress-gated proximity as the main switch or make Can the main testbed.

**P2: Continuation test.** Collect a shared bank of interventions across training stages (synthetic and real), saving full simulator and controller state at branch points $\tau \in \mathcal{B} = \{0, K/4, K/2, 3K/4, K\}$. From each branch, run the *frozen pre-intervention learner* with the original remaining horizon, $M$ stochastic rollouts plus one mean-action rollout, and estimate its success rate $\hat p(\tau)$. Run this on a pre-specified sample regardless of how other results turn out.

- **Do the labels track competence?** Compare $\hat p$ at states labeled cruise vs. recovery. Proximity is useful if cruise-labeled states let the learner finish markedly more often.
- **Where does the learner become able to finish?** If $\hat p(K) < \rho_{\text{succ}}$, the intervention is unresolved (failed or premature handback) and analyzed separately. Otherwise
$$
\tau^* = \min\{\tau \in \mathcal{B} : \hat p(\tau) \ge \rho_{\text{succ}}\}
$$
Compare $\tau^*$ with $\hat\tau^*$. This is a coarse, rollout-informed estimate of when the learner can take over; it measures immediate continuation, not downstream learning, and is not an upper bound on PHASE.
- **Do human actions help early and not late?** From $s_\tau$, execute the next $h$ human actions, then hand to the learner; compare with the learner from $s_\tau$. Early prefixes should raise $\hat p$; late prefixes should not.
- **Is $f$ informative?** Train a nearest-neighbor classifier on $f$ to predict continuation success. Near-chance accuracy means the feature space cannot support the switch.
- **Is recovery monotone?** Look for $\hat p$ rising and falling within an intervention, especially for weak correctors.
- **Is the onset hesitant?** In real interventions, check whether the first steps after takeover make progress (cf. ROVE).

**P3: Throughput and tuning.** Measure wall-clock per run; set $\epsilon$, $\epsilon_t$, $N$, $E$, $\sigma$, $c$ on pilot seeds.

### Main comparison

| Method | BC | Shaping |
| --- | --- | --- |
| RLPD + demos | — | — |
| HIL (interventions in replay only) | — | — |
| BC-all | all intervention actions | — |
| BC-all + shape-all | all intervention actions | all intervention states except takeover |
| Recovery-BC only | recovery actions | — |
| **PHASE** | recovery actions | post-recovery states |
| Q-OIL, with and without BC | as published | — |

All methods share the demos, the RLPD configuration, the expiry rule where BC is used, and the tuning budget. BC-all + shape-all is the key baseline: it asks whether splitting the signals by phase helps beyond using both.

Every method keeps intervention transitions, human actions included, in critic replay. The comparisons therefore concern how *additional* BC and shaping supervision is allocated, not whether human states alone suffice.

### Conditional ablations (run depending on the pilot)

**(a) Switch sweep.** Hard switch at $\lfloor \beta K \rfloor$, $\beta \in \{0, 0.25, 0.5, 0.75, 1\}$, plus proximity-PHASE. At $\beta = 1$ shaping is off entirely ($\Phi \equiv 0$), so that endpoint equals BC-all. At $\beta = 0$ there is no explicit BC, but human actions remain in replay, so a win there shows explicit BC is unnecessary given action-labeled replay plus shaping, not that states alone suffice. An interior optimum $\beta^*$ is the best training mix, not where the transition is; it supports the split only if it also beats the ordering controls in (c). Tune $\lambda$ and $c$ at the endpoints too.

**(b) Mechanism.** *Pressed-axis directional agreement*: quantize actions per axis with deadzone $\delta$,

$$
q(a)_j = \operatorname{sign}(a_j)\,\mathbb{1}\big[|a_j| > \delta\big] \in \{-1, 0, +1\},
$$

and over the axes the human pressed, $P_\tau = \{j : q(a^h_\tau)_j \neq 0\}$ plus the gripper command, compute

$$
A_\tau = \frac{1}{|P_\tau|} \sum_{j \in P_\tau} \mathbb{1}\big[q(\bar a^\pi_\tau)_j = q(a^h_\tau)_j\big], \qquad \bar a^\pi_\tau = \tanh\!\big(\mu_\pi(s_\tau)\big),
$$

with the frozen pre-intervention learner. $A_\tau$ measures direction on pressed axes only; it does not show that the robot's action is correct or that differences are harmless style. A robot action matching the pressed axis but adding a large sideways motion scores 1. So also report off-axis motion, $O_\tau = \max_{j \notin P_\tau} |\bar a^\pi_{\tau,j}|$ over translation axes, and stratify by idle status and number of pressed axes, since an agreement increase could partly reflect humans pressing fewer axes after the switch. Quantize the synthetic correctors' continuous actions with the same $q$, and report curves at two values of $\delta$.

Plot $\bar A(k) = \mathbb{E}[A_{\hat\tau^*+k}]$ on held-out interventions and compare the step at $k = 0$ with steps at shifted boundaries. A step at the switch is *consistent with* the mechanism; it is not independent proof, since agreement and proximity both depend on state familiarity and time since takeover. Also report the saturated fraction of human vs. robot actions before and after the switch. Bin by training stage; the separation should appear mid-training.

**(c) Components and ordering.** The original reversed variant changes the amount of BC as well as its order, so it is replaced by matched-count controls. With $m = \hat\tau^*$ taken from PHASE's labels:

| Variant | BC | Shaping support | Question |
| --- | --- | --- | --- |
| Full PHASE | first $m$ actions | $s_m, \dots, s_K$ | — |
| Recovery-BC only | first $m$ actions | — | Is copying corrective actions enough? |
| Cruise-shaping only | — | $s_m, \dots, s_K$ | Are the post-recovery states enough, given replay? |
| Reversed, matched | last $m$ actions | $s_1, \dots, s_{K-m}$ | Does order matter at equal amounts? |
| Random allocation | $m$ random actions | the other states | Does phase assignment matter at all? |

Report the coverage of each shaping support (distinct support states after merging those within $\sigma$), not just the count. A support control holds recovery BC fixed and varies only the shaping centers: post-recovery states (PHASE), all intervention states, or demo states (a Brys-style potential).

**BC decay.** The five-way ablation in Section 2.

**Attraction.** Measure entry into a common, held-out set of reference regions (from a held-out intervention bank), identical for all methods, by training stage. Entering each method's own support set is not comparable across methods.

### Human models

**Synthetic (main experiments).** Gating: stagnation- and decline-triggered progress rules, intervention probability set from precision. Correctors: RL checkpoints, a scripted waypoint controller with multiple modes, and SpaceMouse-style noise; never script recovery behavior in, or the premise holds by construction.

- Calibrate correctors on the shared takeover-state bank from P2, reporting their *conditional* recovery success and intervention duration there; success from ordinary initial states does not transfer.
- Vary style (motor noise, quantization) separately from decision quality (checkpoint, route choice), so an effect can be attributed to one.
- Sweep handback delay and hysteresis within a range informed by real interventions, since the handback rule controls how much cruise segment exists.
- Synthetic phase structure partly reflects the handback rule, so synthetic results support method comparisons, not claims about humans.

**Real.** The author plus 2 to 3 others, keyboard teleop in simulation, blinded and randomized. Real interventions feed P2, (b) and a small head-to-head. They support a pilot and observations about these operators, not general claims about how humans intervene.

### Metrics

1. Autonomous success rate (interquartile mean and bootstrap CIs over runs).
2. Human-controlled steps and intervention count.
3. Environment steps: training, diagnostic, evaluation.
4. Attraction into common reference regions, by training stage.
5. Copying: similarity to human actions at held-out intervention states, and saturated fraction, split by phase.
6. Recovery under perturbation (object shoved or gripper opened mid-episode).
7. Mechanism: $\hat\tau^*/K$ histogram by trigger type and training stage, with $\hat\tau^* = K$ and $K+1$ reported separately; repeat-check fraction; active BC volume over training; $\bar A(k)$ and off-axis motion; $\Phi$ at takeover states.
8. $\alpha$ and realized policy entropy.

### Hypotheses

- In P2, cruise-labeled states have markedly higher learner continuation success than recovery-labeled states, and early human prefixes help more than late ones.
- The repeat check fires on a minority of Lift interventions.
- PHASE beats BC-all + shape-all and recovery-BC only.
- PHASE beats the matched reversed and random-allocation controls.
- PHASE copies less than action-only methods and enters common reference regions earlier in training, while recovering from perturbations as well.
- Active BC volume falls over training as interventions become rarer and expire.
- How PHASE's advantage changes with corrector quality depends on how quality degrades: worse style may favor PHASE; worse decisions also corrupt the states it shapes toward.

### Claims

**If supported:** allocating BC to the early portion of each intervention and state shaping to the later portion improves learning beyond a well-tuned combination of both signals; P2 shows the proximity labels track learner competence; shaping speeds learning and steers exploration while preserving the optimal-policy set of each fixed shaped objective.

**Not claimed:** that proximity finds the true end of recovery; that human states alone suffice; guaranteed unbiased convergence of the learned policy; general claims about human intervention behavior.

## Out of scope

Evaluator-relative counterfactuals ($\Delta_{\text{eval}}$, $\tau^*_{\text{eval}}$) measure human quality against $V^*$, a different question; strong evaluators recover from nearly every takeover state in Lift and Can, so every intervention looks like cruising. Also deferred: weighting interventions by $\Delta$, grading critic-based $\hat\Delta$, and action-free intervention learning.

## Risks

| Risk | Handling |
| --- | --- |
| Proximity labels don't track competence | P2 before the grid; progress-gated or agreement switch; learner-relative gate if needed |
| Features can't separate success from failure | P2 separability check; revise $f$ before tuning thresholds |
| No two-phase structure | P2 prefix test and histogram; fall back to the better single-signal method |
| Lift's states overlap, so PHASE becomes intervention BC | P1; progress-gated proximity, or Can as main testbed |
| First-time stalls labeled cruise | Accepted; repeat check catches the second; progress gate as ablation |
| Gain comes from amount of BC, not its placement | Matched reversed and random-allocation controls; BC-all + shape-all |
| Expiry too fast or too slow | Sweep $N$ and $E$; compare against no-expiry variants |
| Sensitivity to $\epsilon$, $\epsilon_t$, $\delta$ | Tune on pilot seeds; report at two values |
| Shaping attracts toward takeover neighborhoods | Measure $\Phi$ at takeover states; adjust $\sigma$ |
| Temperature differs across variants | Log $\alpha$ and entropy; fixed-$\alpha$ check if needed |
| Synthetic humans unrealistic | Calibrate on takeover bank; vary style and quality separately; validate on real statistics |
| Too many comparisons for 5 seeds | Pre-specified primary comparisons; separate tuning and evaluation seeds |
| Novelty | Cite Brys, RLIF, PVP, IWR, Sirius, Nair et al., SiLRI, ROVE, JSRL |

To-dos:

- [ ] Confirm Q-OIL's formulation (value bonus, buffer sampling) with its authors, and whether its logs include policy checkpoints.
- [ ] Ask the instructor about human-subjects approval.
- [ ] Build the environment wrapper (reward 10, terminate on success) and relabel demo transitions to match.
- [ ] Pin the shared RLPD configuration.
- [ ] Run pilots P1 to P3 on Lift; set thresholds on pilot seeds.
