# PHASE: Algorithmic Contribution

November 3, 2026 · William Gong

PHASE splits each human intervention in two: imitate the human's actions while they pull the robot out of a bad state, then stop copying actions and instead reward the robot for reaching the states the human took it to.

## Core idea

Humans tend to intervene when the robot's progress stalls or declines. What they do after taking over carries two kinds of information:

- **Recovery (early).** The robot is in a failure state its own policy reached. Its actions there are known to be bad and recovery is hard to find by exploration, so the human's *actions* are worth imitating (DAgger's argument).
- **Cruising (later).** The robot is back on track. The human's actions are now mostly style (axis-aligned, saturated teleop inputs, one of several valid modes). What is worth keeping is *where* they took the robot.

Cloning every intervention action copies style and suboptimality, averages across modes, and biases the final policy if its weight never decays. Decaying the BC weight with training time fades human actions but never separates recovery from cruising and never uses the states. Q-OIL's optimistic bonus only favors human actions once the robot is already in an intervention state; nothing attracts it toward those states.

PHASE uses each signal where it is trustworthy: BC on recovery steps, and potential-based shaping toward post-recovery states. Shaping propagates through the critic, giving attraction toward human-guided regions, and leaves the optimal policy unchanged (this guarantee covers the shaping half only).

**Novelty, scoped.** Neighbors: shaping from demonstrations (Brys et al., 2015); rewards or proxy values from takeovers (RLIF, which penalizes takeover states, the mirror image of PHASE's attraction toward handback-side states; Li et al., 2022; PVP); and weighted BC on interventions (IWR, Sirius). The likely-new part is splitting each intervention into an action phase and a state phase. Without the shaping half, PHASE becomes filtered intervention BC and sits close to IWR and Sirius.

## Algorithm

PHASE adds two terms to an off-policy actor-critic (RLPD). Intervention transitions go in the main replay buffer, so the critic sees their rewards and shaping terms. (Storing them only in a separate BC buffer would hide them from the critic; check this when building on an existing intervention-learning codebase.)

**Notation.** An intervention runs from takeover $s_0$ to handback $s_K$; $\tau = 0, \dots, K$ counts human steps; $a^h_\tau$ is the human action.

### 1. Switch point

Recovery ends when the human has brought the robot somewhere that looks like successful behavior:

$$
\hat\tau^* = \min\{\tau : d(s_\tau, \mathcal{D}_{\text{good}}) < \epsilon\}, \qquad \mathcal{D}_{\text{good}} = \{\text{earlier handback states}\} \cup \{\text{states from successful episodes}\}
$$

$d$ is nearest-neighbor distance over a few normalized features (gripper and object positions). Handback states are a free label: humans hand back when they judge the robot is on track. While $\mathcal{D}_{\text{good}}$ is empty, the whole segment counts as recovery. Labels are computed once, when the intervention arrives.

| Variant | Switch point | Role |
| --- | --- | --- |
| Proximity | first step within $\epsilon$ of $\mathcal{D}_{\text{good}}$ | main method |
| Fixed fraction | $\lfloor \alpha K \rfloor$, swept over $\alpha$ | test (a); $\alpha = 1$ all-BC, $\alpha = 0$ all-shaping |
| Oracle | learner-relative counterfactual $\tau^*$ | conditional diagnostic |

### 2. Recovery BC

$$
\mathcal{L}_{\text{BC}} = -\lambda\, \mathbb{E}_{(s,a)\sim\mathcal{D}_\pi}\big[\mathbb{1}[\text{recovery step}]\cdot \log \pi_\theta(a \mid s)\big]
$$

Sampled from the main buffer, its effective weight is $\lambda$ times the fraction of recovery steps, so it dilutes as successful data accumulates. (Averaging over a separate intervention buffer, as Q-OIL does, keeps it at $\lambda$ forever.)

**Open question: should recovery BC decay?** *Keep:* recovery states are failure states the improving policy rarely visits, and keeping BC preserves recovery skill. *Decay:* human recoveries are suboptimal and may leak into nearby normal states. Four-way ablation, scored on recovery under perturbation:

- fixed $\lambda$;
- diluted (above);
- annealed by intervention rate, $\lambda_t = \lambda_0\, \bar\rho_t / \bar\rho_0$;
- age-decayed per sample, $\lambda(i+1) = (1-\epsilon_\lambda)\,\lambda(i)$ per actor update.

If recovery BC never decays, claim "no asymptotic bias" only for the shaping half.

### 3. State shaping

$\Phi$ is a kernel density over post-switch states:

$$
\Phi(s) = \frac{c}{Z} \sum_i w_i \exp\!\Big(-\frac{\lVert f(s) - f(s_i)\rVert^2}{2\sigma^2}\Big), \qquad F(s, s') = \gamma\,\Phi(s') - \Phi(s)
$$

$s_i$ are stored states with $\tau \ge \hat\tau^*$ (takeover states excluded, so the robot is never drawn toward failure). $f$ is a few normalized task features, $Z$ normalizes the peak to 1, $c$ sets scale against the task reward of 10, and $\Phi = 0$ at terminals. Because $\Phi$ changes as interventions arrive, $F$ is computed at sampling time, never stored.

- **Can't be farmed.** Shaping telescopes to $\gamma^T\Phi(s_T) - \Phi(s_0)$ along any trajectory, leaving the optimal policy unchanged (Ng et al., 1999).
- **Propagates.** It enters critic targets, so value flows backward and attracts the robot from anywhere.
- **Multimodal.** A sum of bumps peaks at every mode instead of averaging them.

### 4. Full objective

$$
\text{critic target: } r^{\text{task}} + F(s,s') + \gamma\,\mathbb{E}_{a'\sim\pi_\theta}Q(s',a'), \qquad \text{actor: } \max_\theta\; \mathbb{E}_{s}\big[Q(s,\pi_\theta(s))\big] - \mathcal{L}_{\text{BC}}
$$

One critic, no bonus terms. Every intervention is kept.

## Evaluation

Robomimic Lift, then Can; RLPD as the base learner. Three tests, each answering something the others cannot.

| Test | Question | Evidence |
| --- | --- | --- |
| (a) Switch sweep | Does splitting help learning, and at roughly what ratio? | Learning outcomes |
| (b) Mechanism | Is the learner wrong where we think, and does proximity find the transition? | Intervention data |
| (c) Components | Which half produces the gain, and does order matter? | Learning outcomes |

### (a) Switch sweep

Train with a hard switch at $\lfloor \alpha K \rfloor$, $\alpha \in \{0, 0.25, 0.5, 0.75, 1\}$, plus proximity-PHASE. All-BC winning means intervention BC suffices; all-shaping winning means states suffice; an interior optimum $\alpha^*$ supports the split *only if* it also beats reversed in (c).

- An interior optimum can come from blending (BC as a regularizer, complementary signals) or mis-tuned endpoints. The endpoints optimize different losses, so tune $\lambda$ and $c$ there too.
- $\alpha^*$ is the best training mix, not where the transition is; location comes from (b).

### (b) Mechanism

**Prerequisite.** Histogram of $\hat\tau^*/K$. Mass at 1 means PHASE reduces to DAgger-style BC; mass at 0, to state-only shaping. Report the fraction labeled all-recovery only because $\mathcal{D}_{\text{good}}$ was empty.

**Learner disagreement.** Recovery actions should differ from the learner in *direction* (the learner is wrong); cruise actions only in *magnitude and style*:

$$
\cos_\tau = \frac{\langle a^h_\tau,\, \mu_\pi(s_\tau)\rangle}{\lVert a^h_\tau\rVert\,\lVert \mu_\pi(s_\tau)\rVert}, \qquad m_\tau = \frac{\lVert a^h_\tau\rVert}{\lVert \mu_\pi(s_\tau)\rVert}
$$

Prediction: low $\cos_\tau$ before the switch, high after, with the remaining gap in $m_\tau$ and saturation statistics. Three conditions for the measurement to be valid:

- **Align on the switch.** Plot $\bar c(k) = \mathbb{E}[\cos_{\hat\tau^*+k}]$ and look for a step at $k = 0$; a sharper step than when aligned on $\lfloor\alpha^*K\rfloor$ means proximity beats a fixed fraction at locating transitions.
- **Compute at arrival**, with the policy before the intervention enters training. Otherwise any variant that clones cruise actions agrees with them by construction.
- **Bin by training stage.** Early, the learner is wrong everywhere; late, interventions are mostly genuine failures. The separation should appear mid-training.

(b) uses every intervention step and is the only premise check that works on real humans.

**Reading (a) and (b) together:**

| | (b) holds | (b) fails |
| --- | --- | --- |
| **(a) holds** (interior optimum, beats reversed) | Works for the stated reason | Works for another reason (blending, regularization) |
| **(a) fails** (an endpoint wins) | Structure exists but isn't exploited well; check switch detection, $\lambda$, $c$ | No split to exploit; fall back to action-only or state-only |

### (c) Components

Hold the split fixed (at $\alpha^*$ or proximity) and remove one half at a time. The sweep endpoints are not this test: $\alpha = 1$ copies cruise actions too.

| Variant | Recovery | Cruise | Question |
| --- | --- | --- | --- |
| Full PHASE | BC | shaping | — |
| Recovery-BC only | BC | — | Is copying corrective actions enough? |
| Cruise-shaping only | — | shaping | Are the states enough? |
| Reversed | shaping | BC | Does order matter, or just having both signals? |

Recovery-BC only $\approx$ PHASE means the states add nothing (filtered intervention BC). Reversed $\approx$ PHASE means phase assignment doesn't matter; this is the key falsification check.

### Switch-point diagnostic (conditional)

Run only to separate "the premise is false" from "proximity finds the wrong switch point." Save simulator states at branch points $\tau \in \{0, K/4, K/2, 3K/4, K\}$, roll out the current learner's mean action from each, and with $G_\pi(s) = 10\gamma^T$ on success in $T$ steps (else 0):

$$
J(\tau) = \gamma^\tau G_\pi(s_\tau), \qquad \tau^* = \min\{\tau \in \text{branches} : J(\tau) \ge J(K) - \delta\}
$$

If the learner fails from every branch, fall back to $\hat\tau^*$, so oracle-PHASE differs from proximity-PHASE only where it has information. Lift only, 3 seeds, synthetic humans only. Branch rollouts count as environment steps; report oracle-PHASE as an upper bound.

- **PHASE clearly wins:** optional; one appendix line on the gap to oracle.
- **PHASE ties baselines:** run it. Oracle wins → switch detection is the bottleneck. Oracle also ties → splitting doesn't help here; fall back to state-only shaping or move to Can.
- **Degenerate histogram in (b):** skip.

### Human models

**Synthetic (main sweep).** Gating: stagnation- and decline-triggered progress rules, intervention probability set from precision. Correctors: RL checkpoints (~70% and <50% success) and a scripted waypoint controller, with multiple modes and SpaceMouse-style noise; never script recovery behavior in, or the premise holds by construction. Handback when progress resumes. Validate against real latency, segment lengths and action style. Synthetic phase structure partly reflects the handback rule, so synthetic results support method comparisons, not claims about humans.

**Real.** The author plus 2 to 3 others, keyboard teleop in simulation, blinded and randomized. Real interventions feed (b) and a small head-to-head.

### Baselines and ablations

| Group | Variants |
| --- | --- |
| Baselines | HIL; BC+RL on all intervention actions with fixed $\lambda$; BC+RL with age-decayed $\lambda$; Q-OIL with and without BC |
| Switch point (a) | fixed fraction $\alpha \in \{0, 0.25, 0.5, 0.75, 1\}$; proximity; oracle (conditional) |
| Components (c) | recovery-BC only; cruise-shaping only; reversed |
| Recovery BC weight | fixed; diluted; annealed; age-decayed |

### Metrics

1. Success rate (5+ seeds, per-seed curves).
2. Intervention count.
3. Attraction vs. chaining: entering human-guided regions from outside vs. following a human route once on it.
4. Copying: similarity to human actions at held-out intervention states, and axis-aligned/saturated fractions, split by phase.
5. Recovery under perturbation (object shoved or gripper opened mid-episode).
6. Mechanism: $\hat\tau^*/K$ histogram; aligned $\bar c(k)$ and $m_\tau$ by training stage.

Train past Q-OIL's budgets so asymptotic differences can appear.

### Predictions

- The sweep has an interior optimum that beats reversed.
- $\cos_\tau$ steps up at $\hat\tau^*$ mid-training, more sharply than at $\lfloor\alpha^*K\rfloor$.
- Full PHASE beats both single-half variants.
- PHASE copies less and attracts more than action-only methods (BC+RL baselines, Q-OIL), while recovering from perturbations as well.
- PHASE's advantage grows as corrector quality drops.

### Claims

**Supported by (a)–(c):** PHASE improves success and efficiency; the split matters; proximity beats a fixed fraction; interventions show measurable phase structure. **Not supported without the oracle:** that proximity finds the true end of recovery. Call it a heuristic switch point that works.

## Out of scope

Evaluator-relative counterfactuals ($\Delta_{\text{eval}}$, $\tau^*_{\text{eval}}$) measure human quality against $V^*$, a different question; strong evaluators recover from nearly every takeover state in Lift and Can, so every intervention looks like cruising. Also deferred: weighting interventions by $\Delta$, and grading critic-based $\hat\Delta$.

## Risks

| Risk | Handling |
| --- | --- |
| No two-phase structure | Check the (b) histogram first; fall back to state-only shaping |
| Interior optimum from blending | Reversed variant; (b) curves; tune endpoints |
| Recovery BC decay | Four-way ablation |
| Sensitivity to $\epsilon$, $f(s)$ | Sweep $\epsilon$; aligned (b) curves; conditional oracle |
| Poor $\Phi$ | Slows learning but can't change the optimum; tune $\sigma$, $c$ |
| Synthetic humans unrealistic | Robustness sweep across variants; validate on real statistics |
| Novelty | Literature search; cite Brys, RLIF, PVP, IWR, Sirius |

To-dos:

- [ ] Literature search: shaping from demonstrations and takeovers; intervention-weighted BC.
- [ ] Ask whether Q-OIL's logs include policy checkpoints, for computing (b) on its interventions.
- [ ] Ask the instructor about human-subjects approval.
- [ ] Pilot Lift with 3 seeds to measure throughput.
