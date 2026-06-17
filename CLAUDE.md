# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A browser-based evolutionary robotics demo: a robot drives around a grid world collecting "cans" while
avoiding walls, using only local (one-cell) sensing, driven by a small feedforward neural-network
controller. Two solvers can train that controller — a **genetic algorithm (GA)** and a **PPO
reinforcement learner** (TensorFlow.js) — and the user picks **one of them per run** via a mode selector.
GA and PPO run **independently, one at a time** (they are *not* trained simultaneously); a random-search
baseline always runs alongside whichever solver is selected, for comparison. Each run generates its own
fresh world. The 3D view (react-three-fiber) replays the active solver's best recorded episode step by
step; recharts plots fitness over generations for the active solver plus the baseline.

## Commands

```bash
npm run dev       # start Vite dev server
npm run build     # vite build — bundles for production (does NOT type-check)
npm run preview   # preview the production build
npm run lint      # eslint .
npx tsc --noEmit  # type-check only (run this separately; build skips it)
```

There is no test suite configured in this repo.

## Architecture

### `src/sim/` — pure simulation logic (framework-agnostic, deterministic)

All randomness flows through the seeded PRNG in `rng.ts` (`mulberry32`); nothing in `sim/` reads
`Math.random()` directly, which keeps a given run reproducible from its seed.

- `types.ts` — shared types: `Grid`/`CellType`, `Action`/`ACTIONS`, `Genome` (a flat `Float64Array`),
  `StepRecord`/`EpisodeResult`.
- `world.ts` — random grid generation (`generateWorld`) and grid helpers (`inBounds`, `getCell`, treats
  out-of-bounds as a wall).
- `neuralNet.ts` — genome <-> network: a genome is a flat `Float64Array` of weights/biases sized by
  `NET_SHAPE` (`genomeSize()`). `encodeObservation` one-hot encodes the robot's **current cell plus its 4
  N/S/E/W neighbors** (15 inputs total — the current cell is included so the network can tell whether it's
  standing on a can and should `PICK_UP`). `forward()` is a single tanh-hidden-layer + softmax-output pass;
  action selection is `argmax` over the softmax output (deterministic given an observation — the only
  built-in source of stochastic behavior is the `MOVE_RANDOM` action itself).
- `env.ts` — `RobotEnv`: the mutable grid-world the robot acts in, encapsulating the dynamics and reward
  shaping **shared by every solver** (GA, random search, PPO) so their fitness numbers are directly
  comparable. `observe()` returns the local one-hot encoding; `step(action, rng)` applies one action,
  advancing position/grid/visited/distance-field state and returning its shaped reward + collision/pickup
  flags (and the resolved action, since `MOVE_RANDOM` is concretized here). Reward shaping constants
  (`REWARD_PICKUP`, `PENALTY_WALL_COLLISION`, `PENALTY_PER_STEP`, `REWARD_NEW_CELL`, `CAN_APPROACH_WEIGHT`)
  live at the top of this file — this is the place to tune agent incentives.
- `episode.ts` — `runEpisode` runs one genome's (argmax/deterministic) controller against a `RobotEnv` for a
  fixed number of steps and records every step for replay. Thin loop; all dynamics/rewards are in `env.ts`.
- `ppo.ts` — `PPOTrainer`: a Proximal Policy Optimization solver built on **TensorFlow.js**, offered as an
  alternative to the GA. An actor-critic net (shared tanh hidden layer → 7-way policy head + scalar
  value head) is trained per iteration by rolling out a batch of *stochastic* episodes against `RobotEnv`,
  estimating advantages with GAE(λ), then taking a few epochs of clipped-surrogate gradient steps. Rollout
  forward passes run in plain JS over weights pulled out of the model (cheap; avoids per-step tensor churn);
  only the gradient update uses tf ops. `initPPOBackend()` forces the **synchronous CPU backend** and must be
  awaited once before constructing a trainer, which is what lets the otherwise-synchronous engine loop call
  `trainer.step()` inline. `DEFAULT_PPO_PARAMS` holds the hyperparameters.
- `ga.ts` — `evolveGeneration` evaluates a population on a fixed grid/start position and produces the next
  generation via elitism + tournament selection + single-point crossover + Gaussian mutation. Elite count and
  tournament size are derived as a percentage of `populationSize` (not fixed constants), and mutation strength
  anneals down over the run (`generationIndex`/`maxGenerations`) so early generations explore broadly and later
  ones fine-tune.
- `randomSearch.ts` — `RandomSearchBaseline` baseline: each "generation" just samples a fresh random
  population (no selection/crossover/mutation) and tracks the best-ever fitness, for comparison against the
  active solver (GA or PPO). It runs alongside whichever solver is selected.
- `replay.ts` — derives stats (score, cans collected, wall collisions, remaining actions) at an arbitrary step
  index within a recorded `EpisodeResult`, used to scrub through a replay.

### `src/hooks/useEvolutionEngine.ts` — the central stateful orchestrator

Owns the run lifecycle (`idle` -> `running` -> `paused`/`done`) and a `requestAnimationFrame` loop that paces
playback at `config.simSpeed` steps/sec. Key behaviors to know before changing this file:

- A `mode: SolverMode` (`'ga' | 'ppo'`) selects which solver this run trains. **Only the selected solver
  runs**, and it is only switchable while `idle` (`setMode` no-ops otherwise) — to switch you `reset()` back
  to idle. The previous design ran GA + PPO + baseline together every generation; that is gone.
- The world grid and start position are generated **fresh on every `start()`** (and `reset()`), giving each
  GA/PPO run its own independent world. The world is **not** regenerated per-generation within a run, so the
  fitness landscape stays stable across generations and GA selection pressure is meaningful.
- `computeNextGeneration()` advances the **active solver only** plus the random-search baseline: in `'ga'`
  mode it calls `evolveGeneration` and seeds the GA population; in `'ppo'` mode it calls `PPOTrainer.step()`.
  The inactive solver's history array stays empty (so its chart line / stats simply don't appear). The active
  solver's best episode is recorded and the rAF loop plays it back; when playback reaches the end it calls
  `computeNextGeneration()` again.
- `start()` is **async**: in `'ppo'` mode it awaits `initPPOBackend()` (CPU backend) and constructs the
  `PPOTrainer` before flipping to `running`, so the synchronous rAF loop can call `trainer.step()` inline; in
  `'ga'` mode it skips TF.js entirely (no backend init, no trainer). The trainer is disposed on `reset()`,
  on the start of a new run, and on unmount to free its tf tensors.
- State is mirrored into refs (`configRef`, `populationRef`, `gridRef`, `ppoTrainerRef`, etc.) alongside React
  state so the rAF loop and callbacks always read the latest values without needing to resubscribe effects.
- No persistence — genomes/history live only in memory for the current run; nothing is written to
  `localStorage` or elsewhere, so a page reload loses progress.

### `src/components/` — presentation only

Tailwind v4 (CSS-first config via `@import "tailwindcss"` in `src/index.css`; there is no
`tailwind.config.js`). `World3D.tsx` renders the grid/robot/cans/walls with `@react-three/fiber` +
`@react-three/drei`. Several components are **mode-aware** (take the `mode` prop and show only the active
solver): `FitnessChart.tsx` plots the active solver's best/average plus the random-search baseline;
`ControlPanel.tsx` shows GA-only sliders (population size, mutation rate) only in `'ga'` mode and relabels
"Number of Generations" → "Training Iterations" for PPO; `EvolutionPanel.tsx`/`StatsPanel.tsx` are read-only
stat displays. All are driven entirely by props from `useEvolutionEngine`. `App.tsx` composes the layout and
hosts the **GA/PPO mode selector** above the 3D view — disabled while a run is active (switch requires a
reset).
