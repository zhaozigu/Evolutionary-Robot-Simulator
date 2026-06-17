# Evolutionary Robot Simulator

A browser-based playground where a robot learns to roam a grid world, collecting soda cans while
avoiding walls — using only **local one-cell sensing**. Three different solvers compete to train the
robot's neural-network controller, side by side on the same world:

- **GA** — a genetic algorithm that evolves a population of network weights.
- **PPO** — a Proximal Policy Optimization reinforcement learner built on [TensorFlow.js](https://www.tensorflow.org/js).
- **Random search** — a baseline that just keeps sampling random controllers, for comparison.

A 3D view (react-three-fiber) replays the best episode step by step, and a chart plots each solver's
fitness over generations so you can watch them race.

## Quick start

```bash
npm install
npm run dev      # start the Vite dev server, then open the printed URL
```

Press **Start**, then watch the fitness curves climb. Use the **GA / PPO** toggle above the 3D view to
replay either learner's best run.

## How it works

The robot sees only its current cell plus the four N/S/E/W neighbors (15 one-hot inputs) and picks one
of 7 actions each step (move in a direction, move randomly, stay, or pick up). Reward shaping
encourages collecting cans, exploring new cells, and approaching the nearest can (wall-aware), while
penalizing wall collisions and idling.

Every generation, all three solvers take one step against the **same fixed world**:

- The **GA** evaluates its population and breeds the next generation (elitism + tournament selection +
  uniform crossover + annealed Gaussian mutation, with a few fresh "immigrant" genomes for diversity).
- **PPO** rolls out a batch of stochastic episodes with its actor-critic network, estimates advantages
  with GAE(λ), and takes a few epochs of clipped-surrogate gradient steps.
- **Random search** samples a fresh batch of random controllers and tracks its best-ever.

The world is generated **once per run** (not per generation) so the fitness landscape stays stable and
the comparison is fair. Everything in `src/sim/` is deterministic and framework-agnostic, driven by a
single seeded PRNG.

## Project layout

| Path | What it does |
| --- | --- |
| `src/sim/env.ts` | `RobotEnv` — the grid-world dynamics and reward shaping shared by every solver. |
| `src/sim/neuralNet.ts` | Genome ↔ network encoding, observation encoding, forward pass. |
| `src/sim/episode.ts` | `runEpisode` — runs one genome's controller and records every step for replay. |
| `src/sim/ga.ts` | `evolveGeneration` — the genetic algorithm. |
| `src/sim/ppo.ts` | `PPOTrainer` — the TensorFlow.js PPO reinforcement learner. |
| `src/sim/randomSearch.ts` | The random-search baseline. |
| `src/hooks/useEvolutionEngine.ts` | Run lifecycle + rAF playback loop; drives all three solvers. |
| `src/components/` | 3D view, fitness chart, control sliders, and stat panels (presentation only). |

Reward incentives live at the top of `src/sim/env.ts`; PPO hyperparameters are in `DEFAULT_PPO_PARAMS`
in `src/sim/ppo.ts`.

## Commands

```bash
npm run dev       # start Vite dev server
npm run build     # type-check (tsc -b) then bundle for production
npm run preview   # preview the production build
npm run lint      # eslint .
npx tsc --noEmit  # type-check only, no build output
```

## Notes

- PPO uses the synchronous TensorFlow.js **CPU backend** so it can train inline with the engine loop;
  the controller network is tiny, so this is plenty fast. TensorFlow.js adds ~2 MB to the production
  bundle.
- No persistence: a run lives only in memory, so reloading the page starts over.

## Tech stack

React 19 · TypeScript · Vite · TensorFlow.js · @react-three/fiber + drei (Three.js) · Recharts ·
Tailwind CSS v4.
