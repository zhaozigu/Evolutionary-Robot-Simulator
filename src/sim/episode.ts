import type { Action, EpisodeResult, Genome, Grid, Position, StepRecord } from './types';
import { ACTIONS } from './types';
import type { RNG } from './rng';
import { argmax, forward } from './neuralNet';
import { RobotEnv } from './env';

/**
 * Runs one episode of the given genome's controller against a world, recording every step for replay.
 * The genome picks actions deterministically (argmax over the policy softmax); all dynamics and reward
 * shaping live in {@link RobotEnv}, shared with the other solvers.
 */
export function runEpisode(
  genome: Genome,
  initialGrid: Grid,
  startPosition: Position,
  maxSteps: number,
  rng: RNG,
): EpisodeResult {
  const env = new RobotEnv(initialGrid, startPosition);

  let fitness = 0;
  let cansCollected = 0;
  let wallCollisions = 0;
  const steps: StepRecord[] = [];

  for (let t = 0; t < maxSteps; t++) {
    const input = env.observe();
    const probs = forward(genome, input);
    const action: Action = ACTIONS[argmax(probs)];

    const { resolvedAction, reward, collided, pickedUp } = env.step(action, rng);
    if (pickedUp) cansCollected += 1;
    if (collided) wallCollisions += 1;

    fitness += reward;
    steps.push({
      position: { ...env.pos },
      action: resolvedAction,
      reward,
      cumulativeReward: fitness,
      collided,
      pickedUp,
    });
  }

  return { fitness, cansCollected, wallCollisions, initialGrid, startPosition, steps };
}
