import type { Action, Grid, Position } from './types';
import { ACTIONS } from './types';
import { type RNG, randInt } from './rng';
import { cloneGrid, getCell, inBounds } from './world';
import { encodeObservation } from './neuralNet';

// Reward shaping constants — the single place to tune agent incentives. Shared by every solver
// (GA, random search, PPO) so their fitness numbers are directly comparable.
export const REWARD_PICKUP = 10;
export const PENALTY_EMPTY_PICKUP = -1;
// Softened from -5: at full strength a couple of wall hits outweighed a whole can, so early random
// genomes' fitness was mostly collision noise, drowning out the (much weaker) can-seeking and
// exploration gradient that selection needs to get traction. -3 still clearly teaches avoidance.
export const PENALTY_WALL_COLLISION = -3;
export const PENALTY_PER_STEP = -0.05;
// Lowered from 0.2: with the directed can-seeking signal below now doing the heavy lifting, a large
// blanket exploration bonus made aimless wandering competitive with collecting nearby cans. Kept
// small so exploration still breaks ties when no can is within reach.
export const REWARD_NEW_CELL = 0.1;
// Potential-based shaping: reward per grid cell of progress toward the nearest remaining can, using
// wall-aware BFS distance so the gradient points *around* walls rather than into them. Because the
// reward is the change in potential (Φ(next) - Φ(prev)), pacing back and forth nets exactly zero
// (it can't be farmed) and the optimal policy is provably unchanged — yet it converts the sparse
// "+10 only at the instant of PICK_UP" landscape into a dense slope a learner can start climbing many
// generations sooner. When a can sits adjacent (which the agent can actually sense), this directly
// rewards stepping onto it; further away it still biases exploration the right way.
export const CAN_APPROACH_WEIGHT = 0.3;

const MOVE_DELTAS: Record<Action, [number, number] | null> = {
  MOVE_N: [0, -1],
  MOVE_S: [0, 1],
  MOVE_E: [1, 0],
  MOVE_W: [-1, 0],
  MOVE_RANDOM: null,
  STAY: [0, 0],
  PICK_UP: [0, 0],
};

const NEIGHBOR_DELTAS: ReadonlyArray<[number, number]> = [
  [0, -1],
  [0, 1],
  [1, 0],
  [-1, 0],
];

/**
 * Multi-source BFS: distance (in steps, treating walls as impassable) from every cell to the
 * nearest can. Recomputed only when a can is picked up, so per-step shaping is an O(1) lookup.
 */
function computeCanDistanceField(grid: Grid): { field: number[][]; hasCan: boolean } {
  const height = grid.length;
  const width = grid[0].length;
  const field = Array.from({ length: height }, () => new Array<number>(width).fill(Infinity));
  const queue: Position[] = [];

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (grid[y][x] === 'can') {
        field[y][x] = 0;
        queue.push({ x, y });
      }
    }
  }
  const hasCan = queue.length > 0;

  for (let head = 0; head < queue.length; head++) {
    const { x, y } = queue[head];
    const nextDist = field[y][x] + 1;
    for (const [dx, dy] of NEIGHBOR_DELTAS) {
      const nx = x + dx;
      const ny = y + dy;
      if (!inBounds(grid, nx, ny) || grid[ny][nx] === 'wall') continue;
      if (field[ny][nx] > nextDist) {
        field[ny][nx] = nextDist;
        queue.push({ x: nx, y: ny });
      }
    }
  }

  return { field, hasCan };
}

/**
 * Shaping potential of a position: higher (closer to 0) the nearer it is to a can. Cells with no
 * can left, or walled off from every can, contribute a flat 0 so they inject no spurious gradient.
 */
function potential(field: number[][], hasCan: boolean, pos: Position): number {
  if (!hasCan) return 0;
  const d = field[pos.y][pos.x];
  if (!Number.isFinite(d)) return 0;
  return -CAN_APPROACH_WEIGHT * d;
}

export interface StepOutcome {
  /** The action actually executed — equals the requested action except that MOVE_RANDOM resolves
   *  to one of the four concrete moves, so replay shows what the robot really did. */
  resolvedAction: Action;
  reward: number;
  collided: boolean;
  pickedUp: boolean;
}

/**
 * Mutable grid-world the robot acts in, encapsulating the dynamics and reward shaping shared by
 * every solver. `observe()` returns the local one-hot encoding the controllers see; `step()` applies
 * one action, advancing position/grid state and returning its shaped reward. Construct one per
 * episode (or call `reset()`); it is deterministic given the same action sequence (and `rng` only
 * affects MOVE_RANDOM).
 */
export class RobotEnv {
  readonly initialGrid: Grid;
  readonly startPosition: Position;
  pos: Position;
  private grid: Grid;
  private visitedCells: Set<string>;
  private field: number[][];
  private hasCan: boolean;

  constructor(initialGrid: Grid, startPosition: Position) {
    this.initialGrid = initialGrid;
    this.startPosition = startPosition;
    this.grid = cloneGrid(initialGrid);
    this.pos = { ...startPosition };
    // The start cell isn't "explored", it's given.
    this.visitedCells = new Set<string>([`${this.pos.x},${this.pos.y}`]);
    ({ field: this.field, hasCan: this.hasCan } = computeCanDistanceField(this.grid));
  }

  /** Restores the environment to its starting state so the same instance can run another episode. */
  reset(): void {
    this.grid = cloneGrid(this.initialGrid);
    this.pos = { ...this.startPosition };
    this.visitedCells = new Set<string>([`${this.pos.x},${this.pos.y}`]);
    ({ field: this.field, hasCan: this.hasCan } = computeCanDistanceField(this.grid));
  }

  observe(): number[] {
    return encodeObservation(this.grid, this.pos);
  }

  step(action: Action, rng: RNG): StepOutcome {
    // MOVE_RANDOM is the only built-in source of stochastic behavior: resolve it to a concrete move
    // before scoring so the reward and the recorded action describe the same physical step.
    const resolvedAction: Action = action === 'MOVE_RANDOM' ? ACTIONS[randInt(rng, 0, 4)] : action;

    // Potential at the cell we're standing on, measured against the field for the current grid
    // state (the field is only refreshed *after* this step's shaping is scored, so a PICK_UP — which
    // doesn't move the robot — is never charged for the goal it just removed).
    const prevPotential = potential(this.field, this.hasCan, this.pos);

    let reward = 0;
    let collided = false;
    let pickedUp = false;

    const delta = MOVE_DELTAS[resolvedAction];
    if (resolvedAction === 'PICK_UP') {
      if (this.grid[this.pos.y][this.pos.x] === 'can') {
        this.grid[this.pos.y][this.pos.x] = 'empty';
        reward = REWARD_PICKUP;
        pickedUp = true;
      } else {
        reward = PENALTY_EMPTY_PICKUP;
      }
    } else if (delta && (delta[0] !== 0 || delta[1] !== 0)) {
      const nx = this.pos.x + delta[0];
      const ny = this.pos.y + delta[1];
      if (!inBounds(this.grid, nx, ny) || getCell(this.grid, nx, ny) === 'wall') {
        reward = PENALTY_WALL_COLLISION;
        collided = true;
      } else {
        this.pos.x = nx;
        this.pos.y = ny;
      }
    }
    // STAY and zero-delta moves: no action-specific reward, but still pay the per-step cost below.

    // Small constant cost per action: makes idling for a full episode a guaranteed negative score.
    reward += PENALTY_PER_STEP;

    // First-visit exploration bonus: rewards reaching a new cell, not re-visiting one.
    const posKey = `${this.pos.x},${this.pos.y}`;
    if (!this.visitedCells.has(posKey)) {
      this.visitedCells.add(posKey);
      reward += REWARD_NEW_CELL;
    }

    // Potential-based shaping: credit (or debit) the change in proximity to the nearest can. Scored
    // against the pre-pickup field so it telescopes cleanly and never penalizes the pickup itself.
    reward += potential(this.field, this.hasCan, this.pos) - prevPotential;

    // A collected can changes every cell's nearest-can distance, so rebuild the field for next step.
    if (pickedUp) ({ field: this.field, hasCan: this.hasCan } = computeCanDistanceField(this.grid));

    return { resolvedAction, reward, collided, pickedUp };
  }
}
