export type CellType = 'empty' | 'wall' | 'can';

export type Grid = CellType[][];

export type Direction = 'N' | 'S' | 'E' | 'W';

export const DIRECTIONS: Direction[] = ['N', 'S', 'E', 'W'];

export const ACTIONS = [
  'MOVE_N',
  'MOVE_S',
  'MOVE_E',
  'MOVE_W',
  'MOVE_RANDOM',
  'STAY',
  'PICK_UP',
] as const;

export type Action = (typeof ACTIONS)[number];

export const ACTION_LABELS: Record<Action, string> = {
  MOVE_N: 'Move North',
  MOVE_S: 'Move South',
  MOVE_E: 'Move East',
  MOVE_W: 'Move West',
  MOVE_RANDOM: 'Move Randomly',
  STAY: 'Stay Still',
  PICK_UP: 'Pick Up Can',
};

export interface Position {
  x: number;
  y: number;
}

/** A single recorded frame of an episode, used to replay an individual's run in the 3D view. */
export interface StepRecord {
  position: Position;
  action: Action;
  reward: number;
  cumulativeReward: number;
  collided: boolean;
  pickedUp: boolean;
}

export interface EpisodeResult {
  fitness: number;
  cansCollected: number;
  wallCollisions: number;
  /** Static wall/can layout at episode start; cans are removed over time per step.pickedUp events. */
  initialGrid: Grid;
  startPosition: Position;
  steps: StepRecord[];
}

export type Genome = Float64Array;

export interface NetShape {
  inputSize: number;
  hiddenSize: number;
  outputSize: number;
}
