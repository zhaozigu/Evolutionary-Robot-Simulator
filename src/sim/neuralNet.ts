import type { CellType, Genome, NetShape } from './types';
import { DIRECTIONS } from './types';
import { type RNG, randRange } from './rng';
import { getCell } from './world';
import type { Grid, Position } from './types';

/** One-hot encodes the current cell plus each of the 4 sensed neighbor cells (empty/wall/can) -> 15 inputs. */
export const NET_SHAPE: NetShape = {
  inputSize: (DIRECTIONS.length + 1) * 3,
  hiddenSize: 12,
  outputSize: 7,
};

export function genomeSize(shape: NetShape = NET_SHAPE): number {
  const { inputSize, hiddenSize, outputSize } = shape;
  return inputSize * hiddenSize + hiddenSize + hiddenSize * outputSize + outputSize;
}

export function createRandomGenome(rng: RNG, shape: NetShape = NET_SHAPE): Genome {
  const size = genomeSize(shape);
  const genome = new Float64Array(size);
  for (let i = 0; i < size; i++) genome[i] = randRange(rng, -1, 1);
  return genome;
}

const CELL_ENCODING: Record<CellType, [number, number, number]> = {
  empty: [1, 0, 0],
  wall: [0, 1, 0],
  can: [0, 0, 1],
};

export function encodeObservation(grid: Grid, pos: Position): number[] {
  const offsets: Record<(typeof DIRECTIONS)[number], [number, number]> = {
    N: [0, -1],
    S: [0, 1],
    E: [1, 0],
    W: [-1, 0],
  };
  // Without this, the network only ever sees its neighbors and has no way to know
  // whether the cell it's currently standing on (and could PICK_UP from) has a can.
  const input: number[] = [...CELL_ENCODING[getCell(grid, pos.x, pos.y)]];
  for (const dir of DIRECTIONS) {
    const [dx, dy] = offsets[dir];
    const cell = getCell(grid, pos.x + dx, pos.y + dy);
    input.push(...CELL_ENCODING[cell]);
  }
  return input;
}

function tanh(x: number): number {
  return Math.tanh(x);
}

function softmax(values: number[]): number[] {
  const max = Math.max(...values);
  const exps = values.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

/** Forward pass: input -> tanh hidden layer -> softmax output (action probabilities). */
export function forward(genome: Genome, input: number[], shape: NetShape = NET_SHAPE): number[] {
  const { inputSize, hiddenSize, outputSize } = shape;
  let offset = 0;

  const w1 = genome.subarray(offset, offset + inputSize * hiddenSize);
  offset += inputSize * hiddenSize;
  const b1 = genome.subarray(offset, offset + hiddenSize);
  offset += hiddenSize;
  const w2 = genome.subarray(offset, offset + hiddenSize * outputSize);
  offset += hiddenSize * outputSize;
  const b2 = genome.subarray(offset, offset + outputSize);

  const hidden = new Array<number>(hiddenSize);
  for (let h = 0; h < hiddenSize; h++) {
    let sum = b1[h];
    for (let i = 0; i < inputSize; i++) {
      sum += w1[h * inputSize + i] * input[i];
    }
    hidden[h] = tanh(sum);
  }

  const output = new Array<number>(outputSize);
  for (let o = 0; o < outputSize; o++) {
    let sum = b2[o];
    for (let h = 0; h < hiddenSize; h++) {
      sum += w2[o * hiddenSize + h] * hidden[h];
    }
    output[o] = sum;
  }

  return softmax(output);
}

export function argmax(values: number[]): number {
  let best = 0;
  for (let i = 1; i < values.length; i++) {
    if (values[i] > values[best]) best = i;
  }
  return best;
}
