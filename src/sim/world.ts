import type { CellType, Grid, Position } from './types';
import { type RNG, randInt } from './rng';

export interface WorldConfig {
  size: number;
  wallDensity: number;
  canDensity: number;
}

export const DEFAULT_WORLD_CONFIG: WorldConfig = {
  size: 10,
  wallDensity: 0.1,
  canDensity: 0.15,
};

/** Generates a random grid plus a guaranteed-empty start position for the robot. */
export function generateWorld(config: WorldConfig, rng: RNG): { grid: Grid; start: Position } {
  const { size, wallDensity, canDensity } = config;
  const grid: Grid = Array.from({ length: size }, () =>
    Array.from({ length: size }, (): CellType => {
      const roll = rng();
      if (roll < wallDensity) return 'wall';
      if (roll < wallDensity + canDensity) return 'can';
      return 'empty';
    }),
  );

  const start: Position = { x: randInt(rng, 0, size), y: randInt(rng, 0, size) };
  grid[start.y][start.x] = 'empty';

  return { grid, start };
}

export function cloneGrid(grid: Grid): Grid {
  return grid.map((row) => [...row]);
}

export function inBounds(grid: Grid, x: number, y: number): boolean {
  return y >= 0 && y < grid.length && x >= 0 && x < grid[0].length;
}

/** Cell state outside the grid boundary is treated as a wall. */
export function getCell(grid: Grid, x: number, y: number): CellType {
  if (!inBounds(grid, x, y)) return 'wall';
  return grid[y][x];
}
