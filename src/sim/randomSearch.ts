import type { Grid, Position } from './types';
import type { RNG } from './rng';
import { createRandomGenome } from './neuralNet';
import { runEpisode } from './episode';
import type { GenerationStats } from './ga';

/** Baseline: each "generation" samples a fresh random population (no selection/crossover/mutation). */
export class RandomSearchBaseline {
  private bestEver = -Infinity;

  step(
    generationIndex: number,
    populationSize: number,
    grid: Grid,
    start: Position,
    episodeLength: number,
    rng: RNG,
  ): GenerationStats {
    const fitnesses: number[] = [];
    for (let i = 0; i < populationSize; i++) {
      const genome = createRandomGenome(rng);
      const result = runEpisode(genome, grid, start, episodeLength, rng);
      fitnesses.push(result.fitness);
    }
    const batchBest = Math.max(...fitnesses);
    const worstFitness = Math.min(...fitnesses);
    const averageFitness = fitnesses.reduce((a, b) => a + b, 0) / fitnesses.length;
    this.bestEver = Math.max(this.bestEver, batchBest);

    return {
      generation: generationIndex,
      bestFitness: this.bestEver,
      averageFitness,
      worstFitness,
    };
  }

  reset(): void {
    this.bestEver = -Infinity;
  }
}
