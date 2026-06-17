import type { EpisodeResult, Genome, Grid, Position } from './types';
import { type RNG, randInt } from './rng';
import { createRandomGenome, genomeSize } from './neuralNet';
import { runEpisode } from './episode';

export interface GAParams {
  populationSize: number;
  mutationRate: number;
  mutationStrength: number;
  episodeLength: number;
}

export const DEFAULT_GA_PARAMS: GAParams = {
  populationSize: 50,
  mutationRate: 0.05,
  mutationStrength: 0.5,
  episodeLength: 200,
};

export interface GenerationStats {
  generation: number;
  bestFitness: number;
  averageFitness: number;
  worstFitness: number;
}

export interface GenerationOutcome {
  stats: GenerationStats;
  population: Genome[];
  bestGenome: Genome;
  bestEpisode: EpisodeResult;
}

export function initPopulation(size: number, rng: RNG): Genome[] {
  return Array.from({ length: size }, () => createRandomGenome(rng));
}

function evaluatePopulation(
  population: Genome[],
  grid: Grid,
  start: Position,
  episodeLength: number,
  rng: RNG,
): EpisodeResult[] {
  return population.map((genome) => runEpisode(genome, grid, start, episodeLength, rng));
}

function tournamentSelect(
  population: Genome[],
  fitnesses: number[],
  tournamentSize: number,
  rng: RNG,
): Genome {
  let bestIdx = randInt(rng, 0, population.length);
  for (let i = 1; i < tournamentSize; i++) {
    const idx = randInt(rng, 0, population.length);
    if (fitnesses[idx] > fitnesses[bestIdx]) bestIdx = idx;
  }
  return population[bestIdx];
}

// Uniform crossover: each weight independently inherited from either parent. A single cut
// point on the flattened weight vector has a positional bias with no relation to network
// structure (weights that happen to sit close together in the flat array always co-inherit,
// far-apart ones almost always end up split) — uniform crossover mixes parents evenly instead.
function crossover(a: Genome, b: Genome, rng: RNG): Genome {
  const child = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) {
    child[i] = rng() < 0.5 ? a[i] : b[i];
  }
  return child;
}

// Keeps weights from random-walking to extreme magnitudes over long runs (mutation is
// unbounded additive noise with no pull back toward zero). Unclipped weights eventually push
// the tanh hidden layer into saturation, flattening gradients-by-selection and stalling
// progress; clipping keeps the search space well-conditioned for the whole run.
const WEIGHT_CLIP = 5;

function mutate(genome: Genome, rate: number, strength: number, rng: RNG): Genome {
  const mutated = new Float64Array(genome);
  for (let i = 0; i < mutated.length; i++) {
    if (rng() < rate) {
      // Box-Muller gaussian perturbation
      const u1 = Math.max(rng(), 1e-9);
      const u2 = rng();
      const gaussian = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      mutated[i] = Math.max(-WEIGHT_CLIP, Math.min(WEIGHT_CLIP, mutated[i] + gaussian * strength));
    }
  }
  return mutated;
}

/** Evaluates the current population on the given world and produces the next generation via elitism + tournament selection + crossover + mutation. */
export function evolveGeneration(
  population: Genome[],
  generationIndex: number,
  maxGenerations: number,
  grid: Grid,
  start: Position,
  params: GAParams,
  rng: RNG,
): GenerationOutcome {
  const results = evaluatePopulation(population, grid, start, params.episodeLength, rng);
  const fitnesses = results.map((r) => r.fitness);

  const order = population.map((_, i) => i).sort((a, b) => fitnesses[b] - fitnesses[a]);
  const bestIdx = order[0];
  const bestFitness = fitnesses[bestIdx];
  const worstFitness = Math.min(...fitnesses);
  const averageFitness = fitnesses.reduce((a, b) => a + b, 0) / fitnesses.length;

  // Elitism and tournament size scale with population so selection pressure stays
  // sensible whether the population slider is set to 10 or 200, instead of the
  // fixed elite-of-2/tournament-of-3 that used to ignore population size entirely.
  const eliteCount = Math.max(1, Math.round(params.populationSize * 0.06));
  const tournamentSize = Math.max(2, Math.min(params.populationSize, Math.round(params.populationSize * 0.08)));

  // A few fresh random genomes each generation keep diversity alive: once the gene pool converges,
  // crossover+mutation can only recombine what's already there, so an injection of brand-new
  // genomes is what lets the run still discover strategies outside the current basin (escaping
  // local optima). Kept tiny so it costs almost nothing early when the pool is already diverse.
  const immigrantCount = Math.min(
    params.populationSize - eliteCount,
    Math.max(0, Math.round(params.populationSize * 0.04)),
  );

  // Anneal both mutation strength and rate from full early (broad exploration) down toward the end
  // of the run (fine-tuning): strength to 40% and rate to 50%, so late generations make smaller,
  // sparser tweaks and settle into a good optimum instead of continuing to jump around it.
  const progress = maxGenerations > 0 ? Math.min(1, generationIndex / maxGenerations) : 0;
  const mutationStrength = params.mutationStrength * (1 - 0.6 * progress);
  const mutationRate = params.mutationRate * (1 - 0.5 * progress);

  const nextPopulation: Genome[] = [];
  for (let i = 0; i < eliteCount && i < order.length; i++) {
    nextPopulation.push(new Float64Array(population[order[i]]));
  }
  while (nextPopulation.length < params.populationSize - immigrantCount) {
    const parentA = tournamentSelect(population, fitnesses, tournamentSize, rng);
    const parentB = tournamentSelect(population, fitnesses, tournamentSize, rng);
    let child = crossover(parentA, parentB, rng);
    child = mutate(child, mutationRate, mutationStrength, rng);
    nextPopulation.push(child);
  }
  while (nextPopulation.length < params.populationSize) {
    nextPopulation.push(createRandomGenome(rng));
  }

  return {
    stats: { generation: generationIndex, bestFitness, averageFitness, worstFitness },
    population: nextPopulation,
    bestGenome: population[bestIdx],
    bestEpisode: results[bestIdx],
  };
}

export function expectedGenomeSize(): number {
  return genomeSize();
}
