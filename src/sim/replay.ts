import type { EpisodeResult } from './types';

export interface ReplayStats {
  score: number;
  cansCollected: number;
  wallCollisions: number;
  remainingActions: number;
}

export function getStatsAtIndex(episode: EpisodeResult, index: number): ReplayStats {
  let cansCollected = 0;
  let wallCollisions = 0;
  for (let i = 0; i <= index && i < episode.steps.length; i++) {
    if (episode.steps[i].pickedUp) cansCollected += 1;
    if (episode.steps[i].collided) wallCollisions += 1;
  }
  const score = episode.steps[Math.min(index, episode.steps.length - 1)]?.cumulativeReward ?? 0;
  const remainingActions = Math.max(0, episode.steps.length - (index + 1));
  return { score, cansCollected, wallCollisions, remainingActions };
}

export function getCollectedCanKeys(episode: EpisodeResult, index: number): Set<string> {
  const collected = new Set<string>();
  for (let i = 0; i <= index && i < episode.steps.length; i++) {
    const step = episode.steps[i];
    if (step.pickedUp) collected.add(`${step.position.x},${step.position.y}`);
  }
  return collected;
}
