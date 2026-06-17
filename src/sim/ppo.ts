import * as tf from '@tensorflow/tfjs';
import type { Action, EpisodeResult, Grid, Position, StepRecord } from './types';
import { ACTIONS } from './types';
import { NET_SHAPE } from './neuralNet';
import { RobotEnv } from './env';
import type { RNG } from './rng';
import type { GenerationStats } from './ga';

export interface PPOParams {
  /** Episodes rolled out per iteration to build the training batch. */
  rolloutEpisodes: number;
  /** Gradient-update passes over each collected batch. */
  epochs: number;
  /** PPO clip range ε on the probability ratio. */
  clipRatio: number;
  /** Discount factor. */
  gamma: number;
  /** GAE(λ) trace-decay. */
  lambda: number;
  learningRate: number;
  /** Entropy bonus weight (encourages exploration). */
  entropyCoef: number;
  /** Value-loss weight in the combined objective. */
  valueCoef: number;
  hiddenSize: number;
}

export const DEFAULT_PPO_PARAMS: PPOParams = {
  rolloutEpisodes: 6,
  epochs: 4,
  clipRatio: 0.2,
  gamma: 0.99,
  lambda: 0.95,
  learningRate: 3e-3,
  entropyCoef: 0.01,
  valueCoef: 0.5,
  hiddenSize: NET_SHAPE.hiddenSize,
};

/**
 * Forces the synchronous pure-JS CPU backend. The whole engine drives generations synchronously
 * inside a rAF loop, and the controller network is tiny, so CPU keeps every PPO op (forward pass,
 * `optimizer.minimize`, tensor reads) synchronous and avoids the async GPU readbacks WebGL needs.
 * Must be awaited once before constructing a {@link PPOTrainer}.
 */
export async function initPPOBackend(): Promise<void> {
  if (tf.getBackend() !== 'cpu') {
    await tf.setBackend('cpu');
  }
  await tf.ready();
}

interface Transition {
  obs: number[];
  actionIndex: number;
  logProb: number;
  value: number;
  reward: number;
}

/** Plain-JS forward pass over weights pulled out of the tf model — far cheaper than per-step tensor
 *  ops during the (long, many-step) rollout. Returns action probabilities and the state value. */
function forwardJS(
  w: ExtractedWeights,
  input: number[],
): { probs: number[]; value: number } {
  const { hiddenKernel, hiddenBias, policyKernel, policyBias, valueKernel, valueBias } = w;
  const hiddenSize = hiddenBias.length;
  const outputSize = policyBias.length;

  const hidden = new Array<number>(hiddenSize);
  for (let h = 0; h < hiddenSize; h++) {
    let sum = hiddenBias[h];
    for (let i = 0; i < input.length; i++) sum += input[i] * hiddenKernel[i][h];
    hidden[h] = Math.tanh(sum);
  }

  const logits = new Array<number>(outputSize);
  for (let o = 0; o < outputSize; o++) {
    let sum = policyBias[o];
    for (let h = 0; h < hiddenSize; h++) sum += hidden[h] * policyKernel[h][o];
    logits[o] = sum;
  }
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max));
  const expSum = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map((v) => v / expSum);

  let value = valueBias[0];
  for (let h = 0; h < hiddenSize; h++) value += hidden[h] * valueKernel[h][0];

  return { probs, value };
}

interface ExtractedWeights {
  hiddenKernel: number[][];
  hiddenBias: number[];
  policyKernel: number[][];
  policyBias: number[];
  valueKernel: number[][];
  valueBias: number[];
}

function sampleFrom(probs: number[], rng: RNG): number {
  const r = rng();
  let acc = 0;
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i];
    if (r < acc) return i;
  }
  return probs.length - 1;
}

/**
 * Proximal Policy Optimization solver, presented as a third "now" alternative to the GA and random
 * search. An actor-critic network (shared tanh hidden layer feeding a 7-way policy head and a scalar
 * value head) is trained by, each iteration: rolling out a batch of stochastic episodes, estimating
 * advantages with GAE(λ), then taking a few epochs of clipped-surrogate gradient steps. Shares the
 * exact {@link RobotEnv} dynamics and reward shaping the GA uses, so fitness is directly comparable.
 */
export class PPOTrainer {
  private readonly params: PPOParams;
  private readonly inputSize: number;
  private readonly outputSize: number;
  private model: tf.LayersModel;
  private optimizer: tf.Optimizer;

  constructor(params: PPOParams = DEFAULT_PPO_PARAMS) {
    this.params = params;
    this.inputSize = NET_SHAPE.inputSize;
    this.outputSize = NET_SHAPE.outputSize;

    const input = tf.input({ shape: [this.inputSize] });
    const shared = tf.layers
      .dense({ units: params.hiddenSize, activation: 'tanh', name: 'shared' })
      .apply(input) as tf.SymbolicTensor;
    const policy = tf.layers
      .dense({ units: this.outputSize, name: 'policy' })
      .apply(shared) as tf.SymbolicTensor;
    const value = tf.layers.dense({ units: 1, name: 'value' }).apply(shared) as tf.SymbolicTensor;

    this.model = tf.model({ inputs: input, outputs: [policy, value] });
    this.optimizer = tf.train.adam(params.learningRate);
  }

  private extractWeights(): ExtractedWeights {
    // getWeights() hands back the model's own live variable tensors (not copies), so we read them to
    // plain JS arrays and must NOT dispose them — doing so would destroy the network.
    const get = (name: string) => this.model.getLayer(name).getWeights();
    const [hk, hb] = get('shared');
    const [pk, pb] = get('policy');
    const [vk, vb] = get('value');
    return {
      hiddenKernel: hk.arraySync() as number[][],
      hiddenBias: hb.arraySync() as number[],
      policyKernel: pk.arraySync() as number[][],
      policyBias: pb.arraySync() as number[],
      valueKernel: vk.arraySync() as number[][],
      valueBias: vb.arraySync() as number[],
    };
  }

  /** Runs one PPO iteration: rollout → GAE → clipped policy/value updates. Returns per-generation
   *  stats (over the rollout episodes) plus the best rollout episode, recorded for replay. */
  step(
    generationIndex: number,
    grid: Grid,
    start: Position,
    episodeLength: number,
    rng: RNG,
  ): { stats: GenerationStats; bestEpisode: EpisodeResult } {
    const weights = this.extractWeights();
    const env = new RobotEnv(grid, start);

    // Flat training batch (all steps across all rollout episodes) plus per-episode bookkeeping so
    // GAE can be computed within each episode's own value trace.
    const batch: Transition[] = [];
    const episodeFitness: number[] = [];
    const advantages: number[] = [];
    const returns: number[] = [];

    let bestEpisode: EpisodeResult | null = null;
    let bestFitness = -Infinity;

    for (let e = 0; e < this.params.rolloutEpisodes; e++) {
      env.reset();
      const epTransitions: Transition[] = [];
      const steps: StepRecord[] = [];
      let fitness = 0;
      let cansCollected = 0;
      let wallCollisions = 0;

      for (let t = 0; t < episodeLength; t++) {
        const obs = env.observe();
        const { probs, value } = forwardJS(weights, obs);
        const actionIndex = sampleFrom(probs, rng);
        const action: Action = ACTIONS[actionIndex];

        const { resolvedAction, reward, collided, pickedUp } = env.step(action, rng);
        if (pickedUp) cansCollected += 1;
        if (collided) wallCollisions += 1;
        fitness += reward;

        epTransitions.push({
          obs,
          actionIndex,
          logProb: Math.log(Math.max(probs[actionIndex], 1e-8)),
          value,
          reward,
        });
        steps.push({
          position: { ...env.pos },
          action: resolvedAction,
          reward,
          cumulativeReward: fitness,
          collided,
          pickedUp,
        });
      }

      // GAE(λ) within this episode. Episodes end on the step budget (a time limit, not a true
      // terminal), but with no successor state available we bootstrap from 0 at the tail.
      let gae = 0;
      for (let t = epTransitions.length - 1; t >= 0; t--) {
        const nextValue = t + 1 < epTransitions.length ? epTransitions[t + 1].value : 0;
        const delta = epTransitions[t].reward + this.params.gamma * nextValue - epTransitions[t].value;
        gae = delta + this.params.gamma * this.params.lambda * gae;
        advantages.push(gae);
        returns.push(gae + epTransitions[t].value);
        batch.push(epTransitions[t]);
      }

      episodeFitness.push(fitness);
      if (fitness > bestFitness) {
        bestFitness = fitness;
        bestEpisode = {
          fitness,
          cansCollected,
          wallCollisions,
          initialGrid: grid,
          startPosition: start,
          steps,
        };
      }
    }

    this.update(batch, advantages, returns);

    const averageFitness = episodeFitness.reduce((a, b) => a + b, 0) / episodeFitness.length;
    const worstFitness = Math.min(...episodeFitness);
    const stats: GenerationStats = {
      generation: generationIndex,
      bestFitness,
      averageFitness,
      worstFitness,
    };

    // bestEpisode is always set: rolloutEpisodes >= 1.
    return { stats, bestEpisode: bestEpisode! };
  }

  private update(batch: Transition[], advantages: number[], returns: number[]): void {
    // Advantage normalization stabilizes the policy-gradient scale across iterations.
    const mean = advantages.reduce((a, b) => a + b, 0) / advantages.length;
    const variance = advantages.reduce((a, b) => a + (b - mean) ** 2, 0) / advantages.length;
    const std = Math.sqrt(variance) + 1e-8;
    const normAdv = advantages.map((a) => (a - mean) / std);

    const obsT = tf.tensor2d(batch.map((b) => b.obs));
    const actionsT = tf.tensor1d(batch.map((b) => b.actionIndex), 'int32');
    const oldLogProbsT = tf.tensor1d(batch.map((b) => b.logProb));
    const advT = tf.tensor1d(normAdv);
    const returnsT = tf.tensor1d(returns);

    const { clipRatio, valueCoef, entropyCoef } = this.params;

    for (let epoch = 0; epoch < this.params.epochs; epoch++) {
      this.optimizer.minimize(() => {
        const [logits, values] = this.model.apply(obsT) as tf.Tensor[];
        const logProbsAll = tf.logSoftmax(logits);
        const mask = tf.oneHot(actionsT, this.outputSize);
        const newLogProbs = tf.sum(tf.mul(logProbsAll, mask), 1);

        const ratio = tf.exp(tf.sub(newLogProbs, oldLogProbsT));
        const surr1 = tf.mul(ratio, advT);
        const surr2 = tf.mul(tf.clipByValue(ratio, 1 - clipRatio, 1 + clipRatio), advT);
        const policyLoss = tf.neg(tf.mean(tf.minimum(surr1, surr2)));

        const valuePred = tf.squeeze(values, [1]);
        const valueLoss = tf.mean(tf.square(tf.sub(returnsT, valuePred)));

        const probs = tf.softmax(logits);
        const entropy = tf.mean(tf.neg(tf.sum(tf.mul(probs, logProbsAll), 1)));

        // Minimize policy + value loss while *maximizing* entropy (hence the negative coefficient).
        return tf.add(
          tf.add(policyLoss, tf.mul(valueLoss, valueCoef)),
          tf.mul(entropy, -entropyCoef),
        ) as tf.Scalar;
      });
    }

    obsT.dispose();
    actionsT.dispose();
    oldLogProbsT.dispose();
    advT.dispose();
    returnsT.dispose();
  }

  dispose(): void {
    this.model.dispose();
    this.optimizer.dispose();
  }
}
