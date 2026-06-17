import { useCallback, useEffect, useRef, useState } from 'react';
import { mulberry32 } from '../sim/rng';
import type { RNG } from '../sim/rng';
import { generateWorld, DEFAULT_WORLD_CONFIG } from '../sim/world';
import type { WorldConfig } from '../sim/world';
import { DEFAULT_GA_PARAMS, evolveGeneration, initPopulation } from '../sim/ga';
import type { GAParams, GenerationStats } from '../sim/ga';
import { RandomSearchBaseline } from '../sim/randomSearch';
import { DEFAULT_PPO_PARAMS, PPOTrainer, initPPOBackend } from '../sim/ppo';
import type { PPOParams } from '../sim/ppo';
import type { EpisodeResult, Genome, Grid, Position } from '../sim/types';

export type RunStatus = 'idle' | 'running' | 'paused' | 'done';

/** Which solver the engine trains and replays. GA and PPO are run independently, one at a time. */
export type SolverMode = 'ga' | 'ppo';

export interface EvolutionConfig {
  world: WorldConfig;
  ga: GAParams;
  ppo: PPOParams;
  maxGenerations: number;
  simSpeed: number;
}

export const DEFAULT_CONFIG: EvolutionConfig = {
  world: DEFAULT_WORLD_CONFIG,
  ga: DEFAULT_GA_PARAMS,
  ppo: DEFAULT_PPO_PARAMS,
  maxGenerations: 100,
  simSpeed: 20,
};

export interface EvolutionSnapshot {
  status: RunStatus;
  config: EvolutionConfig;
  generation: number;
  gaHistory: GenerationStats[];
  rsHistory: GenerationStats[];
  ppoHistory: GenerationStats[];
  grid: Grid;
  startPosition: Position;
  currentEpisode: EpisodeResult | null;
  replayIndex: number;
  bestFitnessEver: number;
  mode: SolverMode;
}

export function useEvolutionEngine() {
  const [config, setConfigState] = useState<EvolutionConfig>(DEFAULT_CONFIG);
  const [status, setStatus] = useState<RunStatus>('idle');
  const [generation, setGeneration] = useState(0);
  const [gaHistory, setGaHistory] = useState<GenerationStats[]>([]);
  const [rsHistory, setRsHistory] = useState<GenerationStats[]>([]);
  const [ppoHistory, setPpoHistory] = useState<GenerationStats[]>([]);
  const [grid, setGrid] = useState<Grid>(() => generateWorld(DEFAULT_CONFIG.world, mulberry32(1)).grid);
  const [startPosition, setStartPosition] = useState<Position>({ x: 0, y: 0 });
  const [currentEpisode, setCurrentEpisode] = useState<EpisodeResult | null>(null);
  const [replayIndex, setReplayIndex] = useState(0);
  const [bestFitnessEver, setBestFitnessEver] = useState(-Infinity);
  const [mode, setModeState] = useState<SolverMode>('ga');

  const configRef = useRef(config);
  const statusRef = useRef(status);
  useEffect(() => {
    configRef.current = config;
  }, [config]);
  useEffect(() => {
    statusRef.current = status;
  }, [status]);

  const rngRef = useRef<RNG>(mulberry32(1));
  const populationRef = useRef<Genome[]>([]);
  const gridRef = useRef<Grid>(grid);
  const startPosRef = useRef<Position>(startPosition);
  const generationRef = useRef(0);
  const gaHistoryRef = useRef<GenerationStats[]>([]);
  const rsHistoryRef = useRef<GenerationStats[]>([]);
  const ppoHistoryRef = useRef<GenerationStats[]>([]);
  const currentEpisodeRef = useRef<EpisodeResult | null>(null);
  const replayIndexRef = useRef(0);
  const bestFitnessRef = useRef(-Infinity);
  const rsBaselineRef = useRef(new RandomSearchBaseline());
  const ppoTrainerRef = useRef<PPOTrainer | null>(null);
  const modeRef = useRef<SolverMode>('ga');
  const rafIdRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number>(0);
  const accumulatorRef = useRef(0);

  const computeNextGeneration = useCallback(() => {
    const cfg = configRef.current;
    // The random-search baseline always runs alongside the active solver for comparison.
    const rsStats = rsBaselineRef.current.step(
      generationRef.current,
      cfg.ga.populationSize,
      gridRef.current,
      startPosRef.current,
      cfg.ga.episodeLength,
      rngRef.current,
    );

    // Only the selected solver trains and records this generation; GA and PPO run independently, so
    // the inactive one's history stays empty and its curve/stats don't appear.
    let activeStats: GenerationStats;
    if (modeRef.current === 'ga') {
      const outcome = evolveGeneration(
        populationRef.current,
        generationRef.current,
        cfg.maxGenerations,
        gridRef.current,
        startPosRef.current,
        cfg.ga,
        rngRef.current,
      );
      populationRef.current = outcome.population;
      gaHistoryRef.current = [...gaHistoryRef.current, outcome.stats];
      currentEpisodeRef.current = outcome.bestEpisode;
      activeStats = outcome.stats;
    } else {
      // The trainer is created in start() when PPO mode is selected.
      const ppo = ppoTrainerRef.current!.step(
        generationRef.current,
        gridRef.current,
        startPosRef.current,
        cfg.ga.episodeLength,
        rngRef.current,
      );
      ppoHistoryRef.current = [...ppoHistoryRef.current, ppo.stats];
      currentEpisodeRef.current = ppo.bestEpisode;
      activeStats = ppo.stats;
    }

    generationRef.current += 1;
    rsHistoryRef.current = [...rsHistoryRef.current, rsStats];
    replayIndexRef.current = 0;
    bestFitnessRef.current = Math.max(bestFitnessRef.current, activeStats.bestFitness);

    setGeneration(generationRef.current);
    setGaHistory(gaHistoryRef.current);
    setRsHistory(rsHistoryRef.current);
    setPpoHistory(ppoHistoryRef.current);
    setCurrentEpisode(currentEpisodeRef.current);
    setReplayIndex(0);
    setBestFitnessEver(bestFitnessRef.current);

    if (generationRef.current >= cfg.maxGenerations) {
      statusRef.current = 'done';
      setStatus('done');
    }
  }, []);

  const advanceReplay = useCallback(() => {
    const episode = currentEpisodeRef.current;
    if (!episode) {
      computeNextGeneration();
      return;
    }
    if (replayIndexRef.current + 1 < episode.steps.length) {
      replayIndexRef.current += 1;
      setReplayIndex(replayIndexRef.current);
    } else {
      computeNextGeneration();
    }
  }, [computeNextGeneration]);

  useEffect(() => {
    if (status !== 'running') return;

    lastTimeRef.current = performance.now();
    accumulatorRef.current = 0;

    const tick = (now: number) => {
      const dt = now - lastTimeRef.current;
      lastTimeRef.current = now;
      accumulatorRef.current += dt;
      const msPerStep = 1000 / Math.max(configRef.current.simSpeed, 1);

      let guard = 0;
      while (accumulatorRef.current >= msPerStep && statusRef.current === 'running' && guard < 1000) {
        accumulatorRef.current -= msPerStep;
        advanceReplay();
        guard += 1;
      }

      if (statusRef.current === 'running') {
        rafIdRef.current = requestAnimationFrame(tick);
      }
    };

    rafIdRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafIdRef.current !== null) cancelAnimationFrame(rafIdRef.current);
      rafIdRef.current = null;
    };
  }, [status, advanceReplay]);

  const start = useCallback(async () => {
    if (statusRef.current === 'paused') {
      setStatus('running');
      return;
    }
    if (statusRef.current === 'idle') {
      const cfg = configRef.current;
      // Each run generates its own fresh world, so GA and PPO runs are fully independent.
      rngRef.current = mulberry32(Date.now() & 0xffffffff);
      const { grid: newGrid, start: newStart } = generateWorld(cfg.world, rngRef.current);
      gridRef.current = newGrid;
      startPosRef.current = newStart;
      setGrid(newGrid);
      setStartPosition(newStart);

      // Only set up the selected solver: GA seeds a population; PPO awaits the synchronous TF.js
      // backend (so the rAF loop's synchronous computeNextGeneration() can call into it) and builds
      // a fresh trainer. The unused solver's machinery is left torn down.
      ppoTrainerRef.current?.dispose();
      ppoTrainerRef.current = null;
      if (modeRef.current === 'ga') {
        populationRef.current = initPopulation(cfg.ga.populationSize, rngRef.current);
      } else {
        await initPPOBackend();
        ppoTrainerRef.current = new PPOTrainer(cfg.ppo);
        populationRef.current = [];
      }

      generationRef.current = 0;
      gaHistoryRef.current = [];
      rsHistoryRef.current = [];
      ppoHistoryRef.current = [];
      currentEpisodeRef.current = null;
      bestFitnessRef.current = -Infinity;
      rsBaselineRef.current.reset();
      setGaHistory([]);
      setRsHistory([]);
      setPpoHistory([]);
      setGeneration(0);
      setBestFitnessEver(-Infinity);

      setStatus('running');
    }
  }, []);

  const pause = useCallback(() => {
    if (statusRef.current === 'running') setStatus('paused');
  }, []);

  const reset = useCallback(() => {
    if (rafIdRef.current !== null) cancelAnimationFrame(rafIdRef.current);
    rafIdRef.current = null;

    const cfg = configRef.current;
    rngRef.current = mulberry32(Date.now() & 0xffffffff);
    const { grid: newGrid, start: newStart } = generateWorld(cfg.world, rngRef.current);

    gridRef.current = newGrid;
    startPosRef.current = newStart;
    populationRef.current = [];
    generationRef.current = 0;
    gaHistoryRef.current = [];
    rsHistoryRef.current = [];
    ppoHistoryRef.current = [];
    currentEpisodeRef.current = null;
    replayIndexRef.current = 0;
    bestFitnessRef.current = -Infinity;
    rsBaselineRef.current.reset();
    ppoTrainerRef.current?.dispose();
    ppoTrainerRef.current = null;

    setGrid(newGrid);
    setStartPosition(newStart);
    setGeneration(0);
    setGaHistory([]);
    setRsHistory([]);
    setPpoHistory([]);
    setCurrentEpisode(null);
    setReplayIndex(0);
    setBestFitnessEver(-Infinity);
    setStatus('idle');
  }, []);

  // Selects which solver the next run trains. GA and PPO are independent runs on their own worlds, so
  // the mode is only switchable while idle; reset() returns the engine to idle to change it.
  const setMode = useCallback((next: SolverMode) => {
    if (statusRef.current !== 'idle') return;
    modeRef.current = next;
    setModeState(next);
  }, []);

  // Tear the trainer down when the component unmounts so its tf tensors don't leak.
  useEffect(() => () => ppoTrainerRef.current?.dispose(), []);

  const setConfig = useCallback((updater: (prev: EvolutionConfig) => EvolutionConfig) => {
    setConfigState((prev) => {
      const next = updater(prev);
      if (statusRef.current === 'idle' && next.world !== prev.world) {
        const { grid: newGrid, start: newStart } = generateWorld(next.world, rngRef.current);
        gridRef.current = newGrid;
        startPosRef.current = newStart;
        setGrid(newGrid);
        setStartPosition(newStart);
      }
      return next;
    });
  }, []);

  const snapshot: EvolutionSnapshot = {
    status,
    config,
    generation,
    gaHistory,
    rsHistory,
    ppoHistory,
    grid,
    startPosition,
    currentEpisode,
    replayIndex,
    bestFitnessEver,
    mode,
  };

  return { ...snapshot, setConfig, start, pause, reset, setMode };
}
