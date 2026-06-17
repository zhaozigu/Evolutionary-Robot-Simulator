import type { GenerationStats } from '../sim/ga';
import type { SolverMode } from '../hooks/useEvolutionEngine';

interface EvolutionPanelProps {
  mode: SolverMode;
  generation: number;
  maxGenerations: number;
  populationSize: number;
  latest: GenerationStats | undefined;
  bestFitnessEver: number;
}

function Stat({ label, value, accent }: { label: string; value: string | number; accent?: string }) {
  return (
    <div className="bg-slate-800/60 rounded-md px-3 py-2 min-w-0">
      <div className="text-xs text-slate-400">{label}</div>
      <div className={`text-lg font-semibold tabular-nums truncate ${accent ?? 'text-slate-100'}`}>{value}</div>
    </div>
  );
}

export function EvolutionPanel({
  mode,
  generation,
  maxGenerations,
  populationSize,
  latest,
  bestFitnessEver,
}: EvolutionPanelProps) {
  const isGa = mode === 'ga';
  const title = isGa ? 'Genetic Algorithm' : 'PPO';
  const bestAccent = isGa ? 'text-emerald-400' : 'text-purple-400';
  const avgAccent = isGa ? 'text-sky-400' : 'text-fuchsia-400';
  const progressLabel = isGa ? 'Generation' : 'Iteration';

  return (
    <div className="bg-slate-900/70 border border-slate-700 rounded-lg p-4 space-y-3">
      <h2 className="text-slate-100 font-semibold text-base">{title}</h2>
      <div className="grid grid-cols-2 gap-2">
        <Stat label={progressLabel} value={`${generation} / ${maxGenerations}`} />
        {isGa && <Stat label="Population Size" value={populationSize} />}
        <Stat label="Best (gen)" value={latest ? latest.bestFitness.toFixed(1) : '—'} accent={bestAccent} />
        <Stat label="Avg (gen)" value={latest ? latest.averageFitness.toFixed(1) : '—'} accent={avgAccent} />
      </div>
      <Stat label="Best Fitness Ever" value={Number.isFinite(bestFitnessEver) ? bestFitnessEver.toFixed(1) : '—'} accent="text-amber-400" />
    </div>
  );
}
