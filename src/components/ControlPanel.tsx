import type { EvolutionConfig, RunStatus, SolverMode } from '../hooks/useEvolutionEngine';

interface ControlPanelProps {
  config: EvolutionConfig;
  status: RunStatus;
  mode: SolverMode;
  onChange: (updater: (prev: EvolutionConfig) => EvolutionConfig) => void;
  onStart: () => void;
  onPause: () => void;
  onReset: () => void;
}

function SliderRow({
  label,
  value,
  min,
  max,
  step,
  disabled,
  onChange,
  format,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  disabled?: boolean;
  onChange: (v: number) => void;
  format?: (v: number) => string;
}) {
  return (
    <label className="block text-sm text-slate-300">
      <div className="flex justify-between mb-1">
        <span>{label}</span>
        <span className="text-slate-400 tabular-nums">{format ? format(value) : value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-sky-400 disabled:opacity-40"
      />
    </label>
  );
}

export function ControlPanel({ config, status, mode, onChange, onStart, onPause, onReset }: ControlPanelProps) {
  const locked = status !== 'idle';
  const startLabel = status === 'paused' ? 'Resume' : 'Start';

  return (
    <div className="bg-slate-900/70 border border-slate-700 rounded-lg p-4 space-y-4">
      <h2 className="text-slate-100 font-semibold text-base">Controls</h2>

      <div className="space-y-3">
        <SliderRow
          label="World Size"
          value={config.world.size}
          min={10}
          max={30}
          step={5}
          disabled={locked}
          format={(v) => `${v}×${v}`}
          onChange={(v) => onChange((prev) => ({ ...prev, world: { ...prev.world, size: v } }))}
        />
        <SliderRow
          label="Wall Density"
          value={config.world.wallDensity}
          min={0}
          max={0.3}
          step={0.01}
          disabled={locked}
          format={(v) => `${Math.round(v * 100)}%`}
          onChange={(v) => onChange((prev) => ({ ...prev, world: { ...prev.world, wallDensity: v } }))}
        />
        <SliderRow
          label="Can Density"
          value={config.world.canDensity}
          min={0.02}
          max={0.4}
          step={0.01}
          disabled={locked}
          format={(v) => `${Math.round(v * 100)}%`}
          onChange={(v) => onChange((prev) => ({ ...prev, world: { ...prev.world, canDensity: v } }))}
        />
      </div>

      <div className="border-t border-slate-700 pt-3 space-y-3">
        {mode === 'ga' && (
          <>
            <SliderRow
              label="Population Size"
              value={config.ga.populationSize}
              min={10}
              max={200}
              step={5}
              disabled={locked}
              onChange={(v) => onChange((prev) => ({ ...prev, ga: { ...prev.ga, populationSize: v } }))}
            />
            <SliderRow
              label="Mutation Rate"
              value={config.ga.mutationRate}
              min={0}
              max={0.5}
              step={0.01}
              disabled={locked}
              format={(v) => `${Math.round(v * 100)}%`}
              onChange={(v) => onChange((prev) => ({ ...prev, ga: { ...prev.ga, mutationRate: v } }))}
            />
          </>
        )}
        <SliderRow
          label={mode === 'ga' ? 'Number of Generations' : 'Training Iterations'}
          value={config.maxGenerations}
          min={5}
          max={500}
          step={5}
          disabled={locked}
          onChange={(v) => onChange((prev) => ({ ...prev, maxGenerations: v }))}
        />
        <SliderRow
          label="Episode Length"
          value={config.ga.episodeLength}
          min={50}
          max={400}
          step={10}
          disabled={locked}
          onChange={(v) => onChange((prev) => ({ ...prev, ga: { ...prev.ga, episodeLength: v } }))}
        />
      </div>

      <div className="border-t border-slate-700 pt-3">
        <SliderRow
          label="Simulation Speed"
          value={config.simSpeed}
          min={1}
          max={120}
          step={1}
          format={(v) => `${v} steps/s`}
          onChange={(v) => onChange((prev) => ({ ...prev, simSpeed: v }))}
        />
      </div>

      <div className="flex gap-2 pt-2">
        <button
          onClick={onStart}
          disabled={status === 'running' || status === 'done'}
          className="flex-1 bg-sky-500 hover:bg-sky-400 disabled:bg-slate-700 disabled:text-slate-500 text-slate-950 font-medium rounded-md py-2 text-sm transition-colors"
        >
          {startLabel}
        </button>
        <button
          onClick={onPause}
          disabled={status !== 'running'}
          className="flex-1 bg-amber-500 hover:bg-amber-400 disabled:bg-slate-700 disabled:text-slate-500 text-slate-950 font-medium rounded-md py-2 text-sm transition-colors"
        >
          Pause
        </button>
        <button
          onClick={onReset}
          className="flex-1 bg-slate-700 hover:bg-slate-600 text-slate-100 font-medium rounded-md py-2 text-sm transition-colors"
        >
          Reset
        </button>
      </div>
      {status === 'done' && (
        <p className="text-emerald-400 text-sm text-center">Evolution complete — press Reset to run again.</p>
      )}
    </div>
  );
}
