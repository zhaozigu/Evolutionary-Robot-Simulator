import { useMemo } from 'react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { GenerationStats } from '../sim/ga';
import type { SolverMode } from '../hooks/useEvolutionEngine';

interface FitnessChartProps {
  mode: SolverMode;
  gaHistory: GenerationStats[];
  rsHistory: GenerationStats[];
  ppoHistory: GenerationStats[];
}

interface ChartPoint {
  generation: number;
  gaBest?: number;
  gaAvg?: number;
  rsBest?: number;
  rsAvg?: number;
  ppoBest?: number;
  ppoAvg?: number;
}

export function FitnessChart({ mode, gaHistory, rsHistory, ppoHistory }: FitnessChartProps) {
  const isGa = mode === 'ga';
  const data = useMemo<ChartPoint[]>(() => {
    const length = Math.max(gaHistory.length, rsHistory.length, ppoHistory.length);
    const points: ChartPoint[] = [];
    for (let i = 0; i < length; i++) {
      points.push({
        generation: i,
        gaBest: gaHistory[i]?.bestFitness,
        gaAvg: gaHistory[i]?.averageFitness,
        rsBest: rsHistory[i]?.bestFitness,
        rsAvg: rsHistory[i]?.averageFitness,
        ppoBest: ppoHistory[i]?.bestFitness,
        ppoAvg: ppoHistory[i]?.averageFitness,
      });
    }
    return points;
  }, [gaHistory, rsHistory, ppoHistory]);

  return (
    <div className="bg-slate-900/70 border border-slate-700 rounded-lg p-4 h-72">
      <h2 className="text-slate-100 font-semibold text-base mb-2">Fitness over Generations</h2>
      {data.length === 0 ? (
        <div className="flex items-center justify-center h-52 text-slate-500 text-sm">
          Start the simulation to see fitness curves.
        </div>
      ) : (
        <ResponsiveContainer width="100%" height="90%">
          <LineChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="generation" stroke="#94a3b8" tick={{ fontSize: 12 }} />
            <YAxis stroke="#94a3b8" tick={{ fontSize: 12 }} />
            <Tooltip
              contentStyle={{ background: '#0f172a', border: '1px solid #334155', fontSize: 12 }}
              labelStyle={{ color: '#e2e8f0' }}
            />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            {isGa ? (
              <>
                <Line type="monotone" dataKey="gaBest" name="GA Best" stroke="#22c55e" dot={false} strokeWidth={2} isAnimationActive={false} connectNulls />
                <Line type="monotone" dataKey="gaAvg" name="GA Average" stroke="#38bdf8" dot={false} strokeWidth={2} isAnimationActive={false} connectNulls />
              </>
            ) : (
              <>
                <Line type="monotone" dataKey="ppoBest" name="PPO Best" stroke="#a855f7" dot={false} strokeWidth={2} isAnimationActive={false} connectNulls />
                <Line type="monotone" dataKey="ppoAvg" name="PPO Average" stroke="#e879f9" dot={false} strokeWidth={2} strokeDasharray="2 3" isAnimationActive={false} connectNulls />
              </>
            )}
            <Line type="monotone" dataKey="rsBest" name="Random Search Best" stroke="#f59e0b" dot={false} strokeWidth={2} strokeDasharray="5 4" isAnimationActive={false} connectNulls />
            <Line type="monotone" dataKey="rsAvg" name="Random Search Average" stroke="#f87171" dot={false} strokeWidth={2} strokeDasharray="5 4" isAnimationActive={false} connectNulls />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
