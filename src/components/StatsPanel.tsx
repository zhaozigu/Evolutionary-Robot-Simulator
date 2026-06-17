import { ACTION_LABELS } from '../sim/types';
import type { EpisodeResult } from '../sim/types';
import { getStatsAtIndex } from '../sim/replay';

interface StatsPanelProps {
  episode: EpisodeResult | null;
  replayIndex: number;
}

function Stat({ label, value, accent }: { label: string; value: string | number; accent?: string }) {
  return (
    <div className="bg-slate-800/60 rounded-md px-3 py-2 min-w-0">
      <div className="text-xs text-slate-400">{label}</div>
      <div className={`text-lg font-semibold tabular-nums truncate ${accent ?? 'text-slate-100'}`}>{value}</div>
    </div>
  );
}

export function StatsPanel({ episode, replayIndex }: StatsPanelProps) {
  const stats = episode ? getStatsAtIndex(episode, replayIndex) : null;
  const currentAction = episode?.steps[replayIndex]?.action;

  return (
    <div className="bg-slate-900/70 border border-slate-700 rounded-lg p-4 space-y-3">
      <h2 className="text-slate-100 font-semibold text-base">Current Run</h2>
      <div className="grid grid-cols-2 gap-2">
        <Stat label="Score" value={(stats?.score ?? 0).toFixed(1)} accent="text-sky-400" />
        <Stat label="Remaining Actions" value={stats?.remainingActions ?? '—'} />
        <Stat label="Cans Collected" value={stats?.cansCollected ?? 0} accent="text-emerald-400" />
        <Stat label="Wall Collisions" value={stats?.wallCollisions ?? 0} accent="text-rose-400" />
      </div>
      <div className="text-sm text-slate-400">
        Action: <span className="text-slate-200">{currentAction ? ACTION_LABELS[currentAction] : '—'}</span>
      </div>
    </div>
  );
}
