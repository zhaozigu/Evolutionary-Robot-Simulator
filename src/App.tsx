import { useEvolutionEngine } from './hooks/useEvolutionEngine';
import { World3D } from './components/World3D';
import { ControlPanel } from './components/ControlPanel';
import { StatsPanel } from './components/StatsPanel';
import { EvolutionPanel } from './components/EvolutionPanel';
import { FitnessChart } from './components/FitnessChart';

function App() {
  const {
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
    setConfig,
    start,
    pause,
    reset,
    setMode,
  } = useEvolutionEngine();

  const running = status !== 'idle';
  const modes = [
    { key: 'ga', label: 'Genetic Algorithm', active: 'bg-emerald-500 text-slate-950' },
    { key: 'ppo', label: 'PPO', active: 'bg-purple-500 text-slate-50' },
  ] as const;

  return (
    <div className="min-h-screen bg-[#0b1020] text-slate-100 p-4 lg:p-6 space-y-4">
      <header>
        <h1 className="text-2xl font-bold text-slate-50">Evolutionary Robot Simulator</h1>
        <p className="text-slate-400 text-sm">
          A genetic algorithm and a PPO reinforcement learner each train a neural-network controller to collect
          soda cans while avoiding walls in a grid world it can only see one cell in each direction. Pick a solver
          to run it on its own world, raced against a random-search baseline.
        </p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_320px] gap-4">
        <div className="space-y-4 min-w-0">
          <div className="flex items-center justify-between">
            <span className="text-sm text-slate-400">
              Solver{running && <span className="text-slate-500"> (reset to switch)</span>}:
            </span>
            <div className="inline-flex rounded-md overflow-hidden border border-slate-700">
              {modes.map(({ key, label, active }) => (
                <button
                  key={key}
                  onClick={() => setMode(key)}
                  disabled={running}
                  className={`px-3 py-1 text-sm font-medium transition-colors disabled:cursor-not-allowed ${
                    mode === key
                      ? active
                      : 'bg-slate-800 text-slate-300 enabled:hover:bg-slate-700 disabled:opacity-40'
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
          <div className="bg-slate-900/70 border border-slate-700 rounded-lg overflow-hidden h-[480px]">
            <World3D grid={grid} startPosition={startPosition} episode={currentEpisode} replayIndex={replayIndex} />
          </div>
          <FitnessChart mode={mode} gaHistory={gaHistory} rsHistory={rsHistory} ppoHistory={ppoHistory} />
        </div>

        <div className="space-y-4">
          <ControlPanel config={config} status={status} mode={mode} onChange={setConfig} onStart={start} onPause={pause} onReset={reset} />
          <EvolutionPanel
            mode={mode}
            generation={generation}
            maxGenerations={config.maxGenerations}
            populationSize={config.ga.populationSize}
            latest={mode === 'ga' ? gaHistory[gaHistory.length - 1] : ppoHistory[ppoHistory.length - 1]}
            bestFitnessEver={bestFitnessEver}
          />
          <StatsPanel episode={currentEpisode} replayIndex={replayIndex} />
        </div>
      </div>
    </div>
  );
}

export default App;
