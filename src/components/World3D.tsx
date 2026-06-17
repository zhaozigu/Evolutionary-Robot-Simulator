import { useMemo, useRef, useEffect, useLayoutEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid as DreiGrid } from '@react-three/drei';
import * as THREE from 'three';
import type { EpisodeResult, Grid, Position } from '../sim/types';
import { getCollectedCanKeys } from '../sim/replay';

interface World3DProps {
  grid: Grid;
  startPosition: Position;
  episode: EpisodeResult | null;
  replayIndex: number;
}

function Walls({ grid }: { grid: Grid }) {
  const size = grid.length;
  const walls: [number, number][] = [];
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < grid[y].length; x++) {
      if (grid[y][x] === 'wall') walls.push([x, y]);
    }
  }
  return (
    <group>
      {walls.map(([x, y]) => (
        <mesh key={`wall-${x}-${y}`} position={[x, 0.5, y]} castShadow receiveShadow>
          <boxGeometry args={[0.92, 1, 0.92]} />
          <meshStandardMaterial color="#7c8595" roughness={0.7} metalness={0.1} />
        </mesh>
      ))}
    </group>
  );
}

function Cans({ grid, episode, replayIndex }: { grid: Grid; episode: EpisodeResult | null; replayIndex: number }) {
  const size = grid.length;
  const cans: [number, number][] = [];
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < grid[y].length; x++) {
      if (grid[y][x] === 'can') cans.push([x, y]);
    }
  }
  const collected = useMemo(
    () => (episode ? getCollectedCanKeys(episode, replayIndex) : new Set<string>()),
    [episode, replayIndex],
  );
  return (
    <group>
      {cans.map(([x, y]) => {
        if (collected.has(`${x},${y}`)) return null;
        return (
          <group key={`can-${x}-${y}`} position={[x, 0.3, y]} castShadow>
            <mesh castShadow>
              <cylinderGeometry args={[0.16, 0.16, 0.5, 16]} />
              <meshStandardMaterial color="#e63946" metalness={0.7} roughness={0.25} />
            </mesh>
            <mesh position={[0, 0.26, 0]}>
              <cylinderGeometry args={[0.17, 0.17, 0.04, 16]} />
              <meshStandardMaterial color="#cfd2d6" metalness={0.9} roughness={0.2} />
            </mesh>
          </group>
        );
      })}
    </group>
  );
}

const FLASH_DURATION_MS = 250;

function Robot({ episode, replayIndex, startPosition }: { episode: EpisodeResult | null; replayIndex: number; startPosition: Position }) {
  const groupRef = useRef<THREE.Group>(null);
  const bodyMaterialRef = useRef<THREE.MeshStandardMaterial>(null);
  const target = useRef(new THREE.Vector3(startPosition.x, 0.4, startPosition.y));
  const flashColorRef = useRef<'collide' | 'pickup' | null>(null);
  const flashUntilRef = useRef(0);

  const currentStep = episode && episode.steps[replayIndex] ? episode.steps[replayIndex] : null;
  const pos = currentStep ? currentStep.position : startPosition;

  useLayoutEffect(() => {
    groupRef.current?.position.set(startPosition.x, 0.4, startPosition.y);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    target.current.set(pos.x, 0.4, pos.y);
    if (currentStep?.collided) {
      flashColorRef.current = 'collide';
      flashUntilRef.current = performance.now() + FLASH_DURATION_MS;
    } else if (currentStep?.pickedUp) {
      flashColorRef.current = 'pickup';
      flashUntilRef.current = performance.now() + FLASH_DURATION_MS;
    }
  }, [pos.x, pos.y, currentStep?.collided, currentStep?.pickedUp]);

  useFrame((_, delta) => {
    if (!groupRef.current) return;
    groupRef.current.position.lerp(target.current, Math.min(1, delta * 8));
    if (bodyMaterialRef.current) {
      const flashing = flashColorRef.current && performance.now() < flashUntilRef.current;
      const targetColor = flashing
        ? flashColorRef.current === 'collide'
          ? '#ef4444'
          : '#22c55e'
        : '#38bdf8';
      bodyMaterialRef.current.color.lerp(new THREE.Color(targetColor), 0.4);
    }
  });

  return (
    <group ref={groupRef}>
      <mesh castShadow>
        <boxGeometry args={[0.55, 0.4, 0.55]} />
        <meshStandardMaterial ref={bodyMaterialRef} color="#38bdf8" roughness={0.4} metalness={0.3} />
      </mesh>
      <mesh position={[0, 0.32, 0]} castShadow>
        <sphereGeometry args={[0.2, 16, 16]} />
        <meshStandardMaterial color="#e2e8f0" roughness={0.3} />
      </mesh>
      <mesh position={[0, 0.32, 0.18]}>
        <sphereGeometry args={[0.05, 8, 8]} />
        <meshStandardMaterial color="#0f172a" />
      </mesh>
      <mesh position={[0.45, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
        <cylinderGeometry args={[0.12, 0.12, 0.1, 12]} />
        <meshStandardMaterial color="#1e293b" />
      </mesh>
      <mesh position={[-0.45, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
        <cylinderGeometry args={[0.12, 0.12, 0.1, 12]} />
        <meshStandardMaterial color="#1e293b" />
      </mesh>
    </group>
  );
}

function Floor({ size }: { size: number }) {
  return (
    <group>
      <mesh
        position={[size / 2 - 0.5, -0.02, size / 2 - 0.5]}
        rotation={[-Math.PI / 2, 0, 0]}
        receiveShadow
      >
        <planeGeometry args={[size, size]} />
        <meshStandardMaterial color="#101a33" roughness={1} />
      </mesh>
      <DreiGrid
        args={[size, size]}
        position={[size / 2 - 0.5, 0, size / 2 - 0.5]}
        cellSize={1}
        cellThickness={0.6}
        cellColor="#3a4a6b"
        sectionSize={5}
        sectionThickness={1.4}
        sectionColor="#5d7bb5"
        fadeDistance={size * 3}
        fadeStrength={1}
        infiniteGrid={false}
      />
    </group>
  );
}

export function World3D({ grid, startPosition, episode, replayIndex }: World3DProps) {
  const size = grid.length;
  const center = (size - 1) / 2;
  const camDistance = Math.max(size * 1.1, 8);

  return (
    <Canvas shadows camera={{ position: [center, camDistance, camDistance + center], fov: 45 }}>
      <color attach="background" args={['#0b1020']} />
      <ambientLight intensity={0.55} />
      <directionalLight
        position={[size + 5, size + 8, size * 0.3]}
        intensity={1.1}
        castShadow
        shadow-mapSize={[1024, 1024]}
      />
      <Floor size={size} />
      <Walls grid={grid} />
      <Cans grid={grid} episode={episode} replayIndex={replayIndex} />
      <Robot episode={episode} replayIndex={replayIndex} startPosition={startPosition} />
      <OrbitControls target={[center, 0, center]} maxPolarAngle={Math.PI / 2.05} />
    </Canvas>
  );
}
