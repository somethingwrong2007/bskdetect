import { createArena, setBroadcastCamera, ballBBox, renderSemanticLabel, cameraWorldToPx } from './arena.js';

const canvas = document.getElementById('arena');
const status = document.getElementById('status');

async function main() {
  if (!navigator.gpu) throw new Error('WebGPU is not available in this browser');
  const [dims, teams] = await Promise.all([
    fetch('/court_dims.json').then((r) => r.json()),
    fetch('/team_palettes.json').then((r) => r.json()),
  ]);
  const params = new URLSearchParams(location.search);
  const requested = params.get('team') || 'Lakers';
  const team = teams.find((t) => t.name.toLowerCase() === requested.toLowerCase()) || teams[0];
  const arena = await createArena(canvas, dims[team.model] || dims.FIBA, {
    seed: 7,
    exposure: 0.95,
    team,
    areaLights: params.has('area'),
    shadows: !params.has('noshadow'),
    environment: !params.has('noenv'),
    humans: params.has('humans'),
    simpleHumans: params.has('simplehumans'),
  });
  arena.controls.update();
  status.textContent = `WebGPU arena ready — ${team.name} — drag to orbit`;

  window.__arena = arena;
  window.captureFrame = async (spec = {}) => {
    arena.captureMode = true;
    try {
      if (spec.camera) setBroadcastCamera(arena, spec.camera);
      if (spec.ball) {
        arena.ball.visible = spec.ball.visible !== false;
        arena.ball.position.set(spec.ball.pos[0], spec.ball.pos[2], -spec.ball.pos[1]);
      }
      arena.renderer.toneMappingExposure = spec.exposure ?? 0.95;
      arena.renderer.render(arena.scene, arena.camera);
      const device = arena.renderer.backend?.device;
      if (device?.queue?.onSubmittedWorkDone) await device.queue.onSubmittedWorkDone();
      const bbox = ballBBox(arena);
      return {
        image: canvas.toDataURL('image/jpeg', 0.92),
        bbox,
        width: canvas.width,
        height: canvas.height,
      };
    } finally {
      arena.captureMode = false;
    }
  };
  window.captureTopView = async () => {
    arena.captureMode = true;
    try {
      const THREE = await import('three/webgpu');
      const dimsNow = arena.dims;
      const w = canvas.width;
      const h = canvas.height;
      const viewW = dimsNow.length + 6;
      const cam = new THREE.OrthographicCamera(-viewW / 2, viewW / 2, viewW * h / w / 2, -viewW * h / w / 2, 0.1, 120);
      cam.position.set(0, 30, 0);
      cam.up.set(0, 0, -1);
      cam.lookAt(0, 0, 0);
      cam.updateMatrixWorld();
      arena.ball.visible = false;
      arena.renderer.render(arena.scene, cam);
      const device = arena.renderer.backend?.device;
      if (device?.queue?.onSubmittedWorkDone) await device.queue.onSubmittedWorkDone();
      return canvas.toDataURL('image/jpeg', 0.95);
    } finally {
      arena.captureMode = false;
    }
  };

  window.captureDatasetSample = async (spec = {}) => {
    arena.captureMode = true;
    try {
      if (spec.camera) setBroadcastCamera(arena, spec.camera);
      arena.ball.visible = false;
      arena.renderer.toneMappingExposure = spec.exposure ?? 0.95;
      arena.renderer.render(arena.scene, arena.camera);
      const device = arena.renderer.backend?.device;
      if (device?.queue?.onSubmittedWorkDone) await device.queue.onSubmittedWorkDone();
      return {
        image: canvas.toDataURL('image/jpeg', 0.94),
        label: await renderSemanticLabel(arena),
        H_world_to_px: cameraWorldToPx(arena),
        width: canvas.width,
        height: canvas.height,
      };
    } finally {
      arena.captureMode = false;
    }
  };

  window.captureDatasetBatch = async (cameras = []) => {
    arena.captureMode = true;
    const samples = [];
    try {
      for (const spec of cameras) {
        setBroadcastCamera(arena, spec);
        arena.ball.visible = false;
        arena.renderer.toneMappingExposure = 0.95;
        arena.renderer.render(arena.scene, arena.camera);
        const device = arena.renderer.backend?.device;
        if (device?.queue?.onSubmittedWorkDone) await device.queue.onSubmittedWorkDone();
        samples.push({
          image: canvas.toDataURL('image/jpeg', 0.94),
          label: await renderSemanticLabel(arena),
          H_world_to_px: cameraWorldToPx(arena),
          width: canvas.width,
          height: canvas.height,
        });
      }
      return samples;
    } finally {
      arena.captureMode = false;
    }
  };

  function animate() {
    if (!arena.captureMode) {
      arena.controls.update();
      arena.renderer.render(arena.scene, arena.camera);
    }
    requestAnimationFrame(animate);
  }
  animate();
  window.__ready = true;
}

main().catch((err) => {
  console.error(err);
  status.textContent = `WebGPU init failed: ${err.message}`;
  window.__ready = true;
  window.__error = String(err.stack || err);
});
