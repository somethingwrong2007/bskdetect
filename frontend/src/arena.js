import * as THREE from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RectAreaLightTexturesLib } from 'three/addons/lights/RectAreaLightTexturesLib.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { clone as cloneSkinned } from 'three/addons/utils/SkeletonUtils.js';

const UP = new THREE.Vector3(0, 1, 0);
const HUMAN_FILES = [
  'base_body_anim.glb',
  'clothing_tshirt.glb',
  'clothing_shorts.glb',
  'clothing_shoes.glb',
  'hair_short02.glb',
];
const SEMANTIC_COLORS = [
  [0, 0, 0], [1, 1, 1], [1, 0.863, 0.235],
  [0.314, 0.863, 0.314], [0.314, 0.549, 1], [0.863, 0.314, 0.863],
];

function rngFrom(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

async function woodTexture(loader, repeatX, repeatY) {
  const textures = await Promise.all(['diffuse', 'normal', 'roughness'].map((kind) => loader.loadAsync(`/assets/court_wood_${kind}.jpg`)));
  textures[0].colorSpace = THREE.SRGBColorSpace;
  for (const tex of textures) {
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(repeatX, repeatY);
    tex.anisotropy = 8;
  }
  return textures;
}

function quadGeometry(segments, width, y = 0.008) {
  const positions = [];
  const indices = [];
  const normals = [];
  for (const [x1, z1, x2, z2] of segments) {
    const dx = x2 - x1;
    const dz = z2 - z1;
    const len = Math.hypot(dx, dz);
    if (len < 1e-7) continue;
    const px = -dz / len * width * 0.5;
    const pz = dx / len * width * 0.5;
    const base = positions.length / 3;
    positions.push(
      x1 + px, y, z1 + pz,
      x2 + px, y, z2 + pz,
      x2 - px, y, z2 - pz,
      x1 - px, y, z1 - pz,
    );
    normals.push(0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0);
    indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  geo.setIndex(indices);
  return geo;
}

function addArc(segments, cx, cz, r, a0, a1, n = 80, dash = 0, gap = 0) {
  if (dash <= 0) {
    let prev = null;
    for (let i = 0; i <= n; i++) {
      const a = a0 + (a1 - a0) * i / n;
      const p = [cx + r * Math.cos(a), cz + r * Math.sin(a)];
      if (prev) segments.push([prev[0], prev[1], p[0], p[1]]);
      prev = p;
    }
    return;
  }
  const total = Math.abs(a1 - a0) * r;
  const step = dash + gap;
  let pos = 0;
  while (pos < total) {
    const end = Math.min(pos + dash, total);
    const s = a0 + Math.sign(a1 - a0) * pos / r;
    const e = a0 + Math.sign(a1 - a0) * end / r;
    addArc(segments, cx, cz, r, s, e, Math.max(3, Math.ceil(dash / 0.025)));
    pos += step;
  }
}

function box(w, h, d, x, y, z, material, shadow = true) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), material);
  mesh.position.set(x, y, z);
  mesh.castShadow = shadow;
  mesh.receiveShadow = true;
  return mesh;
}

async function buildCourt(scene, dims, team) {
  const hl = dims.length / 2;
  const hw = dims.width / 2;
  const lw = dims.line_width;
  // 球场外的 apron 使用球队辅色压暗，既保留场馆层次也统一主题。
  const apronColor = new THREE.Color(team?.secondary || '#3b414a')
    .multiplyScalar(0.38).add(new THREE.Color(0x202631));
  const apron = new THREE.Mesh(
    new THREE.PlaneGeometry(dims.length + 12, dims.width + 10),
    new THREE.MeshStandardMaterial({ color: apronColor, roughness: 0.96, metalness: 0.0, envMapIntensity: 0.08 }),
  );
  apron.rotation.x = -Math.PI / 2;
  apron.position.y = -0.035;
  apron.receiveShadow = true;
  scene.add(apron);
  const [woodMap, woodNormal, woodRoughness] = await woodTexture(new THREE.TextureLoader(), 14, 8);
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(dims.length + lw * 2 + 3.6, dims.width + lw * 2 + 3.6),
    new THREE.MeshPhysicalMaterial({
      map: woodMap,
      color: 0xd8c8aa,
      normalMap: woodNormal,
      normalScale: new THREE.Vector2(0.16, 0.16),
      roughnessMap: woodRoughness,
      roughness: 0.82,
      metalness: 0,
      clearcoat: 0.08,
      clearcoatRoughness: 0.58,
      envMapIntensity: 0.08,
    }),
  );
  floor.rotation.x = -Math.PI / 2;
  floor.receiveShadow = true;
  scene.add(floor);

  const paintColor = new THREE.Color(team?.primary || '#1d428a');
  const paintMat = new THREE.MeshStandardMaterial({ color: paintColor, roughness: 0.70, envMapIntensity: 0.14 });
  const boundarySegments = [];
  const threePointSegments = [];
  const paintSegments = [];
  const centerLineSegments = [];
  const centerCircleSegments = [];
  const bx = hl + lw / 2;
  const by = hw + lw / 2;
  boundarySegments.push([-bx, -by, bx, -by], [bx, -by, bx, by], [bx, by, -bx, by], [-bx, by, -bx, -by]);
  const ext = lw / 2 + (dims.center_line_extension || 0);
  centerLineSegments.push([0, -(hw + ext), 0, hw + ext]);
  addArc(centerCircleSegments, 0, 0, dims.center_circle_radius - lw / 2, 0, Math.PI * 2, 96);

  for (const sign of [-1, 1]) {
    const bl = sign * hl;
    const rimX = bl - sign * dims.rim_offset_from_baseline;
    const ftX = bl - sign * (dims.free_throw_dist - lw / 2);
    const laneHalf = dims.paint_width / 2;
    const laneLine = laneHalf + lw / 2;
    const paint = new THREE.Mesh(new THREE.PlaneGeometry(Math.abs(bl - ftX), laneHalf * 2), paintMat);
    paint.rotation.x = -Math.PI / 2;
    paint.position.set((bl + ftX) / 2, 0.002, 0);
    paint.receiveShadow = true;
    scene.add(paint);

    paintSegments.push([bl, -laneLine, ftX, -laneLine], [bl, laneLine, ftX, laneLine], [ftX, -laneLine, ftX, laneLine]);

    const ftR = dims.free_throw_width / 2 - lw / 2;
    if (dims.free_throw_dashed_inside) {
      if (sign > 0) {
        addArc(paintSegments, ftX, 0, ftR, Math.PI / 2, Math.PI * 1.5, 48);
        addArc(paintSegments, ftX, 0, ftR, -Math.PI / 2, Math.PI / 2, 48, 0.38, 0.28);
      } else {
        addArc(paintSegments, ftX, 0, ftR, -Math.PI / 2, Math.PI / 2, 48);
        addArc(paintSegments, ftX, 0, ftR, Math.PI / 2, Math.PI * 1.5, 48, 0.38, 0.28);
      }
    } else {
      addArc(paintSegments, ftX, 0, ftR, 0, Math.PI * 2, 72);
    }

    const r3 = dims.three_pt_radius - lw / 2;
    const ylim = hw - (dims.three_pt_corner_dist + lw / 2);
    const dx = Math.sqrt(Math.max(0, r3 * r3 - ylim * ylim));
    const arcX = rimX - sign * dx;
    threePointSegments.push([bl, -ylim, arcX, -ylim], [bl, ylim, arcX, ylim]);
    const half = Math.asin(Math.min(1, ylim / r3));
    let prev = null;
    for (let i = 0; i <= 90; i++) {
      const a = -half + 2 * half * i / 90;
      const p = [rimX - sign * r3 * Math.cos(a), r3 * Math.sin(a)];
      if (prev) threePointSegments.push([prev[0], prev[1], p[0], p[1]]);
      prev = p;
    }

    const rr = dims.restricted_radius + lw / 2;
    // 与三分弧一致：右半场（+x 底线）经过 π，左半场经过 0，均朝场内鼓出。
    if (sign > 0) addArc(paintSegments, rimX, 0, rr, Math.PI / 2, Math.PI * 1.5, 40);
    else addArc(paintSegments, rimX, 0, rr, -Math.PI / 2, Math.PI / 2, 40);
    const backX = bl - sign * dims.backboard_from_baseline;
    paintSegments.push([backX, -rr, rimX, -rr], [backX, rr, rimX, rr]);
  }

  const courtLines = [];
  for (const [classId, classSegments] of [[1, boundarySegments], [2, threePointSegments], [3, paintSegments],
    [4, centerLineSegments], [5, centerCircleSegments]]) {
    const lines = new THREE.Mesh(quadGeometry(classSegments, lw),
      new THREE.MeshBasicMaterial({ color: 0xf4f4ee, toneMapped: false }));
    lines.userData.segmentationClass = classId;
    lines.userData.visibleMaterial = lines.material;
    // 语义 pass 单独加粗，避免远处 5cm 线被抗锯齿打散成点。
    lines.userData.semanticGeometry = quadGeometry(classSegments, Math.max(lw * 2.2, 0.10));
    courtLines.push(lines);
    scene.add(lines);
  }
  return courtLines;
}

function buildHoop(scene, dims, sign, team) {
  const hl = dims.length / 2;
  const bl = sign * hl;
  const rimX = bl - sign * dims.rim_offset_from_baseline;
  const bbX = bl - sign * dims.backboard_from_baseline;
  const rimY = dims.rim_height;
  const glass = new THREE.MeshPhysicalMaterial({
    color: 0xdce8f2, roughness: 0.08, metalness: 0, transparent: true, opacity: 0.34,
    clearcoat: 0.9, clearcoatRoughness: 0.05, ior: 1.45, envMapIntensity: 1.1,
  });
  scene.add(box(0.045, 1.05, 1.8, bbX, rimY + 0.375, 0, glass));
  const frontX = bbX - sign * 0.026;
  const border = [];
  const rect = (y0, y1, z0, z1) => {
    border.push(frontX, y0, z0, frontX, y1, z0, frontX, y1, z0, frontX, y1, z1,
      frontX, y1, z1, frontX, y0, z1, frontX, y0, z1, frontX, y0, z0);
  };
  rect(rimY - 0.20, rimY + 0.85, -0.90, 0.90);
  rect(rimY + 0.075, rimY + 0.525, -0.295, 0.295);
  const bg = new THREE.BufferGeometry();
  bg.setAttribute('position', new THREE.Float32BufferAttribute(border, 3));
  scene.add(new THREE.LineSegments(bg, new THREE.LineBasicMaterial({ color: 0x111418 })));

  const rim = new THREE.Mesh(
    new THREE.TorusGeometry(dims.rim_radius, 0.012, 12, 48),
    new THREE.MeshPhysicalMaterial({ color: 0xf05a1f, roughness: 0.26, metalness: 0.62, clearcoat: 0.18 }),
  );
  rim.rotation.x = Math.PI / 2;
  rim.position.set(rimX, rimY, 0);
  rim.castShadow = true;
  scene.add(rim);

  const net = [];
  const n = 12;
  const botY = rimY - 0.45;
  const botR = dims.rim_radius * 0.52;
  for (let i = 0; i < n; i++) {
    const a = Math.PI * 2 * i / n;
    const a2 = Math.PI * 2 * (i + 1) / n;
    const p0 = [rimX + dims.rim_radius * Math.cos(a), rimY - 0.01, dims.rim_radius * Math.sin(a)];
    const p1 = [rimX + botR * Math.cos(a), botY, botR * Math.sin(a)];
    const p2 = [rimX + botR * Math.cos(a2), botY, botR * Math.sin(a2)];
    net.push(...p0, ...p1, ...p1, ...p2);
  }
  for (const y of [rimY - 0.15, rimY - 0.30, botY]) {
    const r = dims.rim_radius * (1 - (rimY - y) / 0.6 * 0.48);
    for (let i = 0; i < n; i++) {
      const a = Math.PI * 2 * i / n;
      const a2 = Math.PI * 2 * (i + 1) / n;
      net.push(rimX + r * Math.cos(a), y, r * Math.sin(a), rimX + r * Math.cos(a2), y, r * Math.sin(a2));
    }
  }
  const ng = new THREE.BufferGeometry();
  ng.setAttribute('position', new THREE.Float32BufferAttribute(net, 3));
  scene.add(new THREE.LineSegments(ng, new THREE.LineBasicMaterial({ color: 0xe8e8e0 })));

  // 底座放在端线外 2.2 m，避免任何部件压住底线。
  const poleX = bl + sign * 2.20;
  const metal = new THREE.MeshStandardMaterial({ color: 0x171d28, roughness: 0.48, metalness: 0.32 });
  scene.add(box(0.18, 3.55, 0.18, poleX, 1.775, 0, metal));
  scene.add(box(Math.abs(bbX - poleX) + 0.08, 0.11, 0.11, (poleX + bbX) / 2, rimY + 0.22, 0, metal));
  const padColor = sign > 0 ? (team?.primary || '#8c1919') : (team?.secondary || '#163c8c');
  scene.add(box(0.85, 0.75, 0.85, poleX, 0.375, 0, new THREE.MeshStandardMaterial({ color: padColor, roughness: 0.85 })));
  scene.add(box(1.00, 0.10, 1.00, poleX, 0.05, 0, new THREE.MeshStandardMaterial({ color: 0x10141c, roughness: 0.9 })));
}

function buildBench(scene, dims, sign, team) {
  const hw = dims.width / 2;
  const z = sign * (hw + 3.35);
  const frame = new THREE.MeshStandardMaterial({ color: 0x252c36, roughness: 0.45, metalness: 0.55 });
  const cushion = new THREE.MeshStandardMaterial({ color: sign > 0 ? (team?.primary || '#1a3d9e') : (team?.secondary || '#9e1a1a'), roughness: 0.78 });
  const bottle = new THREE.MeshStandardMaterial({ color: 0x9fd6e8, roughness: 0.25, transparent: true, opacity: 0.75 });
  const towel = new THREE.MeshStandardMaterial({ color: 0xe7e7e2, roughness: 0.95 });
  // 中线两侧各一段替补席，中间留出解说台位置。
  for (const [x0, x1] of [[-6.0, -1.4], [1.4, 6.0]]) {
    const length = x1 - x0;
    const cx = (x0 + x1) / 2;
    const n = 5;
    const sw = length / n;
    scene.add(box(length, 0.10, 0.56, cx, 0.42, z, frame));
    scene.add(box(length, 0.56, 0.10, cx, 0.73, z + sign * 0.28, frame));
    for (let i = 0; i < n; i++) {
      const x = x0 + sw * (i + 0.5);
      scene.add(box(sw - 0.05, 0.09, 0.52, x, 0.49, z, cushion));
      scene.add(box(sw - 0.05, 0.43, 0.09, x, 0.75, z + sign * 0.23, cushion));
    }
    for (let i = 0; i < 5; i++) {
      const x = x0 + 0.16 + i * (length - 0.32) / 4;
      for (const dz of [-0.20, 0.20]) scene.add(box(0.07, 0.42, 0.07, x, 0.21, z + dz, frame));
    }
    for (let i = 0; i < 3; i++) {
      const x = x0 + 0.6 + i * 1.35;
      const b = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.045, 0.20, 10), bottle);
      b.position.set(x, 0.60, z - sign * 0.20);
      b.castShadow = true;
      scene.add(b);
      scene.add(box(0.32, 0.035, 0.22, x + 0.30, 0.55, z, towel));
    }
  }
  const deskMat = new THREE.MeshStandardMaterial({ color: 0x2c323c, roughness: 0.62, metalness: 0.12 });
  const deskFront = new THREE.MeshStandardMaterial({ color: sign > 0 ? (team?.primary || '#1a3d9e') : (team?.secondary || '#9e1a1a'), roughness: 0.72 });
  scene.add(box(2.5, 0.08, 0.72, 0, 0.76, z, deskMat));
  scene.add(box(2.5, 0.62, 0.06, 0, 0.42, z - sign * 0.34, deskFront));
  for (const dx of [-1.05, 1.05]) scene.add(box(0.08, 0.72, 0.08, dx, 0.37, z, frame));
  const screenMat = new THREE.MeshStandardMaterial({ color: 0x0a0d12, roughness: 0.32, metalness: 0.22 });
  for (const dx of [-0.62, 0.62]) scene.add(box(0.52, 0.32, 0.05, dx, 1.02, z + sign * 0.10, screenMat));
}

function buildStands(scene, dims, rng, team) {
  const hl = dims.length / 2;
  const hw = dims.width / 2;
  const margin = 4.2;
  const rows = 6;
  const depth = 1.05;
  const rise = 0.24;
  const mat = new THREE.MeshStandardMaterial({ color: 0x5a6573, roughness: 0.88 });
  for (let r = 0; r < rows; r++) {
    const top = 0.12 + r * rise;
    const d = margin + (r + 0.5) * depth;
    const xLen = 2 * (hl + margin + (r + 1) * depth);
    const zLen = 2 * (hw + margin + (r + 1) * depth);
    for (const s of [-1, 1]) {
      scene.add(box(xLen, top, depth, 0, top / 2, s * (hw + d), mat));
      scene.add(box(depth, top, zLen, s * (hl + d), top / 2, 0, mat));
    }
  }

  const outerX = hl + margin + rows * depth + 0.8;
  const outerZ = hw + margin + rows * depth + 0.8;
  const ribbonMaterials = [team?.primary, team?.secondary].filter(Boolean).map((color) =>
    new THREE.MeshStandardMaterial({ color, roughness: 0.55, emissive: color, emissiveIntensity: 0.38 }));
  if (ribbonMaterials.length) {
    for (const s of [-1, 1]) {
      const xMat = ribbonMaterials[s > 0 ? 0 : ribbonMaterials.length - 1];
      const zMat = ribbonMaterials[s > 0 ? ribbonMaterials.length - 1 : 0];
      scene.add(box(0.08, 0.52, outerZ * 2 - 0.4, s * (outerX - 0.48), 2.45, 0, xMat, false));
      scene.add(box(outerX * 2 - 0.4, 0.52, 0.08, 0, 2.45, s * (outerZ - 0.48), zMat, false));
    }
  }
  const wallMat = new THREE.MeshStandardMaterial({ color: 0x34465b, roughness: 0.92, envMapIntensity: 0.10 });
  const stripMat = new THREE.MeshStandardMaterial({ color: 0x7f95ae, roughness: 0.55,
    emissive: 0x385573, emissiveIntensity: 0.72 });
  for (const s of [-1, 1]) {
    scene.add(box(0.35, 7.5, outerZ * 2, s * outerX, 3.75, 0, wallMat));
    scene.add(box(outerX * 2, 7.5, 0.35, 0, 3.75, s * outerZ, wallMat));
    for (const y of [4.8, 6.3]) {
      scene.add(box(0.06, 0.08, outerZ * 2, s * (outerX - 0.20), y, 0, stripMat, false));
      scene.add(box(outerX * 2, 0.08, 0.06, 0, y, s * (outerZ - 0.20), stripMat, false));
    }
  }

  const spots = [];
  for (let r = 0; r < rows; r++) {
    const top = 0.12 + r * rise;
    const d = margin + (r + 0.5) * depth;
    const spanX = hl + margin + (r + 0.5) * depth - 0.3;
    const spanZ = hw + margin + (r + 0.5) * depth - 0.3;
    for (const s of [-1, 1]) {
      for (let x = -spanX; x <= spanX; x += 0.78 + rng() * 0.18) {
        if (rng() < 0.82) spots.push([x + (rng() - 0.5) * 0.2, top, s * (hw + d), rng()]);
      }
      for (let z = -spanZ; z <= spanZ; z += 0.78 + rng() * 0.18) {
        if (rng() < 0.82) spots.push([s * (hl + d), top, z + (rng() - 0.5) * 0.2, rng()]);
      }
    }
  }
  const bodies = new THREE.InstancedMesh(
    new THREE.CapsuleGeometry(0.15, 0.30, 4, 8),
    new THREE.MeshStandardMaterial({ color: 0x4a5a70, roughness: 0.9 }),
    spots.length,
  );
  const heads = new THREE.InstancedMesh(
    new THREE.SphereGeometry(0.105, 8, 6),
    new THREE.MeshStandardMaterial({ color: 0x8a6548, roughness: 0.92 }),
    spots.length,
  );
  bodies.castShadow = bodies.receiveShadow = true;
  heads.castShadow = true;
  const dummy = new THREE.Object3D();
  spots.forEach(([x, y, z, tone], i) => {
    dummy.position.set(x, y + 0.28, z);
    dummy.rotation.set(0, tone * Math.PI * 2, 0);
    dummy.scale.set(0.9 + tone * 0.2, 0.9 + tone * 0.18, 0.9 + tone * 0.2);
    dummy.updateMatrix();
    bodies.setMatrixAt(i, dummy.matrix);
    bodies.setColorAt(i, new THREE.Color().setHSL(0.54 + tone * 0.22, 0.22, 0.24 + tone * 0.14));
    dummy.position.y = y + 0.67;
    dummy.scale.setScalar(0.9 + tone * 0.18);
    dummy.updateMatrix();
    heads.setMatrixAt(i, dummy.matrix);
    heads.setColorAt(i, new THREE.Color().setHSL(0.055 + tone * 0.045, 0.28, 0.28 + tone * 0.14));
  });
  scene.add(bodies, heads);
}

function buildLights(scene, options = {}) {
  if (options.areaLights === true) THREE.RectAreaLightNode.setLTC(RectAreaLightTexturesLib.init());
  scene.add(new THREE.HemisphereLight(0xc7d8ff, 0x4a3524, 0.88));
  const key = new THREE.DirectionalLight(0xfff0d8, 1.70);
  key.position.set(0, 13, 1);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  key.shadow.camera.left = -20;
  key.shadow.camera.right = 20;
  key.shadow.camera.top = 16;
  key.shadow.camera.bottom = -16;
  key.shadow.camera.near = 1;
  key.shadow.camera.far = 35;
  key.shadow.bias = -0.0002;
  key.shadow.normalBias = 0.025;
  key.shadow.radius = 4;
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xc3d4ff, 0.45);
  fill.position.set(-10, 8, -7);
  scene.add(fill);
  const rimFill = new THREE.DirectionalLight(0xffe6c8, 0.28);
  rimFill.position.set(9, 6, 8);
  scene.add(rimFill);

  if (options.areaLights === true) {
    // 可选面光；默认使用多方向柔光，避免 WebGPU LTC 形成硬亮带。
    for (const x of [-7.5, 7.5]) {
      const area = new THREE.RectAreaLight(0xfff1dc, 1.45, 18.0, 12.0);
      area.position.set(x, 10.5, 0);
      area.lookAt(0, 0, 0);
      scene.add(area);
    }
  }
  for (const [z, color] of [[-10, 0xc7d8ff], [10, 0xffe4c4]]) {
    const side = new THREE.DirectionalLight(color, 0.32);
    side.position.set(0, 5.5, z);
    scene.add(side);
  }
  return { key, fill };
}

async function loadHumanTemplates() {
  const loader = new GLTFLoader();
  const gltfs = await Promise.all(HUMAN_FILES.map((f) => loader.loadAsync(`/assets/humans/${f}`)));
  return gltfs.map((g) => g.scene);
}

function bonesByName(root) {
  const out = new Map();
  root.traverse((o) => { if (o.isBone) out.set(o.name, o); });
  return out;
}

function applyHumanPose(root, sitting) {
  const bones = bonesByName(root);
  const setRot = (name, axis, angle) => {
    const b = bones.get(name);
    if (!b) return;
    const q = b.quaternion.clone();
    const axisVec = new THREE.Vector3(...axis);
    b.quaternion.copy(q.multiply(new THREE.Quaternion().setFromAxisAngle(axisVec, angle)));
  };
  if (sitting) {
    const pelvis = bones.get('pelvis');
    if (pelvis) pelvis.position.y -= 0.36;
    for (const side of ['l', 'r']) {
      setRot(`upperleg_${side}`, [1, 0, 0], -Math.PI / 2);
      setRot(`lowerleg_${side}`, [1, 0, 0], Math.PI / 2);
      setRot(`upperarm_${side}`, [1, 0, 0], -0.35);
      setRot(`lowerarm_${side}`, [1, 0, 0], -0.55);
    }
  } else {
    for (const side of ['l', 'r']) {
      setRot(`upperarm_${side}`, [0, 0, 1], side === 'l' ? -0.18 : 0.18);
      setRot(`lowerarm_${side}`, [1, 0, 0], -0.08);
    }
  }
}

function makeHuman(templates, teamColor, sitting) {
  const group = new THREE.Group();
  templates.forEach((template, index) => {
    const part = cloneSkinned(template);
    part.traverse((o) => {
      if (o.isMesh && o.material) {
        o.material = o.material.clone();
        if (index === 1 || index === 2) o.material.color.set(teamColor);
        o.castShadow = true;
        o.receiveShadow = true;
      }
    });
    applyHumanPose(part, sitting);
    group.add(part);
  });
  return group;
}

function placeHumans(scene, dims, templates) {
  const hw = dims.width / 2;
  const humans = [];
  for (const sign of [-1, 1]) {
    const z = sign * (hw + 3.35);
    const color = sign > 0 ? 0x1a3d9e : 0x9e1a1a;
    // 中线两侧的替补席。
    for (const x of [-5.55, -4.65, -3.75, -2.85, -1.95, 1.95, 2.85, 3.75, 4.65, 5.55]) {
      const h = makeHuman(templates, color, true);
      h.position.set(x, 0, z - sign * 0.02);
      h.rotation.y = sign > 0 ? Math.PI : 0;
      scene.add(h);
      humans.push(h);
    }
    // 中线位置的解说台。
    for (const x of [-0.85, -0.28, 0.28, 0.85]) {
      const h = makeHuman(templates, 0x2b3038, true);
      h.position.set(x, 0, z + sign * 0.58);
      h.rotation.y = sign > 0 ? Math.PI : 0;
      scene.add(h);
      humans.push(h);
    }
    // 替补席两端站立的人员，增加边线遮挡层次。
    for (const x of [-6.45, -4.15, 4.15, 6.45]) {
      const h = makeHuman(templates, color, false);
      h.position.set(x, 0, z + sign * 0.55);
      h.rotation.y = sign > 0 ? Math.PI : 0;
      scene.add(h);
      humans.push(h);
    }
  }
  return humans;
}

function placeHumanProxies(scene, dims) {
  const hw = dims.width / 2;
  const spots = [];
  for (const sign of [-1, 1]) {
    const z = sign * (hw + 3.35);
    const color = sign > 0 ? 0x1a3d9e : 0x9e1a1a;
    for (const x of [-5.55, -4.65, -3.75, -2.85, -1.95, 1.95, 2.85, 3.75, 4.65, 5.55]) {
      spots.push({ x, z: z - sign * 0.02, y: 0.55, color, sitting: true, sign });
    }
    for (const x of [-0.85, -0.28, 0.28, 0.85]) {
      spots.push({ x, z: z + sign * 0.58, y: 0.58, color: 0x2b3038, sitting: true, sign });
    }
    for (const x of [-6.45, -4.15, 4.15, 6.45]) {
      spots.push({ x, z: z + sign * 0.55, y: 0.85, color, sitting: false, sign });
    }
  }
  const bodies = new THREE.InstancedMesh(
    new THREE.CapsuleGeometry(0.18, 0.55, 4, 8),
    new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.9 }),
    spots.length,
  );
  const heads = new THREE.InstancedMesh(
    new THREE.SphereGeometry(0.12, 8, 6),
    new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.92 }),
    spots.length,
  );
  bodies.castShadow = bodies.receiveShadow = true;
  heads.castShadow = true;
  const dummy = new THREE.Object3D();
  spots.forEach((spot, i) => {
    dummy.position.set(spot.x, spot.y, spot.z);
    dummy.rotation.set(0, spot.sign > 0 ? Math.PI : 0, 0);
    dummy.scale.set(1, spot.sitting ? 0.75 : 1.25, 1);
    dummy.updateMatrix();
    bodies.setMatrixAt(i, dummy.matrix);
    bodies.setColorAt(i, new THREE.Color(spot.color));
    dummy.position.y = spot.sitting ? 1.02 : 1.48;
    dummy.scale.setScalar(1);
    dummy.updateMatrix();
    heads.setMatrixAt(i, dummy.matrix);
    heads.setColorAt(i, new THREE.Color(0x9b7657));
  });
  scene.add(bodies, heads);
  return [bodies, heads];
}

export async function createArena(canvas, dims, options = {}) {
  const renderer = new THREE.WebGPURenderer({ canvas, antialias: true, alpha: false });
  await renderer.init();
  renderer.setPixelRatio(1);
  renderer.setSize(canvas.clientWidth || 960, canvas.clientHeight || 540);
  renderer.shadowMap.enabled = options.shadows !== false;
  renderer.shadowMap.type = THREE.PCFShadowMap;
  renderer.debug.checkShaderErrors = true;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = options.exposure || 0.95;
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x111827);
  scene.fog = new THREE.FogExp2(0x111827, 0.004);
  const device = renderer.backend?.device;
  if (device) {
    device.addEventListener('uncapturederror', (e) => console.error('WebGPU uncaptured:', e.error?.message || e.error));
    device.lost.then((info) => console.error('WebGPU device lost:', info.message));
  }
  if (options.environment !== false) {
    const pmrem = new THREE.PMREMGenerator(renderer);
    scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  }

  const rng = rngFrom(options.seed || 1);
  const team = options.team || { name: 'Lakers', model: 'NBA', primary: '#552583', secondary: '#FDB927' };
  const courtLines = await buildCourt(scene, dims, team);
  buildHoop(scene, dims, -1, team);
  buildHoop(scene, dims, 1, team);
  buildBench(scene, dims, -1, team);
  buildBench(scene, dims, 1, team);
  buildStands(scene, dims, rng, team);
  const lights = buildLights(scene, options);

  const loader = new THREE.TextureLoader();
  const [albedo, normal] = await Promise.all([
    loader.loadAsync('/assets/ball_albedo.png'),
    loader.loadAsync('/assets/ball_normal.png'),
  ]);
  albedo.colorSpace = THREE.SRGBColorSpace;
  const ball = new THREE.Mesh(
    new THREE.SphereGeometry(0.12, 40, 24),
    new THREE.MeshPhysicalMaterial({ map: albedo, normalMap: normal, normalScale: new THREE.Vector2(0.38, 0.38), roughness: 0.62, metalness: 0, envMapIntensity: 0.32 }),
  );
  ball.castShadow = true;
  scene.add(ball);

  const camera = new THREE.PerspectiveCamera(42, 16 / 9, 0.1, 250);
  camera.position.set(18, 12, 18);
  camera.lookAt(0, 0, 0);
  const controls = new OrbitControls(camera, canvas);
  controls.target.set(0, 0.6, 0);
  controls.enableDamping = true;
  controls.maxPolarAngle = Math.PI * 0.48;
  controls.minDistance = 4;
  controls.maxDistance = 60;

  const templates = options.humans === true && !options.simpleHumans ? await loadHumanTemplates() : [];
  const humans = options.humans === true
    ? (options.simpleHumans ? placeHumanProxies(scene, dims) : placeHumans(scene, dims, templates))
    : [];

  return { renderer, scene, camera, controls, ball, humans, lights, dims, team, courtLines, captureMode: false };
}

export async function renderSemanticLabel(arena) {
  const { renderer, scene, courtLines } = arena;
  if (!arena._semanticKit) {
    const black = new THREE.MeshBasicMaterial({ color: 0x000000, toneMapped: false });
    const materials = new Map(courtLines.map((line) => [line,
      new THREE.MeshBasicMaterial({
        color: new THREE.Color().setRGB(...SEMANTIC_COLORS[line.userData.segmentationClass], THREE.SRGBColorSpace),
        toneMapped: false,
      })]));
    arena._semanticKit = { black, materials };
  }
  const { black, materials } = arena._semanticKit;
  const saved = [];
  const geometrySaved = [];
  for (const line of courtLines) {
    geometrySaved.push([line, line.geometry]);
    line.geometry = line.userData.semanticGeometry;
  }
  const background = scene.background;
  const fog = scene.fog;
  scene.traverse((object) => {
    if (!object.material) return;
    saved.push([object, object.material]);
    object.material = materials.get(object) || black;
  });
  scene.background = new THREE.Color(0x000000);
  scene.fog = null;
  const previousTarget = renderer.getRenderTarget();
  const previousShadowAuto = renderer.shadowMap.autoUpdate;
  renderer.shadowMap.autoUpdate = false;
  try {
    renderer.setRenderTarget(null);
    renderer.setClearColor(0x000000, 1);
    renderer.clear();
    renderer.render(scene, arena.camera);
    const device = renderer.backend?.device;
    if (device?.queue?.onSubmittedWorkDone) await device.queue.onSubmittedWorkDone();
    return renderer.domElement.toDataURL('image/png');
  } finally {
    renderer.setRenderTarget(previousTarget);
    renderer.shadowMap.autoUpdate = previousShadowAuto;
    for (const [object, material] of saved) object.material = material;
    for (const [line, geometry] of geometrySaved) line.geometry = geometry;
    scene.background = background;
    scene.fog = fog;
  }
}

export function setBroadcastCamera(arena, spec) {
  const cam = arena.camera;
  const w = arena.renderer.domElement.width;
  const h = arena.renderer.domElement.height;
  const focal = spec.focal_length || 1000;
  cam.fov = THREE.MathUtils.radToDeg(2 * Math.atan(h / (2 * focal)));
  cam.aspect = w / h;
  cam.position.set(spec.pos[0], spec.pos[2], -spec.pos[1]);
  cam.up.set(spec.up[0], spec.up[2], -spec.up[1]).normalize();
  cam.lookAt(spec.look_at[0], spec.look_at[2], -spec.look_at[1]);
  cam.updateProjectionMatrix();
  cam.updateMatrixWorld();
}

export function cameraWorldToPx(arena) {
  const p = new THREE.Matrix4().multiplyMatrices(arena.camera.projectionMatrix, arena.camera.matrixWorldInverse).elements;
  const w = arena.renderer.domElement.width;
  const h = arena.renderer.domElement.height;
  const row = (component) => [p[component], -p[component + 8], p[component + 12]];
  const r0 = row(0);
  const r1 = row(1);
  const r3 = row(3);
  return [
    r0.map((v, i) => w * 0.5 * (v + r3[i])),
    r1.map((v, i) => h * 0.5 * (-v + r3[i])),
    r3,
  ];
}

export function ballBBox(arena) {
  const ball = arena.ball;
  if (!ball.visible) return null;
  const w = arena.renderer.domElement.width;
  const h = arena.renderer.domElement.height;
  const p = ball.position.clone().project(arena.camera);
  if (p.z < -1 || p.z > 1) return null;
  const depth = arena.camera.position.distanceTo(ball.position);
  const f = arena.camera.projectionMatrix.elements[5];
  const r = Math.abs(f) * h * 0.5 * 0.12 / Math.max(depth, 0.1);
  const cx = (p.x * 0.5 + 0.5) * w;
  const cy = (1 - (p.y * 0.5 + 0.5)) * h;
  const x0 = Math.max(0, cx - r);
  const x1 = Math.min(w - 1, cx + r);
  const y0 = Math.max(0, cy - r);
  const y1 = Math.min(h - 1, cy + r);
  if (x1 - x0 < 2 || y1 - y0 < 2 || cx < 0 || cx > w || cy < 0 || cy > h) return null;
  return [x0, y0, x1, y1];
}

export { THREE };
