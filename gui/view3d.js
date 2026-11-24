/**
 * 3D Scene Renderer for Basketball Court + Players
 * 
 * Uses Three.js to render:
 * - 3D court model with proper dimensions (FIBA/NBA)
 * - Player skeletons lifted from 2D pose to 3D world coordinates
 * - Real-time streaming of skeleton data from backend
 */

// Global state
let scene, camera, renderer;
let courtMesh, playerSkeletons = {};
let is3DEnabled = false;
let animationId;

// COCO keypoint connections for skeleton rendering
const SKELETON_CONNECTIONS = [
  [0, 1], [0, 2], [1, 3], [2, 4],  // Head
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],  // Arms
  [5, 11], [6, 12], [11, 12],  // Torso
  [11, 13], [13, 15], [12, 14], [14, 16],  // Legs
];

const PLAYER_COLORS = [
  0xff6464,  // Red
  0x64ff64,  // Green
  0x6464ff,  // Blue
  0xffff64,  // Yellow
  0xff64ff,  // Magenta
  0x64ffff,  // Cyan
];

// Initialize 3D scene
function init3D() {
  const canvas = document.getElementById('view3d');
  if (!canvas) return;
  
  // Create scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1a2e);
  
  // Camera (broadcast view)
  const aspect = canvas.clientWidth / canvas.clientHeight;
  camera = new THREE.PerspectiveCamera(60, aspect, 0.1, 1000);
  camera.position.set(0, -8, 5);
  camera.lookAt(0, 0, 0);
  
  // Renderer
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setSize(canvas.clientWidth, canvas.clientHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  
  // Lighting
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
  scene.add(ambientLight);
  
  const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
  directionalLight.position.set(5, 5, 10);
  scene.add(directionalLight);
  
  // Build court
  buildCourt();
  
  // Start animation loop
  animate();
}

// Build 3D court model
function buildCourt() {
  // FIBA court dimensions (meters)
  const courtLength = 28;
  const courtWidth = 15;
  const lineHeight = 0.02;
  
  // Floor
  const floorGeometry = new THREE.PlaneGeometry(courtLength, courtWidth);
  const floorMaterial = new THREE.MeshStandardMaterial({ 
    color: 0xd4a574,  // Maple wood
    roughness: 0.8,
    metalness: 0.1
  });
  const floor = new THREE.Mesh(floorGeometry, floorMaterial);
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = 0;
  scene.add(floor);
  
  // Court lines material
  const lineMaterial = new THREE.LineBasicMaterial({ color: 0xffffff, linewidth: 2 });
  
  // Boundary lines
  const boundaryPoints = [
    new THREE.Vector3(-courtLength/2, lineHeight, -courtWidth/2),
    new THREE.Vector3(courtLength/2, lineHeight, -courtWidth/2),
    new THREE.Vector3(courtLength/2, lineHeight, courtWidth/2),
    new THREE.Vector3(-courtLength/2, lineHeight, courtWidth/2),
    new THREE.Vector3(-courtLength/2, lineHeight, -courtWidth/2),
  ];
  const boundaryGeometry = new THREE.BufferGeometry().setFromPoints(boundaryPoints);
  const boundaryLine = new THREE.Line(boundaryGeometry, lineMaterial);
  scene.add(boundaryLine);
  
  // Center line
  const centerLinePoints = [
    new THREE.Vector3(0, lineHeight, -courtWidth/2),
    new THREE.Vector3(0, lineHeight, courtWidth/2),
  ];
  const centerLineGeometry = new THREE.BufferGeometry().setFromPoints(centerLinePoints);
  const centerLine = new THREE.Line(centerLineGeometry, lineMaterial);
  scene.add(centerLine);
  
  // Center circle
  const centerCircleGeometry = new THREE.RingGeometry(1.75, 1.85, 32);
  const centerCircleMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide });
  const centerCircle = new THREE.Mesh(centerCircleGeometry, centerCircleMaterial);
  centerCircle.rotation.x = -Math.PI / 2;
  centerCircle.position.y = lineHeight;
  scene.add(centerCircle);
  
  // Three-point arcs (simplified)
  const threePtRadius = 6.75;
  const arcGeometry = new THREE.RingGeometry(threePtRadius - 0.05, threePtRadius + 0.05, 32, 1, 0, Math.PI);
  const arcMaterial = new THREE.MeshBasicMaterial({ color: 0xffdc3c, side: THREE.DoubleSide });
  
  // Team A three-point arc
  const arcA = new THREE.Mesh(arcGeometry, arcMaterial);
  arcA.rotation.x = -Math.PI / 2;
  arcA.rotation.z = Math.PI / 2;
  arcA.position.set(-courtLength/2 + 1.575, lineHeight, 0);
  scene.add(arcA);
  
  // Team B three-point arc
  const arcB = new THREE.Mesh(arcGeometry, arcMaterial);
  arcB.rotation.x = -Math.PI / 2;
  arcB.rotation.z = -Math.PI / 2;
  arcB.position.set(courtLength/2 - 1.575, lineHeight, 0);
  scene.add(arcB);
  
  // Paint areas (rectangles)
  const paintWidth = 4.9;
  const paintLength = 5.8;
  
  // Team A paint
  const paintAGeometry = new THREE.PlaneGeometry(paintLength, paintWidth);
  const paintAMaterial = new THREE.MeshBasicMaterial({ color: 0x50dc50, opacity: 0.3, transparent: true });
  const paintA = new THREE.Mesh(paintAGeometry, paintAMaterial);
  paintA.rotation.x = -Math.PI / 2;
  paintA.position.set(-courtLength/2 + paintLength/2, lineHeight, 0);
  scene.add(paintA);
  
  // Team B paint
  const paintBGeometry = new THREE.PlaneGeometry(paintLength, paintWidth);
  const paintBMaterial = new THREE.MeshBasicMaterial({ color: 0x50dc50, opacity: 0.3, transparent: true });
  const paintB = new THREE.Mesh(paintBGeometry, paintBMaterial);
  paintB.rotation.x = -Math.PI / 2;
  paintB.position.set(courtLength/2 - paintLength/2, lineHeight, 0);
  scene.add(paintB);
  
  courtMesh = { floor, boundaryLine, centerLine, centerCircle };
}

// Update player skeletons from backend data
function updatePlayers(players3D) {
  if (!players3D || players3D.length === 0) return;
  
  // Remove old skeletons
  Object.values(playerSkeletons).forEach(skel => {
    scene.remove(skel.group);
  });
  playerSkeletons = {};
  
  // Add new skeletons
  players3D.forEach((player, idx) => {
    const color = PLAYER_COLORS[idx % PLAYER_COLORS.length];
    const skelGroup = createPlayerSkeleton(player, color);
    playerSkeletons[player.track_id] = { group: skelGroup, color };
    scene.add(skelGroup);
  });
}

// Create a single player skeleton mesh
function createPlayerSkeleton(player, color) {
  const group = new THREE.Group();
  
  // Convert kpts_3d from array format
  const kpts = player.kpts_3d; // [[x,y,z], ...]
  
  // Create joint spheres
  const jointMaterial = new THREE.MeshStandardMaterial({ color });
  kpts.forEach((kpt, i) => {
    if (kpt[2] < 0.1) return; // Skip low-confidence keypoints
    
    const sphereGeometry = new THREE.SphereGeometry(0.08, 8, 8);
    const sphere = new THREE.Mesh(sphereGeometry, jointMaterial);
    sphere.position.set(kpt[0], kpt[2], kpt[1]); // Swap Y/Z for Three.js coords
    group.add(sphere);
  });
  
  // Create bone lines
  const lineMaterial = new THREE.LineBasicMaterial({ color, linewidth: 3 });
  SKELETON_CONNECTIONS.forEach(([i, j]) => {
    if (kpts[i][2] < 0.1 || kpts[j][2] < 0.1) return;
    
    const points = [
      new THREE.Vector3(kpts[i][0], kpts[i][2], kpts[i][1]),
      new THREE.Vector3(kpts[j][0], kpts[j][2], kpts[j][1]),
    ];
    const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);
    const line = new THREE.Line(lineGeometry, lineMaterial);
    group.add(line);
  });
  
  // Add bounding box
  const bbox = player.bbox_3d; // [x0, y0, x1, y1]
  const height = player.height_m || 1.85;
  const boxGeometry = new THREE.BoxGeometry(
    bbox[2] - bbox[0],
    height,
    bbox[3] - bbox[1]
  );
  const boxMaterial = new THREE.MeshBasicMaterial({ 
    color, 
    wireframe: true,
    transparent: true,
    opacity: 0.3
  });
  const box = new THREE.Mesh(boxGeometry, boxMaterial);
  box.position.set(
    (bbox[0] + bbox[2]) / 2,
    height / 2,
    (bbox[1] + bbox[3]) / 2
  );
  group.add(box);
  
  return group;
}

// Animation loop
function animate() {
  animationId = requestAnimationFrame(animate);
  
  if (renderer && scene && camera) {
    renderer.render(scene, camera);
  }
}

// Toggle between 2D image and 3D view
function toggle3DView() {
  const img = document.getElementById('frame');
  const canvas = document.getElementById('view3d');
  const btn = document.getElementById('btn-toggle-3d');
  
  if (!img || !canvas || !btn) return;
  
  is3DEnabled = !is3DEnabled;
  
  if (is3DEnabled) {
    img.style.display = 'none';
    canvas.style.display = 'block';
    btn.classList.add('active');
    btn.textContent = '2D';
    
    // Initialize 3D if needed
    if (!scene) {
      init3D();
    }
  } else {
    img.style.display = 'block';
    canvas.style.display = 'none';
    btn.classList.remove('active');
    btn.textContent = '3D';
  }
}

// Fetch 3D skeleton data from backend
async function fetchSkeletonData(frameNo) {
  if (!S.vid || !is3DEnabled) return;
  
  try {
    // Request frame analysis with 3D data
    const response = await fetch(`/api/videos/${S.vid}/frames?from=${frameNo}&to=${frameNo}&limit=1`);
    const data = await response.json();
    
    if (data.frames && data.frames.length > 0) {
      // Note: Current API doesn't return 3D skeletons yet
      // This is a placeholder for future implementation
      console.log('[3D] Frame', frameNo, 'has', data.frames[0].objects.length, 'objects');
    }
  } catch (err) {
    console.error('[3D] Failed to fetch skeleton data:', err);
  }
}

// Export functions for use in app.js
window.toggle3DView = toggle3DView;
window.updatePlayers = updatePlayers;
window.fetchSkeletonData = fetchSkeletonData;

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
  const btn = document.getElementById('btn-toggle-3d');
  if (btn) {
    btn.addEventListener('click', toggle3DView);
  }
});

console.log('[3D] View3D module loaded');
