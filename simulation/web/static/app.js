let defaultObserver = '';

const angleSlider = document.getElementById('angleSlider');
const posSlider   = document.getElementById('posSlider');
const angleValue  = document.getElementById('angleValue');
const posValue    = document.getElementById('posValue');
const startButton = document.getElementById('startButton');
const resetButton = document.getElementById('resetButton');

angleSlider.addEventListener('input', () => {
  angleValue.textContent = Number(angleSlider.value).toFixed(1);
});

posSlider.addEventListener('input', () => {
  posValue.textContent = Number(posSlider.value).toFixed(2);
});

function renderMetrics(target, entries) {
  target.innerHTML = '';
  for (const [label, value] of entries) {
    const dt = document.createElement('dt');
    dt.textContent = label;
    const dd = document.createElement('dd');
    dd.textContent = value;
    target.append(dt, dd);
  }
}

function fmt(value, digits = 4) {
  if (value === null || value === undefined) return 'n/a';
  return Number(value).toFixed(digits);
}

function updateState(state) {
  const gt  = state.ground_truth || {};
  const est = state.estimated    || {};

  renderMetrics(document.getElementById('groundTruthState'), [
    ['Cart position (m)',   fmt(gt.cart_position_m)],
    ['Cart velocity (m/s)', fmt(gt.cart_velocity_m_s)],
    ['Pole angle (°)',      fmt(gt.pole_angle_deg, 3)],
    ['Pole rate (°/s)',     fmt(gt.pole_angle_rate_deg_s, 3)],
  ]);

  renderMetrics(document.getElementById('estimatedState'), [
    ['Cart position (m)',   fmt(est.cart_position_m)],
    ['Cart velocity (m/s)', fmt(est.cart_velocity_m_s)],
    ['Pole angle (°)',      fmt(est.pole_angle_deg, 3)],
    ['Pole rate (°/s)',     fmt(est.pole_angle_rate_deg_s, 3)],
  ]);
}

async function sendJson(url, payload) {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error(`Request failed: ${response.status}`);
  return response.json();
}

function buildResetPayload(autoStart) {
  return {
    cart_position_m:        Number(posSlider.value),
    cart_velocity_m_s:      0,
    pole_angle_deg:         Number(angleSlider.value),
    pole_angle_rate_deg_s:  0,
    use_image_controller:   true,
    observer_json_path:     defaultObserver,
    auto_start:             autoStart,
  };
}

startButton.addEventListener('click', async () => {
  updateState(await sendJson('/api/reset', buildResetPayload(true)));
});

resetButton.addEventListener('click', async () => {
  updateState(await sendJson('/api/reset', buildResetPayload(false)));
});

async function boot() {
  const response = await fetch('/api/config');
  const data = await response.json();

  const observers = data.available_observers || [];
  defaultObserver = observers.find(o => o.includes('40demos')) || observers[0] || '';

  updateState(data.state);

  setInterval(async () => {
    try {
      const resp = await fetch('/api/state');
      updateState(await resp.json());
    } catch (e) {
      console.error(e);
    }
  }, 200);
}

boot().catch(console.error);
