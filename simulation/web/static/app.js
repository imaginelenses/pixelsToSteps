const stateTargets = {
  groundTruth: document.getElementById('groundTruthState'),
  estimated: document.getElementById('estimatedState'),
  control: document.getElementById('controlState'),
};

const observerSelect = document.getElementById('observerSelect');
const runStatus = document.getElementById('runStatus');
const startButton = document.getElementById('startButton');
const restartButton = document.getElementById('restartButton');
const stopButton = document.getElementById('stopButton');
const stepButton = document.getElementById('stepButton');
const resetForm = document.getElementById('resetForm');
const configForm = document.getElementById('configForm');

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

function formatNumber(value, digits = 4) {
  if (value === null || value === undefined) {
    return 'n/a';
  }
  return Number(value).toFixed(digits);
}

async function sendJson(url, payload = undefined) {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: payload ? JSON.stringify(payload) : undefined,
  });
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

function updateState(state) {
  runStatus.textContent = state.running ? 'Running' : state.capture_end_reason;
  runStatus.classList.toggle('live', Boolean(state.running));

  renderMetrics(stateTargets.groundTruth, [
    ['cart_position_m', formatNumber(state.ground_truth.cart_position_m)],
    ['cart_velocity_m_s', formatNumber(state.ground_truth.cart_velocity_m_s)],
    ['pole_angle_deg', formatNumber(state.ground_truth.pole_angle_deg, 3)],
    ['pole_angle_rate_deg_s', formatNumber(state.ground_truth.pole_angle_rate_deg_s, 3)],
    ['frame_index', String(state.frame_index)],
    ['simulated_time_s', formatNumber(state.simulated_time_s, 3)],
  ]);

  const estimated = state.estimated || {};
  renderMetrics(stateTargets.estimated, [
    ['cart_position_m', formatNumber(estimated.cart_position_m)],
    ['cart_velocity_m_s', formatNumber(estimated.cart_velocity_m_s)],
    ['pole_angle_deg', formatNumber(estimated.pole_angle_deg, 3)],
    ['pole_angle_rate_deg_s', formatNumber(estimated.pole_angle_rate_deg_s, 3)],
    ['theta_pixel_deg', formatNumber(state.theta_pixel_deg, 3)],
  ]);

  renderMetrics(stateTargets.control, [
    ['raw_force_n', formatNumber(state.control.raw_force_n, 3)],
    ['commanded_force_n', formatNumber(state.control.commanded_force_n, 3)],
    ['applied_force_n', formatNumber(state.control.applied_force_n, 3)],
    ['observer', state.observer ? state.observer.path : 'teacher-state-feedback'],
    ['observer_target', state.observer ? state.observer.target : 'n/a'],
    ['theta_blend_weight', state.observer ? formatNumber(state.observer.theta_pixel_blend_weight, 2) : 'n/a'],
    ['terminated', String(state.terminated)],
  ]);
}

function populateConfigForms(state, observers) {
  observerSelect.innerHTML = '';
  for (const name of observers) {
    const option = document.createElement('option');
    option.value = name;
    option.textContent = name;
    if (state.observer && state.observer.path === name) {
      option.selected = true;
    }
    observerSelect.append(option);
  }
  if (!state.observer && observers.length > 0) {
    observerSelect.value = observers[0];
  }

  for (const element of configForm.elements) {
    if (!(element instanceof HTMLInputElement)) {
      continue;
    }
    if (state.config[element.name] !== undefined) {
      element.value = state.config[element.name];
    }
  }

  for (const element of resetForm.elements) {
    if (element instanceof HTMLInputElement && element.type === 'checkbox') {
      if (element.name === 'use_image_controller') {
        element.checked = Boolean(state.observer);
      }
    }
  }
}

async function refreshState() {
  const response = await fetch('/api/state');
  const state = await response.json();
  updateState(state);
}

startButton.addEventListener('click', async () => {
  updateState(await sendJson('/api/start'));
});

restartButton.addEventListener('click', async () => {
  updateState(await sendJson('/api/restart'));
});

stopButton.addEventListener('click', async () => {
  updateState(await sendJson('/api/stop'));
});

stepButton.addEventListener('click', async () => {
  updateState(await sendJson('/api/step'));
});

resetForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const formData = new FormData(resetForm);
  const payload = {
    cart_position_m: Number(formData.get('cart_position_m')),
    cart_velocity_m_s: Number(formData.get('cart_velocity_m_s')),
    pole_angle_deg: Number(formData.get('pole_angle_deg')),
    pole_angle_rate_deg_s: Number(formData.get('pole_angle_rate_deg_s')),
    observer_json_path: String(formData.get('observer_json_path') || ''),
    use_image_controller: document.getElementById('useImageController').checked,
    auto_start: document.getElementById('autoStart').checked,
  };
  updateState(await sendJson('/api/reset', payload));
});

configForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const formData = new FormData(configForm);
  const payload = {
    sample_time_s: Number(formData.get('sample_time_s')),
    frame_width_px: Number(formData.get('frame_width_px')),
    frame_height_px: Number(formData.get('frame_height_px')),
    true_masspole_scale: Number(formData.get('true_masspole_scale')),
    true_half_pole_length_scale: Number(formData.get('true_half_pole_length_scale')),
    process_noise_std_n: Number(formData.get('process_noise_std_n')),
    max_force_n: 10.0,
    control_penalty_r: 1e-6,
    true_gravity_scale: 1.0,
    true_masscart_scale: 1.0,
    seed_truth_from_initial_state: true,
  };
  updateState(await sendJson('/api/config', payload));
});

async function boot() {
  const response = await fetch('/api/config');
  const data = await response.json();
  populateConfigForms(data.state, data.available_observers);
  updateState(data.state);
  setInterval(() => {
    refreshState().catch((error) => console.error(error));
  }, 200);
}

boot().catch((error) => console.error(error));
