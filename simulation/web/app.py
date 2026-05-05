from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import sys

BASE_DIR = Path(__file__).resolve().parents[1]
PYTHON_DIR = BASE_DIR / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from live_cartpole import LiveCartPoleSimulator, ResetRequest, SimulationConfig  # noqa: E402


class SimulationConfigPayload(BaseModel):
    sample_time_s: float = Field(default=1.0 / 60.0, gt=0.0)
    max_force_n: float = Field(default=10.0, gt=0.0)
    control_penalty_r: float = Field(default=1e-6, gt=0.0)
    frame_width_px: int = Field(default=160, gt=0)
    frame_height_px: int = Field(default=125, gt=0)
    true_gravity_scale: float = Field(default=1.0, gt=0.0)
    true_masscart_scale: float = Field(default=1.0, gt=0.0)
    true_masspole_scale: float = Field(default=1.1, gt=0.0)
    true_half_pole_length_scale: float = Field(default=0.9, gt=0.0)
    process_noise_std_n: float = Field(default=0.0, ge=0.0)
    seed_truth_from_initial_state: bool = True


class ResetPayload(BaseModel):
    cart_position_m: float = 0.0
    cart_velocity_m_s: float = 0.0
    pole_angle_deg: float = 12.0
    pole_angle_rate_deg_s: float = 0.0
    use_image_controller: bool = True
    observer_json_path: Optional[str] = None
    auto_start: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    simulator = LiveCartPoleSimulator(BASE_DIR)
    app.state.simulator = simulator
    yield
    simulator.close()


app = FastAPI(title="CartPole Image Control", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=Path(__file__).resolve().parent / "static"), name="static")


def get_simulator() -> LiveCartPoleSimulator:
    return app.state.simulator


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(__file__).resolve().parent / "static" / "index.html")


@app.get("/api/config")
def get_config() -> JSONResponse:
    simulator = get_simulator()
    return JSONResponse(
        {
            "state": simulator.snapshot(),
            "available_observers": simulator.available_observers(),
        }
    )


@app.get("/api/state")
def get_state() -> JSONResponse:
    return JSONResponse(get_simulator().snapshot())


@app.get("/api/observers")
def get_observers() -> JSONResponse:
    simulator = get_simulator()
    return JSONResponse({"observers": simulator.available_observers()})


@app.post("/api/config")
def post_config(payload: SimulationConfigPayload) -> JSONResponse:
    simulator = get_simulator()
    simulator.reconfigure(SimulationConfig(**payload.model_dump()))
    return JSONResponse(simulator.snapshot())


@app.post("/api/reset")
def post_reset(payload: ResetPayload) -> JSONResponse:
    simulator = get_simulator()
    observer_path = payload.observer_json_path
    if observer_path and not Path(observer_path).is_absolute():
        observer_path = str((PYTHON_DIR / observer_path).resolve())
    simulator.reset(
        ResetRequest(
            cart_position_m=payload.cart_position_m,
            cart_velocity_m_s=payload.cart_velocity_m_s,
            pole_angle_deg=payload.pole_angle_deg,
            pole_angle_rate_deg_s=payload.pole_angle_rate_deg_s,
            use_image_controller=payload.use_image_controller,
            observer_json_path=observer_path,
            auto_start=payload.auto_start,
        )
    )
    return JSONResponse(simulator.snapshot())


@app.post("/api/start")
def post_start() -> JSONResponse:
    simulator = get_simulator()
    simulator.start()
    return JSONResponse(simulator.snapshot())


@app.post("/api/restart")
def post_restart() -> JSONResponse:
    simulator = get_simulator()
    simulator.restart()
    return JSONResponse(simulator.snapshot())


@app.post("/api/stop")
def post_stop() -> JSONResponse:
    simulator = get_simulator()
    simulator.stop()
    return JSONResponse(simulator.snapshot())


@app.post("/api/step")
def post_step() -> JSONResponse:
    simulator = get_simulator()
    simulator.step_once()
    return JSONResponse(simulator.snapshot())


@app.get("/stream/{stream_name}")
def stream_frames(stream_name: Literal["rgb", "binary"]) -> StreamingResponse:
    simulator = get_simulator()

    def frame_generator():
        last_frame_counter = -1
        while True:
            last_frame_counter = simulator.wait_for_frame(last_frame_counter, timeout_s=1.0)
            frame_bytes = simulator.get_rgb_jpeg() if stream_name == "rgb" else simulator.get_binary_jpeg()
            if not frame_bytes:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")
