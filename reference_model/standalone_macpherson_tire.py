"""Standalone nonlinear MacPherson suspension model with a tire.

No project-local modules are imported. The model requires NumPy; the optional
2D animation requires Matplotlib. Run this file to save simulation data as CSV
and an interactive HTML demo. Displacements are in metres, velocities in m/s,
and forces are deviations from static equilibrium in newtons.

State variables are deviations from the static flat-road equilibrium:

    x = [z_s, z_s_dot, z_u, z_u_dot]

where z_s is the sprung-mass vertical displacement and z_u is the wheel-centre /
unsprung-mass vertical displacement. The tire is a Kelvin-Voigt vertical element
against the road height obtained with an envelope contact model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


Array = np.ndarray


# ---------------------------------------------------------------------------
# Vector helpers and MacPherson kinematics
# ---------------------------------------------------------------------------


def unit(vector: Array) -> Array:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return vector / norm


def skew(vector: Array) -> Array:
    vx, vy, vz = vector
    return np.array([[0.0, -vz, vy], [vz, 0.0, -vx], [-vy, vx, 0.0]])


def rodrigues_from_to(source: Array, target: Array) -> Array:
    """Minimal rotation matrix that maps ``source`` to ``target``."""

    source = unit(source)
    target = unit(target)
    axis_scaled = np.cross(source, target)
    cosine = float(np.dot(source, target))

    if cosine > 1.0 - 1.0e-12:
        return np.eye(3)
    if cosine < -1.0 + 1.0e-12:
        helper = (
            np.array([1.0, 0.0, 0.0])
            if abs(source[0]) < 0.9
            else np.array([0.0, 1.0, 0.0])
        )
        axis = unit(np.cross(source, helper))
        cross_matrix = skew(axis)
        return np.eye(3) + 2.0 * cross_matrix @ cross_matrix

    cross_matrix = skew(axis_scaled)
    return np.eye(3) + cross_matrix + cross_matrix @ cross_matrix / (1.0 + cosine)


@dataclass(frozen=True)
class KinematicPoint:
    wheel_travel: float
    spring_compression: float
    damper_stroke: float
    spring_ratio: float
    damper_ratio: float


@dataclass(frozen=True)
class KinematicMap:
    wheel_travel: Array
    spring_compression: Array
    damper_stroke: Array
    spring_ratio: Array
    damper_ratio: Array

    @classmethod
    def bmw_e90_front(
        cls, phi_min: float = -0.35, phi_max: float = 0.35, count: int = 1001
    ) -> "KinematicMap":
        """Build a BMW E90 front MacPherson kinematic map.

        Hardpoints are copied into this file to keep the model standalone.
        Coordinates are SI metres in the suspension frame: X forward, Y left, Z up.
        """

        geom = BMWFrontMacPhersonGeometry()
        rows = [geom.sample(phi) for phi in np.linspace(phi_min, phi_max, count)]
        rows = sorted(rows, key=lambda row: row["wheel_travel"])
        wheel = np.array([row["wheel_travel"] for row in rows])
        spring = np.array([row["spring_compression"] for row in rows])
        damper = np.array([row["damper_stroke"] for row in rows])
        spring_ratio = np.gradient(spring, wheel)
        damper_ratio = np.gradient(damper, wheel)
        return cls(wheel, spring, damper, spring_ratio, damper_ratio)

    def sample(self, wheel_travel: float) -> KinematicPoint:
        return KinematicPoint(
            wheel_travel=float(wheel_travel),
            spring_compression=float(
                np.interp(wheel_travel, self.wheel_travel, self.spring_compression)
            ),
            damper_stroke=float(
                np.interp(wheel_travel, self.wheel_travel, self.damper_stroke)
            ),
            spring_ratio=float(
                np.interp(wheel_travel, self.wheel_travel, self.spring_ratio)
            ),
            damper_ratio=float(
                np.interp(wheel_travel, self.wheel_travel, self.damper_ratio)
            ),
        )


@dataclass(frozen=True)
class BMWFrontMacPhersonGeometry:
    """BMW E90 front MacPherson hardpoints used by the kinematic map."""

    a: Array = field(default_factory=lambda: np.array([-0.08382, 0.54102, 0.46863]))
    b0: Array = field(default_factory=lambda: np.array([0.02794, 0.66294, -0.10414]))
    p1: Array = field(default_factory=lambda: np.array([0.27051, 0.34544, -0.05969]))
    p2: Array = field(default_factory=lambda: np.array([-0.04318, 0.37338, -0.12573]))
    cd0: Array = field(default_factory=lambda: np.array([-0.00508, 0.61976, -0.00127]))
    wheel0: Array = field(default_factory=lambda: np.array([0.0, 0.750062, 0.0]))
    wheel_axis_left: Array = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))

    def __post_init__(self) -> None:
        object.__setattr__(self, "r0", self._initial_orientation())
        object.__setattr__(self, "s0", unit(self.a - self.b0))
        object.__setattr__(self, "circle", self._lower_arm_circle())
        object.__setattr__(self, "rho_d", self.r0.T @ (self.cd0 - self.b0))
        object.__setattr__(self, "rho_w", self.r0.T @ (self.wheel0 - self.b0))
        object.__setattr__(
            self, "damper_length0", float(np.linalg.norm(self.cd0 - self.a))
        )

    def _initial_orientation(self) -> Array:
        s0 = unit(self.a - self.b0)
        q0 = unit(self.wheel_axis_left - np.dot(self.wheel_axis_left, s0) * s0)
        t0 = np.cross(s0, q0)
        return np.column_stack((q0, t0, s0))

    def _lower_arm_circle(self) -> tuple[Array, float, Array, Array]:
        e12 = unit(self.p2 - self.p1)
        distance = float(np.linalg.norm(self.p2 - self.p1))
        l1 = float(np.linalg.norm(self.b0 - self.p1))
        l2 = float(np.linalg.norm(self.b0 - self.p2))
        a_len = (l1 * l1 - l2 * l2 + distance * distance) / (2.0 * distance)
        center = self.p1 + a_len * e12
        radius = math.sqrt(max(l1 * l1 - a_len * a_len, 0.0))
        n1 = unit(self.b0 - center)
        n2 = np.cross(e12, n1)
        return center, radius, n1, n2

    def sample(self, phi: float) -> dict[str, float]:
        center, radius, n1, n2 = self.circle
        b = center + radius * (n1 * math.cos(phi) + n2 * math.sin(phi))
        s = unit(self.a - b)
        rotation = rodrigues_from_to(self.s0, s) @ self.r0
        cd = b + rotation @ self.rho_d
        wheel = b + rotation @ self.rho_w

        length = float(np.linalg.norm(cd - self.a))
        stroke = self.damper_length0 - length
        return {
            "wheel_travel": float(wheel[2] - self.wheel0[2]),
            "spring_compression": stroke,
            "damper_stroke": stroke,
        }


# ---------------------------------------------------------------------------
# Spring, shock absorber and tire
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProgressiveSpring:
    k: float = 35_000.0
    k_ex: float = 60.0
    beta: float = 30.0
    preload: float = 7_600.0
    exp_arg_max: float = 8.0

    def force(self, compression: float) -> float:
        arg = min(self.beta * float(compression), self.exp_arg_max)
        return self.preload + self.k * compression + self.k_ex * (math.exp(arg) - 1.0)


@dataclass(frozen=True)
class ValveParams:
    leak_b1: float = 0.0
    leak_b2: float = 0.0
    port_b1: float = 0.0
    port_b2: float = 0.0
    dp0: float = 0.0
    k_spring: float = 1.0e16

    def pressure_drop(self, flow: float) -> float:
        q = abs(float(flow))
        dp_parallel = self._parallel_dp(q)
        dp_port = self.port_b1 * q + self.port_b2 * q * q
        return math.copysign(dp_parallel + dp_port, flow)

    def _leak_flow(self, dp: float) -> float:
        if dp <= 0.0:
            return 0.0
        if self.leak_b2 <= 0.0:
            return dp / self.leak_b1 if self.leak_b1 > 0.0 else 0.0
        return (
            -self.leak_b1
            + math.sqrt(self.leak_b1 * self.leak_b1 + 4.0 * self.leak_b2 * dp)
        ) / (2.0 * self.leak_b2)

    def _blowoff_flow(self, dp: float) -> float:
        if dp < self.dp0:
            return 0.0
        return (dp - self.dp0) * dp / self.k_spring

    def _parallel_dp(self, flow: float) -> float:
        if flow <= 0.0:
            return 0.0

        def residual(dp: float) -> float:
            return self._leak_flow(dp) + self._blowoff_flow(dp) - flow

        lo, hi = 0.0, 1.0e5
        for _ in range(200):
            if residual(hi) >= 0.0:
                break
            hi *= 2.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if residual(mid) > 0.0:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)


def turbulent_b2(orifice_area: float, c_d: float = 0.7, rho: float = 850.0) -> float:
    return rho / (2.0 * (c_d * orifice_area) ** 2)


@dataclass(frozen=True)
class TwinTubeShock:
    area_piston: float = 15.0e-4
    area_rod: float = 6.0e-4
    foot_valve: ValveParams = field(default_factory=ValveParams)
    piston_valve: ValveParams = field(default_factory=ValveParams)
    friction_coulomb: float = 40.0
    friction_viscous: float = 200.0
    friction_v_eps: float = 1.0e-2
    gas_force: float = 0.0
    oil_efficiency: float = 1.0
    gas_efficiency: float = 1.0

    @property
    def area_annular(self) -> float:
        return self.area_piston - self.area_rod

    @classmethod
    def default(cls) -> "TwinTubeShock":
        rho = 850.0
        cd = 0.7
        piston = ValveParams(
            leak_b1=5.0e8,
            leak_b2=turbulent_b2(2.2e-6, cd, rho),
            port_b2=turbulent_b2(50.0e-6, cd, rho),
            dp0=0.5e6,
            k_spring=6.4e15,
        )
        foot = ValveParams(
            leak_b1=5.0e8,
            leak_b2=turbulent_b2(2.4e-6, cd, rho),
            port_b2=turbulent_b2(50.0e-6, cd, rho),
            dp0=0.4e6,
            k_spring=5.0e15,
        )
        gas_force = (0.6e6 - 1.013e5) * 6.0e-4
        return cls(foot_valve=foot, piston_valve=piston, gas_force=gas_force)

    def force(self, compression_velocity: float) -> float:
        """Axial shock force, positive in suspension compression support."""

        vel = -float(compression_velocity)  # TSDA convention: positive extension
        hydraulic = self._hydraulic_force_tsda(vel)
        friction = self._friction_force_tsda(vel)
        gas = self.gas_efficiency * self.gas_force
        return hydraulic + friction + gas

    def _hydraulic_force_tsda(self, vel: float) -> float:
        if vel == 0.0:
            return 0.0
        if vel > 0.0:
            area = self.area_annular
            pressure = self.piston_valve.pressure_drop(area * vel)
            force = -pressure * area
        else:
            area = self.area_rod
            pressure = self.foot_valve.pressure_drop(area * (-vel))
            force = pressure * area
        return self.oil_efficiency * force

    def _friction_force_tsda(self, vel: float) -> float:
        coulomb = self.friction_coulomb * math.tanh(vel / self.friction_v_eps)
        viscous = self.friction_viscous * vel
        return -(coulomb + viscous)


@dataclass(frozen=True)
class TireParams:
    radius: float = 0.31785
    stiffness: float = 180_000.0
    damping: float = 1_500.0

    def force(self, compression: float, compression_velocity: float) -> float:
        raw = self.stiffness * compression + self.damping * compression_velocity
        if compression <= 0.0 and raw <= 0.0:
            return 0.0
        return max(raw, 0.0)


# ---------------------------------------------------------------------------
# Road profile and envelope contact
# ---------------------------------------------------------------------------


class RoadProfile(Protocol):
    def height(self, x: float) -> float: ...


@dataclass(frozen=True)
class StepRoad:
    height_m: float = 0.04
    start_x: float = 1.0

    def height(self, x: float) -> float:
        return self.height_m if x >= self.start_x else 0.0


@dataclass(frozen=True)
class PlateauRoad:
    height_m: float = 0.04
    start_x: float = 1.0
    length: float = 2.0

    def height(self, x: float) -> float:
        return self.height_m if self.start_x <= x <= self.start_x + self.length else 0.0


@dataclass(frozen=True)
class EnvelopeContact:
    tire_radius: float = 0.31785
    samples: int = 161
    dx_for_velocity: float = 1.0e-3

    def __post_init__(self) -> None:
        if self.samples < 3:
            raise ValueError("envelope contact requires at least three samples")

    def height(self, x_center: float, road: RoadProfile) -> float:
        offsets = np.linspace(-self.tire_radius, self.tire_radius, self.samples)
        circle = (
            np.sqrt(
                np.maximum(self.tire_radius * self.tire_radius - offsets * offsets, 0.0)
            )
            - self.tire_radius
        )
        road_values = np.array(
            [road.height(float(x_center + offset)) for offset in offsets]
        )
        return float(np.max(road_values + circle))

    def height_and_velocity(
        self, x_center: float, speed: float, road: RoadProfile
    ) -> tuple[float, float]:
        h0 = self.height(x_center, road)
        hp = self.height(x_center + self.dx_for_velocity, road)
        hm = self.height(x_center - self.dx_for_velocity, road)
        dhdx = (hp - hm) / (2.0 * self.dx_for_velocity)
        return h0, float(speed) * dhdx


# ---------------------------------------------------------------------------
# Full 2-DOF dynamics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SuspensionForceState:
    spring_vertical: float
    damper_vertical: float
    total_vertical: float
    spring_compression: float
    damper_stroke: float
    spring_ratio: float
    damper_ratio: float


@dataclass
class MacPhersonTireModel:
    sprung_mass: float = 365.0
    unsprung_mass: float = 45.0
    speed: float = 2.5
    kinematics: KinematicMap = field(default_factory=KinematicMap.bmw_e90_front)
    spring: ProgressiveSpring = field(default_factory=ProgressiveSpring)
    shock: TwinTubeShock = field(default_factory=TwinTubeShock.default)
    tire: TireParams = field(default_factory=TireParams)
    contact: EnvelopeContact | None = None

    def __post_init__(self) -> None:
        if self.contact is None:
            self.contact = EnvelopeContact(tire_radius=self.tire.radius)
        self.static_suspension_force = self.suspension_force(0.0, 0.0).total_vertical
        self.static_tire_force = (
            self.static_suspension_force + self.unsprung_mass * 9.81
        )
        self.static_tire_compression = self.static_tire_force / self.tire.stiffness

    def suspension_force(
        self, relative_travel: float, relative_velocity: float
    ) -> SuspensionForceState:
        point = self.kinematics.sample(relative_travel)
        damper_velocity = point.damper_ratio * relative_velocity
        spring_axial = self.spring.force(point.spring_compression)
        damper_axial = self.shock.force(damper_velocity)
        spring_vertical = point.spring_ratio * spring_axial
        damper_vertical = point.damper_ratio * damper_axial
        return SuspensionForceState(
            spring_vertical=spring_vertical,
            damper_vertical=damper_vertical,
            total_vertical=spring_vertical + damper_vertical,
            spring_compression=point.spring_compression,
            damper_stroke=point.damper_stroke,
            spring_ratio=point.spring_ratio,
            damper_ratio=point.damper_ratio,
        )

    def rhs(self, time: float, state: Array, road: RoadProfile) -> Array:
        z_s, z_s_dot, z_u, z_u_dot = state
        x = self.speed * time
        road_env, road_env_dot = self.contact.height_and_velocity(x, self.speed, road)

        suspension = self.suspension_force(z_u - z_s, z_u_dot - z_s_dot)
        suspension_relative = suspension.total_vertical - self.static_suspension_force

        tire_compression = self.static_tire_compression + road_env - z_u
        tire_compression_velocity = road_env_dot - z_u_dot
        tire_relative = (
            self.tire.force(tire_compression, tire_compression_velocity)
            - self.static_tire_force
        )

        z_s_ddot = suspension_relative / self.sprung_mass
        z_u_ddot = (tire_relative - suspension_relative) / self.unsprung_mass
        return np.array([z_s_dot, z_s_ddot, z_u_dot, z_u_ddot])

    def simulate(
        self,
        t: Array,
        road: RoadProfile,
        initial_state: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    ) -> dict[str, Array]:
        t = np.asarray(t, dtype=float)
        if len(t) < 2 or np.any(np.diff(t) <= 0.0):
            raise ValueError("time vector must be strictly increasing")

        state = np.array(initial_state, dtype=float)
        states = np.zeros((len(t), 4), dtype=float)
        states[0] = state

        for index in range(1, len(t)):
            dt = float(t[index] - t[index - 1])
            ti = float(t[index - 1])
            k1 = self.rhs(ti, state, road)
            k2 = self.rhs(ti + 0.5 * dt, state + 0.5 * dt * k1, road)
            k3 = self.rhs(ti + 0.5 * dt, state + 0.5 * dt * k2, road)
            k4 = self.rhs(ti + dt, state + dt * k3, road)
            state = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            states[index] = state

        return self._output(t, states, road)

    def _output(self, t: Array, states: Array, road: RoadProfile) -> dict[str, Array]:
        road_env = np.zeros_like(t)
        road_env_dot = np.zeros_like(t)
        spring_force = np.zeros_like(t)
        damper_force = np.zeros_like(t)
        suspension_force = np.zeros_like(t)
        tire_force = np.zeros_like(t)
        tire_compression = np.zeros_like(t)
        spring_compression = np.zeros_like(t)
        damper_stroke = np.zeros_like(t)

        for i, (time, row) in enumerate(zip(t, states)):
            z_s, z_s_dot, z_u, z_u_dot = row
            road_env[i], road_env_dot[i] = self.contact.height_and_velocity(
                self.speed * float(time), self.speed, road
            )
            suspension = self.suspension_force(z_u - z_s, z_u_dot - z_s_dot)
            spring_force[i] = (
                suspension.spring_vertical
                - self.suspension_force(0.0, 0.0).spring_vertical
            )
            damper_force[i] = (
                suspension.damper_vertical
                - self.suspension_force(0.0, 0.0).damper_vertical
            )
            suspension_force[i] = (
                suspension.total_vertical - self.static_suspension_force
            )
            tire_compression[i] = self.static_tire_compression + road_env[i] - z_u
            tire_force[i] = (
                self.tire.force(tire_compression[i], road_env_dot[i] - z_u_dot)
                - self.static_tire_force
            )
            spring_compression[i] = suspension.spring_compression
            damper_stroke[i] = suspension.damper_stroke

        return {
            "t": t,
            "x": self.speed * t,
            "z_s": states[:, 0],
            "z_s_dot": states[:, 1],
            "z_u": states[:, 2],
            "z_u_dot": states[:, 3],
            "road_envelope": road_env,
            "road_envelope_dot": road_env_dot,
            "relative_travel": states[:, 2] - states[:, 0],
            "tire_compression": tire_compression,
            "spring_compression": spring_compression,
            "damper_stroke": damper_stroke,
            "spring_vertical_force": spring_force,
            "damper_vertical_force": damper_force,
            "suspension_vertical_force": suspension_force,
            "tire_vertical_force": tire_force,
        }


# Usage
#
# model = MacPhersonTireModel()
# t_end = 3.0
# dt = 1.0e-3
# t = np.arange(0.0, t_end + 0.5 * dt, dt)
# road = StepRoad(height_m=0.04, start_x=1.0)
# data = model.simulate(t, road)


def animate_simulation(model, road, data, *, fps=30.0, playback_speed=1.0):
    """Animate an existing trajectory as a side-view 2-DOF schematic.

    Requires Matplotlib only for visualization. Positions are in metres without
    amplification; spring/damper drawings are schematic, not 3D hardpoints.
    Keep the returned FuncAnimation alive for plt.show(), or use to_jshtml().
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Ellipse, Rectangle

    if not all(math.isfinite(v) and v > 0 for v in (fps, playback_speed)):
        raise ValueError("fps and playback_speed must be finite and positive")
    keys = ("t", "x", "z_s", "z_u", "road_envelope", "tire_compression")
    values = {key: np.asarray(data[key], dtype=float) for key in keys}
    t = values["t"]
    if t.ndim != 1 or len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("time must be a strictly increasing vector")
    if any(v.shape != t.shape or not np.all(np.isfinite(v)) for v in values.values()):
        raise ValueError("animation channels must be finite vectors matching time")
    count = max(2, math.ceil((t[-1] - t[0]) * fps / playback_speed) + 1)
    frame_t = np.linspace(t[0], t[-1], count)
    frames = {key: np.interp(frame_t, t, v) for key, v in values.items()}
    radius = model.tire.radius
    wheel0 = radius - model.static_tire_compression
    body0 = wheel0 + 0.82
    offsets = np.linspace(-1.2, 1.2, 601)
    fig, (scene, history) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("MacPherson: движение подвески", fontsize=16)
    scene.set(title="Боковая схема: 2 степени свободы, масштаб 1:1",
              xlabel="Расстояние от колеса, м", ylabel="Высота над исходной дорогой, м",
              xlim=(-1.2, 1.2),
              ylim=(min(0, float(values["road_envelope"].min()),
                        wheel0 + float(values["z_u"].min()) - radius) - 0.15,
                    max(body0 + float(values["z_s"].max()) + 0.3,
                        float(values["road_envelope"].max()) + 0.3)))
    scene.set_aspect("equal", adjustable="box")
    scene.grid(alpha=0.15)
    road_line, = scene.plot([], [], color="#78634c", lw=3, label="Дорога")
    envelope, = scene.plot([], [], "o", color="#d28a00", label="Огибающая контакта")
    tire = Ellipse((0, wheel0), 2 * radius, 2 * wheel0,
                   facecolor="#d9e0e7", edgecolor="#263445", lw=3)
    body = Rectangle((-0.62, body0), 1.24, 0.22, color="#3276ac")
    scene.add_patch(tire)
    scene.add_patch(body)
    hub, = scene.plot([], [], "o", color="#263445", ms=8)
    spring, = scene.plot([], [], color="#29854d", lw=2.5, label="Пружина")
    damper, = scene.plot([], [], color="#b34f55", lw=3, label="Амортизатор")
    axle, = scene.plot([], [], color="#263445", lw=3)
    status = scene.text(0.02, 0.98, "", transform=scene.transAxes, va="top",
                        fontsize=9, bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
    scene.legend(loc="lower left", fontsize=8)
    for key, color, label in (("z_s", "#3276ac", "Кузов z_s"),
                              ("z_u", "#263445", "Колесо z_u"),
                              ("road_envelope", "#d28a00", "Огибающая дороги")):
        history.plot(t, 1000 * values[key], color=color, label=label)
    history.set(xlabel="Время, с", ylabel="Перемещение от равновесия, мм",
                title="Вертикальные перемещения", xlim=(t[0], t[-1]))
    history.grid(alpha=0.25)
    history.legend(loc="upper right", fontsize=9)
    cursor = history.axvline(t[0], color="#b34f55", lw=1.5)
    body_dot, = history.plot([], [], "o", color="#3276ac")
    wheel_dot, = history.plot([], [], "o", color="#263445")
    fig.text(0.5, 0.025, "Пружина и амортизатор показаны схематически; "
             "деформация шины — по огибающей контакта.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))

    def update(index):
        time, x = frame_t[index], frames["x"][index]
        zs, zu = frames["z_s"][index], frames["z_u"][index]
        wz, bz = wheel0 + zu, body0 + zs
        compression = frames["tire_compression"][index]
        tire.center = (0, wz)
        tire.height = 2 * max(radius - max(compression, 0), 0.02 * radius)
        body.set_y(bz)
        hub.set_data([0], [wz])
        axle.set_data([-0.36, 0.36], [wz, wz])
        bottom, top = wz + 0.07, bz
        coil_y = np.linspace(bottom + 0.08, top - 0.08, 17)
        coil_x = -0.25 + 0.065 * np.where(np.arange(17) % 2, 1, -1)
        spring.set_data(np.r_[-0.25, -0.25, coil_x, -0.25, -0.25],
                        np.r_[bottom, bottom + 0.08, coil_y, top - 0.08, top])
        mid = bottom + 0.55 * (top - bottom)
        damper.set_data(
            [0.25, 0.25, np.nan, 0.19, 0.19, 0.31, 0.31, np.nan, 0.25, 0.25],
            [bottom, mid, np.nan, bottom + 0.1, mid + 0.07, mid + 0.07,
             bottom + 0.1, np.nan, mid, top])
        road_line.set_data(offsets, [road.height(float(x + dx)) for dx in offsets])
        envelope.set_data([0], [frames["road_envelope"][index]])
        cursor.set_xdata([time, time])
        body_dot.set_data([time], [1000 * zs])
        wheel_dot.set_data([time], [1000 * zu])
        status.set_text(f"t = {time:.2f} с    x = {x:.2f} м\n"
                        f"Кузов: {1000 * zs:+.1f} мм    Колесо: {1000 * zu:+.1f} мм\n"
                        f"Ход подвески: {1000 * (zu - zs):+.1f} мм\n"
                        f"Сжатие шины: {1000 * max(compression, 0):.1f} мм"
                        + ("\nОтрыв шины от дороги" if compression <= 0 else ""))
        return (road_line, envelope, tire, body, hub, spring, damper, axle,
                status, cursor, body_dot, wheel_dot)

    animation = FuncAnimation(fig, update, frames=count, interval=1000 / fps,
                              blit=False, repeat=True)
    update(0)
    return animation


def main():
    """Run the model and save CSV data and an HTML animation with controls."""
    import argparse
    from pathlib import Path
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("macpherson_animation.html"))
    parser.add_argument("--data-output", type=Path,
                        help="CSV path (default: macpherson_sim.csv beside the HTML output)")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--playback-speed", type=float, default=1.0)
    parser.add_argument("--road", choices=("step", "plateau"), default="plateau")
    args = parser.parse_args()
    for name in ("duration", "dt", "fps", "playback_speed"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    model = MacPhersonTireModel()
    road = PlateauRoad() if args.road == "plateau" else StepRoad()
    t = np.linspace(0, args.duration, max(1, math.ceil(args.duration / args.dt)) + 1)
    print(f"Simulating {args.duration:g} s ({len(t)} samples)...", flush=True)
    data = model.simulate(t, road)
    # Export the requested notation while preserving the model's channel names.
    channels = {
        "t": "t",
        "x": "x",
        "z_s": "z_s",
        "v_s": "z_s_dot",
        "z_u": "z_u",
        "v_u": "z_u_dot",
        "H": "road_envelope",
        "H_dot": "road_envelope_dot",
        "q": "relative_travel",
        "delta": "tire_compression",
        "c_s": "spring_compression",
        "c_d": "damper_stroke",
        "spring_vertical_force": "spring_vertical_force",
        "damper_vertical_force": "damper_vertical_force",
        "suspension_vertical_force": "suspension_vertical_force",
        "tire_vertical_force": "tire_vertical_force",
    }
    data_output = args.data_output or args.output.with_name("macpherson_sim.csv")
    data_output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        data_output,
        np.column_stack([data[key] for key in channels.values()]),
        delimiter=",",
        header=",".join(channels),
        comments="",
    )
    print(f"Simulation data saved: {data_output.resolve()}", flush=True)
    animation = animate_simulation(model, road, data, fps=args.fps,
                                   playback_speed=args.playback_speed)
    print("Rendering animation...", flush=True)
    html = animation.to_jshtml(default_mode="loop")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        '<!doctype html><html lang="ru"><meta charset="utf-8">'
        '<title>MacPherson — 2D анимация</title>'
        '<body style="font-family:sans-serif;text-align:center">'
        '<p>Нажмите ▶ для воспроизведения. Ползунок позволяет выбрать момент времени.</p>'
        + html + '</body></html>', encoding="utf-8")
    plt.close(animation._fig)
    print(f"Animation saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
