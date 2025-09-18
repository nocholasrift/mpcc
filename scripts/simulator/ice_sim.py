from imp import create_dynamic
import pygame
import numpy as np

# from mpcc_model import export_mpcc_ode_model_spline_tube_cbf
from mpcc_ocp_horizon import create_ocp_tube_cbf
from scripts.unicycle_mpcc.mpcc_ocp_horizon import create_ocp
from acados_template import AcadosOcpSolver, AcadosSimSolver, AcadosOcp


# --- Physics model ---
class UnicycleSlipAware:
    def __init__(self, mu, mass, wheel_radius, g=9.81):
        self.mu = mu
        self.m = mass
        self.r = wheel_radius
        self.g = g
        self.N = self.m * self.g

        self.v = 0
        self.theta = 0
        self.x = 300
        self.y = 300

    def slip_force(self, slip_ratio):
        peak_slip = 0.15
        if slip_ratio < peak_slip:
            return self.mu * self.N * (slip_ratio / peak_slip)
        else:
            return self.mu * self.N * (1.0 - 0.5 * (slip_ratio - peak_slip))

    def step(self, torque_cmd, omega_cmd, dt):

        v_wheel = self.v + (torque_cmd / self.r / self.m) * dt
        slip = (v_wheel - self.v) / max(abs(v_wheel), 1e-5)
        F = np.clip(self.slip_force(slip), -self.mu * self.N, self.mu * self.N)
        a = F / self.m

        self.v += a * dt
        self.theta += omega_cmd * dt
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt

        return self.x, self.y, self.theta, self.v, slip


def main():

    # --- Pygame setup ---
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()

    robot = UnicycleSlipAware(mu=0.3, mass=10, wheel_radius=0.05)
    torque_cmd = 100.0  # Nm
    omega_cmd = 0.0  # rad/s
    dt = 0.05

    # --- OCP setup ---
    ocp = create_ocp_tube_cbf("/home/bezzo/catkin_ws/src/mpcc/params/mpcc.yaml")
    acados_ocp_solver = AcadosOcpSolver(ocp)
    acados_integrator = AcadosSimSolver(ocp)

    running = True
    count = 0
    while running:
        screen.fill((255, 255, 255))

        count += 1
        if count > 200:
            torque_cmd = -10

        # Physics step
        x, y, theta, v, slip = robot.step(torque_cmd, omega_cmd, dt)

        # Draw robot
        # pygame.draw.circle(screen, (0, 0, 255), (int(x), int(y)), 10)
        pygame.draw.rect(screen, (0, 0, 255), (int(x) - 10, int(y) - 8, 20, 16))
        dx = 15 * np.cos(theta)
        dy = 15 * np.sin(theta)
        pygame.draw.line(screen, (255, 0, 0), (x, y), (x + dx, y + dy), 2)

        # Draw slip info
        font = pygame.font.SysFont(None, 24)
        txt = font.render(f"Slip: {slip:.2f}", True, (0, 0, 0))
        screen.blit(txt, (10, 10))

        pygame.display.flip()
        clock.tick(int(1 / dt))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()


if __name__ == "__main__":
    main()
