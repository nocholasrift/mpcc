import numpy as np

from scipy.interpolate import UnivariateSpline


class BezierCurve:
    def __init__(self, p0, p1, p2, p3, n_samples=20):
        self.p0 = np.array(p0, dtype=float)
        self.p1 = np.array(p1, dtype=float)
        self.p2 = np.array(p2, dtype=float)
        self.p3 = np.array(p3, dtype=float)

        self.knots, self.xs, self.ys = self.reparam_traj(M=n_samples)

        # spline fits for x(s) and y(s)
        self.trajx = UnivariateSpline(self.knots, self.xs, k=3, s=0)
        self.trajy = UnivariateSpline(self.knots, self.ys, k=3, s=0)

        self.trajx_d = self.trajx.derivative(n=1)
        self.trajy_d = self.trajy.derivative(n=1)

    def pos(self, s):
        return np.array([self.trajx(s), self.trajy(s)])

    def vel(self, s):
        return np.array([self.trajx_d(s), self.trajy_d(s)])

    def _pos(self, t):
        """Position on curve at param t ∈ [0,1]"""
        return (
            (1 - t) ** 3 * self.p0
            + 3 * (1 - t) ** 2 * t * self.p1
            + 3 * (1 - t) * t**2 * self.p2
            + t**3 * self.p3
        )

    def _vel(self, t):
        """Derivative wrt t"""
        return (
            3 * (1 - t) ** 2 * (self.p1 - self.p0)
            + 6 * (1 - t) * t * (self.p2 - self.p1)
            + 3 * t**2 * (self.p3 - self.p2)
        )

    def get_arclen(self):
        return self.knots[-1]

    def compute_arclen(self, t0=0.0, tf=1.0, n=100):
        """Numerical arc length using trapezoidal rule"""
        ts = np.linspace(t0, tf, n + 1)
        vels = np.array([self._vel(t) for t in ts])
        speeds = np.linalg.norm(vels, axis=1)
        return np.trapz(speeds, ts)

    def binary_search(self, target_s, start=0.0, end=1.0, tol=1e-4):
        """Find t such that arclen(0,t) ≈ target_s"""
        t_left, t_right = start, end
        prev_s, s = 0, -1e9

        while abs(prev_s - s) > tol:
            prev_s = s
            t_mid = (t_left + t_right) / 2.0
            s = self.compute_arclen(0, t_mid)
            if s < target_s:
                t_left = t_mid
            else:
                t_right = t_mid

        return 0.5 * (t_left + t_right)

    def reparam_traj(self, M=20):
        """Return M+1 samples evenly spaced in arc length"""
        total_len = self.compute_arclen(0, 1)
        ds = total_len / M

        ss, xs, ys = [], [], []
        prev_t = 0.0
        for i in range(M + 1):
            s = i * ds
            ti = self.binary_search(s, prev_t, 1.0)
            pos = self._pos(ti)
            ss.append(s)
            xs.append(pos[0])
            ys.append(pos[1])
            prev_t = ti

        return np.array(ss), np.array(xs), np.array(ys)

    def fill(self, ts):
        """Return positions for a list of t values"""
        return np.array([self._pos(t) for t in ts])
