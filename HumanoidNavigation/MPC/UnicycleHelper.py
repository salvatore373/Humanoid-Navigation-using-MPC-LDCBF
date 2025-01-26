import numpy as np
import sympy as sym

from sympy.core.expr import Expr
from sympy.core.symbol import Symbol


class UnicycleHelper:
    """
    Helper class containing utilities to simulate a unicycle
    """

    @staticmethod
    def _comp_unicycle_pos_vel_acc_component(s: Symbol, init_coordinate: float, goal_coordinate: float,
                                             theta_i: float, theta_f: float,
                                             k_param: float, is_for_x: bool = True):
        """
        Given the initial and goal states of the unicycle, compute the x or y path (in terms of position, velocity and
         acceleration) as functions of a parameter time contained in [0, 1].

        :param s: The parameter of the path: a variable contained in [0, 1].
        :param init_coordinate: The initial X or Y coordinate of the unicycle.
        :param goal_coordinate: The final X or Y coordinate of the unicycle.
        :param theta_i: The initial orientation of the unicycle.
        :param theta_f: The final orientation of the unicycle.
        :param k_param: The value of the translational velocity path at 0 and 1, i.e. v_s(0) = v_s(1) = k_param.
        :param is_for_x: Whether it has to compute the path for X or for Y.
        """
        k_param = 10
        sin_or_cos = np.cos if is_for_x else np.sin
        # Compute the polynomial coefficients
        a = init_coordinate
        b = k_param * sin_or_cos(theta_i)
        c = (3 * goal_coordinate - 3 * init_coordinate - k_param * sin_or_cos(theta_f) -
             2 * k_param * sin_or_cos(theta_i))
        d = 2 * init_coordinate - 2 * goal_coordinate + k_param * sin_or_cos(theta_f) + k_param * sin_or_cos(theta_i)

        # Plug the coefficients in the polynomials to get the path expression
        return (a + b * s + c * np.power(s, 2) + d * np.power(s, 3),  # position path
                b + 2 * c * s + 3 * d * np.power(s, 2),  # velocity path
                2 * c + 6 * d * s)  # acceleration path

    @staticmethod
    def compute_unicycle_params(s: Symbol, initial_state: np.ndarray, goal_state: np.ndarray, init_fin_vel: float) -> \
            tuple[Expr, Expr, Expr, Expr, Expr, Expr, Expr, Expr, Expr]:
        """
        Given the initial and goal states of the unicycle, compute x, y, x_dot, y_dot, theta, v, omega as functions of
        a parameter contained in [0, 1].

        :param s: The parameter of the path: a variable contained in [0, 1].
        :param initial_state: The initial configuration of the unicycle, in the form (x, y, theta).
        :param goal_state: The final configuration of the unicycle, in the form (x, y, theta).
        :param init_fin_vel: The value of the translational velocity path at 0 and 1,
        i.e. v_s(0) = v_s(1) = init_fin_vel.
        :return: x, y, x_dot, y_dot, theta, v, omega as functions of a parameter contained in [0, 1].
        """
        # Get the start and goal configurations
        x_i, y_i, theta_i = initial_state
        x_f, y_f, theta_f = goal_state

        # Compute the unicycle parameters
        x, x_dot, x_ddot = UnicycleHelper._comp_unicycle_pos_vel_acc_component(s, x_i, x_f, theta_i, theta_f,
                                                                               k_param=init_fin_vel, is_for_x=True)
        y, y_dot, y_ddot = UnicycleHelper._comp_unicycle_pos_vel_acc_component(s, y_i, y_f, theta_i, theta_f,
                                                                               k_param=init_fin_vel, is_for_x=False)
        theta = sym.atan2(y_dot, x_dot)
        v = sym.sqrt(sym.Pow(x_dot, 2) + sym.Pow(y_dot, 2))
        omega = (y_ddot * x_dot - y_dot * x_ddot) / (sym.Pow(x_dot, 2) + sym.Pow(y_dot, 2))

        return (x, x_dot, x_ddot,
                y, y_dot, y_ddot,
                theta, v, omega)

    @staticmethod
    def _compute_peak_in_function(s: Symbol, func_s: Expr) -> float:
        """
        Computes max(s in [0, 1]) { funct_s(s) }
        """
        # Compute the derivative of funct_s wrt s
        dx_ds = sym.diff(func_s, s)
        # Find the critical points by solving dfunct_s/ds = 0
        critical_points = sym.solve(dx_ds, s)
        # Keep only critical points within the interval [0, 1]
        valid_critical_points = [p for p in critical_points if p.is_real and sym.Interval(0, 1).contains(p)]
        # Add endpoints to the list of points to evaluate
        points_to_evaluate = valid_critical_points + [0, 1]
        # Evaluate funct_s(s) at all points
        evaluated_points = [(p, func_s.subs(s, p)) for p in points_to_evaluate]

        # Prove it visually
        # from sympy.plotting import plot
        # plot(func_s, (s, 0, 1))
        # plot(dx_ds, (s, 0, 1))

        # Find the maximum value and return it
        return float(max(evaluated_points, key=lambda t: t[1])[1])

    @staticmethod
    def _compute_alpha_for_timing_law(s: Symbol, v_s: Expr, omega_s: Expr, v_max: float, omega_max: float) -> float:
        """
        It computes the value of alpha to be used in the timing law s(t) = s_i + alpha * (t - t_i), where alpha is
        defined as:
            alpha = min { v_max/v_peak, omega_max/omega_peak }
            with v_peak = max(s in [s_i, s_f]) { v_s }    and     omega_peak = max(s in [s_i, s_f]) { omega_s }
                 s_i = 0; s_f = 1.

        :param s: The symbolic parameter of the v_s and omega_s functions.
        :param v_s: The translational velocity defined as a function of s
        :param omega_s: The rotational velocity defined as a function of s.
        :param v_max: The maximum value of the unicycle translational velocity.
        :param omega_max: The maximum value of the unicycle rotational velocity.
        """
        v_peak = UnicycleHelper._compute_peak_in_function(s, v_s)
        omega_peak = UnicycleHelper._compute_peak_in_function(s, omega_s)

        return max(v_max / v_peak, omega_max / omega_peak)

    @staticmethod
    def compute_unicycle_params_trajectory(start_position: np.ndarray, goal_position: np.ndarray,
                                           start_orientation: float = 0.0, goal_orientation: float = 0.0,
                                           num_timesteps: int = 25, v_max: float = 10.0, omega_max: float = 10.0):
        """
        Given the initial and goal states of the unicycle, computes the trajectories of x, y, x_dot, y_dot, theta, v,
        omega.

        :param start_position: The vector of the coordinates (x, y) where the midpoint of the feet starts the motion.
        :param goal_position: The vector of the coordinates (x, y) where the midpoint of the feet should be at the end
         of the motion.
        :param start_orientation: The orientation of the unicycle in the initial state.
        :param goal_orientation: The orientation that the unicycle should have in the goal state.
        :param num_timesteps: The number of time steps to include in the solution, i.e. the number of columns in the
         returned trajectories.
        :param v_max: The maximum value of the unicycle translational velocity.
        :param omega_max: The maximum value of the unicycle rotational velocity.
        """
        # Define the symbolic parameter of the path
        s = sym.symbols('s', nonnegative=True)

        init_state = np.insert(start_position, 2, start_orientation)
        goal_state = np.insert(goal_position, 2, goal_orientation)
        (x_tilde, x_prime, x_second, y_tilde, y_prime, y_second, theta_tilde, v_tilde, omega_tilde) = (
            UnicycleHelper.compute_unicycle_params(s, init_state, goal_state, v_max))

        # Turn the path into a trajectory with a linear timing law
        t = sym.symbols('t', nonnegative=True)
        # alpha = 0.7
        alpha = UnicycleHelper._compute_alpha_for_timing_law(s, v_tilde, omega_tilde, v_max, omega_max)
        tim_law = alpha * t
        time_interval = np.linspace(0, 1 / alpha, num_timesteps)
        # Turn the path into a trajectory
        x_dot = x_prime * alpha
        y_dot = y_prime * alpha
        x_ddot = x_second * alpha + x_prime * 0
        y_ddot = y_second * alpha + y_prime * 0
        v = v_tilde * alpha
        omega = omega_tilde * alpha
        (x, x_dot, x_ddot, y, y_dot, y_ddot, theta, v, omega) = (
            sym.lambdify(t, f.subs(s, tim_law), 'numpy') for f in
            (x_tilde, x_dot, x_ddot, y_tilde, y_dot, y_ddot, theta_tilde, v, omega))

        return (f(time_interval) for f in (x, x_dot, x_ddot, y, y_dot, y_ddot, theta, v, omega))
