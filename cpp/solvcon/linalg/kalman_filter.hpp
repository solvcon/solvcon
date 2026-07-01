#pragma once

/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Kalman filter for linear state estimation with predict and update steps.
 *
 * @ingroup group_numerics
 */

#include <solvcon/linalg/factorization.hpp>
#include <solvcon/math/math.hpp>

namespace solvcon
{

namespace detail
{
template <typename T>
struct select_real_t;
} /* end namespace detail */

/**
 * @brief `KalmanStateInfo` includes prior and posterior states and their covariances,
 *  and it is the return type of `KalmanFilter<T>::batch_filter(...)`.
 *
 * @details
 * Due to the iteration of the predict and update steps in `KalmanFilter<T>::batch_filter(...)`, the
 * intermediate results are stored for each time step, including
 *  -   prior_states
 *  -   prior_states_covariance
 *  -   posterior_states
 *  -   posterior_states_covariance
 *
 * @see KalmanStateInfo<T> KalmanFilter<T>::batch_filter(array_type const & zs)
 * @see KalmanStateInfo<T> KalmanFilter<T>::batch_filter(array_type const & zs, array_type const & us)
 *
 * @ingroup group_numerics
 */
template <typename T>
struct KalmanStateInfo
{
    using tuple_type = std::tuple<SimpleArray<T>, SimpleArray<T>, SimpleArray<T>, SimpleArray<T>>;

    KalmanStateInfo(ssize_t observation_size, ssize_t state_size)
    {
        small_vector<ssize_t> const xs_shape{observation_size, state_size};
        small_vector<ssize_t> const ps_shape{observation_size, state_size, state_size};
        prior_states = SimpleArray<T>(xs_shape);
        prior_states_covariance = SimpleArray<T>(ps_shape);
        posterior_states = SimpleArray<T>(xs_shape);
        posterior_states_covariance = SimpleArray<T>(ps_shape);
    }

    SimpleArray<T> prior_states;
    SimpleArray<T> prior_states_covariance;
    SimpleArray<T> posterior_states;
    SimpleArray<T> posterior_states_covariance;
}; /* end struct KalmanStateInfo */

/**
 * Kalman filter for linear Gaussian state estimation.
 *
 * Implements the discrete predict and update recursion over a state mean
 * vector and its covariance matrix, using the transition matrix F, process
 * noise covariance Q, measurement matrix H, measurement noise covariance R,
 * and an optional control matrix B. The update step forms the Kalman gain
 * through an LLT solve and applies the Joseph-form covariance update for
 * numerical stability.
 *
 * Reference: FilterPy KalmanFilter documentation
 * https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html
 *
 * @ingroup group_numerics
 */
template <typename T>
class KalmanFilter
{

public:

    using value_type = T;
    using real_type = typename detail::select_real_t<value_type>::type;
    using array_type = SimpleArray<T>;

private:

    ssize_t m_state_size; // state dimension
    ssize_t m_measurement_size; // measurement dimension
    ssize_t m_control_size; // control dimension

    array_type m_f; // state transition matrix (state_sizexstate_size)
    array_type m_q; // process noise covariance (state_sizexstate_size)
    array_type m_h; // measurement matrix (measurement_sizexstate_size)
    array_type m_r; // measurement noise covariance (measurement_sizexmeasurement_size)
    array_type m_p; // state covariance (state_sizexstate_size)
    array_type m_b; // control matrix (state_sizexcontrol_size)
    array_type m_x; // state vector (state_sizex1)
    array_type m_i; // identity matrix (state_sizexstate_size)

    real_type m_jitter; // regularization jitter for numerical stability

public:

    /**
     * @brief Construct a Kalman filter.
     *
     * @details
     * Creates a Kalman filter with the specified system matrices and noise
     * parameters. The process and measurement noise covariances are created
     * from scalar variance values (multiplied by identity matrices). The
     * initial state covariance is set to the identity matrix.
     *
     * Control input: an external force or command applied to the system
     * (e.g., thrust in aircraft, motor command in robot). If the control
     * matrix B is empty (size 0), control input u is disabled. See
     * https://kalmanfilter.net/multiExamples.html for examples.
     *
     * @param x Initial state vector.
     * @param f State transition matrix F.
     * @param b Control matrix B (empty to disable control input u).
     * @param h Measurement matrix H.
     * @param process_noise Process noise standard deviation.
     * @param measurement_noise Measurement noise standard deviation.
     * @param jitter Numerical stability jitter.
     */
    KalmanFilter(
        array_type const & x, // FIXME: NOLINT(modernize-pass-by-value)
        array_type const & f, // FIXME: NOLINT(modernize-pass-by-value)
        array_type const & b, // FIXME: NOLINT(modernize-pass-by-value)
        array_type const & h, // FIXME: NOLINT(modernize-pass-by-value)
        real_type process_noise,
        real_type measurement_noise,
        real_type jitter)
        : m_state_size(x.shape(0))
        , m_measurement_size(h.shape(0))
        , m_control_size((b.ndim() == 2) ? b.shape(1) : 0)
        , m_f(f)
        , m_q(array_type::scaled_eye(m_state_size, static_cast<T>(process_noise * process_noise)))
        , m_h(h)
        , m_r(array_type::scaled_eye(m_measurement_size, static_cast<T>(measurement_noise * measurement_noise)))
        , m_p(array_type::eye(m_state_size))
        , m_b(b)
        , m_x(x)
        , m_i(array_type::eye(m_state_size))
        , m_jitter(jitter)
    {
        check_dimensions();
    }

    /**
     * @brief Construct a Kalman filter with explicit covariance matrices.
     *
     * @details
     * Creates a Kalman filter with the specified system matrices and covariance
     * matrices. This overload is useful for examples or physical models whose
     * process noise, measurement noise, or initial state uncertainty are not
     * scaled identity matrices.
     *
     * @param x Initial state vector.
     * @param f State transition matrix F.
     * @param b Control matrix B (empty to disable control input u).
     * @param h Measurement matrix H.
     * @param q Process noise covariance Q.
     * @param r Measurement noise covariance R.
     * @param p Initial state covariance P.
     * @param jitter Numerical stability jitter.
     */
    KalmanFilter(
        array_type const & x, // FIXME: NOLINT(modernize-pass-by-value)
        array_type const & f, // FIXME: NOLINT(modernize-pass-by-value)
        array_type const & b, // FIXME: NOLINT(modernize-pass-by-value)
        array_type const & h, // FIXME: NOLINT(modernize-pass-by-value)
        array_type const & q, // FIXME: NOLINT(modernize-pass-by-value)
        array_type const & r, // FIXME: NOLINT(modernize-pass-by-value)
        array_type const & p, // FIXME: NOLINT(modernize-pass-by-value)
        real_type jitter)
        : m_state_size(x.shape(0))
        , m_measurement_size(h.shape(0))
        , m_control_size((b.ndim() == 2) ? b.shape(1) : 0)
        , m_f(f)
        , m_q(q)
        , m_h(h)
        , m_r(r)
        , m_p(p)
        , m_b(b)
        , m_x(x)
        , m_i(array_type::eye(m_state_size))
        , m_jitter(jitter)
    {
        check_dimensions();
    }

    array_type const & state() const { return m_x; }
    array_type const & covariance() const { return m_p; }

    /**
     * @brief Predict step without control input u.
     *
     * @details
     * Advances the state using only the transition model and process noise.
     * Represents a passive system with no external actuation.
     *
     * @note Common in finance, tracking, or passive simulation tasks.
     *
     * @see predict(const array_type&), update(const array_type&)
     */
    void predict()
    {
        // x <- F x  (m_x <- m_f @ m_x)
        predict_state();

        // P <- F P F^H + Q  (m_p <- m_f @ m_p @ m_f^H + m_q)
        predict_covariance();

        // P <- 0.5(P + P^H)  (m_p <- 0.5(m_p + m_p^H))
        m_p = m_p.symmetrize();
    }

    /**
     * @brief Predict step with control input u.
     *
     * @details
     * Updates the state while applying a known control SimpleArray.
     * The control term affects only the state mean.
     *
     * @note Example: in a vehicle model, the state may contain position and velocity,
     * while the control input represents acceleration (throttle or brake). The
     * control term updates how the state evolves between sensor measurements.
     *
     * @param u Control input.
     *
     * @see predict(), update(const array_type&)
     */
    void predict(array_type const & u)
    {
        check_control(u);

        // x <- F x + B u  (m_x <- m_f @ m_x + m_b @ u)
        predict_state(u);

        // P <- F P F^H + Q  (m_p <- m_f @ m_p @ m_f^H + m_q)
        predict_covariance();

        // P <- 0.5(P + P^H)  (m_p <- 0.5(m_p + m_p^H))
        m_p = m_p.symmetrize();
    }

    /**
     * @brief Update step (measurement correction).
     *
     * @details
     * Incorporates a new measurement to refine the predicted state and covariance
     * using the Kalman gain.
     *
     * @param z Measurement input.
     *
     * @see predict(), predict(const array_type&)
     */
    void update(array_type const & z)
    {
        check_measurement(z);

        // y <- z - H x  (y <- z - m_h @ m_x)
        array_type const y = innovation(z);

        // S <- H P H^H + R + jitter I  (s <- m_h @ m_p @ m_h^H + m_r + m_jitter @ I)
        array_type const s = innovation_covariance();

        // K <- P H^H S^{-1} via LLT solve  (k <- m_p @ m_h^H @ s^{-1})
        array_type const k = kalman_gain(s);

        // x <- x + K y  (m_x <- m_x + k @ y)
        update_state(k, y);

        // P <- (I-K H) P (I-K H)^H + K R K^H  (m_p <- (I-k@m_h) @ m_p @ (I-k@m_h)^H + k @ m_r @ k^H)
        update_covariance(k);
    }

    /**
     * @brief Predict and update in batch mode without a batch of control input `us`.
     *
     * @details
     * Given a lot of observation inputs `zs` in different timestamps,
     * `batch_filter(array_type const & zs)` predicts and updates iteratively,
     * and return prior and posterior states and covariancein in different timestamps.
     * This api design is refered from [filterpy]
     * (https://filterpy.readthedocs.io/en/latest/_modules/filterpy/kalman/kalman_filter.html#KalmanFilter.batch_filter).
     *
     * @param zs A batch of measurement inputs.
     *
     * @see KalmanStateInfo<T> KalmanFilter<T>::batch_filter(array_type const & zs, array_type const & us)
     * @see struct KalmanStateInfo<T>;
     */
    KalmanStateInfo<T> batch_filter(array_type const & zs);

    /**
     * @brief Predict and update in batch mode with a batch of control input `us`.
     *
     * @details
     * Given a lot of observation inputs `zs` and control inputs `us` in different timestamps,
     * `batch_filter(array_type const & zs)` predicts and updates iteratively,
     * and return prior and posterior states and covariancein in different timestamps.
     * This api design is refered from [filterpy]
     * (https://filterpy.readthedocs.io/en/latest/_modules/filterpy/kalman/kalman_filter.html#KalmanFilter.batch_filter).
     *
     * @param zs A batch of measurement inputs.
     * @param us A batch of control inputs.
     *
     * @see KalmanStateInfo<T> KalmanFilter<T>::batch_filter(array_type const & zs)
     * @see struct KalmanStateInfo<T>;
     */
    KalmanStateInfo<T> batch_filter(array_type const & zs, array_type const & us);

private:

    void check_dimensions();
    void check_measurement(array_type const & z);
    void check_control(array_type const & u);

    // Predict
    void predict_state();
    void predict_state(array_type const & u);
    void predict_covariance();

    // Update
    array_type innovation(array_type const & z);
    array_type innovation_covariance();
    array_type kalman_gain(array_type const & s);
    void update_state(array_type const & k, array_type const & y);
    void update_covariance(array_type const & k);

    // Batch filter
    void predict_and_update(array_type const & z, KalmanStateInfo<T> & bfs, ssize_t iter);
    void predict_and_update(array_type const & z, array_type const & u, KalmanStateInfo<T> & bfs, ssize_t iter);

}; /* end class KalmanFilter */

template <typename T>
// FIXME: NOLINTNEXTLINE(readability-function-cognitive-complexity)
void KalmanFilter<T>::check_dimensions()
{
    if (m_f.ndim() != 2 || m_f.shape(0) != m_state_size || m_f.shape(1) != m_state_size)
    {
        std::ostringstream oss;
        oss << "KalmanFilter::check_dimensions: The state transition SimpleArray f must be state_sizexstate_size, but got shape (";
        for (ssize_t i = 0; i < m_f.ndim(); ++i)
        {
            if (i > 0)
            {
                oss << ", ";
            }
            oss << m_f.shape(i);
        }
        oss << ")";
        throw std::invalid_argument(oss.str());
    }
    if (m_h.ndim() != 2 || m_h.shape(0) != m_measurement_size || m_h.shape(1) != m_state_size)
    {
        std::ostringstream oss;
        oss << "KalmanFilter::check_dimensions: The measurement SimpleArray h must be measurement_sizexstate_size, but got shape (";
        for (ssize_t i = 0; i < m_h.ndim(); ++i)
        {
            if (i > 0)
            {
                oss << ", ";
            }
            oss << m_h.shape(i);
        }
        oss << ")";
        throw std::invalid_argument(oss.str());
    }
    if (m_q.ndim() != 2 || m_q.shape(0) != m_state_size || m_q.shape(1) != m_state_size)
    {
        std::ostringstream oss;
        oss << "KalmanFilter::check_dimensions: The process noise covariance SimpleArray q must be state_sizexstate_size, but got shape (";
        for (ssize_t i = 0; i < m_q.ndim(); ++i)
        {
            if (i > 0)
            {
                oss << ", ";
            }
            oss << m_q.shape(i);
        }
        oss << ")";
        throw std::invalid_argument(oss.str());
    }
    if (m_r.ndim() != 2 || m_r.shape(0) != m_measurement_size || m_r.shape(1) != m_measurement_size)
    {
        std::ostringstream oss;
        oss << "KalmanFilter::check_dimensions: The measurement noise covariance SimpleArray r must be measurement_sizexmeasurement_size, but got shape (";
        for (ssize_t i = 0; i < m_r.ndim(); ++i)
        {
            if (i > 0)
            {
                oss << ", ";
            }
            oss << m_r.shape(i);
        }
        oss << ")";
        throw std::invalid_argument(oss.str());
    }
    if (m_p.ndim() != 2 || m_p.shape(0) != m_state_size || m_p.shape(1) != m_state_size)
    {
        std::ostringstream oss;
        oss << "KalmanFilter::check_dimensions: The state covariance SimpleArray p must be state_sizexstate_size, but got shape (";
        for (ssize_t i = 0; i < m_p.ndim(); ++i)
        {
            if (i > 0)
            {
                oss << ", ";
            }
            oss << m_p.shape(i);
        }
        oss << ")";
        throw std::invalid_argument(oss.str());
    }
    if (m_x.ndim() != 1 || m_x.shape(0) != m_state_size)
    {
        std::ostringstream oss;
        oss << "KalmanFilter::check_dimensions: The state SimpleArray x must be 1D of length state_size, but got shape (";
        for (ssize_t i = 0; i < m_x.ndim(); ++i)
        {
            if (i > 0)
            {
                oss << ", ";
            }
            oss << m_x.shape(i);
        }
        oss << ")";
        throw std::invalid_argument(oss.str());
    }

    if (m_control_size > 0)
    {
        if (m_b.ndim() != 2)
        {
            std::ostringstream oss;
            oss << "KalmanFilter::check_dimensions: The control SimpleArray b must be 2D when control_size > 0, but got " << m_b.ndim() << "D";
            throw std::invalid_argument(oss.str());
        }
        if (m_b.shape(0) != m_state_size || m_b.shape(1) != m_control_size)
        {
            std::ostringstream oss;
            oss << "KalmanFilter::check_dimensions: The control SimpleArray b must be state_sizexcontrol_size, but got shape (";
            for (ssize_t i = 0; i < m_b.ndim(); ++i)
            {
                if (i > 0)
                {
                    oss << ", ";
                }
                oss << m_b.shape(i);
            }
            oss << ")";
            throw std::invalid_argument(oss.str());
        }
    }
    else
    {
        if (m_b.ndim() != 2 || m_b.shape(1) != 0)
        {
            std::ostringstream oss;
            oss << "KalmanFilter::check_dimensions: The control SimpleArray b must be state_sizex0 when control_size = 0, but got shape (";
            for (ssize_t i = 0; i < m_b.ndim(); ++i)
            {
                if (i > 0)
                {
                    oss << ", ";
                }
                oss << m_b.shape(i);
            }
            oss << ")";
            throw std::invalid_argument(oss.str());
        }
    }
}

template <typename T>
void KalmanFilter<T>::predict_state()
{
    // x <- F x  (m_x <- m_f @ m_x)
    m_x = m_f.matmul(m_x);
}

template <typename T>
void KalmanFilter<T>::predict_state(array_type const & u)
{
    // x <- F x + B u  (m_x <- m_f @ m_x + m_b @ u)
    m_x = m_f.matmul(m_x).add(m_b.matmul(u));
}

template <typename T>
void KalmanFilter<T>::predict_covariance()
{
    // P <- F P F^H + Q  (m_p <- m_f @ m_p @ m_f^H + m_q)
    array_type const f_h = m_f.hermitian();
    m_p = m_f.matmul(m_p).matmul(f_h).add(m_q);
}

template <typename T>
void KalmanFilter<T>::check_measurement(array_type const & z)
{
    if (z.ndim() != 1 || z.shape(0) != m_measurement_size)
    {
        std::ostringstream oss;
        oss << "KalmanFilter::check_measurement: The measurement SimpleArray z must be 1D of length measurement_size (" << m_measurement_size << "), but got shape (";
        for (ssize_t i = 0; i < z.ndim(); ++i)
        {
            if (i > 0)
            {
                oss << ", ";
            }
            oss << z.shape(i);
        }
        oss << ")";
        throw std::invalid_argument(oss.str());
    }
}

template <typename T>
void KalmanFilter<T>::check_control(array_type const & u)
{
    if (m_control_size == 0)
    {
        std::ostringstream oss;
        oss << "KalmanFilter::check_control: Control input not supported: control_size is 0";
        throw std::invalid_argument(oss.str());
    }
    if (u.ndim() != 1 || u.shape(0) != m_control_size)
    {
        std::ostringstream oss;
        oss << "KalmanFilter::check_control: The control SimpleArray u must be 1D of length control_size (" << m_control_size << "), but got shape (";
        for (ssize_t i = 0; i < u.ndim(); ++i)
        {
            if (i > 0)
            {
                oss << ", ";
            }
            oss << u.shape(i);
        }
        oss << ")";
        throw std::invalid_argument(oss.str());
    }
}

template <typename T>
typename KalmanFilter<T>::array_type KalmanFilter<T>::innovation(array_type const & z)
{
    // y <- z - H x  (y <- z - m_h @ m_x)
    return z.sub(m_h.matmul(m_x));
}

template <typename T>
typename KalmanFilter<T>::array_type KalmanFilter<T>::innovation_covariance()
{
    // S <- H P H^H + R + jitter I  (s <- m_h @ m_p @ m_h^H + m_r + m_jitter @ I)
    array_type const h_h = m_h.hermitian();
    array_type const hph_h = m_h.matmul(m_p).matmul(h_h);
    array_type const s = hph_h.add(m_r).add(array_type::scaled_eye(m_measurement_size, static_cast<T>(m_jitter)));

    // S <- 0.5(S + S^H)  (s <- 0.5(s + s^H))
    return s.symmetrize();
}

template <typename T>
typename KalmanFilter<T>::array_type KalmanFilter<T>::kalman_gain(array_type const & s)
{
    // K <- P H^H S^{-1} via LLT solve
    array_type const B = m_h.matmul(m_p); // H@P = B
    array_type const X = llt_solve(s, B); // S@X = B
    return X.hermitian(); // K = X^H
}

template <typename T>
void KalmanFilter<T>::update_state(array_type const & k, array_type const & y)
{
    // x <- x + K y  (m_x <- m_x + k @ y)
    m_x = m_x.add(k.matmul(y));
}

template <typename T>
void KalmanFilter<T>::update_covariance(array_type const & k)
{
    // P <- (I-K H) P (I-K H)^H + K R K^H  (m_p <- (I-k@m_h)@m_p@(I-k@m_h)^H + k@m_r@k^H)
    array_type const kh = k.matmul(m_h);
    array_type const i_minus_kh = m_i.sub(kh);
    array_type const i_minus_kh_h = i_minus_kh.hermitian();
    array_type const k_h = k.hermitian();
    array_type const term1 = i_minus_kh.matmul(m_p).matmul(i_minus_kh_h);
    array_type const term2 = k.matmul(m_r).matmul(k_h);
    m_p = term1.add(term2);

    // P <- 0.5(P + P^H)  (m_p <- 0.5(m_p + m_p^H))
    m_p = m_p.symmetrize();
}

template <typename T>
KalmanStateInfo<T> KalmanFilter<T>::batch_filter(array_type const & zs)
{
    ssize_t const z_m = zs.shape(0);
    ssize_t const z_n = zs.shape(1);
    array_type z(small_vector<ssize_t>{z_n});
    KalmanStateInfo<T> bfs(z_m, m_state_size);

    for (ssize_t iter = 0; iter < z_m; ++iter)
    {
        for (ssize_t j = 0; j < z_n; ++j)
        {
            z(j) = zs(iter, j);
        }
        predict_and_update(z, bfs, iter);
    }
    return bfs;
}

template <typename T>
void KalmanFilter<T>::predict_and_update(array_type const & z, KalmanStateInfo<T> & bfs, ssize_t iter)
{
    SimpleArray<T> & prior_state = bfs.prior_states;
    SimpleArray<T> & prior_state_covariance = bfs.prior_states_covariance;
    SimpleArray<T> & posterior_state = bfs.posterior_states;
    SimpleArray<T> & posterior_state_covariance = bfs.posterior_states_covariance;

    predict();
    for (ssize_t j = 0; j < m_state_size; ++j)
    {
        prior_state(iter, j) = m_x(j);
        for (ssize_t k = 0; k < m_state_size; ++k)
        {
            prior_state_covariance(iter, j, k) = m_p(j, k);
        }
    }

    update(z);
    for (ssize_t j = 0; j < m_state_size; ++j)
    {
        posterior_state(iter, j) = m_x(j);
        for (ssize_t k = 0; k < m_state_size; ++k)
        {
            posterior_state_covariance(iter, j, k) = m_p(j, k);
        }
    }
}

template <typename T>
KalmanStateInfo<T> KalmanFilter<T>::batch_filter(array_type const & zs, array_type const & us)
{
    ssize_t const z_m = zs.shape(0);
    ssize_t const z_n = zs.shape(1);
    array_type z(small_vector<ssize_t>{z_n});
    KalmanStateInfo<T> bfs(z_m, m_state_size);

    array_type u;
    ssize_t u_n = 0;

    ssize_t const u_m = us.shape(0);
    if (u_m != z_m)
    {
        throw std::invalid_argument("KalmanFilter::batch_filter: The number of control inputs must match the number of measurements.");
    }
    u_n = us.shape(1);
    u = array_type(small_vector<ssize_t>{u_n});

    for (ssize_t iter = 0; iter < z_m; ++iter)
    {
        for (ssize_t j = 0; j < z_n; ++j)
        {
            z(j) = zs(iter, j);
        }
        for (ssize_t j = 0; j < u_n; ++j)
        {
            u(j) = us(iter, j);
        }
        predict_and_update(z, u, bfs, iter);
    }
    return bfs;
}

template <typename T>
void KalmanFilter<T>::predict_and_update(array_type const & z, array_type const & u, KalmanStateInfo<T> & bfs, ssize_t iter)
{
    SimpleArray<T> & prior_state = bfs.prior_states;
    SimpleArray<T> & prior_state_covariance = bfs.prior_states_covariance;
    SimpleArray<T> & posterior_state = bfs.posterior_states;
    SimpleArray<T> & posterior_state_covariance = bfs.posterior_states_covariance;

    predict(u);
    for (ssize_t j = 0; j < m_state_size; ++j)
    {
        prior_state(iter, j) = m_x(j);
        for (ssize_t k = 0; k < m_state_size; ++k)
        {
            prior_state_covariance(iter, j, k) = m_p(j, k);
        }
    }

    update(z);
    for (ssize_t j = 0; j < m_state_size; ++j)
    {
        posterior_state(iter, j) = m_x(j);
        for (ssize_t k = 0; k < m_state_size; ++k)
        {
            posterior_state_covariance(iter, j, k) = m_p(j, k);
        }
    }
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
