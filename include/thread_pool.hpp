#pragma once
// Persistent thread pool + parallel_for. This is the "own threadpool" half
// of what ggml does in ggml-cpu (ggml_threadpool): worker threads are
// created ONCE and parked on a condition variable, so a parallel region
// costs a wake-up, not a pthread_create per call (which is what the old
// matmul_fast paid on every matmul).
//
// Design, kept deliberately small:
//   * parallel_for(n, fn, min_per_task) splits [0, n) into contiguous
//     chunks, one per task, and calls fn(begin, end) for each chunk. The
//     calling thread works too (it is "thread 0"), so N-way parallelism
//     needs only N-1 workers.
//   * Chunks are claimed under a mutex. There are only ~num_threads of
//     them per call, so the lock is cold and the code stays obviously
//     correct -- no lock-free tricks to get wrong.
//   * Nested parallel_for (e.g. multi-head attention parallelizes over
//     heads, and each head calls matmul which would also like to
//     parallelize) runs the inner region inline on the current thread.
//     Only the outermost level fans out, so we never oversubscribe or
//     deadlock waiting on a pool whose workers are all busy.
//   * Exceptions thrown inside a task are captured and re-thrown on the
//     calling thread after all tasks finish.
//   * set_num_threads(n) caps how many tasks a region is split into. The
//     pool itself always owns kMaxThreads-1 workers, so tests can ask for
//     1, 2, 4, 8 threads on any machine and get that many chunks.
//
// All kernels split work so that every OUTPUT element is computed by
// exactly one thread with the same summation order as the serial code,
// so threaded results are bit-identical to single-threaded results. The
// tests rely on (and verify) that property.

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace tinyml {

class ThreadPool {
public:
    static constexpr int kMaxThreads = 8; // total parallelism incl. caller

    ThreadPool() {
        workers_.reserve(kMaxThreads - 1);
        for (int i = 0; i < kMaxThreads - 1; ++i) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
        }
        cv_work_.notify_all();
        for (auto& w : workers_) w.join();
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // Run fn(begin, end) over [0, n) split into up to max_tasks chunks.
    void run(int64_t n, const std::function<void(int64_t, int64_t)>& fn,
             int64_t min_per_task, int max_tasks) {
        if (n <= 0) return;
        min_per_task = std::max<int64_t>(1, min_per_task);
        int64_t by_grain = (n + min_per_task - 1) / min_per_task;
        int tasks = (int)std::min<int64_t>({(int64_t)max_tasks, by_grain, (int64_t)kMaxThreads, n});

        if (tasks <= 1 || in_parallel_region()) {
            fn(0, n);
            return;
        }

        std::lock_guard<std::mutex> submit_guard(submit_m_); // one region at a time
        const int64_t chunk = (n + tasks - 1) / tasks;
        tasks = (int)((n + chunk - 1) / chunk);

        {
            std::lock_guard<std::mutex> lk(m_);
            fn_ = &fn;
            n_ = n;
            chunk_ = chunk;
            n_tasks_ = tasks;
            next_task_ = 0;
            done_ = 0;
            error_ = nullptr;
        }
        cv_work_.notify_all();

        drain(); // the caller helps

        std::unique_lock<std::mutex> lk(m_);
        cv_done_.wait(lk, [&] { return done_ == n_tasks_; });
        fn_ = nullptr;
        n_tasks_ = 0;
        next_task_ = 0;
        std::exception_ptr err = error_;
        error_ = nullptr;
        lk.unlock();
        if (err) std::rethrow_exception(err);
    }

    static bool in_parallel_region() { return tl_depth_() > 0; }

private:
    static int& tl_depth_() {
        thread_local int depth = 0;
        return depth;
    }

    // Claim and run tasks until none are left.
    void drain() {
        for (;;) {
            const std::function<void(int64_t, int64_t)>* fn;
            int64_t b, e;
            {
                std::lock_guard<std::mutex> lk(m_);
                if (next_task_ >= n_tasks_) return;
                int t = next_task_++;
                fn = fn_;
                b = (int64_t)t * chunk_;
                e = std::min<int64_t>(n_, b + chunk_);
            }
            ++tl_depth_();
            try {
                (*fn)(b, e);
            } catch (...) {
                std::lock_guard<std::mutex> lk(m_);
                if (!error_) error_ = std::current_exception();
            }
            --tl_depth_();
            {
                std::lock_guard<std::mutex> lk(m_);
                if (++done_ == n_tasks_) cv_done_.notify_all();
            }
        }
    }

    void worker_loop() {
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_work_.wait(lk, [&] { return stop_ || next_task_ < n_tasks_; });
                if (stop_) return;
            }
            drain();
        }
    }

    std::vector<std::thread> workers_;
    std::mutex m_;
    std::mutex submit_m_;
    std::condition_variable cv_work_, cv_done_;
    bool stop_ = false;

    const std::function<void(int64_t, int64_t)>* fn_ = nullptr;
    int64_t n_ = 0, chunk_ = 1;
    int n_tasks_ = 0, next_task_ = 0, done_ = 0;
    std::exception_ptr error_;
};

namespace detail {
inline std::atomic<int>& thread_limit() {
    static std::atomic<int> limit = [] {
        int n = (int)std::thread::hardware_concurrency();
        if (const char* env = std::getenv("TINYML_THREADS")) {
            int v = std::atoi(env);
            if (v > 0) n = v;
        }
        return std::atomic<int>(std::max(1, std::min(n, ThreadPool::kMaxThreads)));
    }();
    return limit;
}
} // namespace detail

inline ThreadPool& global_pool() {
    static ThreadPool pool;
    return pool;
}

// Cap on how many tasks one parallel region is split into (1 = serial).
inline void set_num_threads(int n) {
    detail::thread_limit().store(std::max(1, std::min(n, ThreadPool::kMaxThreads)));
}
inline int get_num_threads() { return detail::thread_limit().load(); }

// RAII helper: temporarily change the thread count (handy in tests/benchmarks).
class ScopedThreads {
public:
    explicit ScopedThreads(int n) : prev_(get_num_threads()) { set_num_threads(n); }
    ~ScopedThreads() { set_num_threads(prev_); }
    ScopedThreads(const ScopedThreads&) = delete;
    ScopedThreads& operator=(const ScopedThreads&) = delete;
private:
    int prev_;
};

// parallel_for(n, fn, min_per_task, max_tasks): fn(begin, end) is called on
// disjoint chunks covering [0, n). min_per_task keeps chunks big enough to
// be worth a wake-up; max_tasks (0 = use the global limit) caps the split.
template <typename F>
inline void parallel_for(int64_t n, F&& fn, int64_t min_per_task = 1, int max_tasks = 0) {
    if (max_tasks <= 0) max_tasks = get_num_threads();
    if (max_tasks <= 1 || n <= 1) {
        if (n > 0) fn((int64_t)0, n);
        return;
    }
    std::function<void(int64_t, int64_t)> f = std::forward<F>(fn);
    global_pool().run(n, f, min_per_task, max_tasks);
}

} // namespace tinyml
