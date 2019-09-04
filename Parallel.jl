push!(LOAD_PATH, ".")

module Parallel

using Random
using Future

GLOBAL_THREAD_RNG = []

function __init__()
    Random.seed!(trunc(Int64, time()))
    global GLOBAL_THREAD_RNG = [MersenneTwister(rand(1:(1<<30))) for i=1:Threads.nthreads()]
end

function threadRand(args...)
    return rand(GLOBAL_THREAD_RNG[Threads.threadid()], args...)
end

function threadRandn(args...)
    return randn(GLOBAL_THREAD_RNG[Threads.threadid()], args...)
end

function threadShuffle(args...)
    return shuffle(GLOBAL_THREAD_RNG[Threads.threadid()], args...)
end

end