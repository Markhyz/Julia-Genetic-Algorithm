push!(LOAD_PATH, ".")

module Parallel

using Random
using Future

const GLOBAL_THREAD_RNG = [MersenneTwister(rand(1:(1<<30))) for i=1:Threads.nthreads()];

function threadRand(args...)
    return rand(GLOBAL_THREAD_RNG[Threads.threadid()], args...)
end

function threadShuffle(args...)
    return shuffle(GLOBAL_THREAD_RNG[Threads.threadid()], args...)
end

end