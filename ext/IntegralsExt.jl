module IntegralsExt

using LinearAlgebra
using Integrals
using AuxQuad
import AuxQuad:AuxQuadGK

struct AuxQuadGKJL{F,S} <: SciMLBase.AbstractIntegralAlgorithm
    order::Int
    norm::F
    segbuf::S
end
AuxQuadGK(; order = 7, norm = norm, segbuf = nothing) = AuxQuadGKJL(order, norm, segbuf)

function Integrals.__solvebp_call(prob::IntegralProblem, alg::AuxQuadGKJL, sensealg, lb, ub, p;
                        reltol = 1e-8, abstol = 1e-8,
                        maxiters = typemax(Int))
    if isinplace(prob) || lb isa AbstractArray || ub isa AbstractArray
        error("AuxQuadGK only accepts one-dimensional quadrature problems.")
    end
    @assert prob.batch == 0
    @assert prob.nout == 1
    p = p
    f = x -> prob.f(x, p)
    val, err = auxquadgk(f, lb, ub,
                      rtol = reltol, atol = abstol, order = alg.order, norm = alg.norm, segbuf=alg.segbuf)
    SciMLBase.build_solution(prob, AuxQuadGK(), val, err, retcode = ReturnCode.Success)
end

end