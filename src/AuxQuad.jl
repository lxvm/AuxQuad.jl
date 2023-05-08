"""
Package for auxiliary integration, i.e. integrating multiple functions at the same time
while ensuring that each converges to its own tolerance. This has a few advantages over
vector-valued integrands with custom norms in that the errors from different integrands can
be treated separated and the adaptive algorithm can decide which integrand to prioritize
based on whether others have already converged.
This results in conceptually simpler algorithms, especially when the various integrands may
differ in order of magnitude.

# Statement of need

Calculating integrals of the form a^2/(f(x)^2+a^2)^2 is challenging
in the a -> 0 limit because they become extremely localized while also having vanishingly
small tails. I.e. the tails are O(a^2) however the integral is O(a^-1). Thus, setting an
absolute tolerance is impractical, since the tolerance also needs to be O(a^2) to resolve
the tails (otherwise the peaks will be missed) and that may be asking for many more digits
than desired. Another option is to specify a relative tolerance, but a failure mode is that
if there is more than one peak to integrate, the algorithm may only resolve the first one
because the errors in the tails to find the other peaks become eclipsed by the first peak
error magnitudes. When the peak positions are known a priori, the convential solution is to
pass in several breakpoints to the integration interval so that each interval has at most
one peak, but often finding breakpoints can be an expensive precomputation that is better
avoided. Instead, an integrand related to the original may more reliably find the peaks
without requiring excessive integrand evaluations or being expensive to compute. Returning
to the original example, an ideal auxiliary integrand would be 1/(f(x)+im*a)^2, which has
O(1) tails and a O(1) integral. Thus the tails will be resolved in order to find the peaks,
which don't need to be resolved to many digits of accuracy. However, since one wants to find
the original integral to a certain number of digits, it may be necessary to adapt further
after the auxiliary integrand has converged. This is the problem the package aims to solve.

# Example

    f(x)    = sin(x)/(cos(x)+im*1e-5)   # peaked "nice" integrand
    imf(x)  = imag(f(x))                # peaked difficult integrand
    f2(x)   = f(x)^2                    # even more peaked
    imf2(x) = imf(x)^2                  # even more peaked!

    x0 = 0.1    # arbitrary offset of between peak

    integrand(x) = Integrands(f2(x) + f2(x-x0), imf2(x) + imf2(x-x0))

    using QuadGK    # plain adaptive integration

    quadgk(x -> imf2(x) + imf2(x-x0), 0, 2pi, atol = 1e-5)   # 1.4271103714584847e-7
    quadgk(x -> imf2(x) + imf2(x-x0), 0, 2pi, rtol = 1e-5)   # 235619.45750214785

    julia> quadgk(x -> imf2(x), 0, 2pi, rtol = 1e-5)   # 78539.81901117883

julia> quadgk(x -> imf2(x-x0), 0, 2pi, rtol = 1e-5)   # 157079.63263294287
    using AuxQuad   # auxiliary integration

    auxquadgk(integrand, 0, 2pi, atol=1e-2) # 628318.5306870549
    auxquadgk(integrand, 0, 2pi, rtol=1e-2) # 628318.5306872142

As can be seen from the example, plain integration can completely fail to capture the
integral despite using stringent tolerances. With a well-chosen auxiliary integrand, often
arising naturally from the structure of the integrand, the integration is much more robust
to error because it can resolve the regions of interest with the more-easily adaptively
integrable problem.
"""
module AuxQuad

using QuadGK: handle_infinities, Segment, evalrule, cachedrule, InplaceIntegrand, alloc_segbuf
using DataStructures, LinearAlgebra
import Base.Order.Reverse

export auxquadgk, Integrands, Errors

struct Integrands{T<:Tuple}
    vals::T
end
Integrands(args...) = Integrands(args)

struct Errors{T<:Tuple}
    vals::T
end
Errors(args...) = Errors(args)
Base.getindex(e::Errors, i::Integer) = e.vals[i]
LinearAlgebra.norm(x::Integrands) = Errors(map(norm, x.vals))
Base.isnan(e::Errors) = any(isnan, e.vals)
Base.isinf(e::Errors) = any(isinf, e.vals)
Base.zero(e::Errors) = Errors(map(zero, e.vals))
Base.isless(e::Errors, f::Errors) = all(isless(a,b) for (a,b) in zip(e.vals, f.vals))
Base.isless(e::Errors, x::Number) = all(<(x), e.vals)
Base.isless(x::Number, e::Errors) = all(>(x), e.vals)
Base.isequal(e::Errors, x::Number) = all(==(x), e.vals)
Base.isequal(x::Number, e::Errors) = all(==(x), e.vals)

# arithmetic
for T in (:Integrands, :Errors)
    @eval Base.:+(x::$T, y::$T) = $T(map(+, x.vals, y.vals))
    @eval Base.:-(x::$T, y::$T) = $T(map(-, x.vals, y.vals))
    @eval Base.:*(x::$T, y::$T) = $T(map(*, x.vals, y.vals))
    @eval Base.:*(x::$T, y) = $T(map(z -> z*y, x.vals))
    @eval Base.:*(y, x::$T) = $T(map(z -> z*y, x.vals))
end

const IntegrandsSegment{TI,TE} = Segment{<:Any,<:Integrands{TI},<:Errors{TE}}
const IntegrandsVector{TI,TE} = Vector{<:IntegrandsSegment{TI,TE}}

struct IndexedOrdering{T<:Base.Order.Ordering} <: Base.Order.Ordering
    o::T
    n::Integer
end

Base.Order.lt(o::IndexedOrdering, a::Number, b::Errors) =
    Base.Order.lt(o.o, a, b[o.n])
Base.Order.lt(o::IndexedOrdering, a::Errors, b::Number) =
    Base.Order.lt(o.o, a[o.n], b)
Base.Order.lt(o::IndexedOrdering, a::T, b::T) where {T<:Errors} =
    Base.Order.lt(o.o, a[o.n], b[o.n])
Base.Order.lt(o::IndexedOrdering, a::Number, b::IntegrandsSegment) =
    Base.Order.lt(o, a, b.E)
Base.Order.lt(o::IndexedOrdering, a::IntegrandsSegment, b::Number) =
    Base.Order.lt(o, a.E, b)
Base.Order.lt(o::IndexedOrdering, a::T, b::T) where {T<:IntegrandsSegment} =
    Base.Order.lt(o, a.E, b.E)

# Internal routine: integrate f over the union of the open intervals
# (s[1],s[2]), (s[2],s[3]), ..., (s[end-1],s[end]), using h-adaptive
# integration with the order-n Kronrod rule and weights of type Tw,
# with absolute tolerance atol and relative tolerance rtol,
# with maxevals an approximate maximum number of f evaluations.
function do_auxquadgk(f::F, s::NTuple{N,T}, n, atol, rtol, maxevals, nrm, segbuf) where {T,N,F}
    x,w,gw = cachedrule(T,n)

    @assert N ≥ 2
    segs = ntuple(i -> evalrule(f, s[i],s[i+1], x,w,gw, nrm), Val{N-1}())
    if f isa InplaceIntegrand
        I = f.I .= segs[1].I
        for i = 2:length(segs)
            I .+= segs[i].I
        end
    else
        I = sum(s -> s.I, segs)
    end
    E = sum(s -> s.E, segs)
    numevals = (2n+1) * (N-1)

    # logic here is mainly to handle dimensionful quantities: we
    # don't know the correct type of atol115, in particular, until
    # this point where we have the type of E from f.  Also, follow
    # Base.isapprox in that if atol≠0 is supplied by the user, rtol
    # defaults to zero.
    atol_ = something(atol, zero(E))
    rtol_ = something(rtol, iszero(atol_) ? sqrt(eps(one(eltype(x)))) : zero(eltype(x)))

    # optimize common case of no subdivision
    if E ≤ atol_ || E ≤ rtol_ * nrm(I) || numevals ≥ maxevals
        return (I, E) # fast return when no subdivisions required
    end

    segheap = segbuf === nothing ? collect(segs) : (resize!(segbuf, N-1) .= segs)
    for m in eachindex(I.vals)
        ord = IndexedOrdering(Reverse, m)
        heapify!(segheap, ord)
        I, E, numevals = auxadapt(f, segheap, I, E, numevals, x,w,gw,n, atol_, rtol_, maxevals, nrm, ord)
        (E ≤ atol_ || E ≤ rtol_ * nrm(I) || numevals ≥ maxevals) && break
    end
    return (I, E)
end

# internal routine to perform the h-adaptive refinement of the integration segments (segs)
function auxadapt(f::F, segs::Vector{T}, I, E, numevals, x,w,gw,n, atol, rtol, maxevals, nrm, ord) where {F, T}
    # Pop the biggest-error segment and subdivide (h-adaptation)
    # until convergence is achieved or maxevals is exceeded.
    while Base.Order.lt(ord, E, atol) && Base.Order.lt(ord, E, rtol*nrm(I)) && numevals < maxevals
        s = heappop!(segs, ord)
        mid = (s.a + s.b) / 2
        s1 = evalrule(f, s.a, mid, x,w,gw, nrm)
        s2 = evalrule(f, mid, s.b, x,w,gw, nrm)
        if f isa InplaceIntegrand
            I .= (I .- s.I) .+ s1.I .+ s2.I
        else
            I = (I - s.I) + s1.I + s2.I
        end
        E = (E - s.E) + s1.E + s2.E
        numevals += 4n+2

        # handle type-unstable functions by converting to a wider type if needed
        Tj = promote_type(typeof(s1), promote_type(typeof(s2), T))
        if Tj !== T
            return adapt(f, heappush!(heappush!(Vector{Tj}(segs), s1, ord), s2, ord),
                         I, E, numevals, x,w,gw,n, atol, rtol, maxevals, nrm)
        end

        heappush!(segs, s1, ord)
        heappush!(segs, s2, ord)
    end

    # re-sum (paranoia about accumulated roundoff)
    if f isa InplaceIntegrand
        I .= segs[1].I
        E = segs[1].E
        for i in 2:length(segs)
            I .+= segs[i].I
            E += segs[i].E
        end
    else
        I = segs[1].I
        E = segs[1].E
        for i in 2:length(segs)
            I += segs[i].I
            E += segs[i].E
        end
    end
    return (I, E, numevals)
end

auxquadgk(f, segs...; kws...) =
    auxquadgk(f, promote(segs...)...; kws...)

function auxquadgk(f, segs::T...;
       atol=nothing, rtol=nothing, maxevals=10^7, order=7, norm=norm, segbuf=nothing) where {T}
    handle_infinities(f, segs) do f, s, _
        do_auxquadgk(f, s, order, atol, rtol, maxevals, norm, segbuf)
    end
end

end
