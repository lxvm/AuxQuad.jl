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

    function integrand(x)
        re, im = reim(f2(x) + f2(x-x0))
        Integrands(re, imf2(x) + imf2(x-x0))
    end

    using QuadGK    # plain adaptive integration

    quadgk(x -> imf2(x) + imf2(x-x0), 0, 2pi, atol = 1e-5)   # 1.4271103714584847e-7
    quadgk(x -> imf2(x) + imf2(x-x0), 0, 2pi, rtol = 1e-5)   # 235619.45750214785

    quadgk(x -> imf2(x), 0, 2pi, rtol = 1e-5)   # 78539.81901117883

    quadgk(x -> imf2(x-x0), 0, 2pi, rtol = 1e-5)   # 157079.63263294287

    using AuxQuad   # auxiliary integration

    auxquadgk(integrand, 0, 2pi, atol=1e-2) # 628318.5306881254
    auxquadgk(integrand, 0, 2pi, rtol=1e-2) # 628318.5306867635

As can be seen from the example, plain integration can completely fail to capture the
integral despite using stringent tolerances. With a well-chosen auxiliary integrand, often
arising naturally from the structure of the integrand, the integration is much more robust
to error because it can resolve the regions of interest with the more-easily adaptively
integrable problem.
"""
module AuxQuad

using QuadGK: handle_infinities, Segment, cachedrule, InplaceIntegrand, alloc_segbuf
using DataStructures, LinearAlgebra
import Base.Order.Reverse
import QuadGK: evalrule

export auxquadgk, AuxQuadGK, Integrands, Errors, auxquadgk_count

struct Integrands{N,T}
    vals::NTuple{N,T}
end
Integrands(args...) = Integrands(args)
Base.eltype(::Type{Integrands{N,T}}) where {N,T} = T
Base.size(u::Integrands{N,T}) where {N,T} = only(unique(size, u.vals))
eachorder(::Integrands{N}) where N = ntuple(n -> IndexedOrdering(Reverse, n), Val(N))


struct Errors{N,T}
    vals::NTuple{N,T}
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
    @eval Base.:/(x::$T, y::$T) = $T(map(/, x.vals, y.vals))
    @eval Base.:/(x::$T, y) = $T(map(z -> z/y, x.vals))
    @eval Base.:/(y, x::$T) = $T(map(z -> y/z, x.vals))
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

struct Sequential end
struct Parallel{T,S}
    f::Vector{T} # array to store function evaluations
    old_segs::Vector{S} # array to store segments popped off of heap
    new_segs::Vector{S} # array to store segments to add to heap
end


"""
    Parallel(domain_type=Float64, range_type=Float64, error_type=Float64; order=7)

This helper will allocate a buffer to parallelize `quadgk` calls across function evaluations
with a given `domain_type`, i.e. the type of the integration limits, `range_type`, i.e. the
type of the range of the integrand, and `error_type`, the type returned by the `norm` given
to `quadgk`. The keyword `order` allocates enough memory so that the Gauss-Kronrod rule of
that order can initially be evaluated without additional allocations. By passing this buffer
to multiple compatible `quadgk` calls, they can all be parallelized without repeated
allocation.
"""
function Parallel(TX=Float64, TI=Float64, TE=Float64; order=7)
    Parallel(Vector{TI}(undef, 2*order+1), alloc_segbuf(TX,TI,TE), alloc_segbuf(TX,TI,TE, size=2))
end


evalrule(::Sequential, args...) = evalrule(args...)


function batcheval!(fx, f::F, x, a, s, l, n) where {F}
    Threads.@threads for i in 1:n
        z = i <= l ? x[i] : -x[n-i+1]
        fx[i] = f(a + (1 + z)*s)
    end
end
function batcheval!(fx, f::InplaceIntegrand{F}, x, a, s, l, n) where {F}
    Threads.@threads for i in 1:n
        z = i <= l ? x[i] : -x[n-i+1]
        fx[i] = zero(f.fx) # allocate the output
        f.f!(fx[i], a + (1 + z)*s)
    end
end

function parevalrule(fx, f::F, a,b, x,w,gw, nrm, l, n) where {F}
    n1 = 1 - (l & 1) # 0 if even order, 1 if odd order
    s = convert(eltype(x), 0.5) * (b-a)
    batcheval!(fx, f, x, a, s, l, n)
    if n1 == 0 # even: Gauss rule does not include x == 0
        Ik = fx[l] * w[end]
        Ig = zero(Ik)
    else # odd: don't count x==0 twice in Gauss rule
        f0 = fx[l]
        Ig = f0 * gw[end]
        Ik = f0 * w[end] + (fx[l-1] + fx[l+1]) * w[end-1]
    end
    for i = 1:length(gw)-n1
        fg = fx[2i] + fx[n-2i+1]
        fk = fx[2i-1] + fx[n-2i+2]
        Ig += fg * gw[i]
        Ik += fg * w[2i] + fk * w[2i-1]
    end
    Ik_s, Ig_s = Ik * s, Ig * s # new variable since this may change the type
    E = nrm(Ik_s - Ig_s)
    if isnan(E) || isinf(E)
        throw(DomainError(a+s, "integrand produced $E in the interval ($a, $b)"))
    end
    return Segment(oftype(s, a), oftype(s, b), Ik_s, E)
end

function evalrule(p::Parallel, f::F, a,b, x,w,gw, nrm) where {F}
    l = length(x)
    n = 2*l-1   # number of Kronrod points
    n <= length(p.f) || resize!(p.f, n)
    parevalrule(p.f, f, a,b, x,w,gw, nrm, l,n)
end

function eval_segs(p::Sequential, s::NTuple{N}, f::F, x,w,gw, nrm) where {N,F}
    return ntuple(i -> evalrule(p, f, s[i]..., x,w,gw, nrm), Val(N))
end
function eval_segs(p::Parallel, s, f::F, x,w,gw, nrm) where {F}
    l = length(x)
    n = 2*l-1   # number of Kronrod points
    m = length(s)
    resize!(p.new_segs, m)
    (nm = n*m) <= length(p.f) || resize!(p.f, nm)
    segs = collect(enumerate(s))
    Threads.@threads for item in segs
        i, (a, b) = item
        v = view(p.f, (1+(i-1)*n):(i*n))
        p.new_segs[i] = parevalrule(v, f, a, b, x,w,gw, nrm, l,n)
    end
    return p.new_segs
end


# Internal routine: integrate f over the union of the open intervals
# (s[1],s[2]), (s[2],s[3]), ..., (s[end-1],s[end]), using h-adaptive
# integration with the order-n Kronrod rule and weights of type Tw,
# with absolute tolerance atol and relative tolerance rtol,
# with maxevals an approximate maximum number of f evaluations.
function do_auxquadgk(f::F, s::NTuple{N,T}, n, atol, rtol, maxevals, nrm, segbuf, parallel) where {T,N,F}
    x,w,gw = cachedrule(T,n)

    @assert N ≥ 2
    segs = eval_segs(parallel, ntuple(i -> (s[i],s[i+1]), Val(N-1)), f, x,w,gw, nrm)
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
    for ord in eachorder(I)
        heapify!(segheap, ord)
        I, E, numevals = auxadapt(f, segheap, I, E, numevals, x,w,gw,n, atol_, rtol_, maxevals, nrm, ord, parallel)
        (E ≤ atol_ || E ≤ rtol_ * nrm(I) || numevals ≥ maxevals) && break
    end
    return (I, E)
end


# pop segments that contribute most to error
function pop_segs(::Sequential, segs, ord, E, tol)
    return (heappop!(segs, ord),)
end
function pop_segs(p::Parallel, segs, ord, E, tol)
    empty!(p.old_segs)
    while Base.Order.lt(ord, E, tol) && !isempty(segs)
        s = heappop!(segs, ord)
        E -= s.E
        push!(p.old_segs, s)
    end
    return p.old_segs
end

# bisect segments
function bisect_segs(p::Sequential, (s,), f::F, x,w,gw, nrm) where {F}
    mid = (s.a + s.b) / 2
    return eval_segs(p, ((s.a, mid), (mid, s.b)), f, x,w,gw, nrm)
end

function bisect_segs(p::Parallel, old_segs, f::F, x,w,gw, nrm) where {F}
    lims = map(s -> (mid=(s.a+s.b)/2 ; ((s.a,mid), (mid,s.b))), old_segs)
    eval_segs(p, Iterators.flatten(lims), f, x,w,gw, nrm)
end


# internal routine to perform the h-adaptive refinement of the integration segments (segs)
function auxadapt(f::F, segs::Vector{T}, I, E, numevals, x,w,gw,n, atol, rtol, maxevals, nrm, ord, parallel) where {F, T}
    # Pop the biggest-error segment and subdivide (h-adaptation)
    # until convergence is achieved or maxevals is exceeded.
    while (tol = max(atol, rtol*nrm(I)); Base.Order.lt(ord, E, tol)) && numevals < maxevals
        old_segs = pop_segs(parallel, segs, ord, E, tol)
        new_segs = bisect_segs(parallel, old_segs, f, x,w,gw, nrm)

        if f isa InplaceIntegrand
            for i = eachindex(old_segs)
                I .-= old_segs[i].I
            end
            for i = eachindex(new_segs)
                I .+= new_segs[i].I
            end
        else
            I = (I - sum(s -> s.I, old_segs)) + sum(s -> s.I, new_segs)
        end
        E = (E - sum(s -> s.E, old_segs)) + sum(s -> s.E, new_segs)
        numevals += length(new_segs)*(2n+1)

        # handle type-unstable functions by converting to a wider type if needed
        Tj = promote_type(T, typeof.(new_segs)...)
        if Tj !== T
            segs_ = Vector{Tj}(segs)
            foreach(s -> heappush!(segs_, s, ord), new_segs)
            return adapt(f, segs_,
                         I, E, numevals, x,w,gw,n, atol, rtol, maxevals, nrm, parallel)
        end

        foreach(s -> heappush!(segs, s, ord), new_segs)
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
       atol=nothing, rtol=nothing, maxevals=10^7, order=7, norm=norm, segbuf=nothing, parallel=Sequential()) where {T}
    handle_infinities(f, segs) do f, s, _
        do_auxquadgk(f, s, order, atol, rtol, maxevals, norm, segbuf, parallel)
    end
end

function auxquadgk_count(f, args...; kws...)
    count::Int = 0
    i, e = auxquadgk(args...; kws...) do x
        count += 1
        f(x)
    end
    return (i, e, count)
end

"""
Algorithm extension to Integrals.jl
"""
function AuxQuadGK end

end
