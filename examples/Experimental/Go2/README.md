# Unitree Go2

A starting point for a Go2 quadruped case. `test.jl` loads the robot from
`urdf/go2.urdf` with RigidBodyDynamics.jl, simulates 1.5 s at 200 Hz under a
piecewise-constant torque sequence (explicit Euler), and optimizes that sequence
by gradient descent, with ForwardDiff gradients through the whole simulation, to
raise the centre of mass at 0.9 s under a small torque penalty. No decision rule
is trained here yet.

```bash
cd examples/Experimental/Go2
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. test.jl
```
