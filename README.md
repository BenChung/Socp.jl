SOCP.jl
============

The project aims to build a minimal, high-performance, dense, primal-dual interior point solver for second-order cone problems. It implements the same algorithm as [CVXOPT's `coneqp`](http://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf), though is written entirely in Julia. Ideally, it will eventually provide MathOptInterface bindings, allowing its use for generic SOCPs.

This solver is not particularly "clever"; it aims instead to have very low invocation overhead, allowing it to be used for small dense problems efficiently. At the present time, it is not ready for general use.

License
-------

Copyright 2019 Benjamin Chung

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
