# CAPITAL
**C**ommunication-**A**voiding **P**arallelism-**I**ncreasing ma**T**rix f**A**ctorization **L**ibrary

We merge the **CANDMC** library with new algorithms for Cholesky factorization and QR factorization of dense matrices.

Optimized parallel schedules for:
* Cholesky factorization
* QR factorization
* Eigen-decomposition

Highlights include:
* Communication-avoiding Cholesky QR2 (https://ieeexplore.ieee.org/abstract/document/8820981)
* Pipelined Communication-avoiding QR (*add HPDC arxiv link*)
* Symmetric full-to-band reduction with communication-avoiding QR (https://dl.acm.org/citation.cfm?id=3087561)
