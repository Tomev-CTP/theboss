# (Bo)son (S)ampling (S)imulator

The project is meant to implement the ideas proposed in references [1] and [2] for classically simulating 
lossy boson sampling.  

## Summary

  - [Getting Started](#getting-started)
  - [Running the tests](#running-the-tests)
  - [Versioning](#versioning)
  - [Authors](#authors)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

## Getting Started

So far, to get the project one only needs to download it from this repository (or any other source).

### Prerequisites

To run the scripts there will be needed some version of Python. During development I'm using 3.7, but I believe that
it will also work just fine with some earlier / newer version (at least for now and I cannot guarantee it).

### Code style

Throughout this work PEP-8 will be used. There are several cases where this may go south.

* In some versions of the code matrices may be denoted by capital letters (as in standard mathematical notation). In
order to be more PEP-friendly I'll try to use prefix m_ instead of capital letters, e.g. m_u would be the equivalent of
U. Alternatively explicit use of matrix is also acceptable. 

## Running the tests

Just run all test in src_tests folder.

## Versioning

I'll use [SemVer](http://semver.org/) for versioning and try to keep track of the version in the tags. 

## Authors

  - [Tomasz Rybotycki](https://github.com/Tomev)
  - [Malhavok](https://github.com/Malhavok) - *These precious reviews <3*
  - [Billie Thompson](https://github.com/PurpleBooth) - *Provided README Template* 

## License

This project is licensed under the [Apache License, v.2.0](LICENSE.md).
See the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

  - Daniel Brod and Michał Oszmaniec, 
  - [CTP PAS](http://www.cft.edu.pl/new/public/pl).
  
  
## References

The work will follow the ideas presented in following works:

[0] Clifford P., Clifford R., [arXiv:1706.01260](https://arxiv.org/abs/1706.01260) [quant-ph].

[1] Oszmaniec M., Brod D. J., [arXiv:1801.06166](https://arxiv.org/abs/1801.06166) [quant-ph].

[2] Brod D. J., Oszmaniec M., [arXiv:1906.06696](https://arxiv.org/abs/1906.06696) [quant-ph].

[3] Maciejewski F. B., Zimboras Z., Oszmaniec M., [arXiv:1907.08518](https://arxiv.org/abs/1907.08518) [quant-ph].

[4] Lal Mehta M., Random Matrices, Academic Press, ISBN: 9780080474113, 2004.
