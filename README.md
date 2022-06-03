# (Bo)son (S)ampling (S)imulator

The project is meant to implement the ideas proposed in references [1] and [2] for
classically simulating lossy boson sampling.  

## Summary

  - [Getting Started](#getting-started)
  - [Running the tests](#running-the-tests)
  - [Versioning](#versioning)
  - [Authors](#authors)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

## Getting Started

So far, to get the project one only needs to download it from the
[repository](https://github.com/Tomev-CTP/theboss) (or any other source).

It's also possible to download it as a package with

`pip install theboss`

which will download the package from [pypi](https://pypi.org/project/theboss/).

### Prerequisites

To run the scripts there will be needed some version of Python. During the development
I'm using 3.8, but I believe that it will also work just fine with some earlier or newer
version (at least for now). 

The package has been tested for python 3.7 to 3.10.

In some places I use `math.prod` and `typing` package, that's why 3.8 version is
desirable.

### Code style

Throughout this work PEP-8 will be used. There are several cases where this may go south.

* In some versions of the code matrices may be denoted by capital letters (as in standard mathematical notation). In
order to be more PEP-friendly I'll try to use prefix m_ instead of capital letters, e.g. m_u would be the equivalent of
U. Alternatively explicit use of matrix is also acceptable. 

## Running the tests

Just run all test in `tests` folder with `pytest` or via `tox` command.

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

  - [CTP PAS](http://www.cft.edu.pl/new/public/pl),
  - [Michał Oszmaniec](https://www.cft.edu.pl/pracownik/moszmaniec),
  - [Poór Boldizsár](https://github.com/boldar99) - major code review,
  - [Kolarovszki Zoltán](https://github.com/Kolarovszki) - output states _lexicographicalization_, 
  - [sisco0](https://github.com/sisco0) - some minor bugs pointed out.
  - This research was supported in part by [PL-Grid](https://www.plgrid.pl/) Infrastructure.
  
  
  
## References

This project follows / uses the ideas presented in following papers:

[0] Clifford P., Clifford R., [arXiv:1706.01260](https://arxiv.org/abs/1706.01260) [quant-ph].

[1] Oszmaniec M., Brod D. J., [arXiv:1801.06166](https://arxiv.org/abs/1801.06166) [quant-ph].

[2] Brod D. J., Oszmaniec M., [arXiv:1906.06696](https://arxiv.org/abs/1906.06696) [quant-ph].

[3] Maciejewski F. B., Zimboras Z., Oszmaniec M., [arXiv:1907.08518](https://arxiv.org/abs/1907.08518) [quant-ph].

[4] Mezzadri F., [arXiv:math-ph/0609050](https://arxiv.org/abs/math-ph/0609050) [math-ph].

[5] Clifford P., Clifford R., [arXiv:2005.04214](https://arxiv.org/abs/2005.04214) [quant-ph].
