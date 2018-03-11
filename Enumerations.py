class Algorithms:
    __range = 2
    unused, GradientDescent = range(__range)

    @staticmethod
    def size():
        return Algorithms.__range


class Regressions:
    __range = 3
    unused, LinearRegression, LogisticRegression = range(__range)

    @staticmethod
    def size():
        return Regressions.__range
