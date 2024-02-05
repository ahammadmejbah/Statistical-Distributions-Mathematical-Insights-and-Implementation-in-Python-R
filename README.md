<div align="center">
      <h2> <img src="http://bytesofintelligences.com/wp-content/uploads/2023/03/Exploring-AIs-Secrets-1.png" width="300px"><br/> <p> Statistical Distributions: <span style="color: #007BFF;"></span> Mathematical Insights and <span style="color: red;"> Implementation in Python & R</span></p> </h2>
     </div>

<body>
<p align="center">
  <a href="mailto:ahammadmejbah@gmail.com"><img src="https://img.shields.io/badge/Email-ahammadmejbah%40gmail.com-blue?style=flat-square&logo=gmail"></a>
  <a href="https://github.com/BytesOfIntelligences"><img src="https://img.shields.io/badge/GitHub-%40BytesOfIntelligences-lightgrey?style=flat-square&logo=github"></a>
  <a href="https://linkedin.com/in/ahammadmejbah"><img src="https://img.shields.io/badge/LinkedIn-Mejbah%20Ahammad-blue?style=flat-square&logo=linkedin"></a>
  <a href="https://bytesofintelligences.com/"><img src="https://img.shields.io/badge/Website-Bytes%20of%20Intelligence-lightgrey?style=flat-square&logo=google-chrome"></a>
  <a href="https://www.youtube.com/@BytesOfIntelligences"><img src="https://img.shields.io/badge/YouTube-BytesofIntelligence-red?style=flat-square&logo=youtube"></a>
  <a href="https://www.researchgate.net/profile/Mejbah-Ahammad-2"><img src="https://img.shields.io/badge/ResearchGate-Mejbah%20Ahammad-blue?style=flat-square&logo=researchgate"></a>
  <br>
  <img src="https://img.shields.io/badge/Phone-%2B8801874603631-green?style=flat-square&logo=whatsapp">
  <a href="https://www.hackerrank.com/profile/ahammadmejbah"><img src="https://img.shields.io/badge/Hackerrank-ahammadmejbah-green?style=flat-square&logo=hackerrank"></a>
</p>


# Normal Distribution

The normal distribution, also known as the Gaussian distribution, is a probability distribution that is symmetric about the mean, showing that data near the mean are more frequent in occurrence than data far from the mean. It has numerous applications in real life, including in the fields of statistics, finance, natural and social sciences.


The probability density function (PDF) of the normal distribution is given by the formula:

![eqauations](https://latex.codecogs.com/svg.image?&space;f(x|\mu,\sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}})

where:
- ![equations](https://latex.codecogs.com/svg.image?x) is the variable,
- ![equations](https://latex.codecogs.com/svg.image?\mu) is the mean,
- ![equations](https://latex.codecogs.com/svg.image?\sigma^2) is the variance,
- ![equations](https://latex.codecogs.com/svg.image?\sigma) is the standard deviation, and
- ![equations](https://latex.codecogs.com/svg.image?e) is the base of the natural logarithm.

Let's calculate the probability of a random variable ![equations](https://latex.codecogs.com/svg.image?X), which follows a normal distribution with a mean ![equations](https://latex.codecogs.com/svg.image?\mu) = ![equations](https://latex.codecogs.com/svg.image?50) and a standard deviation ![equations](https://latex.codecogs.com/svg.image?\sigma) = ![equations](https://latex.codecogs.com/svg.image?10), taking on a value less than ![equations](https://latex.codecogs.com/svg.image?60).

1. **Identify Parameters**: ![equations](https://latex.codecogs.com/svg.image?\mu) = ![equations](https://latex.codecogs.com/svg.image?50), ![equations](https://latex.codecogs.com/svg.image?\sigma) = ![equations](https://latex.codecogs.com/svg.image?10)
2. **Standardize the Variable** (convert ![equations](https://latex.codecogs.com/svg.image?X) to ![eqauations](https://latex.codecogs.com/svg.image?Z)-![equations](https://latex.codecogs.com/svg.image?score):</br>
	![equations](https://latex.codecogs.com/svg.image?&space;Z=\frac{X-\mu}{\sigma}=\frac{60-50}{10}=1&space;)
3. **Use Z-table** or cumulative distribution function (CDF) to find the probability:
   - For simplicity, let's say ![equations](https://latex.codecogs.com/svg.image?P(X<60)=P(Z<1)) corresponds to approximately ![equations](https://latex.codecogs.com/svg.image?0.8413) from the Z-table.

###  Python Code


```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Parameters
mu = 50
sigma = 10
x = 60

# Plot
x_values = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
y_values = stats.norm.pdf(x_values, mu, sigma)
plt.plot(x_values, y_values, label='Normal Distribution')
plt.fill_between(x_values, y_values, 
where=(x_values < x), color='skyblue', 
alpha=0.5, label='Area under curve')
plt.legend()
plt.title('Normal Distribution ($\mu=50$, $\sigma=10$)')
plt.xlabel('X')
plt.ylabel('Probability Density')
plt.show()

# Probability Calculation
prob = stats.norm.cdf(x, mu, sigma)
print(f'Probability (X < {x}): {prob}')
```


<div align="center">

![image](https://github.com/BytesOfIntelligences/Distributions/assets/56669333/71753e85-cdf9-4635-84c4-5f2696692602) </div>

### R Code 


```r
# Parameters
mu <- 50
sigma <- 10
x <- 60

# Plot
x_values <- seq(mu - 4*sigma, mu + 4*sigma, length.out = 1000)
y_values <- dnorm(x_values, mean = mu, sd = sigma)
plot(x_values, y_values, type = 'l', 
main = 'Normal Distribution (mu=50, sigma=10)',
xlab = 'X', ylab = 'Density')
polygon(c(x_values[x_values<x], x),
c(y_values[x_values<x], 0), col = 'skyblue')

# Probability Calculation
prob <- pnorm(x, mean = mu, sd = sigma)
cat(sprintf('Probability (X < %d): %f', x, prob))
```

Both Python and R code snippets above will visualize the normal distribution curve for a mean (![equations](https://latex.codecogs.com/svg.image?\mu)) of 50 and a standard deviation (![equations](https://latex.codecogs.com/svg.image?\sigma)) of 10. They also shade the area under the curve for values less than 60, which corresponds to the probability calculation for ![equations](https://latex.codecogs.com/svg.image?X) < ![equations](https://latex.codecogs.com/svg.image?60). The area under the curve to the left of ![equations](https://latex.codecogs.com/svg.image?X) = ![equations](https://latex.codecogs.com/svg.image?60) represents the probability we calculated in our step-by-step example.


# 

# Uniform Distribution

The uniform distribution is a type of probability distribution in which all outcomes are equally likely. It can be categorized into two types: discrete uniform distribution and continuous uniform distribution. Here, we'll focus on the continuous uniform distribution, which is relevant for continuous intervals of numbers.

The probability density function (PDF) of the continuous uniform distribution for a random variable ![eqauations](https://latex.codecogs.com/svg.image?X) taking values in the interval ![equations](https://latex.codecogs.com/svg.image?[a,b]) is given by:

![image](https://github.com/BytesOfIntelligences/Distributions/assets/56669333/9bd5dd95-fd39-4fa0-b4b1-91bf6577dce0)


where:
- ![equations](https://latex.codecogs.com/svg.image?a) is the minimum value,
- ![equations](https://latex.codecogs.com/svg.image?b) is the maximum value,
- ![equations](https://latex.codecogs.com/svg.image?f(x|a,b)) is the probability density function.

The mean \(\mu\) and variance \(\sigma^2\) of the uniform distribution are given by:


![equations](https://latex.codecogs.com/svg.image?\mu=\frac{a&plus;b}{2})</br>![equations](https://latex.codecogs.com/svg.image?\sigma^2=\frac{(b-a)^2}{12})


### 3. Random Example with Step-by-Step Calculation

Suppose we have a continuous uniform distribution with the interval \([20, 30]\). Let's calculate the probability that a randomly selected value from this distribution is less than \(25\).

1. **Identify Parameters**: \(a = 20\), \(b = 30\)
2. **Calculate Probability**: For a continuous uniform distribution, the probability that \(X < x\) is proportional to the length of the interval from \(a\) to \(x\), compared to the total interval length from \(a\) to \(b\).

![equations](https://latex.codecogs.com/svg.image?&space;P(X<25)=\frac{25-20}{30-20}=\frac{5}{10}=0.5)

### Python Code

Visualization of uniform distribution for ![equations](https://latex.codecogs.com/svg.image?a) = ![equations](https://latex.codecogs.com/svg.image?20) and ![equations](https://latex.codecogs.com/svg.image?b) = ![equations](https://latex.codecogs.com/svg.image?30) and calculates the probability for ![equations](https://latex.codecogs.com/svg.image?X) < ![equations](https://latex.codecogs.com/svg.image?25).

```python
import numpy as np
import matplotlib.pyplot as plt

# Parameters
a = 20
b = 30
x = 25

# Uniform Distribution
values = np.linspace(a, b, 1000)
pdf_values = np.ones_like(values) / (b - a)

# Plot
plt.plot(values, pdf_values, label='Uniform Distribution')
plt.fill_between(values, pdf_values, 
where=(values < x), color='skyblue', 
alpha=0.5, label='Area under curve (X < 25)')
plt.legend()
plt.title('Uniform Distribution ($a=20$, $b=30$)')
plt.xlabel('X')
plt.ylabel('Probability Density')
plt.show()

# Probability Calculation
prob = (x - a) / (b - a)
print(f'Probability (X < {x}): {prob}')
```

![image](https://github.com/BytesOfIntelligences/Distributions/assets/56669333/6b440a68-6f7a-452e-8936-aa2dadb7ca01)

### R Code 

```r
# Parameters
a <- 20
b <- 30
x <- 25

# Plot
values <- seq(a, b, length.out = 1000)
pdf_values <- dunif(values, min = a, max = b)

plot(values, pdf_values, type = 'l', 
main = 'Uniform Distribution (a=20, b=30)',
xlab = 'X', ylab = 'Density')
polygon(c(a, values[values<x], x), c(0, pdf_values[values<x], 0), col = 'skyblue')

# Probability Calculation
prob <- (x - a) / (b - a)
cat(sprintf('Probability (X < %d): %f', x, prob))
```

The aforementioned Python and R code excerpts both illustrate the uniform distribution for the interval ![equations](https://latex.codecogs.com/svg.image?[20,30]) and emphasize the region beneath the curve corresponding to the probability calculation for values below ![equations](https://latex.codecogs.com/svg.image?[X,25]). The presented visualisation and calculation serve to illustrate how the uniform distribution ensures that every outcome within the designated range is assigned an equal probability.

#



# Binomial Distribution

The binomial distribution is a discrete probability distribution that describes the number of successes in a fixed number of independent trials of a binary experiment, where each trial has only two possible outcomes (commonly termed as success and failure). This distribution is characterized by the number of trials, ![equations](https://latex.codecogs.com/svg.image?n), and the probability of success in each trial, ![equations](https://latex.codecogs.com/svg.image?p).

The probability of getting exactly ![equations](https://latex.codecogs.com/svg.image?k) successes in ![equations](https://latex.codecogs.com/svg.image?n) trials is given by the binomial probability formula:

![equations](https://latex.codecogs.com/svg.image?P(X=K)) = ![equations](https://latex.codecogs.com/svg.image?\binom{n}{k}p^k(1-p)^{n-k})

where:
- ![equations](https://latex.codecogs.com/svg.image?X) is the random variable representing the number of successes,
- ![equations](https://latex.codecogs.com/svg.image?n) is the number of trials,
- ![equations](https://latex.codecogs.com/svg.image?k) is the number of successful trials,
- ![equations](https://latex.codecogs.com/svg.image?p) is the probability of success on a single trial, and
- ![equations](https://latex.codecogs.com/svg.image?\binom{n}{k}) is the binomial coefficient, calculated as ![equations](https://latex.codecogs.com/svg.image?\frac{n!}{k!(n-k)!}).

Find the probability of getting exactly 3 heads when flipping a fair coin 5 times.

- **Identify Parameters**: ![equations](https://latex.codecogs.com/svg.image?n) = ![equations](https://latex.codecogs.com/svg.image?5) (trials), ![equations](https://latex.codecogs.com/svg.image?k) = ![equations](https://latex.codecogs.com/svg.image?3) (successes), ![equations](https://latex.codecogs.com/svg.image?p) = ![equations](https://latex.codecogs.com/svg.image?0.5) (probability of success, since the coin is fair)

- **Calculation**:
  - The binomial coefficient ![equations](https://latex.codecogs.com/svg.image?\binom{5}{3}=\frac{5!}{3!(5-3)!}=10)
  - Therefore, ![equations](https://latex.codecogs.com/svg.image?P(X=3)=10\times(0.5)^3\times(0.5)^{5-3}=10\times&space;0.125\times&space;0.25=0.3125)

### Python Code

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# Parameters
n = 5
p = 0.5
k = np.arange(0, n+1)

# Binomial Distribution
pmf_values = binom.pmf(k, n, p)

# Plot
plt.bar(k, pmf_values, color='skyblue')
plt.title('Binomial Distribution (n=5, p=0.5)')
plt.xlabel('Number of Successes (k)')
plt.ylabel('Probability')
plt.xticks(k)
plt.grid(axis='y', alpha=0.75)
plt.show()
```

![image](https://github.com/BytesOfIntelligences/Distributions/assets/56669333/1ba798d6-9a42-4c52-a806-1e5ce02e5102)

### R Code 

```r
# Parameters
n <- 5
p <- 0.5
k <- 0:n

# Binomial Distribution
pmf_values <- dbinom(k, size = n, prob = p)

# Plot
barplot(pmf_values, names.arg = k, col = 'skyblue', 
main = 'Binomial Distribution (n=5, p=0.5)', 
xlab = 'Number of Successes (k)', ylab = 'Probability')
```

Both the Python and R code snippets above visualize the binomial distribution for \(n = 5\) trials and a success probability ![equations](https://latex.codecogs.com/svg.image?p) = ![equations](https://latex.codecogs.com/svg.image?0.5), demonstrating the probability of achieving ![equations](https://latex.codecogs.com/svg.image?k) successes in those trials. These visualizations help illustrate the discrete nature of the binomial distribution and how the probability of successes varies with ![equations](https://latex.codecogs.com/svg.image?k), given a fixed number of trials and a constant probability of success.


#

# Poisson Distribution

The Poisson distribution is a discrete probability distribution that expresses the probability of a given number of events happening in a fixed interval of time or space if these events occur with a known constant mean rate and independently of the time since the last event. The Poisson distribution can also apply to the number of events in other specified intervals such as distance, area, or volume.


The probability mass function (PMF) of the Poisson distribution for observing ![equations](https://latex.codecogs.com/svg.image?k) events in an interval is given by:

![equations](https://latex.codecogs.com/svg.image?P(X=k)=\frac{\lambda^k&space;e^{-\lambda}}{k!})

where:
- ![equations](https://latex.codecogs.com/svg.image?X) is the random variable representing the number of events,
- ![equations](https://latex.codecogs.com/svg.image?k) is the number of occurrences of an event,
- ![equations](https://latex.codecogs.com/svg.image?\lambda) is the average number of events in an interval,
- ![equations](https://latex.codecogs.com/svg.image?e) is the base of the natural logarithm (approximately equal to 2.71828).

Suppose we are observing the number of cars passing through a toll booth in an hour, and the average number of cars observed in an hour is 10. What is the probability of observing exactly 7 cars in the next hour?

- **Identify Parameters**: ![equations](https://latex.codecogs.com/svg.image?\lambda) = ![equations](https://latex.codecogs.com/svg.image?10) (average rate of success), ![equations](https://latex.codecogs.com/svg.image?k) = ![equations](https://latex.codecogs.com/svg.image?7) (number of cars we're interested in)


  - Using the Poisson formula: ![euations](https://latex.codecogs.com/svg.image?P(X=7)=\frac{10^7&space;e^{-10}}{7!})
  - Calculate the factorial of ![equations](https://latex.codecogs.com/svg.image?7:7!=5040)
  - Compute the probability: ![equations](https://latex.codecogs.com/svg.image?P(X=7)\frac{10000000&space;e^{-10}}{5040}0.0908)

### Python Code

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# Parameters
lambda_ = 10
k = np.arange(0, 20)

# Poisson Distribution
pmf_values = poisson.pmf(k, lambda_)

# Plot
plt.bar(k, pmf_values, color='skyblue')
plt.title('Poisson Distribution ($\lambda=10$)')
plt.xlabel('Number of Cars (k)')
plt.ylabel('Probability')
plt.xticks(k)
plt.grid(axis='y', alpha=0.75)
plt.show()
```

![image](https://github.com/BytesOfIntelligences/Distributions/assets/56669333/fe2cf90e-c35d-428e-8bec-0711636e9048)

### R Code

```r
# Parameters
lambda <- 10
k <- 0:20

# Poisson Distribution
pmf_values <- dpois(k, lambda)

# Plot
barplot(pmf_values, names.arg = k, col = 'skyblue',
main = 'Poisson Distribution (lambda=10)', 
xlab = 'Number of Cars (k)', ylab = 'Probability')
```


#

# Exponential Distribution

The Exponential Distribution is a continuous probability distribution used to model the time between events in a Poisson point process, i.e., a process in which events occur continuously and independently at a constant average rate. It’s a key tool in the field of reliability engineering and queuing theory, describing how long an item of interest will last or the time until the next event (such as system failure or customer arrival) occurs.

The probability density function (PDF) of the Exponential Distribution is given by:

![image](https://github.com/BytesOfIntelligences/Distributions/assets/56669333/b51af63e-68da-4df5-b208-23b20a37d6d2)

where:
- ![equations](https://latex.codecogs.com/svg.image?x) is the random variable (time between events),
- ![equations](https://latex.codecogs.com/svg.image?\lambda) is the rate parameter, which is the reciprocal of the mean ![equations](https://latex.codecogs.com/svg.image?\mu) = ![equations](https://latex.codecogs.com/svg.image?\frac{1}{\lambda}) of the distribution.

A customer service desk receives on average 3 calls per hour. What is the probability that the next call will happen within the next 15 minutes?

**Given**: ![equations](https://latex.codecogs.com/svg.image?\lambda) = ![equations](https://latex.codecogs.com/svg.image?3) calls per hour.

**Find**: Probability that the next call happens within \(t = 0.25\) hours (15 minutes).

**Solution**:

1. **Convert Parameters**: ![equations](https://latex.codecogs.com/svg.image?\lambda) = ![equations](https://latex.codecogs.com/svg.image?3), ![equations](https://latex.codecogs.com/svg.image?t) = ![equations](https://latex.codecogs.com/svg.image?0.25)
2. **Apply PDF**: ![equations](https://latex.codecogs.com/svg.image?P(X\leq&space;t)=1-e^{-\lambda&space;t})
3. **Compute Probability**:
   - ![equations](https://latex.codecogs.com/svg.image?P(X\leq&space;0.25)=1-e^{-3\times&space;0.25}=1-e^{-0.75}0.527)

### Python Code

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon

class ExponentialDistribution:
    def __init__(self, rate):
        self.rate = rate

    def plot_pdf(self, x_range):
        x = np.linspace(0, x_range, 1000)
        y = self.rate * np.exp(-self.rate * x)
        plt.figure(figsize=(8, 4))
        plt.plot(x, y, label=f'λ={self.rate}')
        plt.title('Exponential Distribution PDF')
        plt.xlabel('Time')
        plt.ylabel('PDF')
        plt.legend()
        plt.show()

    def compute_probability(self, time):
        return 1 - np.exp(-self.rate * time)

# Instantiate and use
exp_dist = ExponentialDistribution(3)
exp_dist.plot_pdf(2)  # Plot PDF for 2 hours
print(f'Probability of next call within 15 minutes: {exp_dist.compute_probability(0.25):.3f}')
```

![image](https://github.com/BytesOfIntelligences/Distributions/assets/56669333/c10c2d3b-343a-44b0-ba2f-5d9f44915c7b)

### R Code 

```r
library(ggplot2)
library(R6)

ExponentialDistribution <- R6Class("ExponentialDistribution",
  public = list(
    rate = NULL,
    initialize = function(rate) {
      self$rate <- rate
    },
    plot_pdf = function(x_range) {
      x <- seq(0, x_range, length.out = 1000)
      y <- self$rate * exp(-self$rate * x)
      ggplot(data.frame(x, y), aes(x, y)) +
        geom_line(color = "blue") +
        ggtitle("Exponential Distribution PDF") +
        xlab("Time") + ylab("PDF")
    },
    compute_probability = function(time) {
      return(1 - exp(-self$rate * time))
    }
  )
)

# Instantiate and use
exp_dist <- ExponentialDistribution$new(rate = 3)
exp_dist$plot_pdf(2)  # Plot PDF for 2 hours
cat(sprintf('Probability of next call within 15 minutes: %0.3f', exp_dist$compute_probability(0.25)))
```

# Gamma Distribution

The Gamma Distribution is a two-parameter family of continuous probability distributions. It's widely used in various fields such as statistics, engineering, and physics, particularly for modeling waiting times in processes where events occur continuously and independently at a constant rate, similar to the exponential distribution, but over multiple events. The Gamma distribution generalizes the exponential distribution to model the time until the \(n\)th event in a Poisson process and has applications in Bayesian statistics, survival analysis, and queueing theory.


The probability density function (PDF) of the Gamma distribution for a random variable \(X\) is given by:

![equations](https://latex.codecogs.com/svg.image?f(x;k,\theta)=\frac{x^{k-1}e^{-x/\theta}}{\theta^k\Gamma(k)}\quad\text{for}x>0,)


where:
- ![equations](https://latex.codecogs.com/svg.image?x) is the random variable,
- ![equations](https://latex.codecogs.com/svg.image?k) is the shape parameter (sometimes denoted as \(\alpha\)),
- ![equations](https://latex.codecogs.com/svg.image?\theta) is the scale parameter (sometimes denoted as \(\beta\), where \(\beta = 1/\theta\)),
- ![equations](https://latex.codecogs.com/svg.image?\Gamma(k)) is the gamma function, defined as ![equations](https://latex.codecogs.com/svg.image?\Gamma(k)) = ![equations](https://latex.codecogs.com/svg.image?\int_0^\infty&space;x^{k-1}e^{-x}dx), which generalizes the factorial function to continuous values, with ![equations](https://latex.codecogs.com/svg.image?\Gamma(n)) = ![equations](https://latex.codecogs.com/svg.image?(n-1)!) for positive integer \(n\).

The gamma distribution includes various distributions as special cases; for example, when ![equations](https://latex.codecogs.com/svg.image?k=1), the gamma distribution simplifies to the exponential distribution.

Assume we are interested in the waiting time until the 3rd event occurs in a process with an average rate of 2 events per hour. Let's calculate the probability density of observing the third event at exactly 2 hours, with ![equations](https://latex.codecogs.com/svg.image?k) = ![equations](https://latex.codecogs.com/svg.image?3) and ![equations](https://latex.codecogs.com/svg.image?\theta) = ![equations](https://latex.codecogs.com/svg.image?0.5) hours (since the rate ![equations](https://latex.codecogs.com/svg.image?\lambda) = ![equations](https://latex.codecogs.com/svg.image?2) corresponds to ![equations](https://latex.codecogs.com/svg.image?\theta) = ![equations](https://latex.codecogs.com/svg.image?1/\lambda).

1. **Given Parameters**: ![equations](https://latex.codecogs.com/svg.image?k) = ![equations](https://latex.codecogs.com/svg.image?3), ![equations](https://latex.codecogs.com/svg.image?\theta) = ![equations](https://latex.codecogs.com/svg.image?0.5)
2. **Objective**: Calculate ![equations](https://latex.codecogs.com/svg.image?\(f(2)); ![equations](https://latex.codecogs.com/svg.image?3), ![equations](https://latex.codecogs.com/svg.image?0.5))

**Calculation**:

Substituting the values into the PDF formula, we get:

![equations](https://latex.codecogs.com/svg.image?f(2;3,0.5)=\frac{2^{3-1}e^{-2/0.5}}{0.5^3\Gamma(3)}=\frac{4e^{-4}}{0.125\times&space;2}=\frac{4e^{-4}}{0.25})

Since ![equations](https://latex.codecogs.com/svg.image?\Gamma(3)) = ![equations](https://latex.codecogs.com/svg.image?2!) = ![equations](https://latex.codecogs.com/svg.image?2), we continue:

![equations](https://latex.codecogs.com/svg.image?=16e^{-4}\approx&space;0.146525)

This result gives the probability density at the specified time, not a probability. To get a probability, one would integrate this density over a range of times.

### Python Code

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

class GammaDistribution:
    def __init__(self, k, theta):
        self.k = k
        self.theta = theta

    def plot_pdf(self, x_range):
        x = np.linspace(0, x_range, 1000)
        y = gamma.pdf(x, self.k, scale=self.theta)
        plt.plot(x, y, label=f'k={self.k}, θ={self.theta}')
        plt.title('Gamma Distribution PDF')
        plt.xlabel('x')
        plt.ylabel('PDF')
        plt.legend()
        plt.show()

    def compute_pdf(self, x):
        return gamma.pdf(x, self.k, scale=self.theta)

# Example usage
gamma_dist = GammaDistribution(k=3, theta=0.5)
gamma_dist.plot_pdf(10)  # Plot PDF up to x=10
print(f'PDF at x=2: {gamma_dist.compute_pdf(2):.6f}')
```

### R Code 

```r
library(ggplot2)
library(R6)

GammaDistribution <- R6Class("GammaDistribution",
  public = list(
    k = NULL,
    theta = NULL,
    initialize = function(k, theta) {
      self$k <- k
      self$theta <- theta
    },
    plot_pdf = function(x_range) {
      x <- seq(0, x_range, length.out = 1000)
      y <- dgamma(x, shape = self$k, scale = self$theta)
      df <- data.frame(x, y)
      ggplot(df, aes(x, y)) + geom_line() + ggtitle("Gamma Distribution PDF") +
        xlab("x") + ylab("PDF") + theme

_minimal()
    },
    compute_pdf = function(x) {
      return(dgamma(x, shape = self$k, scale = self$theta))
    }
  )
)

# Example usage
gamma_dist <- GammaDistribution$new(k = 3, theta = 0.5)
gamma_dist$plot_pdf(10)  # Plot PDF up to x=10
cat(sprintf('PDF at x=2: %f', gamma_dist$compute_pdf(2)))
```
#

# Beta Distribution

The Beta Distribution is a versatile family of continuous probability distributions defined on the interval ![equations](https://latex.codecogs.com/svg.image?[0,1]), making it particularly useful for modeling events constrained to finite intervals. Its flexibility and bounded nature allow it to represent a wide range of distributions of probabilities, making it a cornerstone in Bayesian statistics, particularly in the estimation of probabilities and proportions.


The probability density function (PDF) of the Beta distribution for a random variable \(X\) is given by:

![equations](https://latex.codecogs.com/svg.image?f(x;\alpha,\beta)=\frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}\quad\text{for}0\leq&space;x\leq&space;1)

where:
- ![equations](https://latex.codecogs.com/svg.image?x) is the random variable,
- ![equations](https://latex.codecogs.com/svg.image?\alpha) and \(\beta\) are shape parameters, both greater than 0,
- ![equations](https://latex.codecogs.com/svg.image?B(\alpha,\beta)) is the beta function, serving as a normalization constant to ensure the area under the PDF integrates to 1, defined as ![equations](https://latex.codecogs.com/svg.image?B(\alpha,\beta)=\int_0^1&space;t^{\alpha-1}(1-t)^{\beta-1}dt)


**Scenario**: Consider modeling the probability of success in a certain task, where prior knowledge suggests that success is likely but not guaranteed, with \(\alpha = 5\) and \(\beta = 2\).

**Objective**: Calculate the PDF of ![equations](https://latex.codecogs.com/svg.image?X) = ![equations](https://latex.codecogs.com/svg.image?0.5), representing a 50% estimated probability of success.

**Solution**:

1. **Given Parameters**: ![equations](https://latex.codecogs.com/svg.image?\alpha) = ![equations](https://latex.codecogs.com/svg.image?5), ![equations](https://latex.codecogs.com/svg.image?\beta) = ![equations](https://latex.codecogs.com/svg.image?2)
2. **Compute \(B(\alpha, \beta)\)**:
   Using the properties of the beta function, ![equations](https://latex.codecogs.com/svg.image?B(5,2)=\frac{(5-1)!(2-1)!}{(5&plus;2-1)!}=\frac{4!}{6!})
3. **Apply PDF Formula**:
   Substituting ![equations](https://latex.codecogs.com/svg.image?\alpha,\beta,and&space;x=0.5) into the PDF formula gives us the density value.


Given the complexity of calculating the beta function directly, it's common to use software for these computations. The density ![equations](https://latex.codecogs.com/svg.image?f(0.5;5,2)) can be easily computed using specialized functions in Python or R.

### Python Code

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

class BetaDistributionModel:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    @property
    def distribution(self):
        return beta(self.alpha, self.beta)

    def plot_pdf(self):
        x = np.linspace(0, 1, 1000)
        y = self.distribution.pdf(x)
        plt.plot(x, y, label=f'α={self.alpha}, β={self.beta}')
        plt.title('Beta Distribution PDF')
        plt.xlabel('x')
        plt.ylabel('PDF')
        plt.legend()
        plt.show()

# Instantiate and use
model = BetaDistributionModel(5, 2)
model.plot_pdf()
```

![image](https://github.com/BytesOfIntelligences/Distributions/assets/56669333/de215255-b779-4fe8-82cc-4453e1e239bc)

### R Code 
```r
library(ggplot2)
library(R6)

BetaDistributionModel <- R6Class("BetaDistributionModel",
  public = list(
    alpha = NULL,
    beta = NULL,
    initialize = function(alpha, beta) {
      self$alpha <- alpha
      self$beta <- beta
    },
    plot_pdf = function() {
      x <- seq(0, 1, length.out = 1000)
      y <- dbeta(x, shape1 = self$alpha, shape2 = self$beta)
      ggplot(data.frame(x, y), aes(x, y)) + geom_line() +
        ggtitle(paste("Beta Distribution PDF (α=", self$alpha, ", β=", self$beta, ")", sep="")) +
        xlab("x") + ylab("PDF") + theme_minimal()
    }
  )
)

# Instantiate and use
model <- BetaDistributionModel$new(5, 2)
model$plot_pdf()
```


#

The Chi-Squared (\(\chi^2\)) Distribution is a widely used probability distribution in inferential statistics, especially in hypothesis testing and in the construction of confidence intervals. It is a special case of the Gamma distribution, tailored to scenarios where the squared magnitude of normal variables is summed up, providing a fundamental tool for variance analysis and goodness-of-fit tests.

### Mathematical Formulation of the Chi-Squared Distribution

The probability density function (PDF) of the Chi-Squared distribution for a variable \(X\) is given by:

![equations](https://latex.codecogs.com/svg.image?f(x;k)=\frac{1}{2^{k/2}\Gamma(k/2)}x^{(k/2)-1}e^{-x/2})

where:
- ![equations](https://latex.codecogs.com/svg.image?x) is the value of the random variable,
- ![equations](https://latex.codecogs.com/svg.image?k) is the degrees of freedom,
- ![equations](https://latex.codecogs.com/svg.image?\Gamma) is the gamma function, which generalizes the factorial function to non-integer values, with ![equations](https://latex.codecogs.com/svg.image?\Gamma(n)) = ![equations](https://latex.codecogs.com/svg.image?(n-1)!) for an integer ![equations](https://latex.codecogs.com/svg.image?n).

### Example with Step-by-Step Calculation

**Scenario**: Determine the probability density of obtaining a value of 5 from a Chi-Squared distribution with 3 degrees of freedom ![equations](https://latex.codecogs.com/svg.image?k) = ![equations](https://latex.codecogs.com/svg.image?3).

**Solution Steps**:

1. **Given Parameters**: ![equations](https://latex.codecogs.com/svg.image?x) = ![equations](https://latex.codecogs.com/svg.image?5), ![equations](https://latex.codecogs.com/svg.image?k) = ![equations](https://latex.codecogs.com/svg.image?3).
2. **Apply PDF Formula**:
   - Substitute \(x\) and \(k\) into the formula to calculate the PDF value.

**Computation**:

![equations](https://latex.codecogs.com/svg.image?f(5;3)=\frac{1}{2^{3/2}\Gamma(3/2)}5^{(3/2)-1}e^{-5/2})


### Python Code

Chi-Squared distribution to demonstrate a different OOP concept, emphasizing code reuse and composition over inheritance:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

class ChiSquaredMixin:
    def __init__(self, df):
        self.df = df

    def pdf(self, x):
        return chi2.pdf(x, self.df)

class ChiSquaredDistribution(ChiSquaredMixin):
    def __init__(self, df):
        super().__init__(df)
    
    def plot_pdf(self, x_range):
        x = np.linspace(0, x_range, 1000)
        y = self.pdf(x)
        plt.plot(x, y, label=f'DF={self.df}')
        plt.title('Chi-Squared Distribution PDF')
        plt.xlabel('x')
        plt.ylabel('PDF')
        plt.legend()
        plt.show()

# Instantiate and visualize
dist = ChiSquaredDistribution(df=3)
dist.plot_pdf(10)
```

![image](https://github.com/BytesOfIntelligences/Distributions/assets/56669333/4285fd54-0373-46a6-864c-0bb616263abc)

### R Code for Visualization (Functional OOP Style)


```r
library(ggplot2)

create_chi_squared_distribution <- function(df) {
  list(
    df = df,
    pdf = function(x) dchisq(x, df),
    plot_pdf = function(x_range) {
      x <- seq(0, x_range, length.out = 1000)
      y <- dchisq(x, df)
      ggplot(data.frame(x, y), aes(x, y)) + geom_line() +
        ggtitle(paste("Chi-Squared Distribution PDF (DF=", df, ")", sep="")) +
        xlab("x") + ylab("PDF") + theme_minimal()
    }
  )
}

# Instantiate and use
dist <- create_chi_squared_distribution(df=3)
dist$plot_pdf(10)
```
# F-distribution

The F-distribution, also known as Snedecor's F distribution or the Fisher-Snedecor distribution, is a continuous probability distribution that arises frequently in the context of statistical hypothesis testing, especially in the analysis of variance (ANOVA) and in comparing variances across two independent samples. It is named after Ronald Fisher and George W. Snedecor.


The probability density function (PDF) of the F-distribution for a random variable ![equations](https://latex.codecogs.com/svg.image?X) with degrees of freedom ![equations](https://latex.codecogs.com/svg.image?d_1) and ![equations](https://latex.codecogs.com/svg.image?d_2) is given by:

![equations](https://latex.codecogs.com/svg.image?f(x;d_1,d_2)=\frac{\sqrt{\frac{(d_1&space;x)^{d_1}d_2^{d_2}}{(d_1&space;x&plus;d_2)^{d_1&plus;d_2}}}}{x&space;B\left(\frac{d_1}{2},\frac{d_2}{2}\right)}\quad\text{for}x>0)

where:
- ![equations](https://latex.codecogs.com/svg.image?x) is the random variable,
- ![equations](https://latex.codecogs.com/svg.image?d_1) and ![equations](https://latex.codecogs.com/svg.image?d_2) are the degrees of freedom associated with the numerator and the denominator, respectively,
- ![equations](https://latex.codecogs.com/svg.image?B) is the Beta function, used here as a normalization constant to ensure the area under the PDF integrates to 1.

The probability density function value for an F-distribution with ![equations](https://latex.codecogs.com/svg.image?d_1) = ![equations](https://latex.codecogs.com/svg.image?5) degrees of freedom in the numerator and ![equations](https://latex.codecogs.com/svg.image?d_2) = ![equations](https://latex.codecogs.com/svg.image?10) degrees of freedom in the denominator at ![equations](https://latex.codecogs.com/svg.image?x) = ![equations](https://latex.codecogs.com/svg.image?2)

**Given**: ![equations](https://latex.codecogs.com/svg.image?d_1) = ![equations](https://latex.codecogs.com/svg.image?5), ![equations](https://latex.codecogs.com/svg.image?d_2) = ![equations](https://latex.codecogs.com/svg.image?10), ![equations](https://latex.codecogs.com/svg.image?x) = ![equations](https://latex.codecogs.com/svg.image?2)


1. **Identify Parameters**: The degrees of freedom ![equations](https://latex.codecogs.com/svg.image?d_1) = ![equations](https://latex.codecogs.com/svg.image?5), 1. **Identify Parameters**: The degrees of freedom ![equations](https://latex.codecogs.com/svg.image?d_1) = ![equations](https://latex.codecogs.com/svg.image?5), ![equations](https://latex.codecogs.com/svg.image?d_2) = ![equations](https://latex.codecogs.com/svg.image?10), and the value ![equations](https://latex.codecogs.com/svg.image?x) = ![equations](https://latex.codecogs.com/svg.image?2).
![equations](https://latex.codecogs.com/svg.image?d_2) = ![equations](https://latex.codecogs.com/svg.image?10), and the value ![equations](https://latex.codecogs.com/svg.image?x) = ![equations](https://latex.codecogs.com/svg.image?2).
2. **Apply PDF Formula**: Use the given formula to plug in the values of \(d_1\), \(d_2\), and \(x\).
3. **Compute Beta Function**: ![equations](https://latex.codecogs.com/svg.image?B\left(\frac{5}{2},\frac{10}{2}\right)) is required for the denominator of the formula.

### Python Code

```python
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f

class Distribution(ABC):
    
    @abstractmethod
    def plot_pdf(self):
        pass

class FDistribution(Distribution):
    
    def __init__(self, dfn, dfd):
        self.dfn = dfn
        self.dfd = dfd
    
    def plot_pdf(self):
        x = np.linspace(0.01, 5, 1000)
        y = f.pdf(x, self.dfn, self.dfd)
        plt.plot(x, y, label=f'df1={self.dfn}, df2={self.dfd}')
        plt.title('F-Distribution PDF')
        plt.xlabel('x')
        plt.ylabel('PDF')
        plt.legend()
        plt.show()

# Instantiate and visualize
f_dist = FDistribution(dfn=5, dfd=10)
f_dist.plot_pdf()
```

![image](https://github.com/BytesOfIntelligences/Distributions/assets/56669333/b426045e-1435-4a7e-bf41-344801674899)

### R Code 

```r
plot_pdf <- function(dfn, dfd) {
  x <- seq(0.01, 5, length.out = 1000)
  y <- df(x, dfn, dfd)
  plot(x, y, type = 'l', col = 'blue', main = 'F-Distribution PDF', xlab = 'x', ylab = 'PDF')
  legend('topright', legend = paste('df1=', dfn, ', df2=', dfd), col = 'blue', bty = 'n')
}

# Create a list that serves as a prototype
f_dist <- list(dfn=5, dfd=10, plot_pdf=plot_pdf)

# Use the prototype
f_dist$plot_pdf(f_dist$dfn, f_dist$dfd)
```

# Student's t-distribution

The Student's t-distribution is a family of continuous probability distributions that arises when estimating the mean of a normally distributed population in situations where the sample size is small and population standard deviation is unknown. It plays a crucial role in many areas of statistical inference, especially in the development of t-tests for hypothesis testing and confidence interval estimation.


The probability density function (PDF) of the Student's t-distribution for a random variable ![equations](https://latex.codecogs.com/svg.image?T) with ![equations](https://latex.codecogs.com/svg.image?v) degrees of freedom is given by:

![equations](https://latex.codecogs.com/svg.image?f(t;v)=\frac{\Gamma\left(\frac{v&plus;1}{2}\right)}{\sqrt{v\pi}\,\Gamma\left(\frac{v}{2}\right)}\left(1&plus;\frac{t^2}{v}\right)^{-\frac{v&plus;1}{2}})

where:
- ![equations](https://latex.codecogs.com/svg.image?t) is the random variable,
- ![equations](https://latex.codecogs.com/svg.image?v) is the degrees of freedom,
- \![equations](https://latex.codecogs.com/svg.image?\Gamma) denotes the gamma function, an extension of the factorial function with its argument shifted down by 1.

Calculate the probability density of observing a value ![equations](https://latex.codecogs.com/svg.image?t) = ![equations](https://latex.codecogs.com/svg.image?2) from a Student's t-distribution with ![equations](https://latex.codecogs.com/svg.image?v) = ![equations](https://latex.codecogs.com/svg.image?10) degrees of freedom.

**Given**: ![equations](https://latex.codecogs.com/svg.image?t) = ![equations](https://latex.codecogs.com/svg.image?2), ![equations](https://latex.codecogs.com/svg.image?v) = ![equations](https://latex.codecogs.com/svg.image?10)

**Solution**:

1. **Identify Parameters**: ![equations](https://latex.codecogs.com/svg.image?v) = ![equations](https://latex.codecogs.com/svg.image?10), ![equations](https://latex.codecogs.com/svg.image?t) = ![equations](https://latex.codecogs.com/svg.image?2).
2. **Apply PDF Formula**: Plug the values of ![equations](https://latex.codecogs.com/svg.image?v) and ![equations](https://latex.codecogs.com/svg.image?t) into the PDF formula.
3. **Compute Gamma Function Values**: Use ![equations](https://latex.codecogs.com/svg.image?\Gamma\left(\frac{10&plus;1}{2}\right)) and ![equations](https://latex.codecogs.com/svg.image?\Gamma\left(\frac{10}{2}\right)) in the formula.
4. **Calculate the Density**: Substitute all values into the formula to get the density at ![equations](https://latex.codecogs.com/svg.image?t) = ![equations](https://latex.codecogs.com/svg.image?2).

### Python Code

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

def plot_decorator(func):
    def wrapper(*args, **kwargs):
        plt.figure(figsize=(10, 6))
        func(*args, **kwargs)
        plt.title('Student\'s t-Distribution PDF')
        plt.xlabel('t')
        plt.ylabel('PDF')
        plt.grid(True)
        plt.show()
    return wrapper

class StudentTDistribution:
    def __init__(self, degrees_of_freedom):
        self.degrees_of_freedom = degrees_of_freedom

    @plot_decorator
    def plot_pdf(self):
        x = np.linspace(-5, 5, 1000)
        y = t.pdf(x, self.degrees_of_freedom)
        plt.plot(x, y, label=f'v={self.degrees_of_freedom}')
        plt.legend()

# Instantiate and visualize
dist = StudentTDistribution(degrees_of_freedom=10)
dist.plot_pdf()
```


![image](https://github.com/BytesOfIntelligences/Distributions/assets/56669333/61077949-9c6f-483e-8b52-8ecf9596bff5)

### R Code 

```r
library(ggplot2)

plot_t_distribution <- function(degrees_of_freedom) {
  x <- seq(-5, 5, length.out = 1000)
  y <- dt(x, df = degrees_of_freedom)
  data <- data.frame(x, y)
  ggplot(data, aes(x, y)) +
    geom_line() +
    ggtitle(paste("Student's t-Distribution PDF with", degrees_of_freedom, "DF")) +
    xlab("t") + ylab("PDF") +
    theme_minimal()
}

plot_t_distribution(10)
```

# Log-Normal Distribution


The Log-Normal Distribution is a probability distribution of a random variable whose logarithm is normally distributed. This distribution is applicable in various fields, such as finance for stock prices, environmental studies for particle sizes, and anywhere the data is positively skewed. It characterizes phenomena where the product of random variables is central, especially in processes subject to multiplicative effects.

The probability density function (PDF) of the Log-Normal distribution for a random variable \(X\) is given by:

![image](https://github.com/BytesOfIntelligences/Distributions/assets/56669333/00dfe243-963d-4c6b-bcfa-45630bb53a69)

where:
- ![equations](https://latex.codecogs.com/svg.image?x) > ![equations](https://latex.codecogs.com/svg.image?0) is the value of the random variable,
- ![equations](https://latex.codecogs.com/svg.image?\mu) and ![equations](https://latex.codecogs.com/svg.image?\sigma) are the mean and standard deviation of the variable's natural logarithm, not of the variable itself.

### Example with Step-by-Step Calculation

**Scenario**: Calculate the probability density for a value \(x = 1.5\) from a Log-Normal distribution with parameters ![equations](https://latex.codecogs.com/svg.image?\mu) = ![equations](https://latex.codecogs.com/svg.image?0) and ![equations](https://latex.codecogs.com/svg.image?\sigma )= ![equations](https://latex.codecogs.com/svg.image?0.25).

**Given**: ![equations](https://latex.codecogs.com/svg.image?x) = ![equations](https://latex.codecogs.com/svg.image?1.5), ![equations](https://latex.codecogs.com/svg.image?\mu) = ![equations](https://latex.codecogs.com/svg.image?0), ![equations](https://latex.codecogs.com/svg.image?\sigma) = ![equations](https://latex.codecogs.com/svg.image?0.25)

**Solution**:

1. **Identify Parameters**: ![equations](https://latex.codecogs.com/svg.image?x) = ![equations](https://latex.codecogs.com/svg.image?1.5), ![equations](https://latex.codecogs.com/svg.image?\mu) = ![equations](https://latex.codecogs.com/svg.image?0), ![equations](https://latex.codecogs.com/svg.image?\sigma) = ![equations](https://latex.codecogs.com/svg.image?0.25).
2. **Apply PDF Formula**: Substitute the given values into the PDF formula to compute the density.
3. **Compute the Density**:

![image](https://github.com/BytesOfIntelligences/Distributions/assets/56669333/c85d91c0-96a1-43aa-94bc-68a5382da869)
### Python Code 

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

class PlotMixin:
    def plot(self, x, y, title, xlabel, ylabel):
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, label=self.label)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.show()

class LogNormalDistribution(PlotMixin):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.label = f'μ={mu}, σ={sigma}'

    def pdf(self, x):
        return (1 / (x * self.sigma * np.sqrt(2 * np.pi))) * \
               np.exp(-((np.log(x) - self.mu) ** 2) / (2 * self.sigma ** 2))

    def plot_pdf(self, start, end):
        x = np.linspace(start, end, 1000)
        y = self.pdf(x)
        self.plot(x, y, 'Log-Normal Distribution PDF', 'x', 'PDF')

# Instantiate and visualize
distribution = LogNormalDistribution(mu=0, sigma=0.25)
distribution.plot_pdf(0.01, 3)
```

![image](https://github.com/BytesOfIntelligences/Distributions/assets/56669333/485c6118-43f1-4e47-8d00-f793eff98c1f)

### R Code 
```r
library(ggplot2)

plot_strategy <- function(data, title, xlab, ylab) {
  ggplot(data, aes(x=x, y=y)) + 
    geom_line() + 
    ggtitle(title) + 
    xlab(xlab) + 
    ylab(ylab) + 
    theme_minimal()
}

log_normal_distribution <- function(mu, sigma, start, end) {
  x <- seq(start, end, length.out = 1000)
  y <- dlnorm(x, meanlog = mu, sdlog = sigma)
  data <- data.frame(x=x, y=y)
  title <- paste("Log-Normal Distribution PDF: mu =", mu, ", sigma =", sigma)
  plot_strategy(data, title, "x", "PDF")
}

# Instantiate and visualize
log_normal_distribution(mu = 0, sigma = 0.25, start = 0.01, end = 3)
```

#

# Weibull Distribution

The Weibull Distribution is a continuous probability distribution used extensively in reliability engineering, life data analysis, and survival analysis. The distribution is flexible, capable of modeling various types of data behavior — from exponential to Rayleigh distributions, depending on its parameters. It's particularly useful for describing the life duration of objects, failure rates of processes, and times to events in a wide range of disciplines.

The probability density function (PDF) of the Weibull distribution for a random variable ![equations](https://latex.codecogs.com/svg.image?X) is given by:

![equations](https://latex.codecogs.com/svg.image?f(x;\lambda,k)=\frac{k}{\lambda}\left(\frac{x}{\lambda}\right)^{k-1}e^{-(x/\lambda)^k}\quad\text{for}x\geq&space;0&space;)

where:
- ![equations](https://latex.codecogs.com/svg.image?x) is the variable of interest,
- ![equations](https://latex.codecogs.com/svg.image?\lambda) is the scale parameter, which stretches or compresses the distribution and is always positive,
- ![equations](https://latex.codecogs.com/svg.image?k) is the shape parameter, which determines the shape of the distribution. When ![equations](https://latex.codecogs.com/svg.image?k) = ![equations](https://latex.codecogs.com/svg.image?1), the Weibull distribution simplifies to an exponential distribution. For ![equations](https://latex.codecogs.com/svg.image?k) > ![equations](https://latex.codecogs.com/svg.image?1), it models situations where the failure rate increases over time; for ![equations](https://latex.codecogs.com/svg.image?k) < ![equations](https://latex.codecogs.com/svg.image?1), the failure rate decreases over time.

### Cumulative Distribution Function (CDF)

The CDF of the Weibull distribution is given by:

![equations](https://latex.codecogs.com/svg.image?&space;F(x;\lambda,k)=1-e^{-(x/\lambda)^k})

This function provides the probability that a random variable ![equations](https://latex.codecogs.com/svg.image?X) is less than or equal to a particular value ![equations](https://latex.codecogs.com/svg.image?x).

Determine the probability density for a component's failure time \(x = 2\) years, assuming the component's failure time follows a Weibull distribution with \(\lambda = 1.5\) years (scale parameter) and ![equations](https://latex.codecogs.com/svg.image?k) = ![equations](https://latex.codecogs.com/svg.image?3) (shape parameter).

**Given**: ![equations](https://latex.codecogs.com/svg.image?x) = ![equations](https://latex.codecogs.com/svg.image?2), ![equations](https://latex.codecogs.com/svg.image?\lambda) = ![equations](https://latex.codecogs.com/svg.image?1.5), ![equations](https://latex.codecogs.com/svg.image?k) = ![equations](https://latex.codecogs.com/svg.image?3)

**Solution**:

1. **Identify Parameters**: ![equations](https://latex.codecogs.com/svg.image?\lambda) = ![equations](https://latex.codecogs.com/svg.image?1.5), ![equations](https://latex.codecogs.com/svg.image?k) = ![equations](https://latex.codecogs.com/svg.image?3).

   ![equations](https://latex.codecogs.com/svg.image?f(2;1.5,3)=\frac{3}{1.5}\left(\frac{2}{1.5}\right)^{3-1}e^{-(2/1.5)^3})


### Python Code


```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min

def plot_weibull_cdf(scale, shape):
    x = np.linspace(0, 10, 1000)
    cdf = weibull_min.cdf(x, shape, scale=scale)
    plt.figure(figsize=(8, 6))
    plt.plot(x, cdf, label=f'Weibull CDF: λ={scale}, k={shape}')
    plt.title('Weibull Distribution CDF')
    plt.xlabel('x')
    plt.ylabel('CDF')
    plt.legend()
    plt.grid(True)
    plt.show()

# Call the function with specific parameters
plot_weibull_cdf(scale=1.5, shape=3)

```
![image](https://github.com/BytesOfIntelligences/Distributions/assets/56669333/eaabb79b-2318-431a-b555-500907796b92)
### R Code 

```r
library(ggplot2)

plot_weibull_pdf <- function(scale, shape) {
  x <- seq(0, 10, length.out = 1000)
  y <- dweibull(x, shape = shape, scale = scale)
  df <- data.frame(x = x, y = y)
  ggplot(df, aes(x, y)) +
    geom_line() +
    ggtitle(paste("Weibull Distribution PDF: λ=", scale, ", k=", shape)) +
    xlab("x") + ylab("PDF") +
    theme_minimal()
}

# Example usage
plot_weibull_pdf(scale = 1.5, shape = 3)
```

# Geometric Distribution

The Geometric Distribution is a discrete probability distribution that models the number of trials needed to achieve the first success in a sequence of independent Bernoulli trials, each with the same probability of success. This distribution is widely used in scenarios where the objective is to understand the likelihood of achieving a first success after a certain number of attempts, such as in quality control, gambling, and reliability engineering.


The probability mass function (PMF) of the Geometric distribution for a random variable \(X\) (representing the number of trials up to and including the first success) is given by:

![equations](https://latex.codecogs.com/svg.image?P(X=x)=(1-p)^{x-1}p&space;)

where:
- ![equations](https://latex.codecogs.com/svg.image?x\in\{1,2,3,\dots\}) is the trial on which the first success occurs,
- ![equations](https://latex.codecogs.com/svg.image?p) is the probability of success on each trial.

### Cumulative Distribution Function (CDF)

The cumulative distribution function (CDF) for the Geometric distribution is:

![equations](https://latex.codecogs.com/svg.image?F(x)=1-(1-p)^x&space;)

indicating the probability that the first success occurs on or before the ![equations](https://latex.codecogs.com/svg.image?x)th trial.

What is the probability that the first success occurs on the 5th trial in a process where the probability of success on each trial is 0.2?

**Given**: ![equations](https://latex.codecogs.com/svg.image?p=0.2,x=5)

**Solution**:

1. **Identify Parameters**: ![equations](https://latex.codecogs.com/svg.image?p=0.2,x=5).
2. **Apply PMF Formula**: Use the given values to compute the probability.
   ![equations](https://latex.codecogs.com/svg.image?P(X=5)=(1-0.2)^{5-1}\times&space;0.2&space;)
3. **Compute the Probability**:
   ![equations](https://latex.codecogs.com/svg.image?P(X=5)=(0.8)^4\times&space;0.2=0.08192&space;)

### Python Code 

```python
import numpy as np
import matplotlib.pyplot as plt

class GeometricDistribution:
    def __init__(self, p):
        self.p = p
        self.on_plot = None  # Event handler placeholder

    def generate_pmf_data(self, n):
        x = np.arange(1, n+1)
        y = (1 - self.p)**(x-1) * self.p
        return x, y

    def plot_pmf(self, n):
        x, y = self.generate_pmf_data(n)
        if self.on_plot:  # Check if event handler is set
            self.on_plot(x, y)

def on_plot_event(x, y):
    plt.figure(figsize=(8, 6))
    plt.stem(x, y, basefmt=" ", use_line_collection=True)
    plt.title('Geometric Distribution PMF')
    plt.xlabel('Number of Trials')
    plt.ylabel('Probability')
    plt.show()

# Instantiate and set event
geom_dist = GeometricDistribution(p=0.2)
geom_dist.on_plot = on_plot_event  # Set event handler

# Trigger the plotting event
geom_dist.plot_pmf(n=10)
```

![image](https://github.com/BytesOfIntelligences/Distributions/assets/56669333/2d0aff43-2df0-4bda-84ec-f0cbeccb0d59)

### R Code 

```r
library(ggplot2)

plot_pmf <- function(p, n) {
  x <- 1:n
  y <- (1 - p)^(x - 1) * p
  data <- data.frame(Trials = x, Probability = y)
  
  ggplot(data, aes(x = Trials, y = Probability)) +
    geom_point() +
    geom_line() +
    ggtitle("Geometric Distribution PMF") +
    xlab("Number of Trials") + 
    ylab("Probability")
}

# Simulating reactive programming by directly calling the function
plot_pmf(p = 0.2, n = 10)
```

#  Negative Binomial Distribution
 
The Negative Binomial Distribution is a discrete probability distribution that extends the geometric distribution. It represents the number of successes in a sequence of independent and identically distributed Bernoulli trials before a specified number of failures occurs. This distribution is particularly useful in scenarios where we're interested in the probability of achieving a certain number of successes before encountering a certain number of failures, and it finds applications in quality control, insurance, and epidemiology, among other fields.

The probability mass function (PMF) of the Negative Binomial distribution for a random variable ![equations](https://latex.codecogs.com/svg.image?X) (representing the number of successes before ![equations](https://latex.codecogs.com/svg.image?r) failures occur) is given by:

![equations](https://latex.codecogs.com/svg.image?P(X=k)=\binom{k&plus;r-1}{k}p^k(1-p)^r&space;)

where:
- ![equations](https://latex.codecogs.com/svg.image?k=0,1,2,\ldots) is the number of successes,
- ![equations](https://latex.codecogs.com/svg.image?r) is the number of failures until the experiment is stopped,
- ![equations](https://latex.codecogs.com/svg.image?p) is the probability of success on each trial,
- \(\binom{k+r-1}{k}\) is the binomial coefficient, calculated as \(\frac{(k+r-1)!}{k!(r-1)!}\).

### Cumulative Distribution Function (CDF)

The CDF of the Negative Binomial distribution can be expressed using the regularized incomplete beta function, but it's often computed numerically in statistical software. What is the probability of getting 3 successes before achieving 2 failures in a series of trials, where the probability of success on each trial is 0.5?

**Given**: ![equations](https://latex.codecogs.com/svg.image?p=0.5,r=2,k=3)

1. ![equations](https://latex.codecogs.com/svg.image?p=0.5,r=2,k=3).
2. Use the given values to compute the probability.
   ![equations](https://latex.codecogs.com/svg.image?P(X=3)=\binom{3&plus;2-1}{3}(0.5)^3(1-0.5)^2&space;)
3. **Compute the Probability**:
   ![equations](https://latex.codecogs.com/svg.image?P(X=3)=\binom{4}{3}(0.125)(0.25)=4\times&space;0.125\times&space;0.25=0.125&space;)

### Python Code

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import nbinom

def logging_decorator(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print(f"Logging: Called {func.__name__} with args={args}, kwargs={kwargs}")
        return result
    return wrapper

class NegativeBinomialDistribution:
    @logging_decorator
    def plot_pmf(self, r, p, max_k):
        k = np.arange(0, max_k+1)
        pmf_values = nbinom.pmf(k, r, p)
        plt.bar(k, pmf_values, color='skyblue')
        plt.title('Negative Binomial Distribution PMF')
        plt.xlabel('Number of Successes')
        plt.ylabel('Probability')
        plt.show()

# Create an instance and plot
nbd = NegativeBinomialDistribution()
nbd.plot_pmf(r=2, p=0.5, max_k=10)
```
![image](https://github.com/BytesOfIntelligences/Distributions/assets/56669333/7840a25f-a5cb-48bd-908c-dc11e3d2dc86)

### R Code 

```r
library(ggplot2)
library(magrittr)

plot_nbd_pmf <- function(r, p, max_k) {
  data.frame(k = 0:max_k) %>%
    dplyr::mutate(probability = dnbinom(k, size = r, prob = p)) %>%
    ggplot(aes(x = k, y = probability)) +
    geom_col(fill = "skyblue") +
    labs(title = "Negative Binomial Distribution PMF",
    x = "Number of Successes", y = "Probability") +
    theme_minimal()
}

# Example usage
plot_nbd_pmf(r = 2, p = 0.5, max_k = 10)
```

# Hypergeometric Distribution

The Hypergeometric Distribution is a discrete probability distribution that describes the probability of drawing a specific number of successes (without replacement) from a finite population containing a fixed number of successes and failures. This distribution is particularly useful in scenarios such as quality control, lot sampling, and ecological studies, where the sampling does not allow replacement, making it distinct from the binomial distribution which assumes independent trials with replacement.



The probability mass function (PMF) of the Hypergeometric distribution for obtaining exactly \(k\) successes in \(n\) draws, from a finite population of size \(N\) containing exactly \(K\) successes, is given by:

![equations](https://latex.codecogs.com/svg.image?P(X=k)=\frac{\binom{K}{k}\binom{N-K}{n-k}}{\binom{N}{n}})

where:
- ![equations](https://latex.codecogs.com/svg.image?X) is the random variable representing the number of successes,
- ![equations](https://latex.codecogs.com/svg.image?\binom{a}{b}) denotes the binomial coefficient, which calculates the number of ways to choose \(b\) elements out of \(a\) elements without regard to order,
- ![equations](https://latex.codecogs.com/svg.image?N) is the total population size,
- ![equations](https://latex.codecogs.com/svg.image?K) is the total number of successes in the population,
- ![equations](https://latex.codecogs.com/svg.image?n) is the number of draws,
- ![equations](https://latex.codecogs.com/svg.image?k) is the number of observed successes in the draws.

In a lot of 20 items, where 5 are defective, what is the probability of finding exactly 2 defective items when randomly selecting 5 items?

**Given**: ![equations](https://latex.codecogs.com/svg.image?N=20,K=5,n=5,k=2)

**Solution**:

1. **Identify Parameters**: The total population size ![equations](https://latex.codecogs.com/svg.image?N) = ![equations](https://latex.codecogs.com/svg.image?20), the number of successes \(K = 5\) (defective items), the number of draws \(n = 5\), and the number of successes sought \(k = 2\).
2. **Apply PMF Formula**: Calculate using the values:
   ![equations](https://latex.codecogs.com/svg.image?P(X=2)=\frac{\binom{5}{2}\binom{15}{3}}{\binom{20}{5}})
3. Calculate binomial coefficients and substitute: 
     ![equations](https://latex.codecogs.com/svg.image?P(X=2)=\frac{10\times&space;455}{15504}\approx&space;0.293&space;)

### Python Code

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import hypergeom

class PlotMixin:
    def plot(self, x, y, title):
        plt.figure(figsize=(8, 6))
        plt.bar(x, y, color='skyblue')
        plt.title(title)
        plt.xlabel('Number of Defective Items')
        plt.ylabel('Probability')
        plt.show()

class HypergeometricDistribution(PlotMixin):
    def __init__(self, N, K, n):
        self.N = N  # Population size
        self.K = K  # Number of successes in population
        self.n = n  # Number of draws

    def plot_pmf(self):
        k = np.arange(0, min(self.K, self.n) + 1)
        pmf = hypergeom.pmf(k, self.N, self.K, self.n)
        self.plot(k, pmf, 'Hypergeometric Distribution PMF')

# Instantiate and visualize
hypergeo_dist = HypergeometricDistribution(N=20, K=5, n=5)
hypergeo_dist.plot_pmf()
```

![image](https://github.com/BytesOfIntelligences/Distributions/assets/56669333/2690f813-ceea-4119-85df-5f9d43dd11c8)


### R Code

```r
library(ggplot2)

plot_hypergeo_pmf <- function(N, K, n) {
  k <- 0:min(K, n)
  pmf <- dhyper(k, K, N-K, n)
  data <- data.frame(k, pmf)
  ggplot(data, aes(x=k, y=pmf)) +
    geom_col(fill="skyblue") +
    labs(title="Hypergeometric Distribution PMF",
         x="Number of Defective Items",
         y="Probability") +
    theme_minimal()
}

# Example usage
plot_hypergeo_pmf(N=20, K=5, n=5)
```

