# =================================
# Creation of the probability table
# =================================

m <- seq(1, 60, 1)
p <- function(m){
  c(M = m,
    p = 1 - prod((365:(365 - m + 1)/365)))
}
prob_table <- t(sapply(m, p))

# Find the minimum number of people for probability > 0.5
min_people <- prob_table[prob_table[, "p"] > 0.5, "M"][1]
cat("Minimum number of people for P > 0.5:", min_people, "\n")

# ==============================
# Plotting the probability curve
# ==============================

# This section generates a plot for the Birthday Problem.
# It displays the probability of at least two individuals sharing 
# the same birthday as a function of the number of people.
#
# Parameters:
# - number_of_individuals: A sequence from 1 to 60.
# - probability: Computed probabilities for each number of individuals.
#
# The plot includes reference lines at P = 0.5 and the critical 
# number of people where the probability exceeds 50%.

number_of_individuals <- 1:60
probability <- numeric(60)
for (i in m) {
  probability[i] = 1 - prod((365:(365 - i + 1))/365)
}

plot(number_of_individuals, probability, 
     type = "l", col = "blue", pch = 1, main = "Birthday Problem",
     ylab = "P(at least two individuals share the same birthday)",
     xlab = "Number of people")
abline(h = 0.5, lty = 2, col = "black")
abline(v = min_people, lty = 2, col = "black")