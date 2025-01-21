set.seed(42)  # For reproducibility

# Simulation parameters
n_simulations <- 100000  # Number simulations
true_rain_probability <- 0.5  # Assume P(raining) = 50%
truth_probability <- 2/3  # Probability a friend tells the truth
lie_probability <- 1/3    # Probability a friend lies

# Simulate raining scenarios (TRUE means it's raining, FALSE means it's not)
raining <- sample(c(TRUE, FALSE), n_simulations, replace = TRUE, prob = c(true_rain_probability, 1 - true_rain_probability))

# Simulate friends' responses based on whether it's actually raining or not
friend_1 <- ifelse(raining, 
                   sample(c(TRUE, FALSE), n_simulations, replace = TRUE, prob = c(truth_probability, lie_probability)),
                   sample(c(FALSE, TRUE), n_simulations, replace = TRUE, prob = c(truth_probability, lie_probability)))

friend_2 <- ifelse(raining, 
                   sample(c(TRUE, FALSE), n_simulations, replace = TRUE, prob = c(truth_probability, lie_probability)),
                   sample(c(FALSE, TRUE), n_simulations, replace = TRUE, prob = c(truth_probability, lie_probability)))

friend_3 <- ifelse(raining, 
                   sample(c(TRUE, FALSE), n_simulations, replace = TRUE, prob = c(truth_probability, lie_probability)),
                   sample(c(FALSE, TRUE), n_simulations, replace = TRUE, prob = c(truth_probability, lie_probability)))

# Find cases where all three friends said "yes" (TRUE)
all_said_yes <- friend_1 & friend_2 & friend_3

# Compute the probability that it is actually raining given all said yes
estimated_probability <- sum(raining[all_said_yes]) / sum(all_said_yes)

# Print the result
cat(sprintf("Estimated probability that it is actually raining given all friends said 'yes': %.4f\n", estimated_probability))
