package main

import "math"

// ComputeCrossEntropyLoss measures how wrong one prediction was.
// predictedProbabilities holds a probability for every candidate word;
// correctNextWordID is the index of the word that actually came next.
func ComputeCrossEntropyLoss(predictedProbabilities []float32, correctNextWordID int) float32 {
	// The probability the model assigned to the correct word.
	probabilityOfCorrectWord := predictedProbabilities[correctNextWordID]
	// Prevent taking the logarithm of zero by setting a tiny minimum probability.
	if probabilityOfCorrectWord < 1e-12 {
		probabilityOfCorrectWord = 1e-12
	}
	// Lower probability means a higher loss.
	return float32(-math.Log(float64(probabilityOfCorrectWord)))
}
