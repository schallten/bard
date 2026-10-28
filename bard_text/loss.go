package main

import "math"

// CrossEntropy measures how wrong the model's prediction was for one example
// probs contains the probabilities for all candidate words.
// target is the index of the correct next word.
func CrossEntropy(probs []float32, target int) float32 {
	// get probablity the model assigned to correct word.
	p := probs[target]
	// Prevent taking the logarithm of zero by setting a tiny minimum probability
	if p < 1e-12 {
		p = 1e-12
	}
	// lower probability = higher loss
	// return negative logathis of correct word pboelbablty
	return float32(-math.Log(float64(p)))
}
