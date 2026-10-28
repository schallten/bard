package main

import (
	"math"
	"math/rand"
)

// holds learnable params of logistic regression
type Model struct {
	V    int       // vocab size
	W    []float32 // rows are input features, columns are candidates
	Bias []float32 // one bias per candidate word , so V values in total
}

// model with small random weights and zero biases
func NewModel(vocabSize int, seed int64) *Model {
	rng := rand.New(rand.NewSource(seed))

	m := &Model{
		V:    vocabSize,
		W:    make([]float32, 2*vocabSize*vocabSize), // 2V rows of V columns, all zero to start
		Bias: make([]float32, vocabSize),             // One zero bias per candidate
	}

	for i := range m.W {
		// visit every weight
		m.W[i] = float32(rng.NormFloat64() * 0.01) // small random value near zero
	}
	return m // return a pointer to new model
}

// calculates a score for each possible next word based on the two context words
func (m *Model) Logits(c0, c1 int) []float32 {
	// create a slice to store one score for each word in the vocabulary
	z := make([]float32, m.V) // slice to score one score for each word in the vocabulary

	row0 := c0 * m.V
	row1 := (m.V + c1) * m.V
	// m.W is a flat array treated as a table with m.V columns per row.
	// To find a row's starting index, multiply its row number by m.V.
	// The first context word uses row c0, while the second uses row m.V + c1
	// because its rows come after the first m.V rows.

	// calculate a score for every possible next word
	for j := 0; j < m.V; j++ {
		// start with this word's bias then add its weights for btoh context words
		z[j] = m.Bias[j] + m.W[row0+j] + m.W[row1+j]
	}
	// Loop through every word in the vocabulary and calculate its score.
	// For each word, add its bias to the weights from both context words.
	// row0 and row1 point to the start of their respective rows in the
	// flattened weight array, while j selects the column for the current word.
	// Store each score in z, which holds one score per possible next word.
	return z // return score for all possible next words
}

// The softmax function is a mathematical operation that converts a vector of raw, unconstrained numbers (called logits) into a clean probability distribution that sums up to 1

// The function finds the largest score, subtracts it from every score for numerical stability, and applies the exponential function to each result. It adds up those exponential values, divides each one by the total, and returns the resulting probabilities
func Softmax(z []float32) []float32 {
	// largest score in input slice
	maxZ := z[0]
	for _, v := range z[1:] {
		if v > maxZ {
			maxZ = v
		}
	}
	// one output slot for each input score
	p := make([]float32, len(z))

	// keep track of total of all exponential values
	var sum float32

	// convert each score into an exponential value
	// subctracting the largest score helps prevent overlfow
	for j, v := range z {
		e := float32(math.Exp(float64(v - maxZ)))

		// store value temporarily and add it to total
		p[j] = e
		sum += e
	}
	// divide each value by total to get probabilities
	for j := range p {
		p[j] /= sum
	}
	// return one probability for each input score
	return p
}

// Forward takes two context words and returns a probability for every possible next word.
func (m *Model) Forward(c0, c1 int) []float32 {
	// First calculate a score for each candidate word,
	// then convert those scores into probabilities.
	return Softmax(m.Logits(c0, c1))
}
