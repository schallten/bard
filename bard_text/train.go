package main

import (
	"fmt"
	"math"
)

// Step runs one SGD update for a single example and returns the loss
// measured before the update.
func (m *Model) Step(c0, c1, target int, lr float32) float32 {
	p := m.Forward(c0, c1)
	loss := CrossEntropy(p, target)

	row0 := c0 * m.V
	row1 := (m.V + c1) * m.V

	for j, pj := range p {
		// correction = guess minus truth, in one line.
		d := pj
		if j == target {
			d -= 1
		}
		m.Bias[j] -= lr * d
		m.W[row0+j] -= lr * d
		m.W[row1+j] -= lr * d
	}
	return loss
}

// Train sweeps the examples epochs times, reporting average loss per pass.
func (m *Model) Train(examples []Example, epochs int, lr float32) {
	for e := 1; e <= epochs; e++ {
		var total float64
		for _, ex := range examples {
			total += float64(m.Step(ex.Context[0], ex.Context[1], ex.Target, lr))
		}
		avg := total / float64(len(examples))
		fmt.Printf("epoch %2d   loss %.4f   perplexity %.1f\n", e, avg, math.Exp(avg))
	}
}

// AverageLoss measures loss over all examples without changing the model.
func (m *Model) AverageLoss(examples []Example) float32 {
	var total float64
	for _, ex := range examples {
		total += float64(CrossEntropy(m.Forward(ex.Context[0], ex.Context[1]), ex.Target))
	}
	return float32(total / float64(len(examples)))
}

// GradientCheck compares the analytic correction for two weights against a
// numerical one. Run it after some training, when logits are large enough to
// stand above float32 rounding noise.
func GradientCheck(m *Model, ex Example, eps float32) {
	c0, c1, target := ex.Context[0], ex.Context[1], ex.Target
	p := m.Forward(c0, c1)
	row0 := c0 * m.V

	loss := func() float32 {
		return CrossEntropy(m.Forward(c0, c1), target)
	}

	for _, j := range []int{target, (target + 1) % m.V} {
		analytic := p[j]
		if j == target {
			analytic -= 1
		}

		idx := row0 + j
		old := m.W[idx]
		m.W[idx] = old + eps
		up := loss()
		m.W[idx] = old - eps
		down := loss()
		m.W[idx] = old

		fmt.Printf("W[%d][%d]  analytic %+.6f   numeric %+.6f\n",
			c0, j, analytic, (up-down)/(2*eps))
	}
}
