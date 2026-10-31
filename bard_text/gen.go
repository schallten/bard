package main

import (
	"math/rand"
	"strings"
)

// Sample picks the next word from the model's scores, with temperature.
// 1.0 is the trained distribution; below 1 is safer, above 1 is wilder.
func (m *Model) Sample(c0, c1 int, temperature float32, rng *rand.Rand) int {
	z := m.Logits(c0, c1)
	if temperature != 1 {
		for j := range z {
			z[j] /= temperature
		}
	}
	p := Softmax(z)

	r := rng.Float32()
	var acc float32
	for j, pj := range p {
		acc += pj
		if r < acc {
			return j
		}
	}
	return len(p) - 1
}

// Generate continues a two-word start for n more words.
func Generate(m *Model, tk *Tokenizer, start [2]int, n int, temperature float32, seed int64) string {
	rng := rand.New(rand.NewSource(seed))
	ids := []int{start[0], start[1]}
	for i := 0; i < n; i++ {
		next := m.Sample(ids[len(ids)-2], ids[len(ids)-1], temperature, rng)
		ids = append(ids, next)
	}

	var b strings.Builder
	for i, id := range ids {
		if i > 0 {
			b.WriteByte(' ')
		}
		s, _ := tk.Decode(id)
		b.WriteString(s)
	}
	return b.String()
}
